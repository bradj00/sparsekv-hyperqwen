# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""250K tier tests (run_250k). Documents: real code (vLLM sources of the release venv), A forward / B reverse order,
each ~NTOK tokens with two needles (30%, 75%) and two unique code lines (45%, 60%).
usage: chat250.py PORT OUT MODE [args]
  single           user A: cold needle, code, needle2, code2, explain, copy (follow-ups hit the host cache)
  dual             A and B concurrently: cold needle (B after B_DELAY), then rounds code/needle2/code2/explain/copy
                   both at once. PRIME_MT1=1: the cold turn is "Reply with OK." with max_tokens 1 (the re-prefill repro)
  short            4K and 16K documents: explain/copy decode tok/s (2 reps)
  soak MINUTES     two long docs, turns alternate/concurrent (random question), a short request every SHORT_EVERY s;
                   stops at MINUTES or after MAX_ERR wrong answers/errors
env: NTOK (248000), B_DELAY (20), EXPLAIN_TOK / COPY_TOK (384)"""
import glob, json, os, random, sys, threading, time, urllib.request
from transformers import AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(f"{REPO}/models/Qwen3.8-27B-W4A16-AutoRound-fast")
PORT, OUT, MODE = int(sys.argv[1]), sys.argv[2], sys.argv[3]
NTOK = int(os.environ.get("NTOK", "248000"))
KEY = os.environ.get("VLLM_API_KEY") or open(os.environ.get("API_KEY_FILE", "api_key.txt")).read().strip()
base = f"{REPO}/venv/lib/python3.12/site-packages/vllm"
FILES = sorted(glob.glob(f"{base}/**/*.py", recursive=True))
EXPLAIN_TOK = int(os.environ.get("EXPLAIN_TOK", "384"))
COPY_TOK = int(os.environ.get("COPY_TOK", "384"))
LOCK = threading.Lock()
OUTF = open(OUT, "a")


def log(rec):
    rec["t"] = round(time.time(), 1)
    with LOCK:
        OUTF.write(json.dumps(rec) + "\n"); OUTF.flush()
        print(json.dumps(rec)[:400], flush=True)


def build(ntok, rev, tag, magic, pw, skip=0):
    parts, n = [], 0
    files = list(reversed(FILES) if rev else FILES)[skip:]
    for f in files:
        src = open(f).read()
        if len(src) < 2000: continue
        p = f"# ===== FILE: {os.path.relpath(f, base)} =====\n{src}\n"
        t = len(tok.encode(p))
        if n + t > ntok: continue
        parts.append(p); n += t
        if n > ntok - 3000: break
    doc = "".join(parts)
    n1 = f"\n# The special magic number for the project codenamed {tag}-HERON is {magic}.\n"
    c = doc.find("\n", int(len(doc) * 0.3)) + 1; doc = doc[:c] + n1 + doc[c:]
    n2 = f"\n# The secret passphrase for the vault codenamed {tag}-FALCON is {pw}.\n"
    c = doc.find("\n", int(len(doc) * 0.75)) + 1; doc = doc[:c] + n2 + doc[c:]
    lines = doc.split("\n"); cnt = {}
    for ln in lines: cnt[ln.strip()] = cnt.get(ln.strip(), 0) + 1

    def codeq(frac):
        i = int(len(lines) * frac)
        while not (len(lines[i].strip()) > 30 and len(lines[i + 1].strip()) > 20 and cnt[lines[i].strip()] == 1
                   and cnt[lines[i + 1].strip()] == 1 and not lines[i].strip().startswith("#")
                   and doc.count(lines[i].strip()) == 1 and doc.count(lines[i + 1].strip()) == 1): i += 1
        a, b = lines[i].strip(), lines[i + 1].strip()
        return ("In the code above, find the line\n" + a + "\nQuote exactly the line that comes immediately after it. "
                "Answer with that single line only.", b)
    q1, w1 = codeq(0.6); q2, w2 = codeq(0.45)
    qs = {"needle": (f"What is the special magic number for the project codenamed {tag}-HERON? Answer with the number only.", magic, 30),
          "code": (q1, w1, 80), "code2": (q2, w2, 80),
          "needle2": (f"What is the secret passphrase for the vault codenamed {tag}-FALCON? Answer with the passphrase only.", pw, 30),
          "explain": ("Explain in detail what the last file in the code above does, function by function.", None, EXPLAIN_TOK),
          "copy": ("Copy the last function of the last file in the code above verbatim, with no commentary.", None, COPY_TOK),
          "ok": ("Reply with OK.", None, 1)}
    return doc, qs


def ask(user, doc, q, want, mx, label, extra=None):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": doc + "\n\n" + q}], "temperature": 0,
            "max_tokens": mx, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}
    if mx > 100:
        body["ignore_eos"] = True
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); tf = None; text = ""; usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                j = json.loads(line[5:])
                if j.get("usage"): usage = j["usage"]
                for ch in j.get("choices", []):
                    c = ch.get("delta", {}).get("content")
                    if c: tf = tf or time.time(); text += c
    except Exception as e:  # noqa: BLE001
        rec = {"user": user, "q": label, "error": repr(e), "ok": False}; rec.update(extra or {}); log(rec); return rec
    te = time.time(); tf = tf or te
    ct = usage["completion_tokens"]
    rec = {"user": user, "q": label, "prompt": usage["prompt_tokens"],
           "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "ttft": round(tf - t0, 2),
           "ntok": ct, "dec_s": round(te - tf, 2), "tok_s": round((ct - 1) / (te - tf), 1) if ct > 1 and te > tf else None,
           "t0": t0, "tf": tf, "te": te, "ok": (want in text) if want else None, "text": text[:120]}
    rec.update(extra or {}); log(rec); return rec


def turn(user, docs, name, extra=None):
    doc, qs = docs[user]
    q, want, mx = qs[name]
    return ask(user, doc, q, want, mx, name, extra)


def pair(docs, name, delay=0.3, extra=None):
    res = {}
    def run(u): res[u] = turn(u, docs, name, extra)
    ts = [threading.Thread(target=run, args=(u,)) for u in "AB"]
    ts[0].start(); time.sleep(delay); ts[1].start()
    for t in ts: t.join()
    a, b = res["A"], res["B"]
    if a.get("ntok", 0) > 50 and b.get("ntok", 0) > 50:  # combined decode throughput over the overlapping window
        lo, hi = max(a["tf"], b["tf"]), min(a["te"], b["te"])
        span = max(a["te"], b["te"]) - min(a["tf"], b["tf"])
        log({"kind": "combined", "q": name, "combined_tok_s": round((a["ntok"] + b["ntok"] - 2) / span, 1),
             "overlap_s": round(max(0, hi - lo), 1), **(extra or {})})
    return res


def docs2():
    return {"A": build(NTOK, False, "BLUE", "7418-2295", "amber-quartz-41"),
            "B": build(NTOK, True, "RED", "5530-8817", "cobalt-ridge-07")}


if MODE == "single":
    d = {"A": build(NTOK, False, "BLUE", "7418-2295", "amber-quartz-41")}
    log({"kind": "doc", "A": len(tok.encode(d["A"][0]))})
    for name in ("needle", "code", "needle2", "code2", "explain", "copy", "explain"):
        turn("A", d, name, {"mode": "single"})
elif MODE == "dual":
    d = docs2()
    log({"kind": "doc", "A": len(tok.encode(d["A"][0])), "B": len(tok.encode(d["B"][0]))})
    first = "ok" if os.environ.get("PRIME_MT1") == "1" else "needle"
    pair(d, first, float(os.environ.get("B_DELAY", "20")), {"mode": "dual", "phase": "cold"})
    for name in os.environ.get("ROUNDS", "code,needle2,code2,explain,copy,needle").split(","):
        pair(d, name, 0.3, {"mode": "dual"})
elif MODE == "short":
    for L in (4000, 16000):
        dd = {"S": build(L, False, "GREEN", "1111-2222", "x"), }
        for rep in range(2):
            for name in ("explain", "copy"):
                turn("S", dd, name, {"mode": "short", "len": L, "rep": rep})
elif MODE == "soak":
    minutes = float(sys.argv[4]); t_end = time.time() + minutes * 60
    d = docs2()
    short = {"S": build(6000, True, "GREEN", "1111-2222", "plum-delta-3", skip=40)}
    log({"kind": "doc", "A": len(tok.encode(d["A"][0])), "B": len(tok.encode(d["B"][0]))})
    errs = [0]; maxerr = int(os.environ.get("MAX_ERR", "3")); stop = threading.Event()
    rng = random.Random(int(os.environ.get("SEED", "1")))
    names = ["needle", "code", "needle2", "explain", "copy"]  # code2 (B: near-duplicate block, deterministic miss) left out

    def check(r):
        if r.get("error") or r.get("ok") is False:
            errs[0] += 1
            if errs[0] >= maxerr: stop.set()

    def shorts():
        every = float(os.environ.get("SHORT_EVERY", "45"))
        while not stop.is_set() and time.time() < t_end:
            check(turn("S", short, rng.choice(["needle", "needle2", "explain"]), {"mode": "soak"}))
            stop.wait(every)
    th = threading.Thread(target=shorts, daemon=True); th.start()
    k = 0
    while not stop.is_set() and time.time() < t_end:
        if k % 3 == 2:   # alternating: one user at a time
            u = "AB"[(k // 3) % 2]; check(turn(u, d, rng.choice(names), {"mode": "soak", "k": k}))
        else:            # concurrent
            for r in pair(d, rng.choice(names), 0.3, {"mode": "soak", "k": k}).values(): check(r)
        k += 1
    stop.set(); th.join(timeout=600)
    log({"kind": "soak_end", "turns": k, "errors": errs[0], "stopped_early": time.time() < t_end})
