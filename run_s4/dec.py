# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Streams-per-card decode harness (run_s4/dec.py PORT OUT NTOK KS [--log SERVER_LOG] [--ans N] [--rounds R]).
For each k in KS: users 0..k-1 active (new users cold-prefill their own document: needle, serial), then R rounds of
  [code (recall), explain N tokens all k at once, copy N tokens all k at once], then needle2.
Records per request: prompt, cached, ttft, ntok, tok_s, ok; per concurrent batch:
  combined_tok_s  = all tokens / (last end - first first-token)   (old sweep metric, includes others' follow-up prefill)
  steady_tok_s    = tokens produced inside [max first-token, min end] / its length (all k decoding, no prefill in it)
  plus deltas of the server's spec-decode counters (acceptance) and tier stats (hot hit/miss/resident, staged blocks,
  steps) over the batch, PCIe bytes per step from hot misses (2080 B per row-layer) and prefill staging.
Docs: vLLM sources (per-user shuffled), needles at 30%/75%, substring-unique code line at 60% (as run_250k/sweep.py)."""
import argparse, glob, json, os, random, re, threading, time, urllib.request
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("port", type=int); ap.add_argument("out"); ap.add_argument("ntok", type=int); ap.add_argument("ks")
ap.add_argument("--log", default=None); ap.add_argument("--ans", type=int, default=384)
ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--qs", default="code,explain,copy")
ap.add_argument("--uoff", type=int, default=0)  # document seed offset (distinct docs per server)
ap.add_argument("--repo", default=os.environ.get("REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
A = ap.parse_args()
REPO = A.repo
tok = AutoTokenizer.from_pretrained(f"{REPO}/models/Qwen3.8-27B-W4A16-AutoRound-fast")
PORT, OUT, NTOK = A.port, A.out, A.ntok
KS = [int(x) for x in A.ks.split(",")]
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
KEY = os.environ.get("VLLM_API_KEY") or open(os.environ.get("API_KEY_FILE", "api_key.txt")).read().strip()
base = os.environ.get("VLLM_SRC", f"{REPO}/venv/lib/python3.12/site-packages/vllm")  # same corpus as run_250k
FILES = sorted(f for f in glob.glob(f"{base}/**/*.py", recursive=True) if os.path.getsize(f) >= 2000)
LOCK = threading.Lock(); OUTF = open(OUT, "a")
BLOCK_LAYER_BYTES = 864 * 2080


def log(rec):
    rec["t"] = round(time.time(), 1)
    with LOCK:
        OUTF.write(json.dumps(rec) + "\n"); OUTF.flush(); print(json.dumps(rec)[:400], flush=True)


def metrics():
    try:
        txt = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics",
                                     headers={"Authorization": f"Bearer {KEY}"}), timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for ln in txt.splitlines():
        if ln.startswith("#"): continue
        m = re.match(r"(vllm:(spec_decode_num_(accepted|draft)_tokens|spec_decode_num_drafts|generation_tokens|"
                     r"prompt_tokens|num_preemptions)[a-z_]*)(\{[^}]*\})? ([0-9.e+]+)", ln)
        if m: out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(5))
    return out


def tier_stats():
    if not A.log: return None
    last = None
    with open(A.log, "rb") as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4_000_000))
        for ln in f.read().decode(errors="ignore").splitlines():
            if "tier: stats" in ln: last = ln
    if not last: return None
    m = re.search(r"'steps': (\d+)", last); h = re.search(r"hot\[hit,miss,resident\]=\[(\d+), (\d+), (\d+)\]", last)
    p = re.search(r"'prefill_staged_blocks': (\d+)", last)
    return {"steps": int(m.group(1)) if m else 0, "hit": int(h.group(1)) if h else 0, "miss": int(h.group(2)) if h else 0,
            "res": int(h.group(3)) if h else 0, "staged": int(p.group(1)) if p else 0}


def build(u):
    u = u + A.uoff
    files = list(FILES); random.Random(1000 + u).shuffle(files)
    parts, n = [], 0
    for f in files:
        p = f"# ===== FILE: {os.path.relpath(f, base)} =====\n{open(f).read()}\n"
        t = len(tok.encode(p))
        if n + t > NTOK: continue
        parts.append(p); n += t
        if n > NTOK - 3000: break
    doc = "".join(parts)
    magic, pw = f"{4000 + 37 * u}-{7000 + 113 * u}", f"word{u}-quartz-{11 + u}"
    n1 = f"\n# The special magic number for the project codenamed U{u}-HERON is {magic}.\n"
    c = doc.find("\n", int(len(doc) * 0.3)) + 1; doc = doc[:c] + n1 + doc[c:]
    n2 = f"\n# The secret passphrase for the vault codenamed U{u}-FALCON is {pw}.\n"
    c = doc.find("\n", int(len(doc) * 0.75)) + 1; doc = doc[:c] + n2 + doc[c:]
    lines = doc.split("\n"); cnt = {}
    for ln in lines: cnt[ln.strip()] = cnt.get(ln.strip(), 0) + 1
    i = int(len(lines) * 0.6)
    while not (len(lines[i].strip()) > 30 and len(lines[i + 1].strip()) > 20 and cnt[lines[i].strip()] == 1
               and cnt[lines[i + 1].strip()] == 1 and not lines[i].strip().startswith("#")
               and doc.count(lines[i].strip()) == 1 and doc.count(lines[i + 1].strip()) == 1): i += 1
    a, b = lines[i].strip(), lines[i + 1].strip()
    qs = {"needle": (f"What is the special magic number for the project codenamed U{u}-HERON? Answer with the number only.", magic, 30),
          "needle2": (f"What is the secret passphrase for the vault codenamed U{u}-FALCON? Answer with the passphrase only.", pw, 30),
          "code": ("In the code above, find the line\n" + a + "\nQuote exactly the line that comes immediately after it. "
                   "Answer with that single line only.", b, 80),
          "explain": ("Explain in detail what the last file in the code above does, function by function.", None, A.ans),
          "copy": ("Copy the last function of the last file in the code above verbatim, with no commentary.", None, A.ans)}
    return doc, qs


def ask(u, docs, name, extra):
    doc, qs = docs[u]; q, want, mx = qs[name]
    body = {"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\n" + q}], "temperature": 0,
            "max_tokens": mx, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}
    if mx > 100: body["ignore_eos"] = True
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); tf = None; text = ""; usage = None; marks = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                j = json.loads(line[5:])
                if j.get("usage"): usage = j["usage"]
                for ch in j.get("choices", []):
                    c = ch.get("delta", {}).get("content")
                    if c:
                        now = time.time(); tf = tf or now; text += c; marks.append((now, len(text)))
    except Exception as e:  # noqa: BLE001
        rec = {"user": u, "q": name, "error": repr(e), "ok": False, **extra}; log(rec); return rec
    te = time.time(); tf = tf or te; ct = usage["completion_tokens"]
    rec = {"user": u, "q": name, "prompt": usage["prompt_tokens"],
           "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, "ttft": round(tf - t0, 2),
           "ntok": ct, "tok_s": round((ct - 1) / (te - tf), 1) if ct > 1 and te > tf else None,
           "t0": t0, "tf": tf, "te": te, "ok": (want in text) if want else None, "text": text[:100], **extra}
    log(rec)
    rec["_marks"] = marks; rec["_chars"] = max(1, len(text))
    return rec


def tokens_between(r, a, b):
    """Approximate tokens of request r emitted in [a, b] (chars scaled to completion tokens)."""
    c = [n for t, n in r["_marks"] if a <= t <= b]
    if not c: return 0.0
    c0 = max([n for t, n in r["_marks"] if t < a] or [0])
    return (max(c) - c0) * r["ntok"] / r["_chars"]


def batch(users, docs, name, extra, stagger=0.2):
    res = {}
    m0, s0 = metrics(), tier_stats()
    def run(u): res[u] = ask(u, docs, name, extra)
    ts = []
    for u in users:
        t = threading.Thread(target=run, args=(u,)); t.start(); ts.append(t); time.sleep(stagger)
    for t in ts: t.join()
    time.sleep(1.0)
    m1, s1 = metrics(), tier_stats()
    rs = [res[u] for u in users if "te" in res[u]]
    if len(rs) == len(users) and all(r["ntok"] > 50 for r in rs):
        span = max(r["te"] for r in rs) - min(r["tf"] for r in rs)
        a, b = max(r["tf"] for r in rs), min(r["te"] for r in rs)
        steady = round(sum(tokens_between(r, a, b) for r in rs) / (b - a), 1) if b - a > 2 else None
        acc = None
        k_acc, k_dr = "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total"
        if k_acc in m1 and k_dr in m1 and m1[k_dr] > m0.get(k_dr, 0):
            acc = round(1 + (m1[k_acc] - m0.get(k_acc, 0)) / (m1[k_dr] - m0.get(k_dr, 0)), 2)
        rec = {"kind": "combined", "q": name, "combined_tok_s": round(sum(r["ntok"] - 1 for r in rs) / span, 1),
               "steady_tok_s": steady, "steady_window_s": round(b - a, 1), "per_stream": [r["tok_s"] for r in rs],
               "ttft": [r["ttft"] for r in rs], "tokens_per_step": acc, **extra}
        if s0 and s1:
            d = {k: s1[k] - s0[k] for k in s1}
            look = d["hit"] + d["miss"]
            rec.update({"tier_steps": d["steps"], "hot_miss_rate": round(d["miss"] / look, 4) if look else None,
                        "resident_frac": round(d["res"] / (look + d["res"]), 4) if look + d["res"] else None,
                        "miss_MB_per_step": round(d["miss"] * 2080 / max(1, d["steps"]) / 1e6, 2),
                        "staged_GB": round(d["staged"] * BLOCK_LAYER_BYTES / 1e9, 2)})
        log(rec)
    return res


docs = {}
for k in KS:
    for u in range(k):
        if u not in docs:
            docs[u] = build(u)
            log({"kind": "doc", "user": u, "doc_tokens": len(tok.encode(docs[u][0])), "k": k})
    new = [u for u in range(k) if not docs[u][1].get("_warm")]
    if new:
        batch(new, docs, "needle", {"k": k, "phase": "cold"}, stagger=5)
        for u in new: docs[u][1]["_warm"] = True
    for r in range(A.rounds):
        for name in A.qs.split(","):
            batch(list(range(k)), docs, name, {"k": k, "round": r})
    batch(list(range(k)), docs, "needle2", {"k": k})
log({"kind": "sweep_end"})
