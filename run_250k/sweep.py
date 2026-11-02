# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Streams-per-card sweep on one tier server (run_250k/sweep.py PORT OUT NTOK KS [ROUNDS]).
KS e.g. "1,2,3": for each k, users 0..k-1 are active (new users cold-prefill first: needle, staggered), then ROUNDS
rounds of [code, explain, copy] with all k users concurrently, then needle2. Distinct documents per user (vLLM sources
in a per-user shuffled file order), needles at 30%/75%, a substring-unique code line at 60%.
Records go to OUT (jsonl): per request (user, q, k, prompt, cached, ttft, ntok, tok_s, ok) and per concurrent batch
(combined decode tok/s over the span of the decode windows)."""
import glob, json, os, random, sys, threading, time, urllib.request
from transformers import AutoTokenizer

REPO = os.environ.get("REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tok = AutoTokenizer.from_pretrained(f"{REPO}/models/Qwen3.8-27B-W4A16-AutoRound-fast")
PORT, OUT, NTOK = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
KS = [int(x) for x in sys.argv[4].split(",")]
ROUNDS = int(sys.argv[5]) if len(sys.argv) > 5 else 2
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
KEY = os.environ.get("VLLM_API_KEY") or open(os.environ.get("API_KEY_FILE", "api_key.txt")).read().strip()
base = f"{REPO}/venv/lib/python3.12/site-packages/vllm"
FILES = sorted(f for f in glob.glob(f"{base}/**/*.py", recursive=True) if os.path.getsize(f) >= 2000)
LOCK = threading.Lock(); OUTF = open(OUT, "a")


def log(rec):
    rec["t"] = round(time.time(), 1)
    with LOCK:
        OUTF.write(json.dumps(rec) + "\n"); OUTF.flush(); print(json.dumps(rec)[:300], flush=True)


def build(u):
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
          "explain": ("Explain in detail what the last file in the code above does, function by function.", None, 384),
          "copy": ("Copy the last function of the last file in the code above verbatim, with no commentary.", None, 384)}
    return doc, qs


def ask(u, docs, name, extra):
    doc, qs = docs[u]; q, want, mx = qs[name]
    body = {"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\n" + q}], "temperature": 0,
            "max_tokens": mx, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}
    if mx > 100: body["ignore_eos"] = True
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); tf = None; text = ""; usage = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                j = json.loads(line[5:])
                if j.get("usage"): usage = j["usage"]
                for ch in j.get("choices", []):
                    c = ch.get("delta", {}).get("content")
                    if c: tf = tf or time.time(); text += c
    except Exception as e:  # noqa: BLE001
        rec = {"user": u, "q": name, "error": repr(e), "ok": False, **extra}; log(rec); return rec
    te = time.time(); tf = tf or te; ct = usage["completion_tokens"]
    rec = {"user": u, "q": name, "prompt": usage["prompt_tokens"],
           "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, "ttft": round(tf - t0, 2),
           "ntok": ct, "tok_s": round((ct - 1) / (te - tf), 1) if ct > 1 and te > tf else None,
           "t0": t0, "tf": tf, "te": te, "ok": (want in text) if want else None, "text": text[:100], **extra}
    log(rec); return rec


def batch(users, docs, name, extra, stagger=0.2):
    res = {}
    def run(u): res[u] = ask(u, docs, name, extra)
    ts = []
    for u in users:
        t = threading.Thread(target=run, args=(u,)); t.start(); ts.append(t); time.sleep(stagger)
    for t in ts: t.join()
    rs = [res[u] for u in users if "te" in res[u]]
    if len(rs) == len(users) and all(r["ntok"] > 50 for r in rs):
        span = max(r["te"] for r in rs) - min(r["tf"] for r in rs)
        log({"kind": "combined", "q": name, "combined_tok_s": round(sum(r["ntok"] - 1 for r in rs) / span, 1),
             "per_stream": [r["tok_s"] for r in rs], **extra})
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
    for r in range(ROUNDS):
        for name in ("code", "explain", "copy"):
            batch(list(range(k)), docs, name, {"k": k, "round": r})
    batch(list(range(k)), docs, "needle2", {"k": k})
log({"kind": "sweep_end"})
