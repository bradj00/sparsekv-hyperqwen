# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Stream E capacity test: one request whose context exceeds the GPU KV pool (SPARSE_TIER_SPILL=1).
usage: bigctx.py PORT OUT NTOK   -- code corpus of ~NTOK tokens with a needle at 30% depth; asks (1) the needle,
(2) the line following a unique line at 60% depth (same document, prefix-cache hit), streams both."""
import json, os, sys, time, urllib.request, glob
from transformers import AutoTokenizer
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(f"{WT}/models/Qwen3.8-27B-W4A16-AutoRound-fast")
PORT, OUT, NTOK = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
KEY = os.environ.get("VLLM_API_KEY") or open(os.environ.get("API_KEY_FILE", "api_key.txt")).read().strip()
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
base = f"{WT}/venv/lib/python3.12/site-packages/vllm"
files = sorted(glob.glob(f"{base}/**/*.py", recursive=True))
if os.environ.get("DOC_ORDER") == "rev":  # a different document (second user)
    files = files[::-1]
parts, n = [], 0
for f in files:
    src = open(f).read()
    if len(src) < 2000: continue
    p = f"# ===== FILE: {os.path.relpath(f, base)} =====\n{src}\n"
    t = len(tok.encode(p))
    if n + t > NTOK: continue
    parts.append(p); n += t
    if n > NTOK - 3000: break
doc = "".join(parts)
NEEDLE = "\n# The special magic number for the project codenamed BLUE-HERON is 7418-2295.\n"
cut = doc.find("\n", int(len(doc) * 0.3)) + 1
doc = doc[:cut] + NEEDLE + doc[cut:]
NEEDLE2 = "\n# The secret passphrase for the vault codenamed GREEN-FALCON is amber-quartz-41.\n"
cut2 = doc.find("\n", int(len(doc) * 0.75)) + 1
doc = doc[:cut2] + NEEDLE2 + doc[cut2:]
lines = doc.split("\n"); cnt = {}
for ln in lines: cnt[ln.strip()] = cnt.get(ln.strip(), 0) + 1
i = int(len(lines) * 0.6)
while not (len(lines[i].strip()) > 30 and len(lines[i+1].strip()) > 20 and cnt[lines[i].strip()] == 1
           and cnt[lines[i+1].strip()] == 1 and not lines[i].strip().startswith("#")): i += 1
a, b = lines[i].strip(), lines[i+1].strip()
qs = [("needle", "What is the special magic number for the project codenamed BLUE-HERON? Answer with the number only.", "7418-2295", 30),
      ("code", "In the code above, find the line\n" + a + "\nQuote exactly the line that comes immediately after it. Answer with that single line only.", b, 80),
      ("needle2", "What is the secret passphrase for the vault codenamed GREEN-FALCON? Answer with the passphrase only.", "amber-quartz-41", 30),
      ("speed", "Explain in detail what the last file above does.", None, 384)]
res = {"doc_tokens": len(tok.encode(doc))}
print("doc tokens", res["doc_tokens"], flush=True)
WANT = os.environ.get("QS", "needle,code,speed").split(",")
QD = {x[0]: x for x in qs}
for name in WANT:
    name, q, want, mx = QD[name]
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": doc + "\n\n" + q}], "temperature": 0,
            "max_tokens": mx, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}
    if name == "speed": body["min_tokens"] = mx
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); tf = None; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=1500) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    tf = tf or time.time(); text += c
    te = time.time(); nout = usage["completion_tokens"]
    tf = tf or te  # empty answer
    ok = (want in text) if want else None
    res[name] = {"prompt_tokens": usage["prompt_tokens"], "out": nout, "ttft": round(tf - t0, 1),
                 "decode_tps": round((nout - 1) / (te - tf), 1) if nout > 1 else None, "ok": ok, "text": text[:300], "want": want}
    print(f"[{name}] {usage['prompt_tokens']} in, {nout} out, ttft {tf-t0:.1f}s, decode {res[name]['decode_tps']} tok/s, ok={ok} text={text[:120]!r}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    if os.environ.get("RESET_BETWEEN") == "1":  # force-evict the GPU prefix cache (vLLM dev endpoint)
        rq = urllib.request.Request(f"http://127.0.0.1:{PORT}/reset_prefix_cache", data=b"", method="POST",
                                    headers={"Authorization": f"Bearer {KEY}"})
        for _ in range(10):
            ok_reset = json.loads(urllib.request.urlopen(rq, timeout=60).read()).get("success")
            if ok_reset: break
            time.sleep(1)
        res.setdefault("resets", []).append(bool(ok_reset))
        print(f"[reset_prefix_cache] success={ok_reset}", flush=True)
print("CLIENT DONE")
