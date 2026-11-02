# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""sk-release smoke test against a running serve-sparse.sh server.
usage: smoke.py PORT OUT [NTOK]
  1. short prompt (no document)
  2. needle at 30% depth of a ~NTOK-token code document (cold prefill)
  3. follow-up on the same document: quote the line after a unique line at 60% depth (prefix-cache hit, fast TTFT)
  4. second needle at 75% depth (another follow-up)
  5. 256-token decode-speed request on the same document"""
import glob, json, os, sys, time, urllib.request
from transformers import AutoTokenizer
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(f"{REPO}/models/Qwen3.8-27B-W4A16-AutoRound-fast")
PORT, OUT = int(sys.argv[1]), sys.argv[2]
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 128000
KEY = os.environ.get("VLLM_API_KEY") or open(os.environ.get("API_KEY_FILE", "api_key.txt")).read().strip()
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
base = f"{REPO}/venv/lib/python3.12/site-packages/vllm"
parts, n = [], 0
for f in sorted(glob.glob(f"{base}/**/*.py", recursive=True)):
    src = open(f).read()
    if len(src) < 2000: continue
    p = f"# ===== FILE: {os.path.relpath(f, base)} =====\n{src}\n"
    t = len(tok.encode(p))
    if n + t > NTOK: continue
    parts.append(p); n += t
    if n > NTOK - 3000: break
doc = "".join(parts)
N1 = "\n# The special magic number for the project codenamed BLUE-HERON is 7418-2295.\n"
c = doc.find("\n", int(len(doc) * 0.3)) + 1; doc = doc[:c] + N1 + doc[c:]
N2 = "\n# The secret passphrase for the vault codenamed GREEN-FALCON is amber-quartz-41.\n"
c = doc.find("\n", int(len(doc) * 0.75)) + 1; doc = doc[:c] + N2 + doc[c:]
lines = doc.split("\n"); cnt = {}
for ln in lines: cnt[ln.strip()] = cnt.get(ln.strip(), 0) + 1
i = int(len(lines) * 0.6)
while not (len(lines[i].strip()) > 30 and len(lines[i+1].strip()) > 20 and cnt[lines[i].strip()] == 1
           and cnt[lines[i+1].strip()] == 1 and not lines[i].strip().startswith("#")): i += 1
a, b = lines[i].strip(), lines[i+1].strip()
Q = [("short", None, "What is the capital of France? Answer in one word.", "Paris", 16),
     ("short_speed", None, "Write a detailed technical explanation of how a hash map works, with examples.", None, 256),
     ("needle", doc, "What is the special magic number for the project codenamed BLUE-HERON? Answer with the number only.", "7418-2295", 30),
     ("followup_code", doc, "In the code above, find the line\n" + a + "\nQuote exactly the line that comes immediately after it. Answer with that single line only.", b, 80),
     ("followup_needle2", doc, "What is the secret passphrase for the vault codenamed GREEN-FALCON? Answer with the passphrase only.", "amber-quartz-41", 30),
     ("speed", doc, "Explain in detail what the last file above does.", None, 256)]
res = {"doc_tokens": len(tok.encode(doc)), "ntok": NTOK}
print("doc tokens", res["doc_tokens"], flush=True)
for name, d, q, want, mx in Q:
    content = q if d is None else d + "\n\n" + q
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": mx, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}
    if name in ("speed", "short_speed"): body["min_tokens"] = mx
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); tf = None; text = ""; usage = None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                j = json.loads(line[5:])
                if j.get("usage"): usage = j["usage"]
                for ch in j.get("choices", []):
                    cc = ch.get("delta", {}).get("content")
                    if cc: tf = tf or time.time(); text += cc
    except Exception as e:  # noqa: BLE001
        res[name] = {"error": repr(e)}; print(f"[{name}] ERROR {e!r}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1); continue
    te = time.time(); nout = usage["completion_tokens"]
    ok = (want in text) if want else (nout >= mx and len(set(text.split())) > 20)
    res[name] = {"prompt_tokens": usage["prompt_tokens"], "out": nout, "ttft_s": round(tf - t0, 2),
                 "decode_tps": round((nout - 1) / (te - tf), 1) if nout > 1 else None, "ok": ok,
                 "text": text[:300], "want": want}
    print(f"[{name}] {usage['prompt_tokens']} in, {nout} out, ttft {tf-t0:.2f}s, decode {res[name]['decode_tps']} tok/s,"
          f" ok={ok} text={text[:100]!r}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("SMOKE", "PASS" if all(res.get(k, {}).get("ok") for k, *_ in Q) else "FAIL", flush=True)
