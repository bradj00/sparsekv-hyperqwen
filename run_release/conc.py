# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""sk-release 2-user tier test (stream E's conc test, release build): user A ~177K tokens and user B ~120K tokens on
different documents (A = vLLM files in forward order, B = reverse order, each with its own needles), B starts B_DELAY s
after A. Phases (barrier between phases, GPU prefix cache wiped with POST /reset_prefix_cache before phases 2 and 3,
so follow-ups must come back through the host KV blocks + host GDN checkpoints):
  1. both: needle at 30% depth (cold prefill; concurrent, small pool -> spill / preemption)
  2. both: quote the line after a unique code line at 60% depth
  3. both: second needle at 75% depth
usage: conc.py PORT OUT [NTOK_A NTOK_B]   (needs VLLM_SERVER_DEV_MODE=1 on the server)"""
import glob, json, os, sys, threading, time, urllib.request
from transformers import AutoTokenizer
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(f"{REPO}/models/Qwen3.8-27B-W4A16-AutoRound-fast")
PORT, OUT = int(sys.argv[1]), sys.argv[2]
NA = int(sys.argv[3]) if len(sys.argv) > 3 else 177000
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 120000
B_DELAY = float(os.environ.get("B_DELAY", "60"))
KEY = os.environ.get("VLLM_API_KEY") or open(os.environ.get("API_KEY_FILE", "api_key.txt")).read().strip()
base = f"{REPO}/venv/lib/python3.12/site-packages/vllm"
FILES = [f for f in sorted(glob.glob(f"{base}/**/*.py", recursive=True))]


def build(ntok, rev, tag, magic, pw):
    parts, n = [], 0
    for f in (reversed(FILES) if rev else FILES):
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
    i = int(len(lines) * 0.6)
    while not (len(lines[i].strip()) > 30 and len(lines[i+1].strip()) > 20 and cnt[lines[i].strip()] == 1
               and cnt[lines[i+1].strip()] == 1 and not lines[i].strip().startswith("#")): i += 1
    a, b = lines[i].strip(), lines[i+1].strip()
    qs = [("needle", f"What is the special magic number for the project codenamed {tag}-HERON? Answer with the number only.", magic, 30),
          ("code", "In the code above, find the line\n" + a + "\nQuote exactly the line that comes immediately after it. Answer with that single line only.", b, 80),
          ("needle2", f"What is the secret passphrase for the vault codenamed {tag}-FALCON? Answer with the passphrase only.", pw, 30)]
    return doc, qs


docs = {"A": build(NA, False, "BLUE", "7418-2295", "amber-quartz-41"),
        "B": build(NB, True, "RED", "5530-8817", "cobalt-ridge-07")}
res = {u: {"doc_tokens": len(tok.encode(d))} for u, (d, _) in docs.items()}
print({u: r["doc_tokens"] for u, r in res.items()}, flush=True)


def ask(user, qi):
    doc, qs = docs[user]
    name, q, want, mx = qs[qi]
    if qi == 0 and user == "A" and os.environ.get("A_MAXTOK1"):
        # G-23 window: A's phase-1 answer ends within the 2-step GDN save delay while B is still prefilling (needle
        # check then expected to FAIL for A; the host-restored follow-ups in phases 2-3 are what is tested)
        mx = int(os.environ["A_MAXTOK1"])
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": doc + "\n\n" + q}], "temperature": 0,
            "max_tokens": mx, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); tf = None; text = ""; usage = None
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                j = json.loads(line[5:])
                if j.get("usage"): usage = j["usage"]
                for ch in j.get("choices", []):
                    c = ch.get("delta", {}).get("content")
                    if c: tf = tf or time.time(); text += c
    except Exception as e:  # noqa: BLE001
        res[user][name] = {"error": repr(e), "ok": False}; print(f"[{user}] [{name}] ERROR {e!r}", flush=True); return
    te = time.time(); tf = tf or te
    ok = want in text
    res[user][name] = {"prompt_tokens": usage["prompt_tokens"], "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                       "ttft_s": round(tf - t0, 2), "total_s": round(te - t0, 2), "ok": ok, "text": text[:200], "want": want}
    print(f"[{user}] [{name}] {usage['prompt_tokens']} in, ttft {tf-t0:.2f}s, ok={ok} text={text[:90]!r}", flush=True)


def reset():
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}/reset_prefix_cache", data=b"", method="POST",
                                headers={"Authorization": f"Bearer {KEY}"})
    ok = None
    for _ in range(20):
        try:
            raw = urllib.request.urlopen(rq, timeout=60).read()
            try:
                ok = json.loads(raw).get("success", True) if raw.strip() else True
            except ValueError:
                ok = True
        except Exception as e:  # noqa: BLE001
            ok = f"error {e!r}"
        if ok is True: break
        time.sleep(1)
    res.setdefault("resets", []).append(ok)
    print(f"[reset_prefix_cache] {ok}", flush=True)


for qi in range(3):
    if qi > 0:
        reset()
    ta = threading.Thread(target=ask, args=("A", qi)); ta.start()
    time.sleep(B_DELAY if qi == 0 else 0.5)
    tb = threading.Thread(target=ask, args=("B", qi)); tb.start()
    ta.join(); tb.join()
    json.dump(res, open(OUT, "w"), indent=1)
skip = {("A", "needle")} if os.environ.get("A_MAXTOK1") else set()
allok = all(res[u].get(n, {}).get("ok") for u in "AB" for n in ("needle", "code", "needle2") if (u, n) not in skip)
print("CONC", "PASS" if allok else "FAIL", flush=True)
