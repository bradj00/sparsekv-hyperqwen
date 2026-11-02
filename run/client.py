# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Stream B validation client (OpenAI chat API, temperature 0, thinking off).

usage: client.py PORT OUTJSON suite[,suite...]
suites: exact | needle | code | speed | quick
"""
import glob
import json
import os
import sys
import time
import urllib.request

from transformers import AutoTokenizer

WT = os.environ.get("SPARSE_WT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = f"{WT}/models/Qwen3.8-27B-W4A16-AutoRound-fast"
PORT = int(sys.argv[1])
OUT = sys.argv[2]
SUITES = sys.argv[3].split(",")
KEY = os.environ.get("VLLM_API_KEY") or open(os.environ.get("API_KEY_FILE", "api_key.txt")).read().strip()
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
tok = AutoTokenizer.from_pretrained(MODEL)
results = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(results, open(OUT, "w"), indent=1)


def chat(content, max_tokens=200, stream=False, min_tokens=None):
    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if min_tokens:
        body["min_tokens"] = min_tokens
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        if not stream:
            d = json.loads(r.read())
            return d["choices"][0]["message"]["content"], d["usage"], time.time() - t0
        text, tfirst, usage = "", None, None
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if tfirst is None:
                        tfirst = time.time()
                    text += c
        tend = time.time()
        return text, usage, (t0, tfirst, tend)


# ---------------- corpora ----------------
def code_corpus():
    base = f"{WT}/venv/lib/python3.12/site-packages/vllm"
    files = sorted(glob.glob(f"{base}/v1/**/*.py", recursive=True)) + sorted(
        glob.glob(f"{base}/model_executor/layers/**/*.py", recursive=True)) + sorted(
        glob.glob(f"{base}/model_executor/models/*.py"))
    parts = []
    for f in files:
        src = open(f).read()
        if len(src) < 2000:
            continue
        parts.append(f"# ===== FILE: {os.path.relpath(f, base)} =====\n{src}\n")
    return parts


def essay_corpus():
    rows = [json.loads(x) for x in open(os.environ.get("NIAH_JSONL", "niah_32k_samples.jsonl"))]
    parts = []
    for r in rows:
        b = r["body"]
        for i in range(0, len(b), 4000):
            parts.append(b[i:i + 4000])
    return parts


def build(parts, ntok):
    out, n = [], 0
    for p in parts:
        t = len(tok.encode(p))
        if n + t > ntok:
            if ntok - n < 3000:
                break
            continue
        out.append(p)
        n += t
    return "".join(out), n


NEEDLE = "The special magic number for the project codenamed BLUE-HERON is 7418-2295."
NQ = "What is the special magic number for the project codenamed BLUE-HERON? Answer with the number only."


def needle_prompt(ntok, depth=0.5):
    hay, _ = build(essay_corpus() * 2, ntok)
    cut = hay.find(". ", int(len(hay) * depth)) + 2
    doc = hay[:cut] + NEEDLE + " " + hay[cut:]
    return doc + "\n\n" + NQ


def code_prompt(ntok, depth=0.4):
    doc, _ = build(code_corpus(), ntok)
    lines = doc.split("\n")
    counts = {}
    for ln in lines:
        counts[ln.strip()] = counts.get(ln.strip(), 0) + 1
    i = int(len(lines) * depth)
    while True:
        a, b = lines[i].strip(), lines[i + 1].strip()
        if len(a) > 30 and len(b) > 20 and counts[a] == 1 and counts[b] == 1 and not a.startswith("#"):
            break
        i += 1
    q = ("In the code above, find the line\n" + a + "\nQuote exactly the line that comes immediately after it. "
         "Answer with that single line only.")
    return doc, doc + "\n\n" + q, b


def ntoks(s):
    return len(tok.encode(s))


# ---------------- suites ----------------
for suite in SUITES:
    if suite == "exact":
        prompts = {
            "short_poem": "Write a 12-line poem about a lighthouse keeper.",
            "short_code": "Write a Python function that parses ISO-8601 durations like P3DT4H5M, with tests.",
        }
        prompts["needle16k"] = needle_prompt(16000, 0.3)
        _, p, _ = code_prompt(24000, 0.6)
        prompts["code24k"] = p.replace("Answer with that single line only.",
                                       "Then explain in detail what the surrounding function does.")
        for name, p in prompts.items():
            text, usage, dt = chat(p, max_tokens=256)
            results[f"exact/{name}"] = {"prompt_tokens": usage["prompt_tokens"], "text": text}
            print(f"[exact] {name}: {usage['prompt_tokens']} tok in, {usage['completion_tokens']} out, {dt:.1f}s",
                  flush=True)
            save()
    if suite in ("needle", "code", "speed", "quick"):
        lens = {"quick": [32000]}.get(suite, [32000, 64000, 128000])
        if suite == "speed":
            lens = [128000]
        for L in lens:
            if suite in ("needle", "quick"):
                p = needle_prompt(L)
                text, usage, dt = chat(p, max_tokens=30)
                ok = "7418-2295" in text
                results[f"needle/{L}"] = {"prompt_tokens": usage["prompt_tokens"], "text": text, "ok": ok,
                                          "secs": dt}
                print(f"[needle {L}] {usage['prompt_tokens']} tok: {'PASS' if ok else 'FAIL'} {text!r} ({dt:.0f}s)",
                      flush=True)
                save()
            if suite in ("code", "quick"):
                _, p, ans = code_prompt(L)
                text, usage, dt = chat(p, max_tokens=80)
                ok = ans in text
                results[f"code/{L}"] = {"prompt_tokens": usage["prompt_tokens"], "text": text, "want": ans,
                                        "ok": ok, "secs": dt}
                print(f"[code {L}] {usage['prompt_tokens']} tok: {'PASS' if ok else 'FAIL'} got={text!r} "
                      f"want={ans!r} ({dt:.0f}s)", flush=True)
                save()
            if suite == "speed":
                doc, _, _ = code_prompt(L)
                for task, q in [("explain", "Explain in detail what the code above does, file by file."),
                                ("copy", "Reproduce the last file above verbatim.")]:
                    text, usage, (t0, tf, te) = chat(doc + "\n\n" + q, max_tokens=512, min_tokens=512,
                                                     stream=True)
                    n = usage["completion_tokens"]
                    tps = (n - 1) / (te - tf)
                    results[f"speed/{L}/{task}"] = {"prompt_tokens": usage["prompt_tokens"], "completion": n,
                                                    "ttft": tf - t0, "decode_tps": tps, "text": text[:400]}
                    print(f"[speed {L} {task}] {usage['prompt_tokens']} tok in, {n} out, ttft {tf-t0:.1f}s, "
                          f"decode {tps:.1f} tok/s", flush=True)
                    save()
    if suite == "concurrent":
        import threading
        jobs = {"needle64k": (needle_prompt(64000, 0.7), "7418-2295", 30)}
        _, p, ans = code_prompt(100000, 0.55)
        jobs["code100k"] = (p, ans, 80)
        out = {}

        def run(name, p, want, mt):
            text, usage, dt = chat(p, max_tokens=mt)
            out[name] = (want in text, text, usage["prompt_tokens"], dt)

        th = [threading.Thread(target=run, args=(n, *v)) for n, v in jobs.items()]
        [t.start() for t in th]
        [t.join() for t in th]
        for n, (ok, text, pt, dt) in out.items():
            results[f"concurrent/{n}"] = {"ok": ok, "text": text, "prompt_tokens": pt, "secs": dt}
            print(f"[concurrent {n}] {pt} tok: {'PASS' if ok else 'FAIL'} {text!r} ({dt:.0f}s)", flush=True)
        save()
    if suite == "speedshort":
        for L in (4000, 8000):
            doc, _, _ = code_prompt(L)
            for task, q in [("explain", "Explain in detail what the code above does, file by file."),
                            ("copy", "Reproduce the last file above verbatim.")]:
                text, usage, (t0, tf, te) = chat(doc + "\n\n" + q, max_tokens=512, min_tokens=512, stream=True)
                n = usage["completion_tokens"]
                tps = (n - 1) / (te - tf)
                results[f"speedshort/{L}/{task}"] = {"prompt_tokens": usage["prompt_tokens"], "completion": n,
                                                     "decode_tps": tps, "text": text[:300]}
                print(f"[speedshort {L} {task}] {usage['prompt_tokens']} tok in, {n} out, decode {tps:.1f} tok/s",
                      flush=True)
                save()
    if suite == "grow":
        import difflib

        # (a) one generation that crosses the cutoff mid-decode: verbatim copy of the last file
        parts = code_corpus()
        target = int(os.environ.get("GROW_START", "44300"))
        last = next(p for p in parts[400:] if 6000 < len(p) < 9000)
        doc, n = build([p for p in parts if p is not last], target - ntoks(last) - 300)
        doc = doc + last
        q = "Reproduce the last file above verbatim, starting from its FILE header line."
        text, usage, (t0, tf, te) = chat(doc + "\n\n" + q, max_tokens=1200, stream=True)
        out = text.strip().strip("`")
        src = last.strip()
        sm = difflib.SequenceMatcher(None, out[:len(src)], src[:len(out)], autojunk=False)
        ratio = sm.ratio()
        lcp = 0
        o2 = out[out.find("# =====") :] if "# =====" in out else out
        for a, b in zip(o2, src):
            if a != b:
                break
            lcp += 1
        pt, nc = usage["prompt_tokens"], usage["completion_tokens"]
        results["grow/copy"] = {"prompt_tokens": pt, "completion": nc, "end_len": pt + nc, "similarity": ratio,
                                "exact_prefix_chars": lcp, "out_chars": len(o2), "text": text}
        print(f"[grow copy] {pt} -> {pt + nc} tokens (crosses {target}?), similarity {ratio:.3f}, "
              f"exact prefix {lcp}/{min(len(o2), len(src))} chars", flush=True)
        save()
        # (b) multi-turn conversation growing from ~36K to ~60K: code-next questions about old and new parts
        doc1, _ = build(parts, 36000)
        rest = [p for p in parts if p not in doc1]
        doc2, _ = build(rest, 22000)

        def nextline_q(doc, depth):
            lines = doc.split("\n")
            counts = {}
            for ln in lines:
                counts[ln.strip()] = counts.get(ln.strip(), 0) + 1
            i = int(len(lines) * depth)
            while True:
                a, b = lines[i].strip(), lines[i + 1].strip()
                if len(a) > 30 and len(b) > 20 and counts[a] == 1 and counts[b] == 1 and not a.startswith("#"):
                    return a, b
                i += 1

        msgs = []
        turns = [(doc1, doc1, 0.5), (doc2, doc2, 0.5), ("", doc1, 0.2)]
        for ti, (add, src_doc, depth) in enumerate(turns):
            a, b = nextline_q(src_doc, depth)
            content = (add + "\n\n" if add else "") + ("In the code above (anywhere in this conversation), find the "
                                                       "line\n" + a + "\nQuote exactly the line that comes "
                                                       "immediately after it. Answer with that single line only.")
            msgs.append({"role": "user", "content": content})
            body = {"model": "qwen3.8-27b", "messages": msgs, "temperature": 0, "max_tokens": 80,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=1200) as r:
                d = json.loads(r.read())
            ans = d["choices"][0]["message"]["content"]
            msgs.append({"role": "assistant", "content": ans})
            ok = b in ans
            results[f"grow/turn{ti + 1}"] = {"prompt_tokens": d["usage"]["prompt_tokens"], "ok": ok, "text": ans,
                                             "want": b}
            print(f"[grow turn{ti + 1}] {d['usage']['prompt_tokens']} tok: {'PASS' if ok else 'FAIL'} got={ans!r} "
                  f"want={b!r} ({time.time() - t0:.0f}s)", flush=True)
            save()
print("CLIENT DONE")
