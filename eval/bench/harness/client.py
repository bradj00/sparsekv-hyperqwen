"""Benchmark client: drives one OpenAI-compatible /v1/completions server with pre-rendered prompts.

Request lines (reqs/*.jsonl):
  {"id": str, "shard": 0|1, "group": str, "prompt": str, "max_tokens": int,
   "sampling": {"temperature": 0, ...} (optional, default greedy), "seed": int (optional),
   "turns": [{"glue": str, "max_tokens": int}, ...] (optional follow-up turns; turn k's prompt is
            prompt_{k-1} + output_{k-1} + glue_k, i.e. the model's own previous answers stay in the history)}
Output lines: {"id", "outputs": [str per turn], "finish": [...], "usage": [...], "ttft": [...], "secs": [...],
               "spec": [{accepted, drafted, drafts} per turn when conc == 1], "error": str|None}
Resumable: ids already in the output file are skipped.
"""
import asyncio, json, os, re, time
import httpx

MODEL = "qwen3.8-27b"


def bearer_token():
    # optional: the token the server was started with (empty = server without auth)
    return os.environ.get("BENCH_TOKEN", "")


_MET = re.compile(r'^(vllm:(?:spec_decode_num_(?:accepted_tokens|draft_tokens|drafts)|prefix_cache_hits|prefix_cache_queries)(?:_total)?)\{[^}]*\}\s+([0-9.eE+-]+)', re.M)


_HDR = {}


async def metrics(c, port):
    try:
        r = await c.get(f"http://127.0.0.1:{port}/metrics", timeout=10, headers=_HDR)
        d = {}
        for k, v in _MET.findall(r.text):
            k = k.replace("vllm:", "").replace("_total", "")
            d[k] = d.get(k, 0.0) + float(v)
        return d
    except Exception:
        return {}


def mdelta(a, b):
    return {k: round(b.get(k, 0) - a.get(k, 0)) for k in set(a) | set(b)}


async def complete(c, port, hdr, prompt, max_tokens, sampling, seed):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "skip_special_tokens": False}
    body.update({"temperature": 0.0})
    body.update(sampling or {})
    if seed is not None:
        body["seed"] = seed
    t0 = time.time(); ttft = None; parts = []; finish = None; usage = None
    async with c.stream("POST", f"http://127.0.0.1:{port}/v1/completions", json=body, headers=hdr,
                        timeout=httpx.Timeout(7200, connect=30)) as r:
        if r.status_code != 200:
            txt = (await r.aread()).decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {r.status_code}: {txt}")
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                t = ch.get("text") or ""
                if t and ttft is None:
                    ttft = time.time() - t0
                parts.append(t)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return "".join(parts), finish, usage, (round(ttft, 3) if ttft else None), round(time.time() - t0, 3)


def strip_special(t):
    # outputs are kept raw (think tags included); drop a trailing end-of-turn marker if present
    for s in ("<|im_end|>", "<|endoftext|>"):
        if t.endswith(s):
            t = t[: -len(s)]
    return t


async def run_job(job, port, out, bench, server_log, alive):
    reqs = [json.loads(l) for l in open(os.path.join(bench, job["reqs"]))]
    shard = job.get("shard")
    if shard is not None:
        reqs = [r for r in reqs if r.get("shard", 0) == shard]
    if job.get("ids"):
        keep = set(job["ids"]); reqs = [r for r in reqs if r["id"] in keep]
    if job.get("limit"):  # {"<prefix of id>": max count}
        cnt = {}
        sel = []
        for r in reqs:
            k = next((p for p in job["limit"] if r["id"].startswith(p)), None)
            if k is not None:
                cnt[k] = cnt.get(k, 0) + 1
                if cnt[k] > job["limit"][k]:
                    continue
            sel.append(r)
        reqs = sel
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                j = json.loads(l)
                if not j.get("error"):
                    done.add(j["id"])
            except Exception:
                pass
    todo = [r for r in reqs if r["id"] not in done]
    conc = int(job.get("conc", 1))
    hdr = {"Authorization": f"Bearer {bearer_token()}"}
    _HDR.update(hdr)
    stats = {"n_total": len(reqs), "n_skipped": len(reqs) - len(todo), "n_done": len(reqs) - len(todo), "n_err": 0}
    lock = asyncio.Lock()
    died = False
    async with httpx.AsyncClient() as c:
        m_job0 = await metrics(c, port)

        async def one(r):
            nonlocal died
            rec = {"id": r["id"], "cfg": job["cfg"], "gpu": job.get("gpu"), "outputs": [], "finish": [], "usage": [],
                   "ttft": [], "secs": [], "spec": [], "error": None}
            prompt = r["prompt"]
            turns = [{"glue": "", "max_tokens": r["max_tokens"]}] + list(r.get("turns") or [])
            try:
                for k, t in enumerate(turns):
                    if k > 0:
                        prompt = prompt + rec["outputs"][-1] + t["glue"]
                    m0 = await metrics(c, port) if conc == 1 else None
                    seed = r.get("seed")
                    txt, fin, use, ttft, secs = await complete(c, port, hdr, prompt, t["max_tokens"], r.get("sampling"),
                                                               None if seed is None else seed + k)
                    txt = strip_special(txt)
                    rec["outputs"].append(txt); rec["finish"].append(fin); rec["usage"].append(use)
                    rec["ttft"].append(ttft); rec["secs"].append(secs)
                    if conc == 1:
                        rec["spec"].append(mdelta(m0, await metrics(c, port)))
            except Exception as ex:  # noqa: BLE001
                rec["error"] = repr(ex)[:400]
                if not alive():
                    died = True
            async with lock:
                if rec["error"] is None:
                    stats["n_done"] += 1
                else:
                    stats["n_err"] += 1
                if not (died and rec["error"]):
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")

        sem = asyncio.Semaphore(conc)

        async def guarded(r):
            async with sem:
                if died:
                    return
                await one(r)

        await asyncio.gather(*(guarded(r) for r in todo))
        stats["spec_job"] = mdelta(m_job0, await metrics(c, port))
    stats["server_died"] = died
    return stats
