#!/usr/bin/env python3
"""hs_gen.py: "haystack + thinking" suite (community ask 2026-11-02): MMLU-Pro questions, thinking ON, free-running
sampling, asked AFTER a fixed ~150K-token haystack of unrelated LongBench v2 documents. The haystack is identical for
every item so the prefix cache prefills it once. Thinking traces run far past the 1024-token exact tail, so the model
must re-find its own earlier reasoning through the sparse selection. Same ids/gt as reqs/mmlupro.jsonl.
Writes reqs/mmlupro_hs.jsonl"""
import json, os, sys
from transformers import AutoTokenizer
B = os.environ.get("BENCH", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 150000; N = 48
tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL_DIR", "Qwen3.8-27B-W4A16-AutoRound"))
docs, n = [], 0
for i, l in enumerate(open(os.path.join(B, "data", "hf/longbench_v2.jsonl"))):
    r = json.loads(l)
    if r["domain"] in ("Long In-context Learning",) or not r["context"].isascii(): continue
    t = tok(r["context"], add_special_tokens=False)["input_ids"]
    take = t[: min(len(t), 40000, TARGET - n)]
    docs.append(f"=== Document {len(docs)+1} ===\n" + tok.decode(take)); n += len(take)
    if n >= TARGET: break
hay = "Background documents (they may be unrelated to the task below):\n\n" + "\n\n".join(docs) + "\n\n=== End of documents ===\n\n"
src = [json.loads(l) for l in open(f"{B}/reqs/mmlupro.jsonl")]
s0 = [r for r in src if r["shard"] == 0][: N // 2]; s1 = [r for r in src if r["shard"] == 1][: N // 2]
M = "<|im_start|>user\n"
with open(f"{B}/reqs/mmlupro_hs.jsonl", "w") as f:
    for r in s0 + s1:
        assert r["prompt"].count(M) == 1
        r = dict(r, prompt=r["prompt"].replace(M, M + hay), group="mmlupro_hs")
        f.write(json.dumps(r) + "\n")
print("haystack tokens", n, "docs", len(docs), "items", len(s0) + len(s1),
      "prompt tokens ex.", len(tok(r["prompt"], add_special_tokens=False)["input_ids"]))
