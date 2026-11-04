#!/usr/bin/env python3
"""mkjob.py GPU SEQ CFG SUITE [k=v ...] : drop a job file into queue/gpu<GPU>/.
Special: ruler_n=32k:3,64k:3,128k:2,192k:1 -> per (length, task) limit per shard."""
import json, os, sys
B = os.environ.get("BENCH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gpu, seq, cfg, suite = sys.argv[1:5]
kv = dict(a.split("=", 1) for a in sys.argv[5:])
job = {"cfg": cfg, "env": json.loads(kv.pop("env", "{}")), "reqs": f"reqs/{suite}.jsonl", "shard": int(gpu),
       "out": f"out/{cfg}/{suite}.g{gpu}.jsonl", "conc": int(kv.pop("conc", 1)), "gpu": int(gpu),
       "label": kv.pop("label", f"{suite}")}
if "shard" in kv:
    v = kv.pop("shard"); job["shard"] = None if v == "all" else int(v)
if "out" in kv:
    job["out"] = kv.pop("out")
if "ruler_n" in kv:
    from mkreqs import RULER_TASKS
    lim = {}
    for part in kv.pop("ruler_n").split(","):
        ln, n = part.split(":")
        for t in RULER_TASKS:
            lim[f"ruler/{ln}/{t}/"] = int(n)
    job["limit"] = lim
if "ids" in kv:
    job["ids"] = kv.pop("ids").split(",")
assert not kv, kv
p = os.path.join(B, "queue", f"gpu{gpu}", f"{int(seq):03d}-{cfg}-{suite}.json")
json.dump(job, open(p, "w"), indent=1)
print(p)
