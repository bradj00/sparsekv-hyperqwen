# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Summarise the grid.*.jsonl files (produced by an internal eval harness, not included): per (tag, len, users, prompt) over reps.

usage: grid_table.py file.jsonl [file2.jsonl ...]
Columns: mean per-request decode tok/s, mean TTFT per request (s), total throughput (all completion tokens /
(last t_last - first t0)), and how many requests overlapped in decode (both decoding at once).
"""
import json
import sys
from collections import defaultdict
from statistics import mean

rows = defaultdict(list)
for f in sys.argv[1:]:
    for line in open(f):
        d = json.loads(line)
        if d.get("kind") != "batch":
            continue
        rows[(d["tag"], d["len"], d["users"], d["prompt"])].append(d)

print(f"{'tag':<14}{'len':>7}{'U':>3} {'prompt':<8}{'n':>3} {'req tok/s':>10} {'TTFT s':>14} {'total tok/s':>12} "
      f"{'acc len':>8} {'overlap':>8}")
for k in sorted(rows):
    ds = rows[k]
    tps = [r["decode_tok_s"] for d in ds for r in d["req"]]
    ttft = [r["ttft_s"] for d in ds for r in d["req"]]
    tot = []
    ov = 0
    for d in ds:
        rq = d["req"]
        t0 = min(r["t0"] for r in rq)
        t1 = max(r["t_last"] for r in rq)
        tot.append(sum(r["completion_tokens"] for r in rq) / (t1 - t0))
        if len(rq) == 2:
            a, b = rq
            if min(a["t_last"], b["t_last"]) - max(a["t_first"], b["t_first"]) > 0.5 * min(
                    a["t_last"] - a["t_first"], b["t_last"] - b["t_first"]):
                ov += 1
    acc = [d.get("spec_mean_accepted_len") or 0 for d in ds]
    tt = "/".join(f"{mean(r['ttft_s'] for d in ds for r in d['req'][i:i + 1]):.1f}" for i in range(k[2]))
    print(f"{k[0]:<14}{k[1]:>7}{k[2]:>3} {k[3]:<8}{len(ds):>3} {mean(tps):>10.1f} {tt:>14} {mean(tot):>12.1f} "
          f"{mean(acc):>8.2f} {ov:>5}/{len(ds) if k[2] == 2 else 0}")
