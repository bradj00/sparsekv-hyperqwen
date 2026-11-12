#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""mix_sum.py FILE... : one line per agent-mix run (eval/v12/results/mix/*.json).

Follow-up turns by uncached prompt tokens: short (<= 1728), medium (1729-2600), long (> 2600): n, median and max
TTFT in seconds (plain median over the records), plus the output tokens of the whole run. The "summary" block stored
in each file was written by the harness with a nearest-rank percentile, so its medians can differ from these in the
first decimal.

    python eval/v12/mix_sum.py eval/v12/results/mix/*.json
"""
import json
import os
import statistics as st
import sys

print("run | agents | seed | requests ok | short n/med/max | medium n/med/max | long n/med/max | output tokens")
for p in sys.argv[1:]:
    o = json.load(open(p))
    ok = [r for r in o["records"] if r.get("ok")]
    fu = [r for r in ok if r["kind"] == "follow"]

    def cell(lo, hi):
        x = [r["ttft"] for r in fu if lo < r["unc"] <= hi]
        return f"{len(x)}/{st.median(x):.1f}/{max(x):.1f}" if x else "0/-/-"

    print(f"{os.path.basename(p)[:-5]} | {o['args']['agents']} | {o['args']['seed']} | {len(ok)}/{len(o['records'])} | "
          f"{cell(-1, 1728)} | {cell(1728, 2600)} | {cell(2600, 10**9)} | {sum(r['ctok'] for r in ok):,}")
