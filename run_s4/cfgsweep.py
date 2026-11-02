# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Launch-config sweep of select+gather at 4 x 246K (SPARSE_STATS_SUB from env), int4 and int8 sketches."""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LS", "246000"); os.environ.setdefault("NREQS", "4")
import importlib.util
spec = importlib.util.spec_from_file_location("sb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "selbench.py"))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selbench.py")).read().split("print(torch.cuda.get_device_name())")[0]
g = {"__name__": "sb"}
exec(compile(src, "selbench", "exec"), g)
T, bench = g["T"], g["bench"]
base = dict(T.CFG)
res = []
for sw, sa, ba, scw, sb, bb in itertools.product((2, 4), (2, 4, 8), (64, 128), (4, 8), (1, 2, 4), (64, 128)):
    T.CFG.update(base); T.CFG.update(stats_warps=sw, split_a=sa, block_a=ba, score_warps=scw, split_b=sb, block_b=bb)
    try:
        ms = bench(4, 246000, 4)
    except Exception as e:  # noqa: BLE001
        continue
    res.append((ms, sw, sa, ba, scw, sb, bb))
    print(f"{ms:.3f} ms  stats_warps={sw} split_a={sa} block_a={ba} score_warps={scw} split_b={sb} block_b={bb}", flush=True)
T.CFG.update(base)
print("default int4", round(bench(4, 246000, 4), 3), "int8", round(bench(4, 246000, 8), 3))
for r in sorted(res)[:5]:
    print("BEST", r)
