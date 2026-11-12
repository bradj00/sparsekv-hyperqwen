#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""gate_an.py GATE_DIR... : call-site accuracy gate from per-item score records.

GATE_DIR holds base.jsonl and cand.jsonl, one record per item of the call-site suite (eval/bench): id, shard, f1,
precision, recall, exact, out_sha (hash of the model's answer, so identical answers can be counted without
publishing them), token counts and timings. Prints the mean F1 per arm and the paired difference cand - base on the
same items with a 95% paired bootstrap (5000 resamples, seed 0), as eval/bench/harness/analyze.py does.

    python eval/v12/gate_an.py eval/v12/results/gate/vocab eval/v12/results/gate/hotpool
"""
import json
import os
import sys

import numpy as np

for g in sys.argv[1:]:
    arm = {}
    for a in ("base", "cand"):
        arm[a] = {r["id"]: r for r in map(json.loads, open(os.path.join(g, a + ".jsonl")))}
        by = {}
        for k, r in arm[a].items():
            by.setdefault(k.split("/")[2], []).append(r["f1"])
        print(f"{g} {a}: n={len(arm[a])} F1={np.mean([r['f1'] for r in arm[a].values()]):.4f}  by length "
              + " ".join(f"{k}:{np.mean(v):.3f}(n={len(v)})" for k, v in sorted(by.items())))
    ks = sorted(set(arm["base"]) & set(arm["cand"]))
    d = np.array([arm["cand"][k]["f1"] - arm["base"][k]["f1"] for k in ks])
    m = d[np.random.default_rng(0).integers(0, len(d), (5000, len(d)))].mean(1)
    lo, hi = np.percentile(m, 2.5), np.percentile(m, 97.5)
    same = sum(arm["base"][k]["out_sha"] == arm["cand"][k]["out_sha"] for k in ks)
    print(f"{g} cand - base: dF1 {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] n={len(ks)}, identical outputs {same}/{len(ks)}  "
          f"GATE {'PASS (CI includes 0 or is positive)' if hi >= 0 else 'FAIL'}")
