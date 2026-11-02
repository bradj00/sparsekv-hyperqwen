# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""sum.py FILE... : per-k table (decode separated from follow-up time)"""
import json, sys, statistics as st
m = lambda x: f"{st.mean(x):.1f}" if x else "-"
for f in sys.argv[1:]:
    rs = []
    for l in open(f):
        try: rs.append(json.loads(l))
        except Exception: pass
    print(f"== {f}")
    print("k | recall | cold TTFT | fu TTFT min/med/max | re-prefill | per-stream ex/cp | combined ex/cp | STEADY ex/cp | tok/step | hot miss | miss MB/step | staged GB/batch")
    for k in sorted({r["k"] for r in rs if "k" in r and "q" in r}):
        R = [r for r in rs if r.get("k") == k and "user" in r and "q" in r]
        rec = [r for r in R if r["q"] in ("needle", "code", "needle2")]
        cold = [r["ttft"] for r in R if r.get("phase") == "cold" and "ttft" in r]
        fu = [r for r in R if r.get("phase") != "cold" and "ttft" in r]
        rp = sum(1 for r in fu if r["cached"] < r["prompt"] - 2000)
        C = [r for r in rs if r.get("kind") == "combined" and r.get("k") == k]
        g = lambda q, key: [r[key] for r in C if r["q"] == q and r.get(key) is not None]
        ex = [r["tok_s"] for r in fu if r["q"] == "explain" and r["tok_s"]]
        cp = [r["tok_s"] for r in fu if r["q"] == "copy" and r["tok_s"]]
        tt = sorted(r["ttft"] for r in fu) or [0]
        tps = [r["tokens_per_step"] for r in C if r["q"] in ("explain", "copy") and r.get("tokens_per_step")]
        hm = [r["hot_miss_rate"] for r in C if r["q"] in ("explain", "copy") and r.get("hot_miss_rate") is not None]
        mb = [r["miss_MB_per_step"] for r in C if r["q"] in ("explain", "copy") and r.get("miss_MB_per_step") is not None]
        sg = [r["staged_GB"] for r in C if r["q"] in ("explain", "copy") and r.get("staged_GB") is not None]
        err = sum(1 for r in R if "error" in r)
        print(f"{k} | {sum(1 for r in rec if r.get('ok'))}/{len(rec)} | {','.join(f'{c:.0f}' for c in cold) or '-'} | "
              f"{tt[0]:.1f}/{tt[len(tt)//2]:.1f}/{tt[-1]:.1f} | {rp} | {m(ex)}/{m(cp)} | "
              f"{m(g('explain','combined_tok_s'))}/{m(g('copy','combined_tok_s'))} | "
              f"{m(g('explain','steady_tok_s'))}/{m(g('copy','steady_tok_s'))} | {m(tps)} | "
              f"{st.mean(hm) if hm else '-'} | {m(mb)} | {m(sg)}" + (f" | ERR {err}" if err else ""))
