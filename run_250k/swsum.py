# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""swsum.py FILE... : per-k table of a streams sweep"""
import json, sys, statistics as st
for f in sys.argv[1:]:
    rs = []
    for l in open(f):
        try: rs.append(json.loads(l))
        except Exception: pass
    print(f"== {f}")
    print("k | recall ok/total | cold TTFT s | follow-up TTFT min/med/max | re-prefills | explain per-stream | copy per-stream | combined explain / copy")
    for k in sorted({r["k"] for r in rs if "k" in r and "q" in r}):
        R = [r for r in rs if r.get("k") == k and "user" in r and "q" in r]
        rec = [r for r in R if r["q"] in ("needle", "code", "needle2")]
        cold = [r["ttft"] for r in R if r.get("phase") == "cold" and "ttft" in r]
        fu = [r for r in R if r.get("phase") != "cold" and "ttft" in r]
        rp = sum(1 for r in fu if r["cached"] < r["prompt"] - 2000)
        ex = [r["tok_s"] for r in fu if r["q"] == "explain" and r["tok_s"]]
        cp = [r["tok_s"] for r in fu if r["q"] == "copy" and r["tok_s"]]
        cx = [r["combined_tok_s"] for r in rs if r.get("kind") == "combined" and r.get("k") == k and r["q"] == "explain"]
        cc = [r["combined_tok_s"] for r in rs if r.get("kind") == "combined" and r.get("k") == k and r["q"] == "copy"]
        tt = sorted(r["ttft"] for r in fu)
        m = lambda x: f"{st.mean(x):.1f}" if x else "-"
        err = sum(1 for r in R if "error" in r)
        print(f"{k} | {sum(1 for r in rec if r.get('ok'))}/{len(rec)} | {','.join(f'{c:.0f}' for c in cold) or '-'} | "
              f"{tt[0]:.1f}/{tt[len(tt)//2]:.1f}/{tt[-1]:.1f} | {rp} | {m(ex)} | {m(cp)} | {m(cx)} / {m(cc)}" + (f" | ERR {err}" if err else ""))
