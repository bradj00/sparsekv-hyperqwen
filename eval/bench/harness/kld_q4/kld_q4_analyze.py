#!/usr/bin/env python3
"""kld_q4_analyze.py [DATA] [OUT_ROOT] [--md f] [--json f]  -- Q4TEST 2x2 on the teacher-forced KLD harness.

Same samples, same reference (OUT_ROOT/ref, dense, GPU1) and the same estimator as kld_analyze.py (KL on the
reference's top-1024 + tail bucket, decode steps 1..n-1, kl_steps imported from it). Cells, all from GPU0:
  dense int8   = toggle/<sid>.dense.npz        (existing run: SPARSE_KV server, dense switch)
  sparse int8  = toggle/<sid>.sparse.npz       (existing run, sparse switch)
  dense int4   = toggle_q4/<sid>.dense.npz     (new: same server cfg + SPARSE_DEBUG_KV_FAKEQ4=1)
  sparse int4  = toggle_q4/<sid>.sparse.npz
  optional: dense_q4/<sid>.dense_q4.npz (fake int4 on a server WITHOUT SPARSE_KV, cross-check of dense int4),
            toggle_ctl/<sid>.{dense,sparse}.npz (q4test venv with the knob OFF: must equal toggle/, KLD 0)
Pairwise rows report mean KLD + top-1 agreement like kld_analyze.py; the summary gives per-sample paired
interaction estimates with a bootstrap CI over samples:
  sensitivity   S_d = KL(dense int8 || dense int4),  S_s = KL(sparse int8 || sparse int4);   S_s - S_d
  penalty vs dense int8 (same card): P_d = KL(d8 || d4), P_s = KL(d8 || s4) - KL(d8 || s8);   P_s - P_d
Negative interaction = int4 KV moves the sparse model LESS than the dense model (the hypothesis).
"""
import argparse
import json
import os
import sys

import numpy as np

# KLD_DIR = the KLD work dir (data/, out/); kld_analyze.py is imported from eval/kld/ of this repo
KLD_DIR = os.environ.get("KLD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kld"))
sys.path.insert(0, KLD_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kld"))
from kld_analyze import kl_steps, stats  # noqa: E402

D8, S8, D4, S4 = ("toggle", "dense"), ("toggle", "sparse"), ("toggle_q4", "dense"), ("toggle_q4", "sparse")
# (label, P (None = reference), Q, Q expected sparse)
RUNS = [
    ("noise floor: dense int8 vs ref (other card)", None, D8, False),
    ("control: q4test venv knob OFF, dense vs dense int8", D8, ("toggle_ctl", "dense"), False),
    ("control: q4test venv knob OFF, sparse vs sparse int8", S8, ("toggle_ctl", "sparse"), True),
    ("sparse int8 vs dense int8", D8, S8, True),
    ("dense int4 vs dense int8  [S_d]", D8, D4, False),
    ("sparse int4 vs dense int8", D8, S4, True),
    ("sparse int4 vs sparse int8  [S_s]", S8, S4, True),
    ("sparse int4 vs dense int4 (same server)", D4, S4, True),
    ("dense int4 no-SPARSE_KV server vs dense int4 toggle server", D4, ("dense_q4", "dense_q4"), False),
    ("dense int4 vs ref (other card)", None, D4, False),
    ("sparse int4 vs ref (other card)", None, S4, True),
]
RNG = np.random.default_rng(0)


def boot(x, n=5000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    m = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=os.path.join(KLD_DIR, "data"))
    ap.add_argument("root", nargs="?", default=os.path.join(KLD_DIR, "out"))
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    samples = json.load(open(os.path.join(a.data, "samples.json")))["samples"]
    lengths = sorted({s["L"] for s in samples})

    def path(sid, run):
        return os.path.join(a.root, "ref", f"{sid}.ref.npz") if run is None else \
            os.path.join(a.root, run[0], f"{sid}.{run[1]}.npz")

    cache = {}

    def load(sid, run):
        p = path(sid, run)
        if p not in cache:
            try:
                cache[p] = np.load(p) if os.path.exists(p) else None
            except Exception:  # noqa: BLE001
                cache[p] = None
        return cache[p]

    def pair(sid, P, Q):
        zr, zq, zp = load(sid, None), load(sid, Q), (load(sid, P) if P else None)
        if zr is None or zq is None or (P and zp is None):
            return None
        return kl_steps(zr, zq, zp)

    res, problems, md, persample = {}, [], [], {}
    for label, P, Q, want_sparse in RUNS:
        groups = {}
        for s in samples:
            r = pair(s["sid"], P, Q)
            if r is None:
                continue
            kl, top1, fok, sp, _ = r
            persample[(label, s["sid"])] = (float(kl.mean()), float(top1.mean()))
            if not fok.all():
                problems.append(f"{label} {s['sid']}: forced path mismatch at {int((~fok).sum())} steps")
            if want_sparse and not sp.all():
                problems.append(f"{label} {s['sid']}: {int((~sp).sum())}/{len(sp)} decode steps not sparse")
            if not want_sparse and sp.any():
                problems.append(f"{label} {s['sid']}: {int(sp.sum())} dense steps ran sparse")
            for k in (s["L"], "all"):
                g = groups.setdefault(k, {"kl": [], "top1": [], "n": 0})
                g["kl"].append(kl); g["top1"].append(top1); g["n"] += 1
        for k, g in groups.items():
            st = stats(np.concatenate(g["kl"]), np.concatenate(g["top1"]))
            st["samples"] = g["n"]
            res.setdefault(label, {})[str(k)] = st
    md.append("Mean KLD (nats/token) / top-1 agreement, decode steps, all cells on GPU0 unless 'vs ref'")
    md.append("")
    md.append("| comparison (P -> Q) | " + " | ".join(f"{L / 1024:.0f}K" for L in lengths) + " | all | samples |")
    md.append("|---|" + "---|" * (len(lengths) + 2))
    for label, *_ in RUNS:
        if label not in res:
            continue
        cells = [(f"{st['mean']:.4f} / {100 * st['top1']:.1f}%" if (st := res[label].get(str(k))) else "n/a")
                 for k in [*lengths, "all"]]
        md.append(f"| {label} | " + " | ".join(cells) + f" | {res[label]['all']['samples']} |")

    # paired per-sample interactions
    L_SD, L_SS = RUNS[4][0], RUNS[6][0]
    L_D8S8, L_D8S4 = RUNS[3][0], RUNS[5][0]
    summ = {}
    md += ["", "Paired per-sample interaction (bootstrap 95% CI over samples; negative = int4 hurts sparse less)", "",
           "| quantity | " + " | ".join(f"{L / 1024:.0f}K" for L in lengths) + " | all |",
           "|---|" + "---|" * (len(lengths) + 1)]
    defs = [
        ("KLD sensitivity S_s - S_d", lambda sid, i: persample[(L_SS, sid)][i] - persample[(L_SD, sid)][i], 0,
         (L_SS, L_SD)),
        ("KLD penalty P_s - P_d", lambda sid, i: (persample[(L_D8S4, sid)][i] - persample[(L_D8S8, sid)][i])
         - persample[(L_SD, sid)][i], 0, (L_D8S4, L_D8S8, L_SD)),
        ("top-1 sensitivity (pts): (1-A_s) - (1-A_d)", lambda sid, i: persample[(L_SD, sid)][i]
         - persample[(L_SS, sid)][i], 1, (L_SS, L_SD)),
    ]
    for name, f, idx, need in defs:
        cells = []
        for k in [*lengths, "all"]:
            sids = [s["sid"] for s in samples if (k == "all" or s["L"] == k)
                    and all((lab, s["sid"]) in persample for lab in need)]
            m, lo, hi = boot([f(sid, idx) for sid in sids])
            summ.setdefault(name, {})[str(k)] = {"mean": m, "lo": lo, "hi": hi, "samples": len(sids)}
            sc = 100 if idx == 1 else 1
            cells.append("n/a" if not sids else (f"{sc * m:+.4f} [{sc * lo:+.4f}, {sc * hi:+.4f}] (n={len(sids)})"
                                                 if idx == 0 else f"{sc * m:+.2f} [{sc * lo:+.2f}, {sc * hi:+.2f}]"))
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    out = "\n".join(md)
    print(out)
    print("\nchecks:", "OK" if not problems else "\n  " + "\n  ".join(problems[:40]))
    if a.md:
        open(a.md, "w").write(out + "\n")
    if a.json:
        json.dump({"results": res, "interaction": summ, "problems": problems}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
