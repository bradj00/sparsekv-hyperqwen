#!/usr/bin/env python3
"""q4_analyze.py [BENCH] [--shard g1] [--md out.md] [--json out.json]  -- the Q4TEST 2x2 on the bench suites.

{dense, sparse} x {native int8 KV, fake int4 KV (SPARSE_DEBUG_KV_FAKEQ4=1)} from out/{dense,sparse,dense_q4,sparse_q4}/
<suite>.<shard>.jsonl, scored with analyze.py's own scorers (RULER string match; xfile F1/recall/exact). Only items
answered by ALL FOUR cells are used (complete 2x2 per item; default shard g1 = the same card and the same items as
the staged q4 jobs). Reports per length bin:
  - mean score per cell, 95% bootstrap CI
  - int4 damage within dense   = dense_q4 - dense     (paired per item)
  - int4 damage within sparse  = sparse_q4 - sparse
  - interaction (the hypothesis) = (sparse_q4 - sparse) - (dense_q4 - dense); > 0 means int4 hurts sparse LESS
  - sparse - dense at int8 and at int4
"""
import glob
import json
import os
import sys

BENCH = os.environ.get("BENCH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
args = sys.argv[1:]
if args and not args[0].startswith("--"):
    BENCH = args.pop(0)
SHARD = args[args.index("--shard") + 1] if "--shard" in args else "g1"
sys.argv = [sys.argv[0], BENCH]  # analyze.py reads its bench dir from argv[1] at import
sys.path.insert(0, os.path.join(BENCH, "src"))
import analyze as A  # noqa: E402
import numpy as np  # noqa: E402

CELLS = ["dense", "sparse", "dense_q4", "sparse_q4"]
NAME = {"dense": "dense int8", "sparse": "sparse int8", "dense_q4": "dense fake-int4", "sparse_q4": "sparse fake-int4"}
LEN = ["32k", "64k", "128k", "192k", "240k"]


def load(cfg, suite):
    recs = {}
    for p in sorted(glob.glob(os.path.join(BENCH, "out", cfg, f"{suite}.{SHARD}.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            if not r.get("error"):
                recs[r["id"]] = r
    return recs


def fmt(m, lo, hi, signed=False, k=100):
    if m != m:
        return ""
    f = "{:+.1f}" if signed else "{:.1f}"
    s = f.format(k * m)
    if lo == lo:
        s += f" [{f.format(k * lo)}, {f.format(k * hi)}]"
    return s


def two_by_two(title, per):
    """per: {cell: {id: (score, group)}} -> markdown + json."""
    ids = set.intersection(*(set(per[c]) for c in CELLS)) if all(per.get(c) for c in CELLS) else set()
    if not ids:
        have = {c: len(per.get(c) or {}) for c in CELLS}
        return f"### {title}\n\nno complete 2x2 yet (items per cell: {have})\n", {}
    groups = sorted({per["dense"][i][1]["length"] for i in ids}, key=lambda x: (LEN.index(x) if x in LEN else 99, x))
    groups.append("all")
    sel = {g: sorted(i for i in ids if g == "all" or per["dense"][i][1]["length"] == g) for g in groups}
    S = {c: {g: np.array([per[c][i][0] for i in sel[g]]) for g in groups} for c in CELLS}
    rows = [(NAME[c], {g: A.boot_ci(S[c][g]) for g in groups}, False) for c in CELLS]
    diffs = [
        ("int4 damage, dense: dense_q4 - dense", lambda g: S["dense_q4"][g] - S["dense"][g]),
        ("int4 damage, sparse: sparse_q4 - sparse", lambda g: S["sparse_q4"][g] - S["sparse"][g]),
        ("**interaction: (sparse_q4 - sparse) - (dense_q4 - dense)**",
         lambda g: (S["sparse_q4"][g] - S["sparse"][g]) - (S["dense_q4"][g] - S["dense"][g])),
        ("sparse - dense at int8", lambda g: S["sparse"][g] - S["dense"][g]),
        ("sparse - dense at fake-int4", lambda g: S["sparse_q4"][g] - S["dense_q4"][g]),
    ]
    rows += [(lab, {g: A.boot_ci(f(g)) for g in groups}, True) for lab, f in diffs]
    md = [f"### {title}", "", "Items with all four cells (paired): " + ", ".join(f"{g} n={len(sel[g])}" for g in groups),
          "", "| | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    js = {}
    for lab, cells, signed in rows:
        md.append(f"| {lab} | " + " | ".join(fmt(*cells[g], signed=signed) for g in groups) + " |")
        js[lab.strip("*")] = {g: dict(zip(["mean", "lo", "hi"], cells[g]), n=len(sel[g])) for g in groups}
    md.append("")
    md.append("Points (x100). Interaction > 0 with a CI above 0 = int4 KV hurts sparse less than dense (hypothesis).")
    return "\n".join(md) + "\n", js


def main():
    md, js = [f"## Q4TEST 2x2: KV int4 robustness, dense vs sparse (shard {SHARD})", ""], {}
    gt_r = os.path.join(BENCH, "data/ruler_gt.json")
    if os.path.exists(gt_r):
        gt = json.load(open(gt_r))
        per = {c: A.ruler_scores(load(c, "ruler"), gt) for c in CELLS}
        t, j = two_by_two("RULER (13 tasks, string-match score %)", per); md.append(t); js["ruler"] = j
    gt_x = os.path.join(BENCH, "data/xfile_gt.json")
    if os.path.exists(gt_x):
        gt = json.load(open(gt_x))
        for metric in ["f1", "recall", "exact"]:
            per = {c: A.xfile_scores(load(c, "xfile"), gt, metric) for c in CELLS}
            t, j = two_by_two(f"Cross-file call-site retrieval: {metric}", per); md.append(t); js[f"xfile_{metric}"] = j
    text = "\n".join(md)
    print(text)
    if "--md" in args:
        open(args[args.index("--md") + 1], "w").write(text)
    if "--json" in args:
        json.dump(js, open(args[args.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
