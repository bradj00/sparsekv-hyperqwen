#!/usr/bin/env python3
"""analyze.py BENCH_DIR [--md results.md] [--json results.json]

Scores every suite found under out/<cfg>/<suite>.g*.jsonl and prints tables:
per config x length: mean score with a 95% bootstrap CI (resampling items), and the paired difference vs dense on
the items both configs answered (95% paired-bootstrap CI). 'densex' = dense run on the OTHER card (noise floor).

Short suites (gpqa, lcb, ifbench, mmlupro): if every record carries a "score" field (the published per-item records do,
computed by short_score.py with the full ground truth, see package_results.py), that stored score is used and the
ground-truth file is optional; otherwise short_score.py rescores from data/short_gt/<suite>.json.
"""
import glob, json, os, re, sys
import numpy as np

B = sys.argv[1]
sys.path.insert(0, os.path.join(B, "src"))
CFGS = ["dense", "sparse", "tier8", "tier4"]
CFG_NAME = {"dense": "dense (no SPARSE_KV)", "sparse": "sparse GPU-only", "tier8": "sparse + RAM tier int8",
            "tier4": "sparse + RAM tier int4 (prod)", "densex": "dense, other card (noise floor)"}
RNG = np.random.default_rng(0)
NBOOT = 5000
# optional overrides (used by reproduce.py for variant comparisons): --cfgs a,b,c  --label cfg=Name  --suites s1,s2
_A = sys.argv[2:]
if "--cfgs" in _A:
    CFGS = _A[_A.index("--cfgs") + 1].split(",")
for _i, _x in enumerate(_A):
    if _x == "--label":
        _k, _v = _A[_i + 1].split("=", 1)
        CFG_NAME[_k] = _v
SUITES = _A[_A.index("--suites") + 1].split(",") if "--suites" in _A else None


def want(suite):
    return SUITES is None or suite in SUITES


def load(cfg, suite):
    recs = {}
    for p in sorted(glob.glob(os.path.join(B, "out", cfg, f"{suite}.g*.jsonl"))):
        for l in open(p):
            r = json.loads(l)
            if r.get("error"):
                continue
            recs[r["id"]] = r
    return recs


def boot_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    if len(x) == 1:
        return float(x.mean()), float("nan"), float("nan")
    idx = RNG.integers(0, len(x), (NBOOT, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------- scorers: return {id: (score in [0,1], group dict)} ----------------
def ruler_scores(recs, gt):
    out = {}
    for rid, r in recs.items():
        g = gt[rid]
        pred = re.sub(r"[\x00-\x1f]", "\n", r["outputs"][0].strip()).strip().lower()
        refs = g["outputs"]
        if g["metric"] == "part":
            s = max(1.0 if x.lower() in pred else 0.0 for x in refs)
        else:
            s = sum(1.0 if x.lower() in pred else 0.0 for x in refs) / len(refs)
        out[rid] = (s, {"length": g["length"], "task": g["task"]})
    return out


def parse_pairs(text):
    pairs = set()
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        line = re.sub(r"^[-*\d.)\s]+", "", line)
        if "::" not in line:
            continue
        f, fn = line.split("::", 1)
        f = f.strip().strip("`'\"").lstrip("./")
        fn = fn.strip().strip("`'\"").split("(")[0].strip()
        fn = fn.split(".")[-1].strip()
        if f and fn:
            pairs.add((f, fn))
    return pairs


def xfile_prf(text, calls):
    gtp = {(a, b) for a, b in calls}
    pr = parse_pairs(text)
    tp = len(pr & gtp)
    p = tp / len(pr) if pr else 0.0
    rc = tp / len(gtp)
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return p, rc, f1, tp == len(gtp) and len(pr) == len(gtp)


def xfile_bin(n):
    return "1" if n == 1 else "2-3" if n <= 3 else "4-7" if n <= 7 else "8+"


def xfile_scores(recs, gt, metric="f1"):
    out = {}
    for rid, r in recs.items():
        g = gt[rid]
        p, rc, f1, ex = xfile_prf(r["outputs"][0], g["calls"])
        s = {"f1": f1, "precision": p, "recall": rc, "exact": float(ex)}[metric]
        out[rid] = (s, {"length": g["length"], "bin": xfile_bin(g["n_pairs"]), "corpus": g["corpus"]})
    return out


def lbv2_extract(resp):
    resp = resp.replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", resp) or re.search(r"The correct answer is ([A-D])", resp)
    return m.group(1) if m else None


def lbv2_scores(recs, gt):
    return {rid: (float(lbv2_extract(r["outputs"][0]) == gt[rid]["answer"]),
                  {"length": gt[rid]["bin"], "difficulty": gt[rid]["difficulty"]}) for rid, r in recs.items()}


def scbench_scores(recs, gt, turn=None):
    import scbench_score
    out = {}
    os.makedirs(os.path.join(B, "tmp"), exist_ok=True)
    tmp = os.path.join(B, "tmp", "_sc.jsonl")
    with open(tmp, "w") as f:
        for r in recs.values():
            f.write(json.dumps(r) + "\n")
    sc = scbench_score.score_file(gt, tmp)
    for rid, ts in sc.items():
        if rid not in recs or ts is None:
            continue
        ts = [t for t in ts if t is not None]
        if not ts:
            continue
        if turn is None:
            out[rid] = (float(np.mean(ts)), {"length": gt[rid]["task"]})
        elif turn < len(ts):
            out[rid] = (float(ts[turn]), {"length": gt[rid]["task"]})
    return out


def short_scores(recs, gt, suite):
    if recs and all("score" in r for r in recs.values()):  # published records: stored per-item score
        out = {}
        for rid, r in recs.items():
            s = r["score"]
            if isinstance(s, dict):
                s = s.get("loose", s.get("strict"))
            out[rid] = (float(s), {"length": "all"})
        return out
    import short_score
    os.makedirs(os.path.join(B, "tmp"), exist_ok=True)
    tmp = os.path.join(B, "tmp", f"_sh_{suite}.jsonl")
    with open(tmp, "w") as f:
        for r in recs.values():
            f.write(json.dumps(r) + "\n")
    sc = short_score.score_file(suite, gt, tmp)
    out = {}
    for rid, s in sc.items():
        if rid not in recs:
            continue
        if isinstance(s, dict):
            s = s.get("loose", s.get("strict"))
        out[rid] = (float(s), {"length": "all"})
    return out


# ---------------- tables ----------------
def table(title, per_cfg, group_key="length", order=None, cfgs=None, pct=True):
    cfgs = [c for c in (cfgs or CFGS + ["densex"]) if c in per_cfg and per_cfg[c]]
    groups = sorted({g[group_key] for c in cfgs for _, g in per_cfg[c].values()},
                    key=lambda x: (order.index(x) if order and x in order else 99, x))
    groups = [g for g in groups if g != "all"] + ["all"]
    k = 100 if pct else 1
    lines = [f"### {title}", "", "| config | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    res = {}
    for c in cfgs:
        cells = []
        for gname in groups:
            xs = [s for s, g in per_cfg[c].values() if gname == "all" or g[group_key] == gname]
            m, lo, hi = boot_ci(xs)
            res[(c, gname)] = (m, lo, hi, len(xs))
            cells.append("" if not xs else f"{k*m:.1f} [{k*lo:.1f}, {k*hi:.1f}] (n={len(xs)})")
        lines.append(f"| {CFG_NAME.get(c, c)} | " + " | ".join(cells) + " |")
    # paired deltas vs dense
    if "dense" in per_cfg:
        lines += ["", f"Paired difference vs dense (same items; points, 95% bootstrap CI):", "",
                  "| config | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for c in cfgs:
            if c == "dense":
                continue
            cells = []
            for gname in groups:
                ids = [i for i, (s, g) in per_cfg[c].items() if (gname == "all" or g[group_key] == gname)
                       and i in per_cfg["dense"]]
                d = [per_cfg[c][i][0] - per_cfg["dense"][i][0] for i in ids]
                m, lo, hi = boot_ci(d)
                res[(c, gname, "delta")] = (m, lo, hi, len(d))
                cells.append("" if not d else f"{k*m:+.1f} [{k*lo:+.1f}, {k*hi:+.1f}]")
            lines.append(f"| {CFG_NAME.get(c, c)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n", {f"{'|'.join(map(str, kk))}": v for kk, v in res.items()}


def main():
    md, js = [], {}
    LEN = ["32k", "64k", "128k", "192k", "240k"]
    allcfg = CFGS + ["densex"]
    if want("ruler") and os.path.exists(os.path.join(B, "data/ruler_gt.json")):
        gt = json.load(open(os.path.join(B, "data/ruler_gt.json")))
        per = {c: ruler_scores(load(c, "ruler"), gt) for c in allcfg}
        t, j = table("RULER (13 tasks, string-match score %)", per, "length", LEN); md.append(t); js["ruler"] = j
        for ln in LEN[:4]:
            sub = {c: {i: v for i, v in d.items() if v[1]["length"] == ln} for c, d in per.items()}
            if any(sub.values()):
                t, j = table(f"RULER by task at {ln}", sub, "task"); md.append(t); js[f"ruler_{ln}_task"] = j
    if want("xfile") and os.path.exists(os.path.join(B, "data/xfile_gt.json")):
        gt = json.load(open(os.path.join(B, "data/xfile_gt.json")))
        for metric in ["f1", "recall", "precision", "exact"]:
            per = {c: xfile_scores(load(c, "xfile"), gt, metric) for c in allcfg}
            t, j = table(f"Cross-file call-site retrieval: {metric}", per, "length", LEN); md.append(t)
            js[f"xfile_{metric}"] = j
        per = {c: xfile_scores(load(c, "xfile"), gt, "f1") for c in allcfg}
        t, j = table("Cross-file F1 by number of call sites (file, function pairs)", per, "bin", ["1", "2-3", "4-7", "8+"])
        md.append(t); js["xfile_f1_bin"] = j
    if want("lbv2") and os.path.exists(os.path.join(B, "data/lbv2_gt.json")):
        gt = json.load(open(os.path.join(B, "data/lbv2_gt.json")))
        per = {c: lbv2_scores(load(c, "lbv2"), gt) for c in allcfg}
        t, j = table("LongBench v2 subset (accuracy %, no CoT)", per, "length",
                     ["<=32k", "32-64k", "64-128k", "128-192k"]); md.append(t); js["lbv2"] = j
    if want("scbench") and os.path.exists(os.path.join(B, "data/scbench_gt.json")):
        gt = json.load(open(os.path.join(B, "data/scbench_gt.json")))
        per = {c: scbench_scores(load(c, "scbench"), gt) for c in allcfg}
        if any(per.values()):
            t, j = table("SCBench multi-turn (mean over turns, official metric %)", per, "length"); md.append(t)
            js["scbench"] = j
            for turn in range(5):
                pt = {c: scbench_scores(load(c, "scbench"), gt, turn) for c in allcfg}
                t, j = table(f"SCBench turn {turn+1}", pt, "length"); md.append(t); js[f"scbench_t{turn+1}"] = j
    for suite in ["gpqa", "lcb", "ifbench", "mmlupro"]:
        p = os.path.join(B, "data/short_gt", f"{suite}.json")
        if want(suite) and any(glob.glob(os.path.join(B, "out", c, f"{suite}.g*.jsonl")) for c in allcfg):
            gt = json.load(open(p)) if os.path.exists(p) else {}
            per = {c: short_scores(load(c, suite), gt, suite) for c in allcfg}
            t, j = table(f"{suite}", per, "length"); md.append(t); js[suite] = j
    text = "\n".join(md)
    print(text)
    a = sys.argv[2:]
    if "--md" in a:
        open(a[a.index("--md") + 1], "w").write(text)
    if "--json" in a:
        json.dump(js, open(a[a.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
