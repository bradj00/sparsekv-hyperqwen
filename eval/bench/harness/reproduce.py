#!/usr/bin/env python3
"""reproduce.py [--out RESULTS.md] [--json RESULTS.json] [--kld-md FILE ...]

Regenerates every table in RESULTS.md from the published per-item records (../results) and ground truth (../gt),
using the harness's own scorers. analyze.py scores fixed config names out of BENCH/out/<cfg>/<suite>.g*.jsonl, so
each comparison ("view") below is materialised as a temporary BENCH directory holding renamed, decompressed copies of
the published files; analyze.py / q4_analyze.py then run on it unchanged. Deterministic (bootstrap RNG seed 0).

GPQA / LiveCodeBench / IFBench ground truth is not redistributed; for those suites analyze.py uses the per-item
"score" stored in each record (computed by package_results.py with the full ground truth).
"""
import argparse, gzip, json, os, re, shutil, subprocess, sys, tempfile


def h4(text):
    return re.sub(r"^#+ ", "#### ", text, flags=re.M)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
GT = os.path.join(ROOT, "gt")
PY = sys.executable

SH = ("g0", "g1")
CORE = ["xfile", "ruler", "lbv2", "scbench", "gpqa", "lcb", "ifbench", "mmlupro"]


def files(cdir, suite, shards=SH, src_suite=None):
    return [(f"{cdir}/{src_suite or suite}.{s}.jsonl", f"{suite}.{s}.jsonl") for s in shards]


# view = (title, intro, {view_cfg: [(published file, name in view)]}, cfgs, labels, suites, extra analyze args)
VIEWS = []
VIEWS.append(("Headline: fixed release vs dense, both shards where available",
              "dense = no SPARSE_* at all; sparse GPU-only = sparse decode, no RAM tier; fixed int8/int4 = sparse + RAM tier "
              "with the fix (SPARSE_TIER_SUFFIX_SPARSE=0, SPARSE_STATS_SUB=1, SPARSE_SELECT_EVERY=1). Fixed int4 has no "
              "shard-0 cross-file or RULER run; its RULER row is restricted to the 107 shard-1 items the other configs ran "
              "(the full 241-item run is in the RULER fix view). LongBench v2 fixed int4 = shard 0 only (shard 1 ran the "
              "first release).",
              {"dense": sum((files("dense", s) for s in CORE), []),
               "sparse": sum((files("sparse", s) for s in ["xfile", "ruler", "lbv2", "scbench"]), []),
               "fixed8": [("tier8/xfile.g0.jsonl", "xfile.g0.jsonl"), ("fix8/xfile.g1.jsonl", "xfile.g1.jsonl")]
                         + sum((files("tier8", s) for s in ["ruler", "lbv2", "scbench"]), []),
               "fixed4": [("abl_sfxoff/xfile.g1.jsonl", "xfile.g1.jsonl"),
                          ("fix/ruler.g1.jsonl", "ruler.g1.jsonl", "ids_of:dense/ruler.g1.jsonl"),
                          ("tier4/lbv2.g0.jsonl", "lbv2.g0.jsonl")]
                         + sum((files("tier4", s) for s in ["scbench", "gpqa", "lcb", "ifbench", "mmlupro"]), [])},
              ["dense", "sparse", "fixed8", "fixed4"],
              {"dense": "dense", "sparse": "sparse GPU-only", "fixed8": "fixed, RAM tier int8", "fixed4": "fixed, RAM tier int4"},
              None))
VIEWS.append(("MMLU-Pro with thinking after a 150K-token haystack (mmlupro_hs)",
              "48 MMLU-Pro questions asked after the same ~150K-token block of unrelated LongBench v2 documents; thinking on, "
              "sampled (T=1.0, top-p 0.95, top-k 20, per-item seed), 32K max tokens, concurrency 1. Tier = fixed int4.",
              {"dense": files("dense_hs", "mmlupro"), "sparse": files("sparse_hs", "mmlupro"),
               "fixed4": files("tier4_hs", "mmlupro")},
              ["dense", "sparse", "fixed4"],
              {"dense": "dense", "sparse": "sparse GPU-only", "fixed4": "fixed, RAM tier int4"}, ["mmlupro"]))
VIEWS.append(("Before the fix: first release (as shipped) vs dense",
              "release int4/int8 = the production 4 x 250K command of the first release (sparse suffix prefill on, "
              "SPARSE_STATS_SUB=4, SPARSE_SELECT_EVERY=2). Release int8 exists only for cross-file shard 1; LongBench v2 "
              "release int4 only for shard 1.",
              {"dense": sum((files("dense", s) for s in ["xfile", "ruler", "lbv2"]), []),
               "sparse": sum((files("sparse", s) for s in ["xfile", "ruler", "lbv2"]), []),
               "release8": [("tier8/xfile.g1.jsonl", "xfile.g1.jsonl")],
               "release4": files("tier4", "xfile") + files("tier4", "ruler") + [("tier4/lbv2.g1.jsonl", "lbv2.g1.jsonl")]},
              ["dense", "sparse", "release8", "release4"],
              {"dense": "dense", "sparse": "sparse GPU-only", "release8": "first release, RAM tier int8",
               "release4": "first release, RAM tier int4"}, ["xfile", "ruler", "lbv2"]))
X1 = [("abl_knobs", "release int4 + knobs off (STATS_SUB=1, SELECT_EVERY=1)"),
      ("abl_knobs_sk8", "release int4 + knobs off + int8 sketch"),
      ("release8", "release int8"),
      ("abl_knobs_tier8", "release int8 + knobs off"),
      ("abl_sfx16k", "knobs off + suffix KSEL 16384 (2/60 answered, HTTP 500)"),
      ("abl_sfxoff", "knobs off + suffix sparse off (= fixed int4)"),
      ("fix_knobson", "suffix sparse off, knobs on (STATS_SUB=4, SELECT_EVERY=2)"),
      ("fix8", "fixed int8"),
      ("sparse", "sparse GPU-only"),
      ("s5_serial", "S5: fixed + serial staging"),
      ("s5_exact", "S5: fixed + serial staging + exact suffix + DMA"),
      ("s5_hyb128", "S5: hybrid suffix (128 exact rows), knobs off, suffix sparse on"),
      ("s6_old", "S6 control: fixed + serial staging, release basis"),
      ("s6_lmse", "S6: fixed + serial staging, query-aware logit-MSE basis")]
VIEWS.append(("Cross-file ablation ladder (shard 1, RTX 3090, concurrency 1)",
              "Every run below is the same 60 shard-1 cross-file items on the same card, in the order they were run "
              "(see LOG.md). Paired deltas are vs dense on the same items.",
              dict({"dense": [("dense/xfile.g1.jsonl", "xfile.g1.jsonl")],
                    "release4": [("tier4/xfile.g1.jsonl", "xfile.g1.jsonl")],
                    "release8": [("tier8/xfile.g1.jsonl", "xfile.g1.jsonl")],
                    "sparse": [("sparse/xfile.g1.jsonl", "xfile.g1.jsonl")],
                    "s6_old": [("s6_basis/old.g1.jsonl", "xfile.g1.jsonl")],
                    "s6_lmse": [("s6_basis/lmse.g1.jsonl", "xfile.g1.jsonl")]},
                   **{c: [(f"{c}/xfile.g1.jsonl", "xfile.g1.jsonl")] for c, _ in X1
                      if c not in ("release8", "sparse", "s6_old", "s6_lmse")}),
              ["dense", "release4"] + [c for c, _ in X1],
              dict({"dense": "dense", "release4": "release int4"}, **dict(X1)), ["xfile"]))
VIEWS.append(("Four agents at once (concurrency 4), cross-file shard 1, RTX 3090 Ti",
              "S5 staging variants with 4 concurrent requests (the first-release tier OOM-crashed under concurrent "
              "staging; see LOG.md 2026-11-03 02:01Z). Dense = concurrency-1 shard-1 run on the RTX 3090 (cross-card).",
              {"dense": [("dense/xfile.g1.jsonl", "xfile.g1.jsonl")],
               "s5t_s4ref": [("s5t_s4ref/xfile.g1c4.jsonl", "xfile.g1c4.jsonl")],
               "s5t_serial": [("s5t_serial/xfile.g1c4.jsonl", "xfile.g1c4.jsonl")],
               "s5t_exact": [("s5t_exact/xfile.g1c4.jsonl", "xfile.g1c4.jsonl")],
               "s5t_hyb128": [("s5t_hyb128/xfile.g1c4.jsonl", "xfile.g1c4.jsonl")]},
              ["dense", "s5t_s4ref", "s5t_serial", "s5t_exact", "s5t_hyb128"],
              {"dense": "dense (conc 1)", "s5t_s4ref": "release int4 + serial staging", "s5t_serial": "fixed + serial staging",
               "s5t_exact": "fixed + serial staging + exact suffix + DMA", "s5t_hyb128": "hybrid suffix, 128 exact rows"},
              ["xfile"]))
VIEWS.append(("MMLU-Pro after 150K at concurrency 4 (crash repro), shard 0, RTX 3090 Ti",
              "The job that crashed the fixed tier at 01:34Z, re-run with S5 serial staging at concurrency 4.",
              {"dense": [("dense_hs/mmlupro.g0.jsonl", "mmlupro.g0.jsonl")],
               "fixed4": [("tier4_hs/mmlupro.g0.jsonl", "mmlupro.g0.jsonl")],
               "s5_serial": [("s5_serial_hs/mmlupro.g0c4.jsonl", "mmlupro.g0c4.jsonl")],
               "s5_exact": [("s5_exact_hs/mmlupro.g0c4.jsonl", "mmlupro.g0c4.jsonl")]},
              ["dense", "fixed4", "s5_serial", "s5_exact"],
              {"dense": "dense (conc 1)", "fixed4": "fixed int4 (conc 1)", "s5_serial": "fixed + serial staging (conc 4)",
               "s5_exact": "fixed + serial staging + exact suffix (conc 4)"}, ["mmlupro"]))
VIEWS.append(("RULER, shard 1: fixed int4 on the full shard vs first release",
              "fixed int4 ran all 241 shard-1 RULER items (the other configs ran a 107-item subset: 6/6/4/2 items per "
              "task at 32/64/128/192K). Per-config means use each config's own items; paired deltas use the overlap.",
              {"dense": [("dense/ruler.g1.jsonl", "ruler.g1.jsonl")],
               "release4": [("tier4/ruler.g1.jsonl", "ruler.g1.jsonl")],
               "fixed4": [("fix/ruler.g1.jsonl", "ruler.g1.jsonl")],
               "fixed8": [("tier8/ruler.g1.jsonl", "ruler.g1.jsonl")],
               "sparse": [("sparse/ruler.g1.jsonl", "ruler.g1.jsonl")]},
              ["dense", "sparse", "fixed8", "fixed4", "release4"],
              {"dense": "dense", "sparse": "sparse GPU-only", "fixed8": "fixed int8", "fixed4": "fixed int4 (241 items)",
               "release4": "first release int4"}, ["ruler"]))


def read(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f]


def pub(path):
    p = os.path.join(RES, path)
    return p if os.path.exists(p) else p + ".gz"


def build(view_files, tmp):
    os.makedirs(os.path.join(tmp, "out"))
    os.symlink(HERE, os.path.join(tmp, "src"))
    os.symlink(GT, os.path.join(tmp, "data"))
    os.makedirs(os.path.join(tmp, "tmp"))
    for cfg, lst in view_files.items():
        d = os.path.join(tmp, "out", cfg)
        os.makedirs(d, exist_ok=True)
        for ent in lst:
            src, dst = ent[0], ent[1]
            recs = read(pub(src))
            if len(ent) > 2 and ent[2].startswith("ids_of:"):
                keep = {r["id"] for r in read(pub(ent[2][7:]))}
                recs = [r for r in recs if r["id"] in keep]
            with open(os.path.join(d, dst), "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")


def run_analyze(tmp, cfgs, labels, suites):
    cmd = [PY, os.path.join(HERE, "analyze.py"), tmp, "--cfgs", ",".join(cfgs), "--json", os.path.join(tmp, "a.json")]
    for k, v in labels.items():
        cmd += ["--label", f"{k}={v}"]
    if suites:
        cmd += ["--suites", ",".join(suites)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return out, json.load(open(os.path.join(tmp, "a.json")))


def mtp_table():
    """Decode speed with MTP-7 speculative decoding on (served config), free-running greedy 512-token 'explain'."""
    rows = {}
    for cdir, lab in [("dense", "dense"), ("sparse", "sparse GPU-only"), ("tier4", "fixed, RAM tier int4")]:
        for s in SH:
            for r in read(pub(f"{cdir}/mtp.{s}.jsonl")):
                ln, doc = r["id"].split("/")[1:3]
                u, sp = r["usage"][0], (r["spec"] or [{}])[0]
                dec = r["secs"][0] - r["ttft"][0]
                rows.setdefault((lab, ln), []).append(dict(
                    tps=(u["completion_tokens"] - 1) / dec if dec > 0 else float("nan"), ttft=r["ttft"][0],
                    acc=sp.get("spec_decode_num_accepted_tokens", 0), dft=sp.get("spec_decode_num_draft_tokens", 0),
                    drafts=sp.get("spec_decode_num_drafts", 0), gpu=r["gpu"], doc=doc))
    lens = ["32k", "64k", "128k", "192k", "240k"]
    labs = ["dense", "sparse GPU-only", "fixed, RAM tier int4"]
    md = ["#### MTP-7 decode speed (tokens/s after the first token; mean over documents, both cards pooled)", "",
          "| config | " + " | ".join(lens) + " |", "|---|" + "---|" * len(lens)]
    js = {}
    for lab in labs:
        cells = []
        for ln in lens:
            x = rows.get((lab, ln), [])
            if not x:
                cells.append("")
                continue
            tps = sum(e["tps"] for e in x) / len(x)
            acc = sum(e["acc"] for e in x) / max(1, sum(e["drafts"] for e in x))
            cells.append(f"{tps:.1f} tok/s, {acc:.2f} acc/step (n={len(x)})")
            js[f"{lab}|{ln}"] = dict(tps=tps, accepted_per_step=acc, n=len(x),
                                     items=[{k: e[k] for k in ("doc", "gpu", "tps", "ttft")} for e in x])
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    md += ["", "acc/step = accepted draft tokens per speculative step (0-7). Items: dense/sparse 32K-192K on 3 "
           "documents (2 on the RTX 3090 Ti, 1 on the RTX 3090); 240K only fits the tier config. Different documents "
           "and cards per cell: treat as indicative, not paired.", ""]
    return "\n".join(md), js


def headline(allj):
    """The README headline table: per suite, dense score and paired deltas vs dense (points, 95% CI, n paired)."""
    H = allj["Headline: fixed release vs dense, both shards where available"]
    HS = allj["MMLU-Pro with thinking after a 150K-token haystack (mmlupro_hs)"]
    R = allj["Before the fix: first release (as shipped) vs dense"]
    rows = [("RULER, 13 tasks, 32K-192K (score)", H, "ruler", R), ("Cross-file call sites, 64K-192K (F1)", H, "xfile_f1", R),
            ("LongBench v2, no CoT (acc.)", H, "lbv2", R), ("SCBench multi-turn, 4 tasks (official metric)", H, "scbench", None),
            ("MMLU-Pro thinking after 150K haystack (acc.)", HS, "mmlupro", None), ("GPQA Diamond, thinking (acc.)", H, "gpqa", None),
            ("LiveCodeBench v5/v6 window, thinking (pass@1)", H, "lcb", None), ("IFBench, thinking (loose prompt acc.)", H, "ifbench", None),
            ("MMLU-Pro, thinking (acc.)", H, "mmlupro", None)]

    def d(js, key, cfg):
        v = js.get(key, {}).get(f"{cfg}|all|delta")
        return "" if not v or v[3] == 0 else f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}] (n={v[3]})"

    md = ["| suite | dense | sparse GPU-only | fixed, RAM tier int8 | **fixed, RAM tier int4** | first release, int4 (before fix) |",
          "|---|---|---|---|---|---|"]
    for lab, js, key, rel in rows:
        dv = js[key]["dense|all"]
        md.append(f"| {lab} | {100 * dv[0]:.1f} (n={dv[3]}) | {d(js, key, 'sparse')} | {d(js, key, 'fixed8')} | "
                  f"**{d(js, key, 'fixed4')}** | {d(rel, key, 'release4') if rel else ''} |")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "RESULTS.md"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "RESULTS.json"))
    ap.add_argument("--kld-md", nargs="*", default=[], help="markdown files from the KLD scorers to append")
    a = ap.parse_args()
    md = ["# Full results", "",
          "Generated by `harness/reproduce.py` from `results/` and `gt/` (do not edit by hand). Cells: mean score in "
          "points (0-100) with a 95% bootstrap CI over items and n; paired tables: difference vs dense on the items "
          "both configs answered, 95% paired-bootstrap CI. Config definitions: [CONFIGS.md](CONFIGS.md).", ""]
    allj = {}
    for k, (title, intro, vf, cfgs, labels, suites) in enumerate(VIEWS, 1):
        tmp = tempfile.mkdtemp(prefix="bench_view_")
        try:
            build(vf, tmp)
            out, js = run_analyze(tmp, cfgs, labels, suites)
            md += [f"## {k}. {title}", "", intro, "", h4(out), ""]
            allj[title] = js
        finally:
            shutil.rmtree(tmp)
    # q4 2x2 (fake-int4 KV) on the bench suites
    tmp = tempfile.mkdtemp(prefix="bench_q4_")
    try:
        build({"dense": [("dense/ruler.g1.jsonl", "ruler.g1.jsonl"), ("dense/xfile.g1.jsonl", "xfile.g1.jsonl")],
               "sparse": [("sparse/ruler.g1.jsonl", "ruler.g1.jsonl"), ("sparse/xfile.g1.jsonl", "xfile.g1.jsonl")],
               "dense_q4": [("dense_q4/ruler.g1.jsonl", "ruler.g1.jsonl"), ("dense_q4/xfile.g1.jsonl", "xfile.g1.jsonl")],
               "sparse_q4": [("sparse_q4/ruler.g1.jsonl", "ruler.g1.jsonl"), ("sparse_q4/xfile.g1.jsonl", "xfile.g1.jsonl")]},
              tmp)
        out = subprocess.run([PY, os.path.join(HERE, "q4_analyze.py"), tmp, "--json", os.path.join(tmp, "q.json")],
                             check=True, capture_output=True, text=True).stdout
        md += [f"## {len(VIEWS) + 1}. Fake-int4 GPU KV x sparse decode (q4 2x2, shard 1)", "",
               "Does sparse decode tolerate a 4-bit KV cache better than dense? `SPARSE_DEBUG_KV_FAKEQ4=1` rounds the "
               "full-attention KV to int4 in place (64-dim groups, fp16 scales) while keeping the int8 storage. Note: "
               "the sparse fake-int4 RULER cell ran on the RTX 3090 Ti, the other three on the RTX 3090 (cross-card).",
               "", h4(out), ""]
        allj["q4"] = json.load(open(os.path.join(tmp, "q.json")))
    finally:
        shutil.rmtree(tmp)
    t, j = mtp_table()
    md += [f"## {len(VIEWS) + 2}. Decode speed with speculative decoding (mtp suite)", "", t]
    allj["mtp"] = j
    for p in a.kld_md:
        md += [open(p).read().strip(), ""]
    open(a.out, "w").write("\n".join(md).rstrip() + "\n")
    readme = os.path.join(ROOT, "README.md")
    if os.path.exists(readme):  # refresh the generated headline table between the markers
        t = open(readme).read()
        b0, b1 = "<!-- HEADLINE:BEGIN (generated by harness/reproduce.py) -->", "<!-- HEADLINE:END -->"
        if b0 in t and b1 in t:
            t = t[: t.index(b0) + len(b0)] + "\n" + headline(allj) + "\n" + t[t.index(b1):]
            open(readme, "w").write(t)
    print(headline(allj))
    json.dump(allj, open(a.json, "w"), indent=1, sort_keys=True)
    print("wrote", a.out, a.json)


if __name__ == "__main__":
    main()
