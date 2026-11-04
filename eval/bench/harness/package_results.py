#!/usr/bin/env python3
"""package_results.py OUT_DIR GT_DIR MANIFEST.json DST_RESULTS_DIR

Turns the raw client output (OUT_DIR/<config_dir>/<file>.jsonl, written by client.py) into the published per-item
records under DST_RESULTS_DIR/<config_dir>/<file>.jsonl[.gz], adding per-item scores computed with the harness's own
scorers and the FULL ground truth in GT_DIR (ruler/xfile/lbv2/scbench *_gt.json + short_gt/{gpqa,lcb,ifbench,mmlupro}.json).

What is kept per record: id, variant (public config name), config_dir, shard, gpu, job, outputs (model text, all turns),
finish, usage (token counts), ttft, secs, spec (spec-decode + prefix-cache counter deltas), error, score, score_detail.
What is dropped:
  - prompts / contexts (never stored in the client output in the first place);
  - GPQA model output text (reasoning traces restate the gated questions); the extracted answer and correctness stay;
  - the raw "cfg" field (= server_cfg in the manifest).
Records are gzipped when the uncompressed file exceeds 256 KB.
Scores: ruler = analyze.py string match; xfile = F1 (+precision/recall/exact); lbv2 = exact letter; scbench = mean of
per-turn official metric; gpqa/mmlupro = extracted letter == gold; ifbench = loose prompt-level (strict also kept);
lcb = all public+private tests pass (sandboxed). mtp (speed run) and s6_resume (resume smoke test) are unscored.
"""
import gzip, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT, GT, MAN, DST = sys.argv[1:5]
sys.argv = [sys.argv[0], os.path.dirname(OUT.rstrip("/"))]  # analyze.py reads a bench dir from argv[1] at import
import analyze as A  # noqa: E402
sys.path.insert(0, HERE)  # this harness first (analyze.py prepended BENCH/src)
import scbench_score  # noqa: E402
import short_score  # noqa: E402

man = json.load(open(MAN))
gt = {k: json.load(open(os.path.join(GT, f"{k}_gt.json"))) for k in ("ruler", "xfile", "lbv2", "scbench")}
sgt = {k: json.load(open(os.path.join(GT, "short_gt", f"{k}.json"))) for k in ("gpqa", "lcb", "ifbench", "mmlupro")}


def suite_of(key):
    cdir, fname = key.split("/")
    if cdir == "s6_basis":
        return "xfile"
    if cdir == "s6_resume":
        return "s6_resume"
    return fname.split(".")[0]


def score(suite, recs):
    ok = {i: r for i, r in recs.items() if not r.get("error")}
    det = {}
    if suite == "ruler":
        for i, (s, _) in A.ruler_scores(ok, gt["ruler"]).items():
            det[i] = (s, {})
    elif suite == "xfile":
        for i, r in ok.items():
            p, rc, f1, ex = A.xfile_prf(r["outputs"][0], gt["xfile"][i]["calls"])
            det[i] = (f1, {"f1": f1, "precision": p, "recall": rc, "exact": int(ex)})
    elif suite == "lbv2":
        for i, r in ok.items():
            pred = A.lbv2_extract(r["outputs"][0])
            det[i] = (float(pred == gt["lbv2"][i]["answer"]), {"pred": pred})
    elif suite == "scbench":
        recs_sc = scbench_score.score_records(gt["scbench"], list(ok.values()))
        for i, v in recs_sc.items():
            ts = [t for t in (v["scores"] or []) if t is not None]
            if i in ok and ts:
                det[i] = (sum(ts) / len(ts), {"turn_scores": v["scores"]})
    elif suite in ("gpqa", "lcb", "ifbench", "mmlupro"):
        for i, v in short_score.score_records(suite, sgt[suite], ok).items():
            d = {"truncated": v["truncated"], "no_think": v["no_think"]}
            if suite == "gpqa":
                d.update(pred=v["pred"])
            elif suite == "mmlupro":
                d.update(pred=v["pred"], score_lenient=v["score_lenient"])
            elif suite == "ifbench":
                d.update(strict=v["strict"], loose=v["loose"], strict_instr=v["strict_instr"], loose_instr=v["loose_instr"])
            elif suite == "lcb":
                d.update(code_found=v["code_found"], n_tests=v.get("n_tests"), n_pass=v.get("n_pass"))
            det[i] = (float(v["score"]), d)
    return det


index = {}
for key in sorted(man):
    m = man[key]
    src = os.path.join(OUT, key)
    suite = suite_of(key)
    raw = {}
    order = []
    for line in open(src):
        r = json.loads(line)
        order.append(r)
    recs = {}
    for r in order:  # same as analyze.load(): the last non-error record per id is the one scored
        if r["id"] not in recs or not r.get("error"):
            recs[r["id"]] = r
    dup = {i for i in recs if sum(1 for r in order if r["id"] == i and not r.get("error")) > 1}
    det = score(suite, recs)
    rows = []
    for r in order:
        o = {"id": r["id"], "variant": m["variant"], "config_dir": m["config_dir"], "shard": m["shard"],
             "gpu": r.get("gpu"), "job": m["jobs"][-1] if len(m["jobs"]) == 1 else m["jobs"],
             "outputs": r["outputs"], "finish": r["finish"], "usage": r["usage"], "ttft": r["ttft"],
             "secs": r["secs"], "spec": r["spec"], "error": r["error"]}
        if suite == "gpqa":
            o["outputs"] = None
            o["outputs_note"] = "GPQA output text withheld (gated dataset; traces restate the question)"
        if r is recs[r["id"]] and r["id"] in det:
            o["score"], o["score_detail"] = det[r["id"]]
        elif r["id"] in dup:
            o["score_detail"] = {"note": "duplicate request id (RULER generator index collision); only the last "
                                         "record with this id matches the ground truth and is scored"}
        if suite == "mmlupro_hs" or m["config_dir"].endswith("_hs"):
            o["suite"] = "mmlupro_hs"
        else:
            o["suite"] = suite
        rows.append(o)
    blob = "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in rows).encode()
    os.makedirs(os.path.join(DST, m["config_dir"]), exist_ok=True)
    dst = os.path.join(DST, key)
    if len(blob) > 256 * 1024:
        dst += ".gz"
        with gzip.GzipFile(dst, "wb", mtime=0) as f:
            f.write(blob)
    else:
        open(dst, "wb").write(blob)
    n_ok = sum(1 for o in rows if not o["error"])
    index[os.path.relpath(dst, DST)] = dict(m, suite=rows[0]["suite"] if rows else suite, records=len(rows),
                                            records_ok=n_ok, scored=sum(1 for o in rows if "score" in o),
                                            sha256_uncompressed=hashlib.sha256(blob).hexdigest())
    print(f"{key:36s} {m['variant']:26s} n={len(rows):4d} ok={n_ok:4d} scored={index[os.path.relpath(dst, DST)]['scored']:4d}"
          f" -> {os.path.relpath(dst, DST)}", flush=True)
json.dump(index, open(os.path.join(DST, "MANIFEST.json"), "w"), indent=1, sort_keys=True)
