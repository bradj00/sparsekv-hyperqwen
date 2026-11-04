#!/usr/bin/env python
"""Score short-suite outputs.

  short_score.py <suite> GT.json OUT1.jsonl [OUT2 ...] [--json result.json] [--lcb-workers 6]

Output lines: {"id", "cfg", "outputs": [completion], "finish": [...], "usage": [{...}], "error": null|str}
Completion = "<reasoning>...</think>\\n\\n<final>"; only the text after the LAST </think> is scored.
No </think>  -> truncated inside reasoning -> scored wrong (counted in n_no_think).
finish == "length" (even with </think>) is also counted as truncated (n_truncated) but the final part is scored.
Lines with a non-null "error" are excluded (listed under "errors") and count as missing.

Scorers:
  gpqa    : simple-evals ANSWER_PATTERN_MULTICHOICE  (?i)Answer[ \\t]*:[ \\t]*\\$?([A-D])\\$?  via re.search (first
            match) on the final answer with "**" removed; fallback: last \\boxed{X} (X in A-D).
  mmlupro : MMLU-Pro evaluate_from_api.py: response.replace('**',''); re.search(r"answer is \\(?([A-J])\\)?");
            fallback re.search(r'.*[aA]nswer:\\s*([A-J])'); else wrong.  ("lenient" adds the official
            extract_final last-standalone-letter fallback, reported separately.)
  ifbench : IFBench evaluation_lib.test_instruction_following_strict / _loose (strict first, same InputExample,
            like run_eval.py); prompt-level + instruction-level, strict and loose.
  lcb     : lcb_runner extraction_utils.extract_code (last ``` fenced block) + testing_util.run_test on
            public+private tests in a sandboxed subprocess (lcb_worker.py); pass = all results > 0
            (pass_k_utils); timeout 6 s/test, global (timeout+1)*n_tests+5 s like check_correctness.

Python API:  score_file(suite, gt, path, cfg=None) -> {id: score}
  score is 0/1 for gpqa/lcb/mmlupro; for ifbench {"strict": 0/1, "loose": 0/1}.
  score_records(...) returns the full per-id records.
"""
import argparse, collections, copy, json, os, re, shutil, signal, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from short_common import BENCH, IFBENCH_REPO, LCB_REPO, decode_lcb_private, split_final

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ------------------------------------------------------------------ GPQA
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
BOXED_ABCD = r"\\boxed\{\s*(?:\\text\{)?\s*([A-D])\s*\}?\s*\}"


def gpqa_extract(fin):
    t = fin.replace("**", "")
    m = re.search(ANSWER_PATTERN_MULTICHOICE, t)
    if m:
        return m.group(1).upper(), "answer_line"
    bx = re.findall(BOXED_ABCD, t)
    if bx:
        return bx[-1], "boxed"
    return None, None


# ------------------------------------------------------------------ MMLU-Pro
def mmlupro_extract(fin, lenient=False):
    t = fin.replace("**", "")
    m = re.search(r"answer is \(?([A-J])\)?", t)
    if m:
        return m.group(1)
    m = re.search(r'.*[aA]nswer:\s*([A-J])', t)
    if m:
        return m.group(1)
    if lenient:  # official extract_final
        m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", t, re.DOTALL)
        if m:
            return m.group(0)
    return None


# ------------------------------------------------------------------ IFBench
_ifb = None


def _ifbench_lib():
    global _ifb
    if _ifb is None:
        if IFBENCH_REPO not in sys.path:
            sys.path.insert(0, IFBENCH_REPO)
        import evaluation_lib
        _ifb = evaluation_lib
    return _ifb


def ifbench_eval(g, fin):
    el = _ifb or _ifbench_lib()
    inp = el.InputExample(key=g["key"], instruction_id_list=list(g["instruction_id_list"]), prompt=g["prompt"],
                          kwargs=copy.deepcopy(g["kwargs"]))
    p2r = {} if fin is None else {g["prompt"]: fin}
    s = el.test_instruction_following_strict(inp, p2r)
    l = el.test_instruction_following_loose(inp, p2r)
    return s.follow_all_instructions, list(s.follow_instruction_list), l.follow_all_instructions, list(l.follow_instruction_list)


# ------------------------------------------------------------------ LCB
def lcb_extract_code(model_output):
    """lcb_runner/utils/extraction_utils.py extract_code (non-CodeLLaMa, non-base branch)."""
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1: indexlines[-1]])


def lcb_sample(g):
    pub = json.loads(g["public_test_cases"])
    priv = decode_lcb_private(g["private_test_cases"])
    tests = pub + priv
    return {"input_output": json.dumps({"inputs": [t["input"] for t in tests],
                                        "outputs": [t["output"] for t in tests],
                                        "fn_name": g.get("fn_name")})}, len(tests)


_HAVE_UNSHARE = None


def _unshare_ok():
    global _HAVE_UNSHARE
    if _HAVE_UNSHARE is None:
        try:
            _HAVE_UNSHARE = subprocess.run(["unshare", "-rn", "true"], timeout=10).returncode == 0
        except Exception:
            _HAVE_UNSHARE = False
    return _HAVE_UNSHARE


def lcb_run(sample, n_tests, code, timeout=6, mem_gb=2.5):
    """Run code against tests in a sandboxed subprocess. Returns (results list, metadata dict)."""
    gto = (timeout + 1) * n_tests + 5
    root = f"{BENCH}/tmp/lcb_sandbox"
    os.makedirs(root, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=root)
    res_path = os.path.join(tmp, "result.json")
    job = json.dumps({"sample": sample, "code": code, "timeout": timeout,
                      "mem_bytes": int(mem_gb * (1 << 30)), "cpu_secs": gto + 5,
                      "testing_util": os.path.join(LCB_REPO, "lcb_runner", "evaluation", "testing_util.py")}).encode()
    cmd = ([ "unshare", "-rn"] if _unshare_ok() else []) + [PY, "-I", os.path.join(HERE, "lcb_worker.py"), res_path]
    env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2", "LANG": "C.UTF-8"}
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         cwd=tmp, env=env, start_new_session=True)
    timed_out = False
    try:
        p.communicate(job, timeout=gto)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)  # explicit pgid of the child we started
        except ProcessLookupError:
            pass
        p.wait()
    try:
        out = json.load(open(res_path))
        res, meta = out["result"], out["metadata"]
    except Exception:
        # same as check_correctness: no result -> all tests failed (-1)
        res, meta = [-1] * n_tests, {"error": "global timeout" if timed_out else f"worker died rc={p.returncode}"}
    shutil.rmtree(tmp, ignore_errors=True)
    return res, meta


def lcb_passed(res):
    return bool(res) and all((r is True) or (not isinstance(r, bool) and isinstance(r, (int, float)) and r > 0)
                             for r in res)


# ------------------------------------------------------------------ generic
def load_gt(gt):
    return json.load(open(gt)) if isinstance(gt, str) else gt


def load_outputs(paths, cfg=None):
    """-> {cfg: {id: line}}; first non-error line per (cfg,id) wins, else the last error line."""
    by = collections.defaultdict(dict)
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            c = o.get("cfg") or "default"
            if cfg is not None and c != cfg:
                continue
            prev = by[c].get(o["id"])
            if prev is None or (prev.get("error") and not o.get("error")):
                by[c][o["id"]] = o
    return by


def score_records(suite, gt, lines, lcb_workers=6, lcb_mem_gb=2.5, lcb_timeout=6):
    """lines: {id: output line}. Returns {id: record}."""
    gt = load_gt(gt)
    recs, lcb_jobs = {}, []
    for rid, o in lines.items():
        if rid not in gt or o.get("error"):
            continue
        text = (o.get("outputs") or [""])[0]
        finish = (o.get("finish") or [None])[0]
        usage = (o.get("usage") or [{}])[0] or {}
        fin, has_think = split_final(text)
        r = {"score": 0, "no_think": not has_think, "length": finish == "length",
             "truncated": (not has_think) or finish == "length",
             "completion_tokens": usage.get("completion_tokens")}
        g = gt[rid]
        if suite == "gpqa":
            pred, how = gpqa_extract(fin) if fin is not None else (None, None)
            r.update(pred=pred, how=how, gold=g["answer"], score=int(pred == g["answer"]))
        elif suite == "mmlupro":
            pred = mmlupro_extract(fin) if fin is not None else None
            predl = mmlupro_extract(fin, lenient=True) if fin is not None else None
            r.update(pred=pred, gold=g["answer"], score=int(pred == g["answer"]),
                     score_lenient=int(predl == g["answer"]), category=g["category"])
        elif suite == "ifbench":
            s, sl, l, ll = ifbench_eval(g, fin)
            r.update(strict=int(s), loose=int(l), strict_instr=sl, loose_instr=ll, score=int(l))
        elif suite == "lcb":
            code = lcb_extract_code(fin) if fin is not None else ""
            r.update(difficulty=g["difficulty"], code_found=bool(code.strip()))
            if code.strip():
                lcb_jobs.append((rid, code))
        else:
            raise ValueError(suite)
        recs[rid] = r
    if lcb_jobs:
        workers = max(1, min(12, lcb_workers))

        def run(job):
            rid, code = job
            sample, n = lcb_sample(gt[rid])
            res, meta = lcb_run(sample, n, code, timeout=lcb_timeout, mem_gb=lcb_mem_gb)
            return rid, res, meta, n

        with ThreadPoolExecutor(workers) as ex:
            for rid, res, meta, n in ex.map(run, lcb_jobs):
                ok = lcb_passed(res)  # official: np.all(results > 0)
                recs[rid].update(score=int(ok), n_tests=n, n_pass=sum(1 for x in res if x is True or (
                    not isinstance(x, bool) and isinstance(x, (int, float)) and x > 0)),
                    first_fail=next((x for x in res if not (x is True)), None),
                    meta={k: (str(v)[:300]) for k, v in (meta or {}).items()})
    return recs


def score_file(suite, gt, path, cfg=None, **kw):
    """-> {id: score}. 0/1 for gpqa/lcb/mmlupro; {"strict":0/1,"loose":0/1} for ifbench.
    If the file has several cfgs, pass cfg= (otherwise ids from the first cfg are scored)."""
    by = load_outputs([path], cfg)
    if not by:
        return {}
    c = cfg if cfg is not None else next(iter(by))
    recs = score_records(suite, gt, by[c], **kw)
    if suite == "ifbench":
        return {k: {"strict": v["strict"], "loose": v["loose"]} for k, v in recs.items()}
    return {k: v["score"] for k, v in recs.items()}


def summarize(suite, gt, recs, lines):
    gt = load_gt(gt)
    ids = list(gt)
    n = len(ids)
    scored = [i for i in ids if i in recs]
    errors = sorted(i for i, o in lines.items() if i in gt and o.get("error") and i not in recs)
    missing = sorted(i for i in ids if i not in recs)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    ct = [recs[i]["completion_tokens"] for i in scored if recs[i]["completion_tokens"] is not None]
    s = {"suite": suite, "n_gt": n, "n_scored": len(scored), "n_missing": len(missing), "n_errors": len(errors),
         "accuracy": mean([recs[i]["score"] for i in scored]),
         "accuracy_missing_as_wrong": sum(recs[i]["score"] for i in scored) / n if n else None,
         "n_truncated": sum(recs[i]["truncated"] for i in scored),
         "n_no_think": sum(recs[i]["no_think"] for i in scored),
         "n_finish_length": sum(recs[i]["length"] for i in scored),
         "mean_completion_tokens": mean(ct), "max_completion_tokens": max(ct) if ct else None,
         "missing": missing, "errors": errors}
    if suite == "ifbench":
        s["prompt_level_strict"] = mean([recs[i]["strict"] for i in scored])
        s["prompt_level_loose"] = mean([recs[i]["loose"] for i in scored])
        si = [x for i in scored for x in recs[i]["strict_instr"]]
        li = [x for i in scored for x in recs[i]["loose_instr"]]
        s["instruction_level_strict"] = mean(si)
        s["instruction_level_loose"] = mean(li)
        s["instr_counts"] = {"total": len(si), "strict_followed": sum(si), "loose_followed": sum(li)}
        s["accuracy"] = s["prompt_level_loose"]
        s["accuracy_note"] = "accuracy = prompt-level loose; see prompt_level_strict"
        s["per_id"] = {i: {"strict": recs[i]["strict"], "loose": recs[i]["loose"],
                           "strict_instr": recs[i]["strict_instr"], "loose_instr": recs[i]["loose_instr"],
                           "truncated": recs[i]["truncated"]} for i in scored}
    else:
        s["per_id"] = {i: recs[i]["score"] for i in scored}
        s["per_id_detail"] = {i: {k: v for k, v in recs[i].items() if k != "score"} for i in scored}
    if suite == "mmlupro":
        s["accuracy_lenient"] = mean([recs[i]["score_lenient"] for i in scored])
        bc = collections.defaultdict(list)
        for i in scored:
            bc[recs[i]["category"]].append(recs[i]["score"])
        s["by_category"] = {k: mean(v) for k, v in sorted(bc.items())}
    if suite == "lcb":
        bd = collections.defaultdict(list)
        for i in scored:
            bd[recs[i]["difficulty"]].append(recs[i]["score"])
        s["by_difficulty"] = {k: {"n": len(v), "pass@1": mean(v)} for k, v in sorted(bd.items())}
        s["n_no_code_block"] = sum(1 for i in scored if not recs[i]["code_found"])
    if suite == "gpqa":
        s["n_boxed_fallback"] = sum(1 for i in scored if recs[i].get("how") == "boxed")
        s["n_no_answer_extracted"] = sum(1 for i in scored if recs[i].get("pred") is None)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["gpqa", "lcb", "ifbench", "mmlupro"])
    ap.add_argument("gt")
    ap.add_argument("outs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--lcb-workers", type=int, default=6, help="sandbox processes (max 12)")
    ap.add_argument("--lcb-mem-gb", type=float, default=2.5, help="RLIMIT_AS per sandbox process")
    ap.add_argument("--lcb-timeout", type=int, default=6, help="per-test timeout (LCB default 6)")
    a = ap.parse_args()
    gt = load_gt(a.gt)
    by = load_outputs(a.outs, a.cfg)
    result = {"suite": a.suite, "gt": a.gt, "outs": a.outs, "cfgs": {}}
    for c, lines in by.items():
        recs = score_records(a.suite, gt, lines, a.lcb_workers, a.lcb_mem_gb, a.lcb_timeout)
        s = summarize(a.suite, gt, recs, lines)
        result["cfgs"][c] = s
        brief = {k: v for k, v in s.items() if k not in ("per_id", "per_id_detail", "missing", "errors")}
        brief["missing"] = s["missing"][:20] + (["..."] if len(s["missing"]) > 20 else [])
        brief["errors"] = s["errors"][:20]
        print(f"== {a.suite} cfg={c}")
        print(json.dumps(brief, indent=1))
    if a.json:
        json.dump(result, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
