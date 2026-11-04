#!/usr/bin/env python3
"""Score SCBench multi-turn outputs (from client.py) with SCBench's official per-task metrics.

Usage: scbench_score.py GT.json OUT1.jsonl [OUT2.jsonl ...] [--json result.json] [--verbose]

Scores are in 0..1 per (sample, turn). Official metric per task (ext/MInference/scbench/compute_scores.py,
MInference@29ef1974; Multiturnbench_to_Infinitebench mapping):
  scbench_kv, scbench_prefix_suffix -> "kv_retrieval"        get_score_one_kv_retrieval: label in pred   (0/1)
  scbench_qa_eng                    -> "longdialogue_qa_eng" get_score_one_longdialogue_qa_eng:
                                       any(label.upper() in pred.upper())                               (0/1)
  scbench_repoqa                    -> repo_qa_utils.compute_score, reported number = pass@1 at similarity
                                       threshold 0.8: 1 iff the BLEU-closest needle among all needles of the
                                       repo (whole split) is the asked function AND its BLEU >= 0.8     (0/1)
Secondary (not official, reported alongside): qa_eng token F1 (compute_scores.qa_f1_score), qa_eng
"substr_noquote" (official check after stripping stray '"' from labels; 15/40 selected labels contain them), and
repoqa raw best-match BLEU similarity.
Known official-metric ceiling: repoqa/5 turn 4 (locust execute_next_task, a 3-token function) scores BLEU 0.576 even
for an exact copy, so a perfect model gets 0.975 on repoqa with this sample set.
The official functions are copied verbatim below (importing compute_scores/eval_utils drags in torch, jieba,
evaluate.load("rouge"), args.parse_args; repo_qa_utils needs tempdir). repoqa needs tree_sitter==0.21.3 +
tree_sitter_languages==1.10.2 (installed in the bench venv) and nltk.

Records with error != null (or missing ids) are excluded from the means and listed under "missing"/"errors"; a
record with fewer outputs than GT turns is scored on the turns it has and listed under "incomplete".
If several output files hold different "cfg" values, results are reported per cfg. The overall mean is the mean of
the per-task means (macro); the per-turn rows are means over all scored samples of all tasks at that turn index.
"""
import argparse, json, re, statistics, string, sys
from collections import Counter, defaultdict

# ======================= verbatim from scbench/compute_scores.py =======================


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths) -> float:
    f1 = 0
    for ground_truth in ground_truths:
        prediction_tokens = normalize_answer(pred).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        this_f1, _, _ = f1_score(prediction_tokens, ground_truth_tokens)
        f1 = max(f1, this_f1)
    return f1


def get_score_one_kv_retrieval(pred, label, model_name: str) -> bool:
    return label in pred


def get_score_one_longdialogue_qa_eng(pred, label, model_name: str) -> bool:
    pred = pred.strip()
    pred = pred.upper()
    for item in label:
        if item.upper() in pred:
            return 1
    return 0


# ======================= verbatim (minimal) from scbench/repo_qa_utils.py =======================

FUNCTION_QUERY = {
    "python": "(function_definition name: (_)) @fdef",
    "java": "(method_declaration name: (_)) @fdef",
    "typescript": "(function_declaration name: (_)) @fdef",
    "rust": "(function_item name: (_)) @fdef",
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier))) @fdef",
    "go": "(function_declaration name: (_)) @fdef",
}
REPOQA_THRESHOLD = 0.8  # compute_scores reports acc[...]["scores"]["all"][0.8]["pass@1"]


def compute_function_similarity(candidate_function: str, reference_function: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    candidate_tokens = [item for item in re.split(r"\s+", candidate_function.strip())]
    reference_tokens = [item for item in re.split(r"\s+", reference_function.strip())]
    chencherry = SmoothingFunction()
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=chencherry.method4)


def sanitize_output(model_output: str, lang: str) -> str:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tree_sitter_languages import get_language, get_parser

        model_output = model_output.strip()
        search_pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
        code_blocks = re.findall(search_pattern, model_output, re.DOTALL | re.MULTILINE)
        parser = get_parser(lang)
        fn_query = get_language(lang).query(FUNCTION_QUERY[lang])
    if not code_blocks:
        return model_output
    processed_blocks = []
    for block in code_blocks:
        processed_blocks.append(block)
        try:
            block_bytes = bytes(block, "utf8")
            tree = parser.parse(block_bytes)
            for capture in fn_query.captures(tree.root_node):
                node, _ = capture
                function_content = block_bytes[node.start_byte: node.end_byte]
                return function_content.decode("utf8")
        except:  # noqa: E722
            pass
    return processed_blocks[0]


def needle_evaluator(model_output, ground_truth, needles, lang):
    # ignore_comments=False (compute_scores default)
    best_target = None
    best_similarity = 0
    sanitized_output = sanitize_output(model_output, lang)
    for needle in needles:
        current_similarity = compute_function_similarity(sanitized_output, needle["needle"])
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_target = needle["name"]
    return best_target == ground_truth["func_name"], best_target, best_similarity


# ==================================================================================================


def score_turn(ent, k, pred):
    """-> (official score 0/1 float, extra dict)"""
    task, gt = ent["task"], ent["turns"][k]["ground_truth"]
    pred = pred or ""
    if task in ("scbench_kv", "scbench_prefix_suffix"):
        return float(get_score_one_kv_retrieval(pred, gt, "")), {}
    if task == "scbench_qa_eng":
        # "substr_noquote": same as official but with the dataset's stray '"' quote characters stripped from the
        # label (15/40 selected qa_eng answers carry them, e.g. '"Annalisa'; the official check then needs the
        # model to emit the literal quote).
        nq = [g.strip().strip('"').strip() for g in gt]
        return (float(get_score_one_longdialogue_qa_eng(pred, gt, "")),
                {"f1": float(qa_f1_score(pred, gt)), "substr_noquote": float(get_score_one_longdialogue_qa_eng(pred, nq, ""))})
    if task == "scbench_repoqa":
        ok, best, sim = needle_evaluator(pred, gt, ent["needles"], ent["lang"])
        return float(ok and sim >= REPOQA_THRESHOLD), {"bleu": sim, "best": best}
    raise ValueError(task)


def _load_gt(gt):
    if isinstance(gt, str):
        gt = json.load(open(gt))
    return {k: v for k, v in gt.items() if not k.startswith("_")}


def _load_records(path):
    recs = []
    for l in open(path):
        l = l.strip()
        if l:
            recs.append(json.loads(l))
    return recs


def score_records(gt, recs):
    """-> {id: {"scores": [..], "extra": [..], "error": str|None, "cfg": ...}} for ids present in GT."""
    gt = _load_gt(gt)
    out = {}
    for r in recs:
        rid = r.get("id")
        if rid not in gt:
            continue
        prev = out.get(rid)
        if prev is not None and prev["error"] is None and r.get("error"):
            continue  # keep the good record over an errored retry/duplicate
        ent = gt[rid]
        outs = r.get("outputs") or []
        sc, ex = [], []
        if not r.get("error"):
            for k in range(min(len(outs), len(ent["turns"]))):
                s, e = score_turn(ent, k, outs[k])
                sc.append(s); ex.append(e)
        out[rid] = {"scores": sc, "extra": ex, "error": r.get("error"), "cfg": r.get("cfg")}
    return out


def score_file(gt, path):
    """Importable API: dict id -> list of per-turn official scores (0..1). Errored records -> []."""
    return {k: v["scores"] for k, v in score_records(gt, _load_records(path)).items()}


def summarize(gt, scored):
    gt = _load_gt(gt)
    mean = lambda v: round(statistics.mean(v), 4) if v else None  # noqa: E731
    by_task, by_task_turn, by_turn = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    sec = defaultdict(lambda: defaultdict(list))
    missing, errors, incomplete, per_sample = [], {}, [], {}
    for rid, ent in gt.items():
        s = scored.get(rid)
        if s is None:
            missing.append(rid); continue
        if s["error"]:
            errors[rid] = s["error"]; continue
        if len(s["scores"]) < len(ent["turns"]):
            incomplete.append(rid)
        per_sample[rid] = {"scores": s["scores"]}
        for k, v in enumerate(s["scores"]):
            by_task[ent["task"]].append(v)
            by_task_turn[ent["task"]][k].append(v)
            by_turn[k].append(v)
            for ek, ev in s["extra"][k].items():
                if isinstance(ev, float):
                    sec[ent["task"]][ek].append(ev)
        if s["extra"] and "bleu" in s["extra"][0]:
            per_sample[rid]["bleu"] = [round(e["bleu"], 3) for e in s["extra"]]
            per_sample[rid]["best_match"] = [e["best"] for e in s["extra"]]
        if s["extra"] and "f1" in s["extra"][0]:
            per_sample[rid]["f1"] = [round(e["f1"], 3) for e in s["extra"]]
            per_sample[rid]["substr_noquote"] = [e["substr_noquote"] for e in s["extra"]]
    n_expected = Counter(e["task"] for e in gt.values())
    tasks = {}
    for t in sorted(n_expected):
        tasks[t] = {"mean": mean(by_task[t]),
                    "n_samples_scored": sum(1 for r in per_sample if gt[r]["task"] == t),
                    "n_samples_expected": n_expected[t],
                    "per_turn": [mean(by_task_turn[t][k]) for k in sorted(by_task_turn[t])],
                    "secondary": {k: mean(v) for k, v in sec[t].items()}}
    task_means = [v["mean"] for v in tasks.values() if v["mean"] is not None]
    return {"scale": "0..1", "overall_mean_macro": mean(task_means),
            "per_turn_all_tasks": [mean(by_turn[k]) for k in sorted(by_turn)],
            "tasks": tasks, "missing": missing, "errors": errors, "incomplete": incomplete,
            "per_sample": per_sample}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("outs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    gt = _load_gt(a.gt)
    recs = [r for p in a.outs for r in _load_records(p)]
    cfgs = sorted({str(r.get("cfg")) for r in recs}) or ["None"]
    result = {}
    for cfg in cfgs:
        rs = [r for r in recs if str(r.get("cfg")) == cfg]
        result[cfg] = summarize(gt, score_records(gt, rs))
    for cfg, res in result.items():
        print(f"== cfg {cfg}   (scores 0..1; overall = mean of task means)")
        for t, v in res["tasks"].items():
            pt = " ".join("-" if x is None else f"{x:.3f}" for x in v["per_turn"])
            secd = " ".join(f"{k}={x:.3f}" for k, x in v["secondary"].items() if x is not None)
            m = "-" if v["mean"] is None else f"{v['mean']:.3f}"
            print(f"  {t:24s} mean {m}  n={v['n_samples_scored']}/{v['n_samples_expected']}  turns: {pt}  {secd}")
        pt = " ".join("-" if x is None else f"{x:.3f}" for x in res["per_turn_all_tasks"])
        print(f"  per-turn (all tasks): {pt}")
        print(f"  overall: {res['overall_mean_macro']}")
        for key in ("missing", "errors", "incomplete"):
            if res[key]:
                print(f"  {key} ({len(res[key])}): {res[key] if key != 'errors' else list(res[key].items())[:5]}")
        if a.verbose:
            for rid, s in res["per_sample"].items():
                print("   ", rid, s)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
