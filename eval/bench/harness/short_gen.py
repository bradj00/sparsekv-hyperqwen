#!/usr/bin/env python
"""Generate request files + ground truth for the short (thinking-mode) accuracy suites.

Writes  reqs/{gpqa,lcb,ifbench,mmlupro}.jsonl   and   data/short_gt/{suite}.json (+ manifest.json)

Prompt sources (verbatim wording):
  gpqa    : openai/simple-evals common.py QUERY_TEMPLATE_MULTICHOICE (gpqa_eval.py), zero-shot, 4 choices
            shuffled with a fixed per-question seed.
  lcb     : LiveCodeBench lcb_runner/prompts/code_generation.py, LMStyle.OpenAIReasonPreview branch:
            single user message = SYSTEM_MESSAGE_GENERIC + "\\n\\n" + get_generic_question_template_answer().
  ifbench : the IFBench_test prompt verbatim as the user message.
  mmlupro : TIGER-AI-Lab/MMLU-Pro evaluate_from_api.py single_request() header + format_example(),
            with the 5 CoT exemplars dropped (zero-shot).
All rendered with the model's chat template (thinking ON, template default xhigh system message).
"""
import argparse, collections, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from short_common import BENCH, LCB_REPO, decode_lcb_private, seed_for

MODEL_DIR = os.environ.get("MODEL_DIR", "Qwen3.8-27B-W4A16-AutoRound")
MAX_TOKENS = 32768
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

# ---- simple-evals common.py (openai/simple-evals @ 652c89d) ----
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# ---- LiveCodeBench lcb_runner/prompts/code_generation.py (PromptConstants) ----
LCB_SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
LCB_FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def lcb_generic_question_template_answer(question_content, starter_code):
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def lcb_user_message(question_content, starter_code):
    # LMStyle.OpenAIReasonPreview: system message folded into the single user turn
    return LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + lcb_generic_question_template_answer(question_content, starter_code)


# ---- MMLU-Pro evaluate_from_api.py (TIGER-AI-Lab/MMLU-Pro @ f418b11) ----
def mmlupro_format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def mmlupro_user_message(category, question, options):
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    return prompt + mmlupro_format_example(question, options)


# ----------------------------------------------------------------------------------------------
def build_gpqa():
    import pandas
    df = pandas.read_csv(os.path.join(BENCH, "data", "gpqa/dataset/gpqa_diamond.csv"))
    items, gt = [], {}
    for _, row in df.iterrows():
        rid = f"gpqa/{row['Record ID']}"
        # whitespace-stripped (the raw CSV has stray trailing newlines in some cells)
        choices = [str(row[c]).strip() for c in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
        perm = random.Random(seed_for(rid)).sample(range(4), 4)
        choices = [choices[i] for i in perm]
        letter = "ABCD"[perm.index(0)]
        msg = QUERY_TEMPLATE_MULTICHOICE.format(A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=str(row["Question"]).strip())
        items.append((rid, msg))
        gt[rid] = {"answer": letter, "permutation": perm, "subdomain": row.get("Subdomain"),
                   "domain": row.get("High-level domain")}
    return items, gt


def build_lcb(n_target=80, start="2025-02-01", end="2025-05-31"):
    rows = []
    for f in ("test5.jsonl", "test6.jsonl"):
        p = os.path.join(BENCH, "data", f"lcb/{f}")
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                if start <= r["contest_date"][:10] <= end:
                    rows.append(r)
    by_diff = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["question_id"]):
        by_diff[r["difficulty"]].append(r)
    n_all = len(rows)
    if n_all <= n_target:
        chosen = rows
    else:  # proportional allocation, largest remainder
        quota = {d: n_target * len(v) / n_all for d, v in by_diff.items()}
        alloc = {d: int(q) for d, q in quota.items()}
        for d in sorted(quota, key=lambda d: -(quota[d] - alloc[d]))[: n_target - sum(alloc.values())]:
            alloc[d] += 1
        chosen = []
        for d in sorted(by_diff):
            chosen += random.Random(0).sample(by_diff[d], alloc[d])
    chosen.sort(key=lambda r: (r["contest_date"], r["question_id"]))
    items, gt = [], {}
    for r in chosen:
        rid = f"lcb/{r['question_id']}"
        priv = decode_lcb_private(r["private_test_cases"])  # validate decode now
        meta = json.loads(r["metadata"])
        items.append((rid, lcb_user_message(r["question_content"], r["starter_code"])))
        gt[rid] = {"question_id": r["question_id"], "platform": r["platform"], "difficulty": r["difficulty"],
                   "contest_date": r["contest_date"], "fn_name": meta.get("func_name"),
                   "starter_code": r["starter_code"], "n_public": len(json.loads(r["public_test_cases"])),
                   "n_private": len(priv), "public_test_cases": r["public_test_cases"],
                   "private_test_cases": r["private_test_cases"]}  # kept in dataset encoding
    info = {"window": [start, end], "n_in_window": n_all,
            "window_by_difficulty": {d: len(v) for d, v in by_diff.items()},
            "chosen_by_difficulty": dict(collections.Counter(g["difficulty"] for g in gt.values())),
            "data_max_contest_date": max((r["contest_date"] for r in rows), default=None)}
    return items, gt, info


def build_ifbench():
    items, gt = [], {}
    for line in open(os.path.join(BENCH, "data", "hf/ifbench.jsonl")):
        r = json.loads(line)
        rid = f"ifbench/{r['key']}"
        items.append((rid, r["prompt"]))
        gt[rid] = {"key": r["key"], "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
                   "kwargs": r["kwargs"]}
    return items, gt


def build_mmlupro(per_cat=20, seed=0):
    by_cat = collections.defaultdict(list)
    for line in open(os.path.join(BENCH, "data", "hf/mmlu_pro.jsonl")):
        r = json.loads(line)
        by_cat[r["category"]].append(r)
    items, gt = [], {}
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat], key=lambda r: r["question_id"])
        for r in sorted(random.Random(seed).sample(pool, per_cat), key=lambda r: r["question_id"]):
            rid = f"mmlupro/{r['question_id']}"
            items.append((rid, mmlupro_user_message(cat, r["question"], r["options"])))
            gt[rid] = {"answer": r["answer"], "answer_index": r["answer_index"], "category": cat,
                       "n_options": len(r["options"]), "src": r["src"]}
    return items, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="gpqa,lcb,ifbench,mmlupro")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    os.makedirs(f"{BENCH}/reqs", exist_ok=True)
    os.makedirs(os.path.join(BENCH, "data", "short_gt"), exist_ok=True)
    mpath = os.path.join(BENCH, "data", "short_gt/manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for suite in a.suites.split(","):
        extra = {}
        if suite == "gpqa":
            items, gt = build_gpqa()
        elif suite == "lcb":
            items, gt, extra = build_lcb()
        elif suite == "ifbench":
            items, gt = build_ifbench()
        elif suite == "mmlupro":
            items, gt = build_mmlupro()
        else:
            raise SystemExit(f"unknown suite {suite}")
        assert len({i for i, _ in items}) == len(items), "duplicate ids"
        lens = []
        with open(f"{BENCH}/reqs/{suite}.jsonl", "w") as f:
            for k, (rid, msg) in enumerate(items):
                prompt = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                                 add_generation_prompt=True)
                assert prompt.endswith("<think>\n"), "thinking template expected"
                lens.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
                f.write(json.dumps({"id": rid, "shard": k % 2, "group": suite, "prompt": prompt,
                                    "max_tokens": a.max_tokens, "sampling": SAMPLING,
                                    "seed": seed_for(rid)}) + "\n")
        with open(os.path.join(BENCH, "data", f"short_gt/{suite}.json"), "w") as f:
            json.dump(gt, f)
        manifest[suite] = {"n": len(items), "max_tokens": a.max_tokens, "sampling": SAMPLING,
                           "prompt_tokens_mean": round(sum(lens) / len(lens), 1),
                           "prompt_tokens_max": max(lens), **extra}
        print(suite, json.dumps(manifest[suite]))
    json.dump(manifest, open(mpath, "w"), indent=1)


if __name__ == "__main__":
    main()
