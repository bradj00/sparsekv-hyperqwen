#!/usr/bin/env python3
"""SCBench multi-turn request generator (shared long context, follow-up queries, model's OWN previous answers kept).

Usage: scbench_gen.py [--out reqs/scbench.jsonl] [--gt data/scbench_gt.json] [--per 8] [--max-turns 5]

Source: microsoft/SCBench (local jsonl dumps in data/hf/scbench_<task>.jsonl, revision in data/hf/manifest_hf.json),
official code pinned at ext/MInference/scbench (MInference commit 29ef1974fcb5a1440ccf92d7d7996900230fddcf).

What is reproduced from the official code (eval_utils.py / run_scbench.py):
  * turn-1 user text   = eval_utils.multiturn_templates[task].format(context=..., input|question=turn0.input)
  * follow-up user text = eval_utils.multiturn_follow_up_templates_in_chat_tempate[task].format(...)
    (this is what create_multiturn_prompt(..., use_chat_template=True, disable_golden_context=True) uses, i.e.
    SCBench's "model's own answers stay in the history" mode; the default golden-context mode would instead insert
    the gold answer as the previous assistant turn).
  * per-turn generation budget = eval_utils.DATA_NAME_TO_MAX_NEW_TOKENS[task]
    (kv 150, prefix_suffix 150, repoqa 1024, qa_eng 40), same budget for every turn.
  * no assistant answer prefix for these four tasks (only scbench_summary appends "This paper"), so the scorer needs
    no prefix handling.
  * truncation (only if needed): run_scbench.truncate_by_tokens(rendered turn-1 prompt, tok, cap, manner="middle"),
    i.e. keep the first cap//2 and last cap//2 tokens of the whole chat-rendered turn-1 prompt.

Deviations (deliberate):
  1. Chat template rendered with enable_thinking=False (empty <think>\n\n</think>\n\n block opening every assistant
     turn, also kept in history). Official run passes no enable_thinking (Qwen3 default = thinking on).
     Verified per sample: prompt + a1 + glue2 + a2 + ... + glueN == tok.apply_chat_template(msgs, enable_thinking=False,
     add_generation_prompt=True) (with gold answers standing in for model outputs).
  2. The model's previous answer enters the history verbatim; apply_chat_template would |trim it. Answers with
     leading/trailing whitespace therefore differ from a template re-render by that whitespace only.
     Official multi-turn re-encodes each follow-up prompt and appends the EOS id after the previous answer; here the
     client works on text: prompt_{k-1} + raw output_{k-1} + glue_k, glue starting with "<|im_end|>\n". Equivalent
     except that the model's raw output (incl. any leading/trailing whitespace) is kept verbatim.
  3. Length cap: official uses max_seq_length (131072 in their scripts) minus budget*turns. Here: estimated final-turn
     prompt (turn-1 tokens + all glue tokens + full answer budget of every turn) must be <= 190,000 tokens.
     Samples that fit naturally are preferred (first --per qualifying samples in dataset order, no random choice);
     only if fewer than --per fit, the earliest non-fitting samples are middle-truncated to
     min(180000, 190000 - glue - answer budget) tokens and marked truncated=true in the GT.
  4. At most --max-turns (5) turns per sample (scbench_qa_eng has 5-7 turns; the others have exactly 5), same as
     run_scbench.py --max_turns 5.
  5. Only --per samples per task (official runs the whole split).
"""
import argparse, json, os, statistics, sys

from transformers import AutoTokenizer

BENCH = os.environ.get("BENCH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("MODEL_DIR", "Qwen3.8-27B-W4A16-AutoRound")
TASKS = ["scbench_kv", "scbench_repoqa", "scbench_prefix_suffix", "scbench_qa_eng"]

# --- verbatim from ext/MInference/scbench/eval_utils.py -------------------------------------------------------
DATA_NAME_TO_MAX_NEW_TOKENS = {
    "scbench_qa_eng": 40,
    "scbench_kv": 150,
    "scbench_prefix_suffix": 150,
    "scbench_repoqa": 1024,
}
multiturn_templates = {
    "scbench_kv": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    "scbench_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",  # noqa
    "scbench_repoqa": "Based on the function description and code context, please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:\n\n{context}\n\n{input}",
    "scbench_prefix_suffix": "{context}\n\n{input}",
}
multiturn_follow_up_templates_in_chat_tempate = {
    "scbench_kv": "{input}",  # noqa
    "scbench_qa_eng": "Question: {question}\nAnswer:",  # noqa
    "scbench_repoqa": "{input}",
    "scbench_prefix_suffix": "{input}",
}
# ---------------------------------------------------------------------------------------------------------------

USER_OPEN = "<|im_start|>user\n"
ASSIST_OPEN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
GLUE_OPEN = "<|im_end|>\n<|im_start|>user\n"
FINAL_CAP = 190_000
TRUNC_CAP = 180_000


def user_text(task, eg, k):
    # .strip(): the Qwen chat template applies |trim to message content, so official apply_chat_template rendering
    # strips e.g. the leading "\n" and trailing " " of scbench_kv inputs; we reproduce that exactly.
    return _user_text(task, eg, k).strip()


def _user_text(task, eg, k):
    t = eg["multi_turns"][k]
    if k == 0:
        ctx = eg["context"] if "context" in eg else eg["input"]
        if task == "scbench_qa_eng":
            return multiturn_templates[task].format(context=ctx, question=t["input"])
        return multiturn_templates[task].format(context=ctx, input=t["input"])
    if task == "scbench_qa_eng":
        return multiturn_follow_up_templates_in_chat_tempate[task].format(question=t["input"])
    return multiturn_follow_up_templates_in_chat_tempate[task].format(input=t["input"])


def ground_truth(task, eg, t):
    # matches eval_utils.get_ground_truth + extra fields run_scbench.py stores for repoqa
    if task == "scbench_qa_eng":
        return [t["answer"]]
    if task == "scbench_repoqa":
        return {"answer": t["answer"], "func_name": t["name"]}
    return t["answer"]


def gold_text(task, t):
    """What a perfect model answer would look like (used for the template check and the scorer self-test)."""
    if task == "scbench_repoqa":
        return f"```{t.get('lang', '')}\n{t['answer']}\n```"
    return str(t["answer"])


def truncate_middle_tokens(tok, text, cap):
    """run_scbench.truncate_by_tokens(input, tok, cap, manner='middle') + decode back to text."""
    ids = tok.encode(text)
    if len(ids) <= cap:
        return text, len(ids)
    split = cap // 2
    out = tok.decode(ids[:split] + ids[-split:], skip_special_tokens=False)
    n = len(tok.encode(out))
    while n > cap:  # seam re-tokenization can add a token or two
        split -= (n - cap + 1) // 2 + 1
        out = tok.decode(ids[:split] + ids[-split:], skip_special_tokens=False)
        n = len(tok.encode(out))
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{BENCH}/reqs/scbench.jsonl")
    ap.add_argument("--gt", default=os.path.join(BENCH, "data", "scbench_gt.json"))
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ntok = lambda s: len(tok.encode(s))  # noqa: E731
    manifest = {}
    mp = os.path.join(BENCH, "data", "hf/manifest_hf.json")
    if os.path.exists(mp):
        manifest = json.load(open(mp))

    lines, gt, summary = [], {}, {}
    for task in TASKS:
        rows = [json.loads(l) for l in open(os.path.join(BENCH, "data", f"hf/{task}.jsonl"))]
        budget = DATA_NAME_TO_MAX_NEW_TOKENS[task]
        # repoqa scorer needs every needle of the repo across the WHOLE split (run_scbench builds needle_by_repo from
        # all predictions of all samples/turns of the repo); store them so scoring 8 samples matches a full-split run.
        needles_by_repo = {}
        if task == "scbench_repoqa":
            for eg in rows:
                for t in eg["multi_turns"][: a.max_turns]:
                    needles_by_repo.setdefault(eg["repo"], []).append({"needle": t["answer"], "name": t["name"]})

        fit, nofit = [], []
        for idx, eg in enumerate(rows):
            eg = dict(eg, multi_turns=eg["multi_turns"][: a.max_turns])
            for t in eg["multi_turns"]:
                if task == "scbench_repoqa":
                    t["lang"] = eg["lang"]
            nturn = len(eg["multi_turns"])
            prompt = USER_OPEN + user_text(task, eg, 0) + ASSIST_OPEN
            glues = [GLUE_OPEN + user_text(task, eg, k) + ASSIST_OPEN for k in range(1, nturn)]
            gl = sum(ntok(g) for g in glues)
            allowance = budget * nturn
            t1 = ntok(prompt)
            rec = dict(idx=idx, eg=eg, prompt=prompt, glues=glues, t1=t1, gl=gl, allowance=allowance)
            (fit if t1 + gl + allowance <= FINAL_CAP else nofit).append(rec)
            if len(fit) >= a.per:
                break
        sel = [dict(r, truncated=False) for r in fit[: a.per]]
        for r in nofit[: a.per - len(sel)]:
            cap = min(TRUNC_CAP, FINAL_CAP - r["gl"] - r["allowance"])
            p, n = truncate_middle_tokens(tok, r["prompt"], cap)
            assert p.startswith(USER_OPEN) and p.endswith(ASSIST_OPEN), "truncation cut the chat frame"
            sel.append(dict(r, prompt=p, t1=n, truncated=True, t1_orig=r["t1"]))
        sel.sort(key=lambda r: r["idx"])
        if len(sel) < a.per:
            print(f"WARNING {task}: only {len(sel)} samples available", file=sys.stderr)

        short = task.replace("scbench_", "")
        for j, r in enumerate(sel):
            eg, idx = r["eg"], r["idx"]
            turns = eg["multi_turns"]
            # template check: our concatenation == apply_chat_template for the same conversation (gold answers)
            user0 = r["prompt"][len(USER_OPEN): -len(ASSIST_OPEN)]
            msgs = [{"role": "user", "content": user0}]
            mine = r["prompt"]
            for k in range(1, len(turns)):
                ans = gold_text(task, turns[k - 1])
                msgs += [{"role": "assistant", "content": ans},
                         {"role": "user", "content": user_text(task, eg, k)}]
                mine += ans + r["glues"][k - 1]
            ref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert ref == mine, f"chat-template mismatch on {task}/{idx}"

            rid = f"scbench/{short}/{idx}"
            lines.append({"id": rid, "shard": j % 2, "group": f"{short}/{idx}", "prompt": r["prompt"],
                          "max_tokens": budget,
                          "turns": [{"glue": g, "max_tokens": budget} for g in r["glues"]]})
            ent = {"task": task, "idx": idx, "prompt_tokens_turn1": r["t1"],
                   "approx_final_tokens": r["t1"] + r["gl"] + r["allowance"], "truncated": r["truncated"],
                   "n_turns": len(turns), "max_tokens_per_turn": budget,
                   "turns": [{"query": t["input"].strip()[:200], "ground_truth": ground_truth(task, eg, t)}
                             for t in turns]}
            if r["truncated"]:
                ent["prompt_tokens_turn1_untruncated"] = r["t1_orig"]
            if task == "scbench_repoqa":
                ent["lang"], ent["repo"] = eg["lang"], eg["repo"]
                ent["needles"] = needles_by_repo[eg["repo"]]
            gt[rid] = ent
        summary[task] = sel

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    meta = {"source": "microsoft/SCBench", "revision": (manifest.get("scbench_kv") or {}).get("revision"),
            "official_code": "MInference@29ef1974fcb5a1440ccf92d7d7996900230fddcf scbench/",
            "selection": f"first {a.per} samples per task in dataset order whose estimated final prompt <= {FINAL_CAP}"
                         f" tokens (truncate earliest non-fitting only if too few fit)",
            "mode": "multi-turn, disable_golden_context (own answers), chat template enable_thinking=False"}
    with open(a.gt, "w") as f:
        json.dump({"_meta": meta, **gt}, f, ensure_ascii=False, indent=1)

    print(f"wrote {len(lines)} requests -> {a.out}\nGT -> {a.gt}   (revision {meta['revision']})")
    print(f"{'task':24s} {'n':>2s} {'trunc':>5s} {'turn1 tok min/med/max':>26s} {'final est min/med/max':>26s} turns")
    for task, sel in summary.items():
        t1 = [r["t1"] for r in sel]
        fe = [r["t1"] + r["gl"] + r["allowance"] for r in sel]
        nt = sorted({len(r["eg"]["multi_turns"]) for r in sel})
        f3 = lambda v: f"{min(v)}/{int(statistics.median(v))}/{max(v)}"  # noqa: E731
        print(f"{task:24s} {len(sel):2d} {sum(r['truncated'] for r in sel):5d} {f3(t1):>26s} {f3(fe):>26s} {nt}"
              f"  idx={[r['idx'] for r in sel]}")


if __name__ == "__main__":
    main()
