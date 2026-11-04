import sys, json, os
from datasets import load_dataset
from huggingface_hub import HfApi
out = sys.argv[1]
man = {}
api = HfApi()
def save(name, ds_id, config=None, split="test"):
    try:
        info = api.dataset_info(ds_id); sha = info.sha
    except Exception as e:
        sha = f"? {e!r}"[:80]
    ds = load_dataset(ds_id, config, split=split) if config else load_dataset(ds_id, split=split)
    p = os.path.join(out, f"{name}.jsonl"); ds.to_json(p, lines=True, force_ascii=False)
    man[name] = {"hf_id": ds_id, "config": config, "split": split, "revision": sha, "rows": len(ds)}
    print(name, len(ds), sha, flush=True)
save("longbench_v2", "THUDM/LongBench-v2", split="train")
for c in ["scbench_kv", "scbench_repoqa", "scbench_prefix_suffix", "scbench_qa_eng", "scbench_many_shot", "scbench_summary", "scbench_vt", "scbench_mf", "scbench_choice_eng", "scbench_qa_chn", "scbench_repoqa_and_kv", "scbench_summary_with_needles"]:
    try: save(c, "microsoft/SCBench", c, split="test")
    except Exception as e: print("skip", c, repr(e)[:200])
save("mmlu_pro", "TIGER-Lab/MMLU-Pro", split="test")
save("ifbench", "allenai/IFBench_test", split="train")
json.dump(man, open(os.path.join(out, "manifest_hf.json"), "w"), indent=1)
