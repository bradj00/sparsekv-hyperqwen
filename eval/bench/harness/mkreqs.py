#!/usr/bin/env python3
"""Request files for RULER and LongBench v2 (client.py format).

  mkreqs.py ruler  RULER_DATA_DIR OUT.jsonl GT.json
  mkreqs.py lbv2   LONGBENCH_V2.jsonl OUT.jsonl GT.json MODEL_DIR [--n-per-bin 12]
"""
import json, os, random, sys

RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
               "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
RULER_GEN = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
RULER_LEN = {32768: "32k", 65536: "64k", 131072: "128k", 196608: "192k"}


def ruler(data, out, gtp):
    reqs, gt = [], {}
    for L, ln in RULER_LEN.items():
        for t in RULER_TASKS:
            p = os.path.join(data, str(L), t, "validation.jsonl")
            fam = "niah" if t.startswith("niah") else ("qa" if t.startswith("qa") else t)
            for line in open(p):
                r = json.loads(line)
                rid = f"ruler/{ln}/{t}/{r['index']:03d}"
                reqs.append({"id": rid, "shard": r["index"] % 2, "group": f"{ln}/{t}", "prompt": r["input"],
                             "max_tokens": RULER_GEN[fam]})
                gt[rid] = {"task": t, "length": ln, "outputs": r["outputs"], "tokens": r.get("length"),
                           "metric": "part" if fam == "qa" else "all"}
    with open(out, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    json.dump(gt, open(gtp, "w"))
    print(len(reqs), "ruler requests")


LB_TMPL = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lbv2_0shot.txt")).read() \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lbv2_0shot.txt")) else None
LB_BINS = [("<=32k", 0, 32768), ("32-64k", 32768, 65536), ("64-128k", 65536, 131072), ("128-192k", 131072, 194000)]


def lbv2(src, out, gtp, model, n_per_bin=12):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(l) for l in open(src)]
    rng = random.Random(0)
    rng.shuffle(rows)
    picked = {b[0]: [] for b in LB_BINS}
    for r in rows:
        if all(len(v) >= n_per_bin for v in picked.values()):
            break
        if len(r["context"]) > 900_000:  # > ~200K tokens for sure
            continue
        user = (LB_TMPL.replace("$DOC$", r["context"].strip()).replace("$Q$", r["question"].strip())
                .replace("$C_A$", r["choice_A"].strip()).replace("$C_B$", r["choice_B"].strip())
                .replace("$C_C$", r["choice_C"].strip()).replace("$C_D$", r["choice_D"].strip()))
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
        for name, lo, hi in LB_BINS:
            if lo < n + 128 <= hi and len(picked[name]) < n_per_bin:
                picked[name].append((r, prompt, n))
    reqs, gt = [], {}
    i = 0
    for name, _, _ in LB_BINS:
        for r, prompt, n in picked[name]:
            rid = f"lbv2/{name}/{r['_id']}"
            reqs.append({"id": rid, "shard": i % 2, "group": rid, "prompt": prompt, "max_tokens": 128})
            gt[rid] = {"bin": name, "tokens": n, "answer": r["answer"], "domain": r["domain"],
                       "difficulty": r["difficulty"], "length": r["length"]}
            i += 1
        print(name, len(picked[name]))
    with open(out, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    json.dump(gt, open(gtp, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "ruler":
        ruler(*sys.argv[2:5])
    elif sys.argv[1] == "lbv2":
        npb = int(sys.argv[7]) if len(sys.argv) > 7 else 12
        lbv2(*sys.argv[2:6], n_per_bin=npb)
