#!/usr/bin/env python3
"""MTP-on decode-speed / acceptance requests: 3 KLD documents x 5 lengths, free-running greedy 'explain' (512 tokens).
Prompts of one document are prefixes of each other (prefix cache reuses the shorter prefill).
Usage: mtp_gen.py KLD_DATA_DIR MODEL_DIR OUT.jsonl"""
import json, sys
import numpy as np
from transformers import AutoTokenizer
data, model, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model)
LENS = {"32k": 32768, "64k": 65536, "128k": 131072, "192k": 196608, "240k": 240000}
Q = ("\n\n[End of document]\n\nExplain in detail what the final part of the document above "
     "(the last file, or the last few paragraphs) is about.")
pre = "<|im_start|>user\nBelow is a long document. Read it carefully; questions about it follow at the end.\n\n"
post = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
with open(out, "w") as f:
    for di, doc in enumerate(["code_vllm", "code_torch", "prose_wp"]):
        ids = np.load(f"{data}/{doc}.npy")
        for ln, L in LENS.items():
            n = L - 600 - 512
            text = tok.decode(ids[:n].tolist())
            f.write(json.dumps({"id": f"mtp/{ln}/{doc}", "shard": di % 2, "group": doc,
                                "prompt": pre + text + Q + post, "max_tokens": 512}) + "\n")
