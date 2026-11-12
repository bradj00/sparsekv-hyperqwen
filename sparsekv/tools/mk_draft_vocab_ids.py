# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Build a larger MTP draft-vocabulary id list from token-frequency coverage (v1.2: 29,670 -> 63,510 tokens).

The MTP draft head is a slice of the quantized lm_head (HyperQwen prepare/build_draft_vocab.py). A draft token that
is not in the slice ends the draft chain, so a larger list accepts more tokens per step at the cost of a larger head
read per draft token. This tool only writes an id LIST (JSON); mk_draft_vocab_variant.sh then builds a model
directory that uses it. No list is shipped with this repository: the one behind the v1.2 measurements was counted
partly over private agent traffic. Count your own on text that looks like your workload.

Method (as used for v1.2):
  1. Base = the ids already in use, kept and kept first: the model directory's current mtp_draft_vocab_ids.pt
     (--current, default on) and any --base-ids JSON lists (e.g. HyperQwen's prepare/draft_vocab_ids.json).
  2. Corpus = one or more named sources (--source NAME=GLOB[,GLOB...]). The v1.2 list used: rendered agent prompts
     (system prompt, tool definitions, prior turns, tool results), the model's own benchmark outputs, and source code
     (Python, shell / Markdown / YAML / JSON, JS / TS / C). Files: .jsonl (fields prompt / text / response, and every
     string in "outputs"), .jsonl.gz, or anything else read as plain text (--min-chars / --max-chars per file,
     --cap-chars per source, files shuffled with a fixed seed).
  3. Every fifth text of each source is held out; token counts come from the other four fifths.
  4. For each --target N: base + the most frequent counted tokens not in the base, until N ids (fewer if the corpus
     has no more distinct tokens: the v1.2 run asked for 65,536 and got 63,510).
  5. Held-out coverage (share of held-out tokens that are in the list) is printed per source and per candidate, the
     base included, so the gain of each size is measured before any GPU run.

Usage (CPU only; needs `tokenizers`, and `torch` for --current):
  python sparsekv/tools/mk_draft_vocab_ids.py MODEL_DIR --out-dir vocab \\
      --base-ids prepare/draft_vocab_ids.json \\
      --source prompts=my_prompts.jsonl --source code='src/**/*.py,scripts/**/*.sh' \\
      --target 49152 57344 65536
  -> vocab/u49152.json, vocab/u57344.json, vocab/u65536.json (sorted id lists)

Pick the size on the GPU, not on coverage alone (docs/OPTIMIZATION_NOTES.md: the full head accepts slightly more
but is slower at one stream; 64K was the best point measured), and run the accuracy gate: a different draft list
changes the output text when draft acceptance is probabilistic.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import random


def read_texts(path, min_chars, max_chars):
    if path.endswith((".jsonl", ".jsonl.gz")):
        op = gzip.open if path.endswith(".gz") else open
        out = []
        with op(path, "rt", errors="ignore") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(r, dict):
                    continue
                for k in ("prompt", "text", "response"):
                    if isinstance(r.get(k), str) and r[k]:
                        out.append(r[k])
                out += [o for o in (r.get("outputs") or []) if isinstance(o, str) and o]
        return out
    try:
        with open(path, errors="ignore") as f:
            s = f.read()
    except Exception:  # noqa: BLE001
        return []
    return [s] if min_chars < len(s) < max_chars else []


def source_texts(globs, cap_chars, min_chars, max_chars):
    files = []
    for pat in globs:
        files += glob.glob(pat, recursive=True)
    files = sorted(set(files))
    random.Random(0).shuffle(files)
    out, n = [], 0
    for f in files:
        for t in read_texts(f, min_chars, max_chars):
            out.append(t)
            n += len(t)
        if cap_chars and n > cap_chars:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir", help="model directory (tokenizer.json; mtp_draft_vocab_ids.pt for --current)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", action="append", default=[], metavar="NAME=GLOB[,GLOB...]", required=True)
    ap.add_argument("--base-ids", nargs="*", default=[], help="JSON id lists that stay in every candidate")
    ap.add_argument("--no-current", action="store_true", help="do not keep the model directory's current draft ids")
    ap.add_argument("--target", type=int, nargs="+", default=[49152, 57344, 65536])
    ap.add_argument("--cap-chars", type=float, default=12e6, help="per source, 0 = no cap")
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-chars", type=int, default=200000)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    cands, base = {}, []
    if not a.no_current:
        import torch
        cur = torch.load(os.path.join(a.model_dir, "mtp_draft_vocab_ids.pt")).tolist()
        cands[f"current{len(cur)}"] = cur
        base += cur
    for p in a.base_ids:
        ids = json.load(open(p))
        cands[f"{os.path.splitext(os.path.basename(p))[0]}{len(ids)}"] = ids
        base += ids
    base = list(dict.fromkeys(base))
    in_base = set(base)
    if base:
        cands["base"] = base

    counts, held = collections.Counter(), {}
    for spec in a.source:
        name, _, pats = spec.partition("=")
        texts = source_texts(pats.split(","), a.cap_chars, a.min_chars, a.max_chars)
        h, n = collections.Counter(), 0
        for i, t in enumerate(texts):
            ids = tok.encode(t, add_special_tokens=False).ids
            n += len(ids)
            (h if i % 5 == 0 else counts).update(ids)
        held[name] = h
        print(f"{name:16s} {len(texts):7d} texts {n:10d} tokens")
    ranked = [t for t, _ in counts.most_common() if t not in in_base]
    for n in a.target:
        cands[f"u{n}"] = base + ranked[:max(0, n - len(base))]
    print("sizes", {k: len(v) for k, v in cands.items()})
    sets = {k: set(v) for k, v in cands.items()}
    for name, h in held.items():
        tot = sum(h.values()) or 1
        print(f"{name:16s} held-out coverage %: "
              + "  ".join(f"{c}:{100 * sum(v for t, v in h.items() if t in s) / tot:5.2f}" for c, s in sets.items()))
    os.makedirs(a.out_dir, exist_ok=True)
    for c, ids in cands.items():
        if c.startswith("u"):
            with open(os.path.join(a.out_dir, f"{c}.json"), "w") as f:
                json.dump(sorted(ids), f)
            print("wrote", os.path.join(a.out_dir, f"{c}.json"), len(ids), "ids")


if __name__ == "__main__":
    main()
