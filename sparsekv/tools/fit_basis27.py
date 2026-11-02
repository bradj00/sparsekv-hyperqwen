#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Fit the global rank-64 key-sketch basis for Qwen3.8-27B (sparse-KV contract) on the real model.

Phase GPU (needs GPU1, ~5 min):  python fit_basis27.py grams
  Loads the packed W4A16 model in HF (q27loader.py next to this script), prefills calibration and held-out
  documents in chunks of <= 4096 tokens with a growing KV cache (so positions are realistic), and for every
  full-attention layer accumulates per-KV-head Gram matrices of the post-k_norm, post-RoPE keys (exactly what the
  pthfa cache stores, before int8) and of the post-q_norm post-RoPE queries (summed over the 6 q-heads per KV head).
  Saves them to WORK/grams.pt.
Phase CPU:  python fit_basis27.py fit
  eigh of the Grams -> top-64 eigenvectors per (layer, KV head) -> basis27.pt (mixed corpus) and basis27_essay.pt
  (essay only); also the lc27-style "foreign" basis (first 4096 essay tokens) for reference; held-out validation:
  key-energy fraction  tr(P Gk)/tr(Gk)  and logit-energy fraction  tr(P Gq P Gk)/tr(Gq Gk)
  (= E[(q.Pk)^2]/E[(q.k)^2] over independent held-out q,k), at native positions and with the held-out keys/queries
  RoPE-shifted by +64K and +128K.
"""
import copy, glob, json, math, os, random, re, sys, time
import torch

WORK = os.environ.get("WORK", "basis_work")
OUT = os.environ.get("OUT", "basis27.pt")
OUT_ESSAY = os.environ.get("OUT_ESSAY", "basis27_essay.pt")
M = os.environ.get("MODEL_DIR", "models/Qwen3.8-27B-W4A16-AutoRound-fast")     # read-only
VLLM_SRC = os.environ.get("VLLM_SRC", os.path.join(sys.prefix, "lib/python3.12/site-packages/vllm"))
NIAH = os.environ.get("NIAH_JSONL", "niah_32k_samples.jsonl")
LAYERS = list(range(3, 64, 4))
R = 64
CH = 4096
DOC_TOK = int(os.environ.get("DOC_TOK", "9000"))        # tokens per calibration document (3 docs -> ~27K)
HO_TOK = int(os.environ.get("HO_TOK", "6000"))          # tokens per held-out document
HEAD = "<|im_start|>user\nAnswer using only the document below. Be exact.\n\nDocument:\n"

# ------------------------------------------------------------------ corpus
def code_text(files_seed, n_tok, tok, exclude=()):
    files = sorted(glob.glob(f"{VLLM_SRC}/**/*.py", recursive=True))
    random.Random(files_seed).shuffle(files)
    out, used, n = [], [], 0
    for f in files:
        if f in exclude: continue
        t = open(f, errors="ignore").read()
        if 2000 < len(t) < 30000:
            out.append(f"# file: {f.split('site-packages/')[-1]}\n{t}\n"); used.append(f)
            n += len(tok(t, add_special_tokens=False).input_ids)
        if n > n_tok * 1.05: break
    return "".join(out), used

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"},
                    "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command and return stdout/stderr.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                    "required": ["command"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents with a regex.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                    "glob": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace an exact string in a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                    "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
]
ASKS = ["Why does the scheduler preempt requests when the KV cache is full? Find where that happens.",
        "Add a debug log line whenever a speculative draft is rejected.",
        "The server fails to start with a CUDA graph capture error. Investigate.",
        "Summarize how prefix caching hashes blocks.", "Rename the helper and update all call sites.",
        "Write a unit test for the tokenizer detokenize path.", "What does the sampler do with temperature 0?",
        "Find any place that still uses the deprecated config field and fix it."]
THINK = ["I should look at the relevant module first.", "Let me search for the function definition before editing.",
         "The traceback points at the model runner; I will read that file.", "I need to check the tests to see the expected behavior.",
         "This looks like a shape mismatch; let me print the tensor sizes.", "Good, that confirms the hypothesis."]

def chat_text(seed, n_tok, tok, files):
    """Synthetic agent transcript in the Qwen chat format: system prompt with JSON tool schemas, user turns,
    assistant <think> blocks and JSON tool calls, tool responses holding real source excerpts / command output."""
    rng = random.Random(seed)
    parts = ["<|im_start|>system\nYou are a coding agent working in a vLLM checkout. Use the tools.\n\n# Tools\n\n"
             "You may call one or more functions to assist with the user query.\n\n<tools>\n"
             + "\n".join(json.dumps(t) for t in TOOLS) + "\n</tools>\n\nFor each function call, return a json object "
             "with function name and arguments within <tool_call></tool_call> XML tags.<|im_end|>\n"]
    n = len(tok(parts[0], add_special_tokens=False).input_ids)
    while n < n_tok:
        seg = [f"<|im_start|>user\n{rng.choice(ASKS)}<|im_end|>\n"]
        for _ in range(rng.randint(2, 5)):
            f = rng.choice(files); src = open(f, errors="ignore").read().split("\n")
            a = rng.randrange(max(1, len(src) - 60)); rel = f.split("site-packages/")[-1]
            kind = rng.random()
            if kind < 0.45:
                call = {"name": "read_file", "arguments": {"path": rel, "offset": a + 1, "limit": 40}}
                resp = "\n".join(f"{a + 1 + i:6d}\t{l}" for i, l in enumerate(src[a:a + 40]))
            elif kind < 0.75:
                ident = rng.choice(re.findall(r"def (\w+)", "\n".join(src)) or ["forward"])
                call = {"name": "grep", "arguments": {"pattern": f"def {ident}\\b", "path": "vllm/", "glob": "*.py"}}
                hits = [f"{rel}:{i + 1}:{l}" for i, l in enumerate(src) if ident in l][:8]
                resp = "\n".join(hits) or "No matches found"
            else:
                cmd = rng.choice([f"python -m pytest tests/{rel.split('/')[1]} -x -q", f"git log --oneline -5 -- {rel}",
                                  f"wc -l {rel}", "nvidia-smi --query-gpu=memory.used --format=csv"])
                call = {"name": "bash", "arguments": {"command": cmd, "timeout": 120}}
                resp = json.dumps({"exit_code": rng.choice([0, 0, 1]), "stdout": "\n".join(src[a:a + 12]),
                                   "stderr": "" if rng.random() < .7 else "Traceback (most recent call last):\n  ..."})
            seg.append(f"<|im_start|>assistant\n<think>\n{rng.choice(THINK)}\n</think>\n\n<tool_call>\n"
                       f"{json.dumps(call)}\n</tool_call><|im_end|>\n")
            seg.append(f"<|im_start|>user\n<tool_response>\n{resp}\n</tool_response><|im_end|>\n")
        if rng.random() < 0.5:
            f = rng.choice(files); src = open(f, errors="ignore").read().split("\n"); a = rng.randrange(max(1, len(src) - 8))
            old = "\n".join(src[a:a + 3]); call = {"name": "edit_file", "arguments": {"path": f.split("site-packages/")[-1],
                                                    "old_string": old, "new_string": old + "\n    # TODO(agent): checked"}}
            seg.append(f"<|im_start|>assistant\n<tool_call>\n{json.dumps(call)}\n</tool_call><|im_end|>\n"
                       f"<|im_start|>user\n<tool_response>\n{{\"ok\": true}}\n</tool_response><|im_end|>\n")
        seg.append("<|im_start|>assistant\n<think>\nI have enough information now.\n</think>\n\nThe change is in place: "
                   "the function now validates its inputs before scheduling, and the tests pass locally.<|im_end|>\n")
        s = "".join(seg); parts.append(s); n += len(tok(s, add_special_tokens=False).input_ids)
    return "".join(parts)

def build_corpus(tok):
    S = [json.loads(l) for l in open(NIAH)]
    code_cal, used = code_text(11, DOC_TOK, tok)
    code_ho, used_ho = code_text(23, HO_TOK, tok, exclude=set(used))
    cal_files = used; ho_files = used_ho
    docs = {  # name -> (text, max tokens, split)
        "essay": (HEAD + S[4]["body"], DOC_TOK, "cal"),
        "code": (HEAD + code_cal, DOC_TOK, "cal"),
        "chat": (chat_text(5, DOC_TOK, tok, cal_files), DOC_TOK, "cal"),
        "ho_essay": (HEAD + S[1]["body"], HO_TOK, "ho"),       # sample 1 shares no sentences with sample 4 (checked)
        "ho_code": (HEAD + code_ho, HO_TOK, "ho"),
        "ho_chat": (chat_text(99, HO_TOK, tok, ho_files), HO_TOK, "ho"),
    }
    meta = {"cal_code_files": [f.split("site-packages/")[-1] for f in cal_files],
            "ho_code_files": [f.split("site-packages/")[-1] for f in ho_files]}
    return docs, meta

# ------------------------------------------------------------------ GPU phase
def grams():
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AttentionInterface
    from transformers import cache_utils
    LONG = os.environ.get("LONG") == "1"                              # real long-position check (128K docs)
    LONG_LEN = int(os.environ.get("LONG_LEN", "128000")); LONG_FROM = LONG_LEN - 9216
    CAP = (LONG_LEN if LONG else max(DOC_TOK, HO_TOK)) + 64
    fa = torch.ops.aten._scaled_dot_product_flash_attention

    def _update(self, key_states, value_states, *args, **kwargs):       # preallocated KV (copied from lc27.py)
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            B, H, _, D = key_states.shape
            self.buf_k = torch.empty(B, H, CAP, D, dtype=key_states.dtype, device=key_states.device)
            self.buf_v = torch.empty_like(self.buf_k); self.n = 0
        m = key_states.shape[-2]
        self.buf_k[:, :, self.n:self.n + m] = key_states; self.buf_v[:, :, self.n:self.n + m] = value_states
        self.n += m
        self.keys, self.values = self.buf_k[:, :, :self.n], self.buf_v[:, :, :self.n]
        return self.keys, self.values
    cache_utils.DynamicLayer.update = _update

    def prefill_attn(q, k, v, scaling):                                  # copied from lc27.py
        B, Hq, Lq, D = q.shape; Hkv, Lk = k.shape[1], k.shape[2]; g = Hq // Hkv; off = Lk - Lq
        kc = k[:, :, off:].repeat_interleave(g, 1); vc = v[:, :, off:].repeat_interleave(g, 1)
        ob, lb = fa(q.contiguous(), kc, vc, 0.0, True, False, scale=scaling)[:2]
        if off == 0:
            return ob
        qg = q.reshape(B, Hkv, g * Lq, D).contiguous()
        oa, la = fa(qg, k[:, :, :off].contiguous(), v[:, :, :off].contiguous(), 0.0, False, False, scale=scaling)[:2]
        oa = oa.reshape(B, Hq, Lq, D); la = la.reshape(B, Hq, Lq)
        m = torch.maximum(la, lb); wa = (la - m).exp()[..., None]; wb = (lb - m).exp()[..., None]
        return ((oa.float() * wa + ob.float() * wb) / (wa + wb)).to(q.dtype)

    ST = {"name": None, "G": {}, "a": 0}

    def acc_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        scaling = scaling or 1 / math.sqrt(q.shape[-1])
        li = module.layer_idx; Lq = q.shape[2]; Hkv = k.shape[1]; g = q.shape[1] // Hkv
        kn = k[0, :, -Lq:].double()                                        # this chunk's keys [Hkv, Lq, D]
        qn = q[0].reshape(Hkv, g * Lq, q.shape[-1]).double()
        name = ST["name"]
        if LONG:                                  # only the first 8K and the last ~9K positions of a long document
            if ST["a"] < 8192: name += "_head"
            elif ST["a"] >= LONG_FROM: name += "_tail"
            else: return prefill_attn(q, k, v, scaling).transpose(1, 2).contiguous(), None
        d = ST["G"].setdefault(name, {}).setdefault(li, {"k": 0, "q": 0, "kfirst": None, "n": 0})
        d["k"] = d["k"] + (kn.transpose(1, 2) @ kn).cpu()
        d["q"] = d["q"] + (qn.transpose(1, 2) @ qn).cpu()
        if d["n"] == 0:                                                    # lc27-style foreign basis: first 4096 tokens
            d["kfirst"] = (kn.transpose(1, 2) @ kn).cpu()
        d["n"] += Lq
        o = prefill_attn(q, k, v, scaling)
        return o.transpose(1, 2).contiguous(), None
    AttentionInterface.register("acc", acc_attn)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from q27loader import load as _load27
    t0 = time.time()
    model, _cfg = _load27(); model.eval()
    for _m in model.modules():
        if hasattr(_m, "config") and hasattr(_m.config, "_attn_implementation"): _m.config._attn_implementation = "acc"
    tok = AutoTokenizer.from_pretrained(M)
    print(f"loaded in {time.time() - t0:.0f}s, mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    docs, meta = build_corpus(tok)
    if LONG:                                      # held-out only: essays 0-3 (no overlap with the tail region) and fresh code
        S = [json.loads(l) for l in open(NIAH)]
        lc, _ = code_text(37, LONG_LEN + 2000, tok, exclude=set(VLLM_SRC.rsplit('/vllm', 1)[0] + '/' + f for f in meta['cal_code_files']))
        docs = {"long_essay": (HEAD + "\n\n".join(s["body"] for s in S[:4]), LONG_LEN, "ho"),
                "long_code": (HEAD + lc, LONG_LEN, "ho")}
    ntok = {}
    with torch.no_grad():
        for name, (text, cap, split) in docs.items():
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :cap].cuda()
            ST["name"] = name; cache = None; t = time.time()
            for a in range(0, ids.shape[1], CH):
                ST["a"] = a
                out = model(ids[:, a:a + CH], past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
            ntok[name] = ids.shape[1]
            print(f"{name}: {ids.shape[1]} tok in {time.time() - t:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush=True)
            del cache, out; torch.cuda.empty_cache()
    # rotation matrices for RoPE shifts (via the model's own rotary module and apply_rotary_pos_emb)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    rot = model.model.language_model.rotary_emb
    Rm = {}
    for delta in [0, 64000, 128000] + list(range(0, 150001, 2500)):
        if delta in Rm: continue
        pid = torch.full((1, 1), delta, device="cuda", dtype=torch.long)
        x = torch.zeros(1, 1, 1, 256, device="cuda", dtype=torch.float32)
        cos, sin = rot(x, pid)
        eye = torch.eye(256, device="cuda", dtype=torch.float32)[None, None]      # rows e_j -> R e_j  => matrix = R^T
        _, rt = apply_rotary_pos_emb(eye, eye, cos.float(), sin.float())
        Rm[delta] = rt[0, 0].T.double().cpu()                                      # R: k_rot(p+delta) = R @ k_rot(p)
    os.makedirs(WORK, exist_ok=True)
    if LONG:
        ntok = {n: G_[li]['n'] for n, G_ in ST['G'].items() for li in [LAYERS[0]]}
        torch.save({"G": ST["G"], "ntok": ntok, "long_len": LONG_LEN, "tail_from": LONG_FROM}, f"{WORK}/grams_long.pt")
        print("saved", f"{WORK}/grams_long.pt", ntok, flush=True); return
    torch.save({"G": ST["G"], "ntok": ntok, "R": Rm, "meta": meta, "layers": LAYERS}, f"{WORK}/grams.pt")
    with open(f"{WORK}/corpus_sample.txt", "w") as f:                               # for inspection
        f.write(docs["chat"][0][:6000])
    print("saved", f"{WORK}/grams.pt", ntok, flush=True)

# ------------------------------------------------------------------ CPU phase
def save_atomic(obj, path):  # readers (other streams) never see a half-written file
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)

def top_basis(G):            # G [Hkv, D, D] -> [Hkv, R, D] rows = top-R eigenvectors
    w, V = torch.linalg.eigh(G)
    return V.flip(-1)[..., :R].transpose(-1, -2).contiguous()

def rot_avg(G, Rm, deltas):  # mean over shifts of R G R^T
    return sum(Rm[d] @ G @ Rm[d].T for d in deltas) / len(deltas)

def fit():
    torch.set_num_threads(4)
    D = torch.load(f"{WORK}/grams.pt", weights_only=False)
    G, ntok, Rm = D["G"], D["ntok"], D["R"]
    cal = ["essay", "code", "chat"]
    for d in Rm.values():
        assert torch.allclose(d @ d.T, torch.eye(256, dtype=d.dtype), atol=1e-5), "rotation not orthogonal"
    nk = lambda name, li: G[name][li]["k"] / ntok[name]       # per-token-normalized Gram (equal weight per corpus)
    shifts = sorted(d for d in Rm if d % 2500 == 0)            # 0..150K step 2.5K
    bases = {}
    for tag in ("mixed", "mixed_rope", "essay", "foreign4k"):
        B = []
        for li in LAYERS:
            if tag == "mixed": g = sum(nk(n, li) for n in cal)
            elif tag == "mixed_rope": g = rot_avg(sum(nk(n, li) for n in cal), Rm, shifts)
            elif tag == "essay": g = nk("essay", li)
            else: g = G["essay"][li]["kfirst"]
            B.append(top_basis(g))
        bases[tag] = torch.stack(B)                            # [16, 4, 64, 256] float64
    # held-out validation
    def metrics(Bl, li, name, delta):
        Gk = G[name][li]["k"]; Gq = G[name][li]["q"]
        if delta: Gk = Rm[delta] @ Gk @ Rm[delta].T; Gq = Rm[delta] @ Gq @ Rm[delta].T
        Bt = Bl.transpose(-1, -2)
        Kr = Bl @ Gk @ Bt; Qr = Bl @ Gq @ Bt                    # [Hkv, R, R] projected Grams
        ke = torch.diagonal(Kr, dim1=-2, dim2=-1).sum(-1) / torch.diagonal(Gk, dim1=-2, dim2=-1).sum(-1)
        le = (Qr * Kr).sum((-1, -2)) / (Gq * Gk).sum((-1, -2))  # tr(P Gq P Gk) / tr(Gq Gk), Grams symmetric
        return ke, le                                           # per KV head
    rep = {}
    for tag, BB in bases.items():
        rep[tag] = {}
        for name in ("ho_essay", "ho_code", "ho_chat", "essay", "code", "chat"):
            for delta in (0, 64000, 128000):
                key = f"{name}@+{delta // 1000}K"
                rows = []
                for i, li in enumerate(LAYERS):
                    ke, le = metrics(BB[i], li, name, delta)
                    rows.append({"layer": li, "key_energy": ke.tolist(), "logit_energy": le.tolist()})
                rep[tag][key] = rows
    if os.path.exists(f"{WORK}/grams_long.pt"):                # REAL long positions (no shift assumption)
        GL = torch.load(f"{WORK}/grams_long.pt", weights_only=False)
        G.update(GL["G"])
        names = sorted(GL["G"])
        for tag, BB in bases.items():
            for name in names:
                rows = []
                for i, li in enumerate(LAYERS):
                    ke, le = metrics(BB[i], li, name, 0)
                    rows.append({"layer": li, "key_energy": ke.tolist(), "logit_energy": le.tolist()})
                rep[tag][f"{name}@real"] = rows
        print(f"\n== REAL positions (LONG_LEN={GL['long_len']}; head = first 8K tokens, tail = positions >= {GL['tail_from']})")
        print(f"{'basis':12s} " + " ".join(f"{n:>17s}" for n in names) + "   (key_energy / logit_energy, mean over layers+heads)")
        for tag in bases:
            vals = []
            for n in names:
                r = rep[tag][f"{n}@real"]
                vals.append(f"{sum(sum(x['key_energy']) for x in r) / 64:.3f}/{sum(sum(x['logit_energy']) for x in r) / 64:.3f}")
            print(f"{tag:12s} " + " ".join(f"{v:>17s}" for v in vals))
        print("per-layer logit_energy on long_essay_tail / long_code_tail:")
        print("layer " + " ".join(f"{t:>21s}" for t in bases))
        for i, li in enumerate(LAYERS):
            print(f"{li:5d} " + " ".join(f"{sum(rep[t]['long_essay_tail@real'][i]['logit_energy']) / 4:10.3f}/"
                                        f"{sum(rep[t]['long_code_tail@real'][i]['logit_energy']) / 4:.3f}" for t in bases))
    json.dump(rep, open(f"{WORK}/validation.json", "w"), indent=1)
    # print summary: held-out key-energy per layer (mean over 4 KV heads), and logit energy
    for metric in ("key_energy", "logit_energy"):
        print(f"\n== held-out {metric} (mean over KV heads), positions native / +128K shift")
        cols = [(t, n, d) for t in ("mixed", "mixed_rope", "essay", "foreign4k") for n in ("ho_essay", "ho_code", "ho_chat")
                for d in (0,)]
        print("layer " + " ".join(f"{t[:6]}:{n[3:]:>5}" for t, n, d in cols))
        for i, li in enumerate(LAYERS):
            print(f"{li:5d} " + " ".join(f"{sum(rep[t][f'{n}@+0K'][i][metric]) / 4:12.4f}" for t, n, d in cols))
        print("mean  " + " ".join(f"{sum(sum(rep[t][f'{n}@+0K'][i][metric]) / 4 for i in range(16)) / 16:12.4f}"
                                 for t, n, d in cols))
        for d in (64000, 128000):
            print(f"mean@+{d // 1000}K " + " ".join(
                f"{sum(sum(rep[t][f'{n}@+{d // 1000}K'][i][metric]) / 4 for i in range(16)) / 16:10.4f}" for t, n, _ in cols))
    meta_common = {"model": "Qwen3.8-27B-W4A16-AutoRound-fast", "keys": "post-k_norm post-RoPE (HF Qwen3_5 attention input)",
                   "fit": "top-64 eigenvectors of per-(layer,KV head) key Gram K^T K", "ntok": ntok,
                   "date": time.strftime("%Y-%m-%d %H:%M"), "validation": f"{WORK}/validation.json",
                   "script": "sparsekv/tools/fit_basis27.py"}
    choice = os.environ.get("CHOICE", "mixed_rope")
    save_atomic({"layers": LAYERS, "basis": bases[choice].float().contiguous(), "r": R,
                "meta": dict(meta_common, corpus=f"{choice}: essay(niah sample 4)+vllm code+synthetic agent chat, "
                             "equal weight per corpus" + (", Gram averaged over RoPE shifts 0..150K" if "rope" in choice else ""),
                             code_files=D["meta"]["cal_code_files"])}, OUT)
    save_atomic({"layers": LAYERS, "basis": bases["essay"].float().contiguous(), "r": R,
                "meta": dict(meta_common, corpus="essay only (niah sample 4, first %d tokens)" % ntok["essay"])}, OUT_ESSAY)
    for tag, BB in bases.items():
        save_atomic({"layers": LAYERS, "basis": BB.float().contiguous(), "r": R, "meta": dict(meta_common, corpus=tag)},
                   f"{WORK}/basis27_{tag}.pt")
    print("\nwrote", OUT, "(", choice, ")", OUT_ESSAY, "and", WORK + "/basis27_*.pt")

if __name__ == "__main__":
    {"grams": grams, "fit": fit}[sys.argv[1]]()
