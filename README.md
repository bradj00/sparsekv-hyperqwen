<p align="center"><img src="docs/banner.png" alt="Sparse-KV for HyperQwen" width="100%"></p>

# sparsekv-hyperqwen

Query-aware sparse KV decode + a host-RAM KV tier for [HyperQwen](https://github.com/syv-ai/HyperQwen) (vLLM 0.31 + pthfa).

Keep the full KV cache in host RAM at int4, and at each decode step attend only to the ~1% of tokens that matter most, picked by a cheap low-rank key sketch.

> **~4 concurrent 250K-context agents per 24 GB GPU** (Qwen3.8-27B W4A16), with output divergence comparable to dense run on a second GPU.

## What's new in v1.1

Less time waiting for prefill, at no measured accuracy cost. All changes are on by default; details are in
[`docs/CHANGELOG.md`](docs/CHANGELOG.md) and [`eval/prefill/`](eval/prefill/README.md).

- **Accuracy fix shipped:** exact follow-up prefill and crash-safe staging, the fixed configuration from the
  benchmark: call-site F1 −13.7 → −2.1 vs dense, within noise.
- **No more full re-prefills after tool waits:** 32 host GDN checkpoint slots plus decode-boundary saves. Re-prefills
  went from 43% of prefill to 0, and per-step prefill from 6.4 to 4.0 s. Turn-2 TTFT after eviction: 5.75 → 2.06 s.
- **Faster follow-ups:** +864 cached tokens per turn (MTP block drop off). Follow-up TTFT: 2.82 → 1.93 s.
- **FlashAttention-2 prefill** over the int8 cache. Cold ~240K prefill: 397 → 337 s on the 3090 Ti.
- **Short turns skip the prefill queue.** With 4 agents, median follow-up TTFT went 9.8 → 7.1 s, at 4-10% lower
  decode throughput.
- **Shared agent prefixes:** a pin knob plus [harness tips](docs/AGENT_HARNESS_TIPS.md). The common first-request
  prefix grew from 7,262 to 16,026 tokens.

The throughput table under Results was measured with the v1.0 knobs (`SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2`).

---

## How it works

| Piece | What it does |
|---|---|
| **Sparse decode** | On each full-attention layer, score all tokens with a rank-64 int4 key sketch, then run exact int8 attention on the **top 2048 + 64 sink + last 1024** tokens. Prefill and the MTP drafter stay dense. |
| **Frozen sketch basis** | Fitted once offline from real keys and the same for every document, so **prefix caching still works**. |
| **Host-RAM tier** | Full KV lives in pinned host RAM at **int4** (4.4 GB per 250K agent vs 8.4 GB at int8). The GPU keeps only the sketch, sinks, recent tail and a 2.5% hot buffer. |
| **Zero-copy misses** | The GPU reads only the picked rows straight from host memory over PCIe (UVA). Miss rate is 14–16%, and it **works even on a PCIe x4 link**. |
| **No re-staging** | Follow-up turns attend sparsely and fetch only the picked rows, so agents don't stall copying GBs per turn. |
| **Off by default** | Every change is gated by a `SPARSE_*` env var. Short contexts replay the dense CUDA graphs. |

**GPU cost per 250K agent: ~1.6 GB** (0.77 GDN+MTP state · 0.53 sketch · 0.22 hot buffer · ~0.1 tail)

---

## Results

Temperature 0, ~244K-token documents, 1024-token answers. Throughput is combined decode across agents.

| Setup | Decode tok/s (explain / copy) | Follow-up TTFT | Recall |
|---|---|---|---|
| 4 × 250K per GPU | 268–285 / 375–415 | 5.2–5.9 s | 31/32 |
| 4 × 128K per GPU | 312 / 501 | 4.4–5.6 s | 11/13 |
| 1 × ~113K vs dense | **+15% to +41%** faster | — | — |

Throughput scales linearly across GPUs (2 GPUs at 8 × 250K: ~545 / ~790 tok/s, 0 errors, 0 re-prefills).

### Quality (KL divergence vs dense, teacher-forced, 15,300 decode tokens)

| Config | Mean KLD | Top-1 agree |
|---|---|---|
| Dense, same arch, different GPU (noise floor) | 0.0200 | 96.2% |
| Sparse decode, GPU-only | **0.0188** | **96.4%** |
| Sparse + int4 RAM tier (default) | 0.0269 | 95.6% |

- GPU-only sparse diverges from dense **less** than dense does across two GPUs of the same architecture.
- Recall-style answers: 99.6–99.9% top-1 agreement.
- Divergence grows slowly with context (0.024 at 32K → 0.033 at 192K).

Full method: [`eval/kld/README.md`](eval/kld/README.md) · every measurement and knob: `sparsekv/RELEASE.md` · **full benchmark (RULER, call-site, SCBench, GPQA, LiveCodeBench, IFBench, MMLU-Pro; per-item data + runtimes): [`eval/bench/`](eval/bench/README.md)**

### Limitations

- This is approximate attention: greedy output can diverge from dense after some tokens.
- Selection cost and host misses grow with context × agents.
- A cold 244K prefill takes ~390 s, and prefills run one at a time.
- 4 agents per 24 GB GPU is the ceiling at MTP-7.
- The tier hooks patch vLLM scheduler internals, so they may break on vLLM upgrades.

---

## Install

**Needs:** NVIDIA 24 GB GPU · ~18 GB free host RAM per GPU (at 4 × 250K) · CUDA 13.x · a working HyperQwen venv

```bash
# 1. Get HyperQwen at the tested commit and apply this repo
git clone https://github.com/syv-ai/HyperQwen && cd HyperQwen && git checkout ce9615c
git apply ../sparsekv-hyperqwen/patches/hyperqwen-base.diff
cp -r ../sparsekv-hyperqwen/{sparsekv,pthfa,serve-sparse.sh} .
cp ../sparsekv-hyperqwen/patches/sparse-*.patch patches/

# 2. Build a normal HyperQwen venv at ./venv (patches/apply.sh + pthfa), then:
bash sparsekv/install.sh venv --basis sparsekv/basis/basis27.pt   # idempotent; --check / --uninstall
```

## Run

```bash
HQ_GPU=0 SPARSE_PROFILE=tier MAX_LEN=262144 MAX_SEQS=4 GPU_UTIL=0.94 \
SPARSE_TIER_HOT_FRAC=0.025 SPARSE_SKETCH_BITS=4 SPARSE_TIER_HOST_BITS=4 \
SPARSE_TIER_HOST_GB=17.8 SPARSE_STATS_SUB=1 SPARSE_SELECT_EVERY=1 \
  ./serve-sparse.sh          # serves on 127.0.0.1:8093
```

| Profile | `MAX_LEN` | `GPU_UTIL` | `SPARSE_TIER_HOT_FRAC` | `SPARSE_TIER_HOST_GB` |
|---|---|---|---|---|
| 4 × 250K | 262144 | 0.94 | 0.025 | 17.8 |
| 4 × 128K | 131072 | 0.93 | 0.05 | 8.8 |

- **API key:** read from `VLLM_API_KEY`, else from the file at `API_KEY_FILE` (default `./api_key.txt`).
- **Healthy boot log:** shows `host pool 17.7 GB` and `GPU pool 157 blocks`, with no `allocated > reserved`.

## Uninstall

```bash
bash sparsekv/install.sh venv --uninstall   # byte-identical to the pre-install venv
```

To go back to dense without uninstalling, start the server with no `SPARSE_*` variables set and it behaves like stock.

---

## Layout

```
sparsekv/       Triton kernels, host tier + vLLM KV connector, tests, basis, install.sh
patches/        vLLM patches + hyperqwen-base.diff
pthfa/          decode hook (python/pth_flash_attn/sparse_kv.py), FA2 prefill (fa2_prefill.py)
eval/kld/       KL-divergence test vs dense
eval/bench/     full benchmark; eval/prefill/: v1.1 prefill results
docs/           CHANGELOG, agent harness tips
examples/       opencode stable-prefix plugin
run*/ results/  harness scripts + raw results (inputs via env vars, see RELEASE.md)
serve-sparse.sh launcher
```

<details>
<summary>Test hardware</summary>

Two 24 GB Ampere GPUs (one on PCIe Gen4 x16, one on x4), both capped at 250 W · Intel i9-12900K · 96 GB DDR5 · VM with 78 GB RAM · Ubuntu 22.04, driver 595.99.02, CUDA 13.2, torch 2.13, Triton 3.7.1.
</details>

## License & credits

Apache-2.0 (`LICENSE`); third-party attributions are in `NOTICE`.

- Built on [HyperQwen](https://github.com/syv-ai/HyperQwen), [vLLM](https://github.com/vllm-project/vllm) and [pth-flash-attn](https://github.com/noon-at-cgn/pth-flash-attn) (all Apache-2.0). Only patches are shipped, not their source.
- Model: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). `basis27.pt` is derived from its key activations and contains no weights or text.
- The needle-in-a-haystack method follows Greg Kamradt's [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) (MIT). The essay haystacks are not included.
- Related research: [`sparsekv-lab`](../sparsekv-lab)
