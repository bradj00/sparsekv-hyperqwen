# sparsekv-hyperqwen

Query-aware sparse KV decode plus a host-RAM KV tier for HyperQwen (vLLM 0.31 + pthfa). Result: **8 concurrent agents at 250K context each on one RTX 3090 Ti + one RTX 3090** (Qwen3.8-27B W4A16), all from one box.

## What we did to make it work on our rig

- **Sparse decode.** Each decode step, each of the 16 full-attention layers (layers 3, 7, ..., 63; the other 48 are GDN linear attention) scores every token through a rank-64 key sketch. It then runs exact pthfa int8 attention on the top 2048 tokens (~0.8% of 250K), 64 sink tokens and the last 1024 tokens. Prefill and the MTP drafter stay dense.
- **One pick per layer, shared by all 4 KV heads.** The sketch is int4 (`SPARSE_SKETCH_BITS=4`, ~2.1 KB per token over all 16 layers). Top-k is a deterministic Triton radix select.
- **Frozen sketch basis.** `basis27.pt` holds rank-64 eigenvectors of real post-RoPE keys, fitted once offline (`sparsekv/tools/fit_basis27.py`). Because it is the same for every document, prefix caching keeps working.
- **The full KV lives in pinned host RAM at int4** (`SPARSE_TIER_HOST_BITS=4`: 64-dim groups with fp16 scales, 272 B per token-head; it is dequantised back to pthfa int8 rows on fetch). That is 4.4 GB per 250K agent, against 8.4 GB at int8. The GPU keeps the sketch, the sinks, the recent tail and a 2.5% hot buffer (`SPARSE_TIER_HOT_FRAC=0.025`, windowed CLOCK).
- **Hot-buffer misses use zero-copy reads.** The GPU reads only the picked rows straight from pinned memory over PCIe (UVA). Measured: ~10 GB/s on the x16 card and 6.2 GB/s on the x4 card, even with 512 B rows. Misses run 14-16%, about 30-39 MB per step per server.
- **Follow-ups are not re-staged.** A follow-up turn's uncached suffix (≤2048 tokens) attends sparsely: 16-row question segments plus 256-row tail segments, suffix KSEL 4096, and only the picked rows are fetched. Before this fix, every turn copied 2.4-15 GB and the agents serialised.
- **Prefix lookups prefer the host copy** for mid-prefix blocks. A returning agent therefore no longer grabs GPU pool blocks, which had kept the 4th agent stuck in Waiting.
- **Cheaper selection.** `SPARSE_STATS_SUB=4` samples every 4th tile in the normaliser pass. `SPARSE_SELECT_EVERY=2` reuses the top-k on alternate steps. Together: 4×246K select 18.9 → 12.0 ms/step.
- **GPU memory budgeting.** Sparse buffers are reserved out of the KV budget (`SPARSE_MEM_RESERVE=1`), and the CUDA-graph padding slot gets no hot rows. Each agent costs ~1.6 GB of GPU: 0.77 GB GDN+MTP-7 state, 0.53 GB int4 sketch, 0.22 GB hot share, ~0.1 GB tail. **4 slots per 24 GB card at 250K** (GPU_UTIL 0.94, pool 157 blocks vs ~124 needed).
- **Dense/sparse switch per step.** Two CUDA-graph families exist; short contexts replay the dense graphs, long ones the sparse graphs. All changes are inert unless their `SPARSE_*` knob is set.

## Results (measured, temperature 0, documents = ~244K-token vLLM source dumps, 1024-token answers)

| setup | combined decode, explain / copy (tok/s) | follow-up TTFT | recall (needle + code line) |
|---|---|---|---|
| **8 × 250K, both cards at once (box)** | **~545 / ~790** (540-556 / 770-810) | 5.2-5.9 s | 31/32 |
| 4 × 250K, 3090 Ti (x16) | 268-271 / 394-415 | 5.2 / 5.5 s med/max | 15/16 |
| 4 × 250K, 3090 (x4) | 268-285 / 375-395 | 5.3 / 5.9 s | 16/16 |
| 4 × 128K, 3090 Ti | 312 / 501 | 4.4 / 5.6 s | 11/13 |
| go-live A/B, 1 user @ ~113K vs dense baseline, same card | +15% (deep3 127→146) to +36% (copy 140→191); second server (3090) +29% / +41% | | |

Throughout the 8 × 250K run: MemAvailable ≥ 11.6 GB, 0 errors, 0 re-prefills, 0 GB re-staged. A cold prefill of one 244K document takes 390 s, and prefills run one at a time.

**Honest notes**
- **Missed target:** we aimed for 300+ explain tok/s per card at 4 × 250K and got 268-285. Selection cost (3.3 ms/agent/step at 246K) and host misses grow with context × agents.
- **Accuracy is approximate attention.** Needles were always found. The code-line misses are near-duplicate lines (CuTe/asm blocks) that every build missed. Greedy output can diverge from dense after some tokens.
- **No 5th agent per card** at MTP-7. That would need the GDN spec states outside the pool. MTP-3/5 change the block size to 816, and the kernels are built for 864.
- The tier hooks monkeypatch vLLM scheduler internals and are fragile across vLLM upgrades.

## The rig

| | |
|---|---|
| host | consumer desktop: Z790 board, Intel i9-12900K (16C/24T), 96 GB DDR5 (2 × 48 GB, running at 4800 MT/s); serving runs in a VM with GPU passthrough |
| GPU0 | RTX 3090 Ti 24 GB, PCIe Gen4 **x16** (passthrough) |
| GPU1 | RTX 3090 24 GB, PCIe Gen4 slot running at **x4** (downgraded link, measured 6.2 GB/s host reads) |
| VM | Ubuntu 22.04, 24 vCPU, **78 GB RAM** (of the host's 96 GB), NVIDIA driver 595.99.02 / CUDA 13.2, torch 2.13 + cu130, Triton 3.7.1 |
| power / cooling | both cards capped at **250 W** (max 450 W), fans pinned at 100%, 1000 W PSU. A 2-card soak at ~750 W tripped the UPS, which is why the cards are capped |
| model | Qwen3.8-27B W4A16 (AutoRound), MTP-7 speculative decode, int8 KV |
| stack | vLLM 0.31 (HyperQwen fork, github.com/syv-ai/HyperQwen @ `ce9615c`) + pthfa int8 attention kernel + this repo |

## Layout

- `sparsekv/`: `triton_ops.py` (sketch write, select/radix top-k, gather, select_suffix), `tier.py` and `tier_connector.py` (host tier, `HostTierConnector` vLLM KV connector), `ref.py` (PyTorch reference), `install.sh`, tests `test_*.py`, `patches/pthfa-sparse-kv.patch`, `basis/basis27.pt` (4.2 MB, sha256 `dee8b282…`), `tools/fit_basis27.py` + `tools/q27loader.py` (basis fitting). `RELEASE.md` has every measurement and knob; `STREAM_*.md` are the design notes.
- `patches/`: our two vLLM patches (`sparse-switch-dual-graphs.patch`, `sparse-tier-spill-kvcheck.patch`) and `hyperqwen-base.diff`. The diff holds our edits to upstream HyperQwen files (applied with `git apply`, so no modified HyperQwen files are shipped here): the `patches/series` / `patches/apply.sh --sparse` registration, the pthfa `vllm_backend.py` hook, `start_qwen.sh` and `.gitignore`.
- `pthfa/python/pth_flash_attn/sparse_kv.py`: the decode hook.
- `serve-sparse.sh` (launcher).
- `run_s4/` (4 agents/card, 8 × 250K headline: `d.joint_g{0,1}.jsonl`), `run_250k/` (250K tier and streams-per-card sweep), `run_release/` (release smoke, grid, concurrency), `run_tier/`, `run/`, `results/` (kernel benches per GPU). These hold the harness scripts plus jsonl/json results.

## Install / run

```bash
git clone https://github.com/syv-ai/HyperQwen && cd HyperQwen && git checkout ce9615c
git apply ../sparsekv-hyperqwen/patches/hyperqwen-base.diff
cp -r ../sparsekv-hyperqwen/{sparsekv,pthfa,serve-sparse.sh} .
cp ../sparsekv-hyperqwen/patches/sparse-*.patch patches/
# build a normal HyperQwen venv at ./venv first (this tree's patches/apply.sh, now carrying the sparse entries from the diff, + pthfa), then:
bash sparsekv/install.sh venv --basis sparsekv/basis/basis27.pt   # idempotent; --check / --uninstall
HQ_GPU=0 SPARSE_PROFILE=tier MAX_LEN=262144 MAX_SEQS=4 GPU_UTIL=0.94 SPARSE_TIER_HOT_FRAC=0.025 \
  SPARSE_SKETCH_BITS=4 SPARSE_TIER_HOST_BITS=4 SPARSE_TIER_HOST_GB=17.8 SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2 \
  ./serve-sparse.sh          # dev default 127.0.0.1:8093
```

The 4 × 128K config is `MAX_LEN=131072 MAX_SEQS=4 GPU_UTIL=0.93 SPARSE_TIER_HOT_FRAC=0.05 SPARSE_TIER_HOST_GB=8.8` (other knobs as above). `serve-sparse.sh` takes the API key from `VLLM_API_KEY`, or else reads it from the file `API_KEY_FILE` (default `./api_key.txt`, gitignored). To check the boot log, look for `host pool 17.7 GB` and `GPU pool 157 blocks` and confirm there is no `allocated > reserved`.

**Harness scripts** (`run*/`) are the scripts that produced the results, kept as a record. They find the repo from their own location and take their machine-specific inputs from the environment: `VLLM_API_KEY` / `API_KEY_FILE`, `HF_HOME`, `GPU_LOCK_DIR` (flock files, default `/tmp`), `DEV_PY` (python of a dev venv, default `python3`), `NIAH_JSONL` (needle-in-a-haystack samples), `TRACE_RUNS` / `TIER_TRACES` (recorded selection traces), `VLLM_SRC` (the vLLM source tree used as the code corpus). `run_release/grid.*.jsonl` and `run/harness.hsp.json` were produced by an internal eval harness (not included). `sparsekv/tools/fit_basis27.py` (with its model loader `sparsekv/tools/q27loader.py`) takes `MODEL_DIR`, `NIAH_JSONL`, `WORK`, `OUT`. Placeholders such as `$REPO` in the notes and logs stand for your checkout.

## Uninstall / rollback

The release never modifies a separate dense install: it lives in its own venv, and every patch is inert unless its `SPARSE_*` knob is set.
- **Back to dense serving:** stop `serve-sparse.sh` (by PID / process group, or `systemctl stop` if you run it as a unit) and start your normal HyperQwen launcher (`single-user/start_qwen.sh` without `SPARSE_*` variables). The same venv also serves dense: with no `SPARSE_*` set it behaves like stock (`run_release/dense-control.sh`).
- **Remove the release from a venv:** `bash sparsekv/install.sh venv --uninstall` reverses exactly what the install applied (vLLM and pthfa patches, the `sparsekv` module, the basis); the result is byte-identical to the pre-install venv.
- Then check: a short chat completion, and `curl -s localhost:<port>/metrics | grep -m3 spec_decode` shows speculative decoding counters moving.

## Related

Research, experiments and reports behind this release: [`sparsekv-lab`](../sparsekv-lab).

## License & credits

Our code is Apache-2.0 (`LICENSE`); third-party attributions are in `NOTICE`.

- Built on [HyperQwen](https://github.com/syv-ai/HyperQwen) (Apache-2.0), a fork of
  [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0), and
  [pth-flash-attn](https://github.com/noon-at-cgn/pth-flash-attn) (Apache-2.0).
  We ship patches against them, not their source.
- Model: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0);
  `sparsekv/basis/basis27.pt` is derived from its key activations (no weights, no text).
- Uses PyTorch (BSD-3-Clause), Triton (MIT), transformers and compressed-tensors (Apache-2.0).
- Needle-in-a-haystack method after Greg Kamradt's
  [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) (MIT). The essay haystacks
  are not included (the essays are copyrighted by their author, and quoted essay text in recorded
  results is redacted); point `NIAH_JSONL` at TokenGate's `examples/data/niah_32k_samples.jsonl`.
- Shared for non-commercial, open use in the spirit of "power to the people";
  Apache-2.0 allows any use.
