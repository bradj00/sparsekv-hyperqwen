# Stream B (backend): sparse decode in PthFlashAttnImpl

Branch `sk-backend`, its own worktree.

## What it does (SPARSE_KV=1)
- `pth_flash_attn/sparse_kv.py` (new) + about 35 lines in `pth_flash_attn/vllm_backend.py` (`PthFlashAttnImpl.forward`).
- **Per-layer state**: created at the first forward that has metadata. The layer index comes from `layer.layer_name` (`...layers.N...`).
  - Layers whose name contains `mtp`, or whose index is not in the basis file, stay dense (no sketches).
  - The basis comes from `SPARSE_BASIS` (default `sparsekv/basis/basis27.pt`). If the file is missing, a seeded random-orthonormal placeholder is used and a warning is logged.
- **Allocation**: lazy, in the first eager forward that has metadata.
  - The sketch is `int8 [NB*BS+1, 4, 64]` plus an `fp16 [NB*BS+1, 4]` scale, keyed on the KV tensor's data_ptr and shape.
  - It is reallocated if the KV tensor changes. The profile run uses a tiny dummy cache first; the real allocation happens at the warm-up after KV init.
  - It raises if it would allocate inside a CUDA-graph capture. The V2 runner always does an eager warm-up before each capture, so this never triggered.
  - One **staging cache** is shared by all layers: `max_num_seqs x ceil(C/864)` blocks (2 x 4 = 13.7 MiB at C=3136), with a fixed staging block table.
- **Every forward** (prefill and decode): `write_sketch(key, attn_metadata.slot_mapping)`. Padding slots (-1) go to the dummy row.
- **Decode slice**:
  1. `select(q, qsl, seq_lens, block_table, ...)`
  2. `gather_rows(kv -> staging)`
  3. the same `LaunchPlan`, run by `_run_slice` with `dataclasses.replace(decode, seq_lens=n_valid, block_table=staging_bt)` and staging cache views (`staging_views`).

  The prefill slice is unchanged (dense).
- **Graph-safe**: there are no host syncs and no data-dependent shapes on the decode path. The pthfa plan has no seq-len input, so reusing it for the staging launch is valid.
- **Ops**: `SPARSE_IMPL=triton` imports `sparsekv.triton_ops` and falls back to `sparsekv.ref` if the import fails. `SPARSE_IMPL=ref` forces the reference.
  - `select` kwarg names differ (ref: `max_query_len`, `block_size`; triton: `max_q_len`, fixed BS=864). `_Cfg.select_kwargs` adapts by signature.
- Knobs: `SPARSE_KSEL` (2048), `SPARSE_SINK` (64), `SPARSE_WIN` (1024), `SPARSEKV_ROOT` (directory containing `sparsekv/`; default cwd, which is the repo because start_qwen.sh cd's there).

## Files
- `patches/pthfa-sparse-kv.patch`: the pthfa change (diff of `pthfa/python/pth_flash_attn/`).
- `pthfa/python/pth_flash_attn/{sparse_kv.py,vllm_backend.py}`: edited sources.
- `sparsekv/ref.py`: stream B's minimal reference (contract signatures plus `max_query_len`/`block_size` kwargs).
- `sparsekv/triton_ops.py`: snapshot of stream A's (kernel worktree, commit 1fde056).
- `sparsekv/test_backend_sparse.py`: unit test, select -> gather -> pthfa over staging.
- `run/serve.sh`, `run/stop.sh`, `run/session.sh`, `run/gpurun.sh`, `run/client.py`: launch, test and stop harness. `run/results.*.json` holds the outputs.

## Apply to a venv
```
cd <venv>/lib/python3.12/site-packages && patch -p3 < patches/pthfa-sparse-kv.patch
# or copy pthfa/python/pth_flash_attn/{sparse_kv.py,vllm_backend.py} over the installed package
```
The repo root (holding `sparsekv/`) must be the server's cwd, or be set with `SPARSEKV_ROOT`.

## Launch (inside a GPU lock)
```
cd $REPO
GPUS=1 SPARSE_KV=1 SPARSE_IMPL=triton run/gpurun.sh "run/session.sh <tag> 8096 needle,code,speed"
```
`run/serve.sh` sets the production flags: PTH_FA=1 SPEC=mtp DRAFT_TOKENS=7 CTX=long MAX_LEN=153600 MAX_SEQS=2 PREFIX_CACHE=1 EMBED_UVA=1 INT8_ACT=int8 PREFILL_ATTN=int8. It also sets **GPU_UTIL=0.88**, HOST=127.0.0.1, its own TMPDIR/HF_HOME, and SPARSEKV_ROOT.

## Memory
Measured with GPU_UTIL=0.88, MAX_LEN=153600, MAX_SEQS=2.
- Sketch per attention layer: `(NB*BS+1) x 4 x (64 B + 2 B)`. At 231 blocks that is 199,585 rows = **50.2 MiB/layer, 16 layers = 803 MiB**.
  - At the production pool (GPU_UTIL=0.93, about 250 blocks) it would be about 57 MiB/layer, or about 0.9 GiB.
  - It is allocated **after** vLLM sizes the KV pool, so it comes out of the (1-GPU_UTIL) slack. Production's 0.93 would leave only about 0.75 GiB for transients, so the launch here uses **0.88**.
  - 0.90 would probably fit, but it is untested.
- Staging cache: 13.7 MiB (2 requests x 4 blocks at C=3136). With KSEL=30000 it is 123 MiB.
- KV pool tokens on GPU0 (3090 Ti) at GPU_UTIL 0.88: dense 181,322; sparse+triton 180,573; sparse+ref 173,080. The ref's select temporaries enlarge the profiled CUDA-graph pool.
  - For comparison, production at 0.93 has 215,789.

## Results
All of these ran on GPU0 (RTX 3090 Ti) with real basis27.pt, temperature 0 and thinking off (`run/results.<tag>.json`).

| check | dense | sparse |
|---|---|---|
| (a) boot with SPARSE_KV=1, PIECEWISE 7 + FULL 2 graphs captured | ok | ok (ref and triton) |
| (b) greedy outputs, SPARSE_KSEL=30000 (C >= context), ref impl: poem, code, 16K needle, 24K code+explain (256 tok) | - | **4/4 identical** to dense |
| (c) needle 32K / 64K / 128K (KSEL=2048) | PASS x3 | PASS x3 (ref), PASS x3 (triton) |
| (c) quote-next-line code 32K / 64K / 128K | PASS x3 | PASS x3 (ref), PASS x3 (triton) |
| 2 concurrent requests: needle 64K and code 100K (triton) | needle PASS, code FAIL | needle PASS, code FAIL with the same wrong line (dense skips the same line) |
| (d) decode tok/s at 127K, single request, 512 tok, "explain" | 89.9 | ref 48.1, **triton 118.7 (+32%)** |
| (d) decode tok/s at 127K, single request, 512 tok, "copy" | 148.6 | ref 73.6, **triton 181.5 (+22%)** |

Unit test `sparsekv/test_backend_sparse.py`, on 2 requests (60K + 3K) over random int8 caches:
- With KSEL covering the context, staging attention is bit-equal to dense.
- The short request stays exact.
- KSEL=2048 output matches an fp32 torch attention over the selected set (max abs error 8e-3).
- Graph replay equals eager for ref. Triton is nondeterministic eagerly on random data (ties or atomics), so its check is relaxed.
- Select + gather + attention per layer-step: ref 2.9 ms, triton 0.26 ms.

**Stream C harness** (an internal eval harness, not included; `eval_api.py --lens 32000,64000,128000 --speed`) against this server (sparse, triton, KSEL=2048) on **GPU1 (3090)**. It is compared with stream C's dense baseline `harness_dense_validate.json` (same GPU and harness). Output is in `run/harness.hsp.json`.

| len | needle | essay-complete | code-next |
|---|---|---|---|
| 32K dense / sparse | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 64K sparse (no dense baseline) | 1.00 | 1.00 | 1.00 |
| 128K dense / sparse | 1.00 / 1.00 | 1.00 / 1.00 | 0.902 / 0.783 |

At 128K code-next, `--merge` shows exactly one answer that differs from dense (d=0.94). The sparse answer has the right line but wraps it in a ```python fence, and the edit-similarity score penalises that. The other low item (d=0.11) is the same as dense.

| 128K decode (GPU1) | dense | sparse | per-step (tok/s / mean accepted len) |
|---|---|---|---|
| copy | 148.7 tok/s, acc len 6.17 | **179.1 tok/s**, acc len 6.19 | +20% |
| explain | 94.8 tok/s, acc len 3.91 | **103.1 tok/s**, acc len 3.52 | +21% (the output diverged, so acceptance is lower) |

## Open problems
- **Triton ops (stream A snapshot) fail for large KSEL.** KSEL=30000 leaves a CUDA "invalid argument" that is reported by the next pthfa launch; this is probably the top-k kernel, which is built for k=2048.
  - Exactness (b) is therefore proven with SPARSE_IMPL=ref only. With triton, keep KSEL at or below about 2048, or fix it in stream A.
  - The backend has a static guard: if `block_table.shape[1]*BS <= C`, the decode slice runs dense.
- **The triton select is not bit-deterministic** (eager vs eager differs on random data). Greedy outputs can therefore differ run to run under sparse+triton.
- **Ref select scans the full block-table width** (max_model_len, about 154K tokens) every step, whatever the actual seq_len. It is correct but about 2x slower than dense at 128K. It is for tests only.
- **Sketch memory sits outside vLLM's KV budget.** It needs the lower GPU_UTIL above. A cleaner fix would be to reserve it in the profile (for example allocate at Impl init from max_num_blocks) or to register it as a KV-cache spec.
- **Speed numbers include the change in MTP acceptance** caused by sparse outputs differing from dense. The 128K numbers are one run each, on GPU0, with my prompts. Stream C's harness numbers are above.
- Prefix-cache hits were exercised: the speed requests reuse the 128K code document of the code question, and the server logged a prefix hit rate of 17-29%. They answered correctly.
- **Not tested:**
  - preemption;
  - the 150K max context;
  - GPU1 (3090) speed for my own client.
