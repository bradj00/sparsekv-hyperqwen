# sk-switch: per-step dense/sparse switch with zero per-step cost

Branch `sk-switch`, its own worktree. It is based on sk-tier (a4d8dca) with sk-backend merged in, and uses stream A's triton_ops at a834af8 (da022bc kernels).
The worktree's `venv` and `models` are symlinks to the backend stream's `venv` and `models`.

## Design
- **Host decision per step.** `PthFlashAttnMetadataBuilder.build` calls `sparse_kv.decide(cam.max_seq_len, n_real)`.
  - `max_seq_len` is the host upper bound vLLM already computes for every step.
  - `n_real` is the number of non-padded requests, from `seq_lens_cpu_upper_bound > 0`.
  - The step goes sparse iff `max_seq_len >= cutoff(n_real)`. Mixed batches therefore go sparse.
  - The result is stored in `PthAttnMetadata.sparse_decode` and in a module global for the replay.
  - The MTP drafter's layer is in the same 17-layer KV group, so drafter builds call `decide` too. That is harmless: the drafter build comes after the target replay of the step, and the next target build overwrites the decision.
- **Two FULL graph families.** `patches_switch/sparse-switch-dual-graphs.patch` (vLLM `v1/worker/gpu/cudagraph_utils.py`, about 60 lines) changes only the target model's `ModelCudaGraphManager`:
  - It captures every FULL descriptor twice: `graphs[desc]` with `set_family("dense")` and `graphs_sparse[desc]` with `set_family("sparse")`.
  - `run_fullgraph` replays `graphs_sparse[desc]` iff `sparse_kv.step_sparse()`. That is one Python bool check, with no device work.
  - Speculator, drafter and encoder managers are unchanged. The profile's FULL-graph extrapolation counts both families.
  - Piecewise and eager steps (prefill and mixed) run attention eagerly and read `attn_metadata.sparse_decode`.
- **Sketches are written in both families** (prefill and decode), so switching to sparse at any step finds complete sketches.
- **Cutoff table** (`sparse_kv.cutoffs()`), keyed by real batch size; the largest key at or below n applies:
  - per-GPU defaults from stream A's crossover: RTX 3090 Ti `{1: 44000, 2: 29000}`, RTX 3090 `{1: 20000, 2: 19000}`; other GPUs are always sparse.
  - `SPARSE_CUTOFF="1:44000,2:29000"` or `SPARSE_CUTOFF=30000` overrides; `SPARSE_CUTOFF=0` means always sparse.
  - `SPARSE_SWITCH=0` gives one family, always sparse (previous behaviour).
  - `SPARSE_TIER_SPILL=1` forces always-sparse, because spilled blocks are not readable by the dense kernel.
  - The tier in shadow mode is fine with dense, since all blocks stay resident.
- **Logging:**
  - `sparse_kv: switch -> dense|SPARSE (max_seq_len, reqs, cutoff)` is logged on every flip (`SPARSE_SWITCH_LOG=0` silences it).
  - `sparse-kv switch: FULL graphs captured per family ...` reports graph count and reserved-memory growth per family at boot.

## Apply
```
cd <venv>/lib/python3.12/site-packages/vllm && patch -p1 < patches_switch/sparse-switch-dual-graphs.patch
cp pthfa/python/pth_flash_attn/{sparse_kv.py,vllm_backend.py} <venv>/lib/python3.12/site-packages/pth_flash_attn/
```
Launch: `cd $REPO && SPARSE_KV=1 SPARSE_IMPL=triton run/lockrun.sh <gpu> "run/session.sh <tag> <port> quick,speedshort,speed,grow"`.
- `run/serve.sh` uses the production flags with GPU_UTIL=0.88, binds 127.0.0.1, and stops by PID.
- `run/client.py` suites:
  - `speedshort`: 4K/8K decode tok/s;
  - `grow`: (a) a verbatim copy whose generation crosses the cutoff, start at `GROW_START`; (b) a 3-turn conversation growing from 36K to 57K, with quote-next-line questions about the new and the old part.

## Results
Decode tok/s for one request, 512 tokens, temperature 0 (`run/results.<tag>.json`, logs in `run/logs/`). "always-sparse" means SPARSE_SWITCH=0, or the first run, where the decision was never published.

**GPU0, RTX 3090 Ti** (cutoff 44K at B=1). Tags: dense `dn0` (128K row from sk-backend `dense`), always-sparse `sw0` (128K row from `sw1`), switch `sw2`.

| prompt | dense | always-sparse | **switch** |
|---|---|---|---|
| 4K explain / copy | 157.7 / 229.4 | 152.6 / 225.0 | **157.7 / 230.6** |
| 8K explain / copy | 128.4 / 219.8 | 125.7 / 221.3 | **128.7 / 220.1** |
| 127K explain / copy | 89.9 / 148.6 | 120.1 / 184.5 | **120.3 / 183.2** |

**GPU1, RTX 3090** (cutoff 20K). Tags: dense `g1dn`, always-sparse `g1sw0` (`g1sw` for 128K), switch `g1sw2`.

| prompt | dense | always-sparse | **switch** |
|---|---|---|---|
| 4K explain / copy | 154.4 / 219.9 | 150.4 / 206.8 | **153.3 / 218.2** |
| 8K explain / copy | 122.2 / 209.7 | 118.5 / 202.2 | **121.2 / 208.1** |
| 127K explain / copy | 88.9 / 139.8 | 109.5 / 165.4 | **110.2 / 174.1** |

- **Short prompts:** the switch equals dense within noise (at most 1%). Always-sparse loses 2-6% there.
- **128K:** the switch equals always-sparse, at +34% / +23% over dense on GPU0 and +24% / +25% on GPU1.
- **Both graph families are captured.** The boot log shows `FULL graphs captured per family {'dense': 2, 'sparse': 2}` (decode descriptors B=1 and B=2 at q_len 8).
- **A request that crosses the cutoff stays correct:**
  - `grow copy`: a verbatim copy of a 1.8K-token file, starting just under the cutoff. The log flips `switch -> SPARSE (max_seq_len=44001)` on GPU0 and `20003` on GPU1 in the middle of the generation.
  - Its output is the exact file, similarity 1.000 with a 4246/4246-character exact prefix, on both GPUs. The GPU0 dense run gave 4240/4240 (0.998).
- **A 3-turn conversation from 35.8K to 56.5K tokens** (crossing 44K on GPU0, already above 20K on GPU1) passes all 3 quote-next-line questions, about the new part and the old part. The dense server gives the same answers.
- needle and code at 32K pass on the switch server.
- **Memory of the extra graph family:**
  - KV pool is the same with the switch on or off: 180,573 tokens (GPU0), 183,570 (GPU1).
  - vLLM's CUDA graph pool is 0.10 GiB (actual) in both cases.
  - The driver free-memory delta of the profile capture is 0.28 vs 0.26 GiB on GPU0, so about 20 MiB.
  - The per-family reserved-memory growth during the real capture is about 0 (the FULL graphs reuse the pool the piecewise graphs already grew).
  - Capture time is unchanged at about 4 s.
- **Not tested:**
  - two concurrent requests across the B=2 cutoff (the B=2 table entry and mixed-batch path are exercised only by the warm-up);
  - SPARSE_TIER=1 together with the switch (spill forces always-sparse by design; shadow mode should be fine, but it is unvalidated).
