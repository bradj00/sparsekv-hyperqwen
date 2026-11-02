# Stream E ("tier"): host-RAM KV tier for sparse decode

Branch `sk-tier`, its own worktree. Responsible for plan Stage 3 (host tier) + risks.

## Status (2026-11-01)

- **Working: context larger than the GPU KV pool on a dev server.**
  - A 238K-token prompt ran on an 82K-token pool (3090 Ti): needle PASS, 84.6 tok/s decode.
  - A 177K-token prompt ran on a 139K/140K pool, on the 3090 Ti and on the 3090 (x4): 115 / 117 tok/s. Needle PASS
    on both; code-line retrieval PASS on the 3090 Ti, FAIL on the 3090.
  - No vLLM fork: an external KV connector, stream B's backend with small hooks, and one 6-line vLLM patch that skips
    the startup pool check.
- **Bit-exact tier data path** in the server (shadow + verify): 0 differing bytes in 2.1 GB (3090 Ti) and 1.36 GB
  (3090, where the code suite passed at 32/64/128K).
- **Standalone cost** for 16 layers per step at 128K with a 10% hot buffer:
  - 1.08 ms (x16) / 1.61 ms (x4) on synthetic lab traces;
  - 1.0–1.9 ms (x16) on stream F's real 64K–100K traces;
  - fixed cost when everything is resident: 0.41–0.44 ms.
- **Pressure policy (gap 2) done.** A 98K context that fits does no host reads (121.6 tok/s). 177K spills only
  under pressure: 106.9 tok/s, 2.3 s follow-ups.
- **Stage 4 (branch `sk-tier-h4`):**
  - GDN checkpoints in host RAM: a forced wipe of the GPU prefix cache still gives **2.3 s** follow-ups at 177K,
    all PASS.
  - Two users (177K + 120K) on an 81K GPU pool with forced preemption: 5 of 6 answers PASS. The failure is a
    selection-margin question at 177K on the 3090.
  - Host pool knobs: 14 GB per server for 2 × 200K users.
- **Host prefix cache (gap 1) done.** Follow-up questions on a 177K spilled chat: TTFT **197.6 s → 2.3 s**, and
  needle / code line / second needle all PASS.

## Files

| file | what |
|---|---|
| `sparsekv/tier.py` | `HostKVMirror` (exact-size pinned host mirror, UVA view, block DMA mirroring), `HotBuffer` (device-side token-granular hot buffer, windowed-CLOCK replacement), `gather_rows_tiered` (2 Triton kernels, graph-capturable), `gather_rows_tiered_ref` (torch reference) |
| `sparsekv/test_tier.py` | Triton vs reference vs ground truth over 30 steps (mixed residency, 2 reqs, padded columns, 2 layers) + CUDA-graph replay check |
| `sparsekv/tier_trace.py` | synthetic selection traces (61% step-to-step overlap; presets `lab`, `brief`, `heavy`) |
| `sparsekv/tier_sim.py` | CPU miss-rate simulation: exact LRU vs the device policy (runs the actual reference code) |
| `sparsekv/tier_bench.py` | standalone GPU benchmark: ms/step for 16 layers, eager + CUDA graph, resident baseline, raw UVA bandwidth |
| `sparsekv/tier_admission.py` | SKELETON (not wired) of the admission path: spec, manager, connector, worker runtime |
| `sparsekv/tier_connector.py` | **server integration**: `HostTierConnector` (loaded with `kv_connector_module_path`, no vLLM fork needed) + `TierRuntime`. Modes: shadow (mirror only, verify) and **SPILL** (GPU blocks are really freed) |
| `pthfa/python/pth_flash_attn/{sparse_kv,vllm_backend}.py` | stream B's backend (merged from `sk-backend`) plus tier hooks: tiered gather, sketch lookup through host-indexed rows, prefill staging, drafter window |
| `patches_tier/sparse-tier-spill-kvcheck.patch` | vLLM: skip the max_model_len vs GPU pool startup check when `SPARSE_TIER_SPILL=1` (the only vLLM source change) |
| `single-user/start_qwen.sh` | allows `HostTierConnector` with `PTH_FA=1` |
| `run_tier/{serve,session,two,verify,spill}.sh`, `run_tier/bigctx.py` | dev-server sessions (port 8094/8098); `bigctx.py` = needle / code-line / speed on one long prompt |
| `sparsekv/tier_kbench.py`, `sparsekv/tier_all.sh` | kernel split; one-shot test + bench job |
| `sparsekv/gpurun.sh` | polls `gpu0.lock`/`gpu1.lock` non-blockingly, runs a ≤18 min job on whichever card is free |

Run: `sparsekv/gpurun.sh python sparsekv/test_tier.py`;
`python sparsekv/tier_bench.py gen --ctx 131072 262144 --preset lab brief heavy` (CPU) then
`GPU=1 sparsekv/gpurun.sh python sparsekv/tier_bench.py run --ctx 131072 262144 --preset lab --hot 0.10 --baseline`.

## Design (Track A)

Builds on B's compaction path: B's `select_and_stage` calls `gather_rows(kv, block_table, idx, n_valid, st_kv, bt)`.
`gather_rows_tiered` has the same inputs plus the tier state and is a drop-in replacement there. Each picked token
resolves to one of three sources:

1. **resident**: its block is still in the GPU pool (`gpu_bt[r, pos//BS] >= 0` and `!= null_block`), copied from `kv`;
2. **hot**: `pos2row[slot, pos] >= 0`, copied from the request's GPU hot buffer;
3. **host**: read from the pinned host mirror through a UVA device view (zero-copy, no CPU), written to staging
   **and** into a victim hot row.

- **Host mirror** `[16, HB, Hkv, BS, 520]` int8: the same block layout as the GPU cache, so mirroring a filled block
  is one contiguous DMA (1.8 MB per layer-block) on a side stream.
  - Allocated with `cudaHostRegister` on a plain allocation. torch `pin_memory=True` would round a 17 GB pool up
    to 32 GB.
  - Host gathers read 520 B per (token, head) row. The lab showed 512 B rows saturate the link.
- **Hot buffer** `[16, slots, H, Hkv, 520]`: token-granular (one row = 2,080 B, all 4 heads). Per row it keeps
  `tag` (position) and `last` (step). Per slot it keeps `pos2row` (`int32 [max_ctx]`, 1 MB per 256K) and a clock
  `hand`.
- **Replacement: windowed CLOCK, fully on device.**
  1. Hits refresh `last` first.
  2. The n_miss victims come from the 2C rows after the hand: first rows idle for more than `age` steps, then any
     row not used in this step.
  3. Rows used in this step are never evicted. 2C always holds enough candidates because hits + misses ≤ C. The
     hand moves past the last row it consumed.
  4. No sort and no top-k over H. Exact LRU would need a top-k over 13–52K rows per layer-step.
  - The simulation below shows the policy within 0–1.2 points of exact LRU.
- **Kernels** (per layer):
  - `_tier_resolve_kernel`, one program per request: classify picks, refresh hits, prefix-sum miss ranks, choose
    victims in the clock window.
  - `_tier_copy_kernel`, grid `(nreq, C/4)`: copy rows into staging, fill victim rows, update `tag`/`pos2row`/`last`
    (CAS on the old `pos2row`).
  - Shapes depend only on `(nreq, C)`. The step counter is a device tensor. No host syncs. Capturable: the test
    replays a captured multi-layer step and matches the reference bit-for-bit.
- **Correctness:** `test_tier.py` compares Triton to the reference to ground truth for staging rows, hot-row
  contents and all bookkeeping, over 30 steps with mixed residency, two requests with different slots, padded
  columns, plus 8 graph replays.

## Results

All numbers are for **16 sparse layers per decode step, one request, C = 2048 top-k + 64 sink + 1024 window**,
measured in a CUDA graph. Raw data: `results/<gpu>/bench.jsonl`.

**Worst-case residency.** Every block except block 0 and the tail is host-only, so every top-k pick is resident,
hot, or a host miss. "resident" is the same two kernels with everything on the GPU: the cost of the compaction gather
plus tier bookkeeping, with no transfers.

**Miss rate: simulation (`tier_sim.py`, CPU, the actual reference code).** Windowed CLOCK against exact LRU, as a
fraction of top-k picks:

| trace | ctx | hot 5% | hot 10% | hot 20% |
|---|---|---|---|---|
| lab (calibrated: LRU@10% ≈ 10–11% at 128K) | 128K | 27.1 / 27.1 | 11.4 / 11.4 | 2.9 / 3.0 |
| lab | 256K | 13.6 / 13.6 | 3.6 / 3.9 | 3.5 / 3.5 |
| brief (all picks inside a fixed 10.5% working set) | 128K | 24.2 / 24.2 | 2.9 / 3.6 | 0.2 / 0.2 |
| heavy (stress) | 128K | 32.0 / 32.0 | 21.9 / 21.9 | 9.1 / 9.2 |

(LRU / CLOCK with age=8 in each cell.) CLOCK is within 0–1.2 points of exact LRU everywhere.

- Caveat on the synthetic traces: at 256K the same trace parameters give a smaller working set relative to the
  context. 2048 picks is only 0.8% of 256K. So the 256K miss rates are optimistic unless real traces confirm them.

**Speed: tier on vs off, graph ms per decode step (16 layers).**

| trace, ctx, hot buffer | top-k miss % | MB/step from host | 3090 Ti x16: tier (resident) | 3090 x4: tier (resident) |
|---|---|---|---|---|
| lab 128K, 5% | 26.1 | 17.6 | 2.54 (0.41) | 3.63 (0.44) |
| **lab 128K, 10%** | **10.0** | **6.7** | **1.08 (0.41)** | **1.61 (0.44)** |
| lab 128K, 20% | 3.0 | 2.0 | 0.61 (0.42) | 0.81 (0.44) |
| heavy 128K, 10% | 20.0 | 13.4 | 1.90 | 2.81 |
| brief 128K, 10% | 3.6 | 2.5 | 0.60 | 0.89 |
| lab 256K, 5% | 12.3 | 8.3 | 1.26 | 1.90 |
| **lab 256K, 10%** | **3.7** | **2.5** | **0.65 (0.42)** | **0.90 (0.44)** |
| heavy 256K, 10% | 9.5 | 6.4 | 1.03 | 1.60 |
| lab 128K, 10%, **2 requests** | 10.8 | 14.6 | 2.08 (0.67) | 2.99 (0.71) |
| **real 27B picks** (stream F, t0_32k q0), 32K, H = 19% | 5.8 | 3.4 | 0.73 | 0.98 |
| real (t0_32k q2), 32K, H = 19% | 9.8 | 5.9 | 1.02 | 1.47 |
| **real (t2_64k q0), 64K, 10%** | **16.3** | **10.0** | **1.53** | **2.16** |
| real (t2_64k q0), 64K, 20% | 6.8 | 4.2 | 0.81 | 1.13 |

**All of stream F's real 27B traces** (sketch arm: actual sparse decoding; `results/NVIDIA_GeForce_RTX_3090_Ti/real.jsonl`,
3090 Ti graph ms per step for 16 layers). Range across the traces in each length bucket:

| ctx (traces) | hot 10%: miss %, ms | hot 15% | hot 20% |
|---|---|---|---|
| 32K (5 runs; H floor 2C = 19% of ctx) | 6.3–18.4 %, 0.75–1.34 ms | same | 5.9–18.1 %, 0.73–1.32 ms |
| 64K (4 runs) | 10.4–21.9 %, 1.01–1.94 ms | 8.3–11.9 %, 0.87–1.13 ms | 7.2–9.1 %, 0.81–0.94 ms |
| 100K (4 runs) | 9.4–16.1 %, 0.95–1.44 ms | 6.9–13.8 %, 0.83–1.28 ms | 5.4–13.2 %, 0.74–1.23 ms |

- On real traces the miss rate keeps falling up to a 15–20% hot buffer, then flattens. The floor is new tokens that
  were never touched before.
- Sketch-arm picks miss 0.5–3 points more than the full-attention shadow picks.
- On x4, per-step cost is roughly the x16 figure + 0.07 ms/MB × MB/step (see the synthetic table).

- **Marginal cost of host traffic:** about 0.10 ms/MB on x16 (≈10 GB/s effective) and about 0.17 ms/MB on x4
  (≈5.8 GB/s). This matches the lab's UVA numbers (10 / 6.2 GB/s). The copy kernel reaches link speed with 520 B
  rows.
  - Raw bandwidth on this box: block DMA is 22–25 GB/s on the 3090 Ti and 6.5 GB/s on the 3090, in both directions.
    `torch.index_select` over UVA reaches only 7.2 / 5.2 GB/s.
- **Fixed cost:** about 0.41–0.44 ms/step for 16 layers when everything is resident, which is roughly the compaction
  gather itself. Kernel split on x16 (eager, `tier_kbench.py`): resolve ≈ 0.23–0.29 ms, copy ≈ 0.32 ms resident /
  0.55 ms with 6% misses.
  - The resolve kernel runs one program per request (serial prefix sums). Fusing it into stream A's select kernel
    would remove most of that.
- **Real traces miss more than the synthetic lab preset.** At 64K with 10% hot, 16.3% of picks miss, against the
  lab's 9.6%. At 20% it is 6.8%. Size the hot buffer at **15–20% for x4**, or accept about 2 ms/step at 64K.
- **First decode step after a new question is cold:** ~99% misses (stream F's sim). That is a one-time burst of up to
  16 × 2048 × 2,080 B = 68 MB: about 7 ms (x16) or 11 ms (x4).
- **Against a decode step:** sparse decode at 128K on the 3090 Ti runs 118.7–181.5 tok/s with MTP-7 (stream B), so
  about 25–40 ms per verify step. At 10% hot the tier adds about 0.7 ms (x16) or 1.2 ms (x4) per step on synthetic
  lab traces, and about 1.5 / 2.2 ms on the real 64K trace: roughly **2–8%**.

## In the vLLM server (Track A done: context larger than the GPU pool)

The tier runs inside the dev server as an **external KV connector**: `--kv-transfer-config
{"kv_connector":"HostTierConnector","kv_connector_module_path":"sparsekv.tier_connector","kv_role":"kv_both"}`.

- The connector subclasses `SupportsHMA`. Without that, vLLM turns off the hybrid manager and the GDN model fails to
  start.
- It sets `has_sync_kv_loads`, so `start_load_kv` runs before each forward, and it carries the `SchedulerOutput` in
  its metadata. Model Runner V2's `pre_forward` does not pass it on.

Everything else is stream B's compaction path plus small hooks.

**Two bugs found on the way, both fixed:**

- vLLM's per-layer cache view `[NB, 4, 864, 520]` is not contiguous. Its strides are `(1797120, 520, 2080, 1)`: the
  memory is token-major, 2,080 B per token. The first copy kernel assumed HND contiguity and corrupted outputs from
  the first decode token. It now takes strides, and `test_tier.py` uses a strided cache.
- The scheduler output was missing in `start_load_kv`, so the first "shadow" runs never mirrored anything. Their
  speed numbers measured only the all-resident path, and I have discarded them.

**Shadow mode with verification** (`SPARSE_TIER_VERIFY=1`, 3090 Ti, exact suite). Every mirrored block outside
sink + tail is read from the hot buffer or host, and the plain GPU gather runs again for comparison.

- Over 150 steps: 330K hot hits, 136K host misses, 551K resident rows, **0 differing bytes in 2.1 GB of staged rows**.
- Greedy outputs match the tier-off session on 3 of 4 prompts. `code24k` diverges after ~90 tokens, although the
  staged bytes are identical. Stream B's sparse path also diverges from dense on these prompts (`short_code` at
  char 280), so I attribute it to run-to-run numerical nondeterminism. I have **not proven** this: I did not run the
  baseline twice.

**SPILL mode: real capacity.** The scheduler frees GPU blocks.

| card | GPU KV pool (sparse group) | prompt | needle | code line | TTFT | decode (384 tok, `speed`) | GPU KV usage while running |
|---|---|---|---|---|---|---|---|
| 3090 Ti x16, `GPU_UTIL=0.78`, `max_model_len=200000` | 161 blocks = **139K tokens** | **177K** | PASS | PASS | 199 s | **115.3 tok/s** | 17–24% |
| 3090 Ti x16, `GPU_UTIL=0.70`, `max_model_len=250000` | 95 blocks = **82K tokens** | **238K** (2.9× the pool) | PASS | (not asked) | 321 s | **84.6 tok/s** | ≤ 40% |
| 3090 x4, `GPU_UTIL=0.78`, `max_model_len=200000` | 162 blocks = **140K tokens** | **177K** | PASS | **FAIL** (quoted a nearby line) | 196 s | **117.0 tok/s** | — |

- The x4 run's code-line FAIL is unexplained. Possible causes:
  - selection quality at 177K: stream B validated sparse retrieval only up to 128K;
  - numerical differences between the two cards.

  The tier's data path is bit-exact on both cards. On the 3090 (x4), shadow + verify over the code suite gave
  **32K / 64K / 128K all PASS**, with 194K hot hits and 200K host misses, and **0 differing bytes in 1.36 GB**
  (`run_tier/sess.verify4.out`). So the tier is not the cause. Selection or numerics above 128K are the open
  suspects.
- Hot-buffer misses on x4 at 177K: 2.83M hits against 0.46M misses, 14.1%. Decode speed matches x16, so the x4 link
  does not cost decode throughput at this miss rate.
- The needle sits at 30% depth and the code line at 60%. Both are far outside sink + tail, so they live only in host
  RAM. They are found through the host-indexed sketch region and fetched through the hot buffer or UVA.
- Reference point: stream B's sparse decode at 128K on the same card, no tier, is 118.7 tok/s on the same "explain"
  type of task.
- Hot buffer (10% of max_model_len) on real decoding:
  - at 177K: 3.09M hits against 0.49M host misses, a **13.6% miss rate**;
  - at 238K: 4.20M hits against 0.41M host misses, **8.9%**.
- Every question re-prefills its whole context (TTFT 199 s / 321 s). Spilled blocks are not prefix-cacheable yet:
  that is gap (a) below.
- `max_model_len` is capped by RoPE at 262,144 (`max_position_embeddings`). The host pool sets the real limit:
  8.4 GB pinned per 250K tokens.

## Host prefix cache: multi-turn on a spilled context (gap 1, done)

**How it works.** Commit 8ba7049, `sparsekv/tier_connector.py`. No vLLM change beyond the startup-check patch.

- **Scheduler-owned host pool** (`HostCache`): host blocks keyed by the vLLM block hash, refcounted per request, with
  unreferenced cached blocks evicted LRU. The host block for each logical block of a running request is chosen in
  `build_connector_meta`.
  - A block whose hash is already on host is shared, with no copy.
  - Otherwise a block is allocated and a copy is scheduled.
  - The worker receives `(logical block, host block)` maps and copy lists in the connector metadata. The worker no
    longer owns an allocator.
- **Prefix hits on host blocks.** `bind_kv_cache_manager` swaps the hybrid coordinator's manager class for the
  sparse group with `HostTierFullAttentionManager`. Its `find_longest_cache_hit` walks the block hashes:
  - GPU-cached block → that block;
  - host-cached block → the null block, the same mechanism as sliding-window hits;
  - anything else → stop.

  The coordinator then trims the hit to the GDN/mamba "align" checkpoint as usual. The mamba state at the boundary
  still came from the GPU prefix cache in these runs: the request-1 checkpoint survives because recently freed
  blocks are evicted last.
- **Worker.** A block is read from host when it is host-mapped and either null in the GPU block table (a host hit)
  or freed by the scheduler.
  - Sparse decode reads it through the hot buffer / UVA.
  - Its sketch comes from the host-indexed rows, which persist with the host block, so hits need no sketch
    recompute.
  - The new turn's dense prefill stages it from host.
  - The MTP drafter drops null and freed blocks through a general, graph-safe compaction (argsort keep-mask).

**Result** (3090 Ti, `max_model_len` 200K, GPU pool 139K tokens, one 177,4xx-token document, four questions in a
row, `run_tier/results.hp1.json`):

| question | prompt tokens | TTFT | decode tok/s | answer |
|---|---|---|---|---|
| needle (cold, full prefill) | 177,457 | **197.6 s** | 96.9 (10 tokens) | PASS `7418-2295` |
| code line at 60% depth | 177,469 | **2.3 s** | 137.7 (short) | PASS |
| second needle at 75% depth | 177,456 | **2.3 s** | 63.3 (short) | PASS `amber-quartz-41` |
| explain the last file (384 tokens) | 177,445 | **2.4 s** | **106.6** | coherent |

- Host cache: 205 blocks allocated, 615 host hits, 0 evictions. Each follow-up mapped 204 host blocks; only the last
  block plus the new turn was recomputed.
- Hot buffer over the run: 3.41M hits against 0.56M misses (14%).
- Before this change, every follow-up re-prefilled the whole context at about 199 s. Now the cost is staging
  205 blocks × 16 layers from host (~6 GB at 22–25 GB/s) plus the ~900-token prefill.

## Free blocks only under memory pressure (gap 2, done)

**Mechanism.** `SPARSE_TIER_SPILL_POLICY=pressure` is now the default; `eager` is the old behaviour.

- The scheduler hook frees `[1, (processed − tail) // 864)` only while free GPU blocks are below a watermark.
  The default watermark is `num_gpu_blocks // 10` (16 here); `SPARSE_TIER_WATERMARK_BLOCKS` overrides it.
- It reports `freed_upto` per request in the connector metadata. The worker reads a block from host only if it is
  null in the GPU table or below `freed_upto`, not merely because it is outside the window. A context that fits
  therefore never touches host RAM on reads.
- Blocks are still mirrored to host as they fill: about 29 MB of DMA per 864 tokens, and it keeps the host prefix
  cache warm.

**Result.** One server, 3090 Ti, GPU pool 139K tokens (`run_tier/results.press1.{fit,big}.json`):

| phase | prompt | Q1 TTFT | follow-up TTFT | answers | decode (384 tok) | host reads |
|---|---|---|---|---|---|---|
| fits in the pool | 97.7K | 83.5 s | 1.3 / 1.2 s | needle PASS, code PASS | **121.6 tok/s** | **none**: 5.6M picks, all resident; 0 blocks staged |
| larger than the pool | 177.5K | 202.2 s | **2.3 / 2.3 / 2.3 s** | needle, code, needle2 PASS | **106.9 tok/s** | 1.39M hot hits, 0.24M misses (vs 0.56M with eager) |

- The host cache evicted 51 blocks (LRU): the two documents together exceed the 233-block host pool.
- The fit phase decodes at stream B's no-tier speed. B measured 118.7 tok/s at 128K on the same card.

## Stage 4 hardening (branch `sk-tier-h4`, its own worktree)

### 1. GDN/mamba checkpoints in host RAM

- **What the GPU path does.** A follow-up turn's prefix hit needs the GDN state at an aligned boundary, from vLLM's
  align-mode checkpoint blocks. Under GPU churn those checkpoints are evicted, the hit length drops to 0, and the
  turn falls back to a full prefill. Measured: in the first two-user run (`results.conc1.*.noMamba.json`) the
  follow-ups took 286–290 s.
- **Save.** vLLM hands connectors the exact align-mode boundary blocks in
  `scheduler_output.kv_connector_block_state.boundary_state_offloads`.
  - The connector keeps the last `SPARSE_TIER_HOST_MAMBA_PER_REQ` (2) boundaries at or below the prompt end, keyed by
    the boundary block's hash.
  - The worker copies the raw state pages of all 3 mamba groups (48 GDN layers × 1.78 MB = **85.3 MB per
    checkpoint**) into pinned slots. There are `SPARSE_TIER_HOST_MAMBA_SLOTS` (8) slots, evicted LRU.
  - The copy runs `SPARSE_TIER_MAMBA_SAVE_DELAY` (2) steps after the offer, at the start of a step, before its
    forward. This is conservative: a freed block can't be rewritten before that forward.
  - Verified: with `SPARSE_TIER_DEBUG_MAMBA_CMP=1`, the host copy equals the GPU checkpoint that a normal GPU hit
    uses, page for page (48/48 identical, `sess.mh9.out`).
- **Restore.** `HostTierMambaManager.find_longest_cache_hit` runs when no GPU checkpoint reaches the full-attention
  hit length. It takes one free GPU block per mamba group, registers it under the boundary hash, fills it from host,
  frees it as a hashed cached block (evicted last), and returns the ordinary hit.
  - **The fill must be enqueued at scheduling time, not in the connector hook.** In the V2 runner, vLLM's mamba
    preprocessing copies the hit checkpoint into the running state during `execute_model` preparation, which comes
    *before* `kv_connector.pre_forward`. A load in `start_load_kv` is therefore too late, and the answers were garbage
    or an immediate EOS (`sess.mh3`–`mh7`).
  - The scheduler and worker share one process here (single GPU, UniProc), so the restore calls
    `TierRuntime.load_mamba_now` directly. The copy then sits on the stream ahead of the next step.
  - With multi-process workers (TP > 1) this still needs a proper path, for example a pre-preprocess hook in the
    runner. **Not done.**
- **Debugging notes:**
  - Saving at request end from the request's block table finds nothing, because align mode has already retired
    the checkpoints.
  - `SPARSE_TIER_DEBUG_FA_HOST_ONLY=1` makes full-attention hits ignore GPU-cached blocks. It showed the all-host
    full-attention path is correct (`sess.mh8b.out`), which isolated the bug to the GDN restore.

**Result:** forced eviction of the GPU prefix cache before every follow-up. That is vLLM's dev endpoint
`POST /reset_prefix_cache` (`VLLM_SERVER_DEV_MODE=1`, test only); it wipes all GPU block hashes, including every
GDN checkpoint and every attention block. 3090 Ti, 177K-token document, `run_tier/results.mhbig.json`:

| question | GPU prefix cache | TTFT | answer |
|---|---|---|---|
| needle (cold) | — | 203.7 s | PASS |
| code line at 60% | wiped | **2.3 s** | PASS |
| second needle at 75% | wiped | **2.3 s** | PASS |
| explain, 384 tokens | wiped | **2.3 s** | coherent, **126.2 tok/s** |

- 3 GDN restores from host, 612 host attention blocks mapped, 0 host evictions.
- Without the host checkpoint, a wiped GPU cache means a full re-prefill of about 200 s.
- Same test at 60K on the 3090 (`sess.mh10.out`): 1.5 s follow-ups, all PASS.


### 2. Two concurrent long requests

Setup: 3090 (x4), `GPU_UTIL=0.70`, so the sparse group's GPU pool is 94 blocks (81K tokens); host pool 13.4 GB.
Request A is 177K tokens; request B is 120K tokens on a different document and starts 60 s later. Each asks
needle, code line, second needle (`run_tier/conc.sh`, `results.conc{2,3,4,5}.{A,B}.json`).

| run | spill trigger | preemptions | A: needle / code / needle2 | B: needle / code / needle2 | follow-up TTFT (A ; B) |
|---|---|---|---|---|---|
| conc1 (before host checkpoints) | pressure | 0 | PASS / PASS / — | PASS / PASS / — | 290 s ; 286 s (full re-prefill) |
| conc2 | pressure (watermark 9) | 0 | PASS / PASS / PASS | PASS / PASS / PASS | 104 s*, 4.7 s ; 4.7 s, 2.2 s |
| conc3 | watermark 1 (forced) | ≥1 (A re-admitted with host hit) | PASS / PASS / PASS | PASS / PASS / PASS | 98 s*, 4.8 s ; 4.9 s, 2.3 s |
| conc4 (= conc3 + preemption log) | watermark 1 | **1, logged**: A re-admitted with 53,568 tokens from cache | PASS / **FAIL** / PASS | PASS / PASS / PASS | 98 s*, 4.7 s ; 4.6 s, 2.2 s |
| conc5 (final code) | watermark 1 | 1, same | PASS / **FAIL** / PASS | PASS / PASS / PASS | 98 s*, 4.7 s ; 4.8 s, 2.2 s |

\* A's first follow-up waits behind B's 120K prefill. The time is scheduler queueing (2,048-token chunks are
shared), not tier work.

- Preemption and resume work. A preempted request comes back as a new request (Model Runner V2), hits its own host
  blocks and host GDN checkpoint, and resumes instead of re-prefilling.
- **A's code-line question at 177K on the 3090 failed in 2 of 4 runs**: conc4 and conc5, but not conc3, with no
  behaviour change in between. It also failed in the earlier single-user x4 run. On the 3090 Ti it passes every
  time.
  - The tier's data path is bit-exact on both cards (verify mode).
  - So this looks like a marginal sparse-selection case at 177K on this card. Stream B validated retrieval up to
    128K. It needs a look by B/C: run selection quality at 150–180K.
- Decode speed of a request while the other user prefills is low (2.8–48 tok/s). That is batching with a long
  prefill, not the tier.


### 3. Host pool sizing for 2 users (knobs and RAM budget)

| item | cost | knob (default) |
|---|---|---|
| KV mirror of the 16 sparse layers | **33.3 KB per token** (16 × 2,080 B): 6.7 GB per 200K tokens | `SPARSE_TIER_HOST_GB`, or `SPARSE_TIER_HOST_TOKENS`; default `max_model_len × max_num_seqs`, capped by `SPARSE_TIER_HOST_GB_CAP` (16 GB) |
| GDN/mamba checkpoints | **85.3 MB each** | `SPARSE_TIER_HOST_MAMBA_SLOTS` (8 → 0.68 GB), `SPARSE_TIER_HOST_MAMBA_PER_REQ` (2) |
| GPU side of the host pool | host-indexed sketch rows: 4.2 KB of GPU memory per host-pool token (400K → 1.7 GB) | grows with the host pool, so a larger host pool shrinks the GPU KV pool |
| safety check | the worker refuses to start if the pinned pool would leave less than 8 GB of `MemAvailable` | `HOST_HEADROOM_GB` = 8 |

**RAM budget on this VM** (78 GB total, two servers):

- One server process uses about 7.5 GB RSS, measured: EngineCore 5–6 GB plus API server about 1.8 GB.
- OS and other services take about 3 GB, and I keep at least 8 GB headroom for page cache and model loading.
- That leaves about 52 GB for the two servers' host tiers, about 26 GB each at most.

**Defaults chosen:**

- **2 users at `max_model_len` 200K: 400K tokens = 13.3 GB KV plus 0.7 GB checkpoints, about 14 GB per server.**
  Two such servers come to about 28 GB of pinned memory plus 15 GB of processes, or 43 of 78 GB.
- The default cap of 16 GB per server holds about 480K tokens (2 × 240K).
- When both servers run, do not raise `SPARSE_TIER_HOST_GB` above about 24 GB each.
- When the host pool fills, unreferenced host-cached blocks are evicted LRU. Running requests' blocks are never
  evicted; a full pool of referenced blocks raises a clear error.

### 4. Scheduler hook as a class (partial)

- The spill rule is now `HostTierSpillPolicy`: one object holding tail, sink, policy and watermark, plus a
  reference to the connector's freed bookkeeping. It is installed on the live sparse-group manager.
- Host prefix hits are the `HostTier*Manager` classes put into the hybrid coordinator's lookup table.
- vLLM 0.31's manager instances cannot be re-classed (`__class__ assignment: object layout differs`). A fully
  static install needs a `HostTierFullAttentionSpec` emitted by the attention layer and registered with
  `KVCacheSpecRegistry`. The MTP drafter layer shares that KV group, so this touches stream B's backend and the
  group planning. I left it as the documented next step (`tier_admission.py` skeleton).

## Admission: what is implemented, and the full plan

**Implemented now (`SPARSE_TIER=1 SPARSE_TIER_SPILL=1`; minimal positional tier, no vLLM fork):**

1. **Startup check.** `patches_tier/sparse-tier-spill-kvcheck.patch` skips `_check_enough_kv_cache_memory`, so
   `max_model_len` can exceed the pool.
   - Run with `--no-scheduler-reserve-full-isl`, so admission only needs the next chunk's blocks.
   - The host pool is sized by `SPARSE_TIER_HOST_TOKENS` (default `max_model_len × max_num_seqs`).
2. **Scheduler.** `HostTierConnector.bind_kv_cache_manager` wraps `remove_skipped_blocks` of the KV group that
   holds the 16 sparse layers.
   - It frees blocks in `[1, (processed − TAIL) // 864)` through `_remove_blocks_in_range`, which nulls the entry and
     frees the block. That is the sliding-window mechanism.
   - Null entries count toward `len(req_to_blocks)`, so a long request asks only for new tail blocks.
3. **Worker.** In `start_load_kv`, before the forward, every block that ends at least 64 tokens below
   `num_computed_tokens` gets a host block from a worker-side free list. Its KV (16 layers) and its sketch rows are
   copied on the compute stream.
   - The scheduler frees a block only once `processed − 1088` has passed it, and `processed ≤ num_computed`. The
     copy is therefore enqueued no later than the forward of the first step that can reuse the block. This needs no
     acknowledgement round trip.
   - The worker derives the spilled set from the same rule (mirrored and outside sink + tail). That is a superset of
     what the scheduler freed, so stale block ids are never read, and no `block_table_updates` are needed.
4. **Readers of spilled blocks.**
   - Sparse decode: hot buffer or host (UVA).
   - Sketch: rows `NB*BS + host_slot`, a host-indexed region appended to stream B's sketch cache. Without it, a
     reused GPU block would overwrite the sketches of the spilled tokens.
   - Dense prefill of a sparse layer: every block staged into a scratch cache.
   - The **MTP drafter layer sits in the same KV group**: it attends the sink block plus the non-spilled tail, with a
     compacted block table (graph-safe). Drafts at long range may get worse; target outputs stay exact.

**Known limits of the minimal version** (in priority order):

- (a) ~~No prefix caching for spilled blocks~~: **done**, see "Host prefix cache" above (follow-up TTFT 2.3 s at
  177K).
- (b) Prefill staging copies all of a request's blocks per layer per chunk, and it uses a host sync (`.cpu()` of the
  block table) on the prefill path.
- (c) ~~Eager spill policy~~: **done**, the pressure policy is now the default (see above).
- (d) Tested with one sequence (`MAX_SEQS=1`). Preemption and resume are handled in code but untested.
- (e) The scheduler hook is a monkeypatch. It belongs in a `HostTierFullAttentionSpec`/manager (skeleton in
  `tier_admission.py`).
- (f) Hot buffer sizing is per slot × `max_model_len`, about 0.85 GB per slot at 250K.

**Full plan (reference, from the code study):**

File references are to the installed tree `venv/.../vllm` (0.31.0 + patches).

**What 0.31 already provides.** HiSparse is the only code path where a running request has KV that is not on the
GPU during decode.

- It has a private host block pool:
  - `HiSparseSourceManager.bind_host_pool` (`single_type_kv_cache_manager.py:2367`);
  - an admission sentinel when the host pool is short (`:2412`);
  - GPU blocks "unpinned" so admission can exceed the pool (`block_pool.py:711-731`, reuse callback → `_lose_page`,
    `coordinator.py:376-399`);
  - an acknowledge-then-free spill protocol (`coordinator.py:705-766`);
  - `scheduler_output.block_table_updates`, which Model Runner V2 already applies as overwrite rows
    (`v1/worker/gpu/model_runner.py:1216`);
  - startup checks that skip host groups (`kv_cache_utils.py:2752, 3056`).
- It is gated to MLA/DSA models (`config/vllm.py:1838` `index_topk`, `layout.py:58-137`, `binding.py:47`).
- It needs BLNHC token-major rows with `row_bytes % 16 == 0`. Our HND 520 B head rows are neither token-major nor
  16-aligned.
- Its resolver's shared-memory need at MTP k=7 does not fit sm_86 unless selection is shared (stream D added
  `shared_selection`).
- `kv_offload/*`, the OffloadingConnector and the SimpleCPU connector are prefix-cache offload only. The worker
  asserts that no group is host-resident (`offloading/worker.py:88`).
- None of HyperQwen's `offload-*`, `pinned-kv-empty-cache`, `kv-cache-seat-budget` or `hybrid-kv-groups-v2-cudagraph`
  patches helps with residency. `kv-cache-seat-budget` would report a capped spec correctly.
  `hybrid-kv-groups-v2-cudagraph` touches `kv_cache_utils.py` (merge friction only).
- `SlidingWindowManager` and `RSWAManager` show how to null and free old blocks (`_remove_blocks_in_range`, `:669`).
  - They never tell the worker: the stale IDs stay in the worker's table.
  - pthfa has no window mask at D=256, so dense pthfa would read whatever block now sits there. Every non-resident
    access must therefore go through our selection plus a republished table.

**Smallest viable path: a positional host tier** (skeleton in `sparsekv/tier_admission.py`). It reuses HiSparse's
pieces, not its MLA-bound groups.

1. **Spec.** `HostTierFullAttentionSpec(FullAttentionSpec)` for the 16 sparse layers. The MTP drafter layer keeps a
   plain `FullAttentionSpec` in its own group.
   - `max_admission_blocks_per_request = sink blocks + cdiv(win + verify + max_in_flight, BS) + 1`, with
     `max_memory_usage_bytes` derived from it, as `SlidingWindowSpec` does.
   - Add it to the cap tuple in `get_manager_for_kv_cache_spec` (`:2745`).
   - Together these fix both the startup check and the per-request admission gate.
   - Add a host-capacity check in their place: host blocks ≥ `cdiv(max_model_len, BS) + 2`, as `layout.py:246` does.
2. **Manager.** `HostTierFullAttentionManager(FullAttentionManager)`:
   - Every GPU block gets a host block from a private host `BlockPool` (`bind_host_pool` pattern). That is a
     write-through mirror target, so a spill moves no data.
   - Host pool short → return the sentinel `num_gpu_blocks + 1`.
   - `remove_skipped_blocks` (already called from `allocate_slots` before counting) spills mirrored-and-acked blocks
     in `[sink, processed - tail)`: null entry, free GPU block, add to `block_table_updates`.
   - Policy `pressure` (default) spills only while free blocks are below a watermark. Then a context that fits pays
     nothing; this is risk 1. Policy `eager` spills everything outside sink and tail.
   - Running requests count null entries in `len(req_to_blocks)`, so they ask only for new tail blocks.
     `FullAttentionManager`'s fast path already does this (`:247`).
3. **Connector** (`HiSparseConnector` pattern).
   - Scheduler: `build_connector_meta` carries the (gpu, host) mirror pairs of newly full blocks, the host
     block-table rows, and new or finished requests. It sets `block_table_updates`.
   - Worker: `register_kv_caches` creates `HostKVMirror`. `start_load_kv` mirrors full blocks on a side stream; the
     next forward waits on its event. It acknowledges blocks via a `get_finished`-style report, and the manager
     spills only acknowledged blocks. That is HiSparse's protocol, and it makes async scheduling safe.
4. **Worker runtime** (`TierRuntime`), hooked into stream B's `pth_flash_attn/sparse_kv.py`:
   - `host_bt` and `HotBuffer` are indexed by `req_state_idx`, the same index as the V2 block tables. Batch rows are
     gathered with `idx_mapping` into static buffers.
   - Decode: `gather_rows_tiered(..., null_block=0)` replaces `gather_rows`.
   - **The sketch must be indexed by host slot** (`host_bt*BS + off`), not GPU slot. B's sketch cache is indexed by
     GPU slot, so a spilled block's sketch rows are overwritten as soon as its GPU block is reused. Indexed by host
     slot, sketches survive spills and follow host prefix caching. Cost: ~4.2 KB/token on GPU, sized by the host
     pool (16 GB host ≈ 480K tokens ≈ 2 GB of sketch).
   - Prefill over spilled blocks: stage the host blocks into a per-layer GPU scratch pool (double-buffered,
     prefetching layer l+1) and run dense pthfa on a patched block table. Prefill attention runs outside the FULL
     decode graphs.
     - Cost per 2048-token chunk ≈ spilled context × 2,080 B × 16 layers. For 150K that is 5 GB: about 0.2 s at the
       22–25 GB/s block-DMA rate measured on the 3090 Ti (x16), or about 0.8 s at 6.5 GB/s on the 3090 (x4).
     - That is fine for a new agent turn (1–3 chunks). It is expensive for a first prefill that is larger than the
       pool: about +60 s (x16) to +200 s (x4) at 400K. Mitigate with bigger chunks for long prompts.
5. **Prefix caching.** Host blocks keep their hash (the host pool has caching on). `find_longest_cache_hit` extends a
   GPU hit with host-only blocks: null GPU entry plus host block id, staged for the prefill. Without this every
   agent turn re-prefills its context, so it belongs in v1, not later.

GPU memory per context token after this: 4.2 KB sketch + 2.1 KB MTP-drafter KV (dense, one layer) + hot buffer
(10% × 33 KB = 3.3 KB) ≈ **9.6 KB, against ~35 KB today (≈3.6×)**. GDN state per request is unchanged. The drafter
layer's KV becomes the next-largest item.

**Track B (HiSparse port, stream D's `sk-port`)** is the alternative. Keep HiSparse's scheduler side (coordinator,
unpin/reuse, acknowledgements, prefix tail restore). On the worker:

- (a) feed `sparsekv.select` output, one row per request (`shared_selection`), to `swap_in` → physical hot rows,
  then gather into B's HND staging (or call `gather_selected_cache`, which stages without mutating LRU);
- or (b) keep `gather_rows_tiered` and use only HiSparse's host pool and tables.

Blockers to check first:

- BLNHC: HiSparse forces token-major GPU KV, while pthfa prefill and drafter read HND (risk 7). Compaction itself is
  indifferent, because it gathers into an HND staging cache.
- `row_bytes % 16`: 2,080 is fine; 520 is not.
- The int16 LRU (H ≤ 32K rows, i.e. 12.5% of 256K).
- Upstream reports 17–20% overhead at low concurrency.

Recommendation: the positional tier (its own scheduler pieces plus `tier.py`) is less code and keeps HND. Use
Track B only if D's port boots with pthfa under BLNHC.

## Track B (HiSparse, stream D's `sk-port` e918b70): status and plan

- **Done:** `tier.gather_physical_rows` (Triton + reference, tested). It turns HiSparse physical row indices
  (token-major 2,080 B rows of the resident/hot cache) into the HND staging cache that stock pthfa reads. That covers
  the compaction end of `swap_in` and `gather_selected_cache`.
- **Not started:** emitting `SparseFullAttentionSpec` from the pthfa layer, a sketch INDEXER group, the `swap_in`
  call with one row per request (`shared_selection`), and booting HiSparse. I put Track A's server path first
  because it reached "context larger than the pool" sooner.
- **Integration plan:**
  1. `PthFlashAttnImpl`/`Attention.get_kv_cache_spec` returns `SparseFullAttentionSpec(top_k=C=3136,
     total_num_kv_heads=4)` for the 16 layers when `hisparse_config.selector == "sketch"`.
  2. Stream B's sketch moves into a child `SketchKeyCache` layer with `MLAAttentionSpec(cache_role=INDEXER)`, so it
     stays GPU-resident and prefix-cached. Track A's host-indexed sketch region does the same job.
  3. In `select_and_stage`: `idx`/`n_valid` → `hisparse_cache.swap_in(req_id_per_token = one row per request,
     logical_topk_indices = idx, num_valid_rows = n_valid, return_valid_counts=True)` → physical rows →
     `gather_physical_rows` → pthfa over staging. MTP verify rows share the request's row.
  4. The MTP drafter is in the same KV group. HiSparse would spill its pages too, so it needs Track A's window trick
     or its own group.
- **Finding relevant to D's risk 7:** vLLM already hands pthfa a token-major cache. The strides of
  `[NB, 4, 864, 520]` are `(1797120, 520, 2080, 1)`, so BLNHC is the layout already in use.
- **Assessment:** HiSparse would bring host prefix caching, pressure-based unpinning and an acknowledgement protocol.
  Those are exactly gaps (a), (c) and (e) of Track A. It costs a sketch INDEXER layer, the spec plumbing, an
  unbooted subsystem, and upstream's reported 17–20% low-concurrency overhead. The cheaper next step is probably a
  host prefix cache inside Track A: host blocks keep their block hash, and a scheduler hit on a host-only block maps
  to a null GPU entry plus a host block id, staged for the prefill.

## What is left

1. **Host prefix caching** (gap a): it removes the full re-prefill per turn, which is the main agent blocker.
2. Pressure-based spill (gap c). Then contexts that fit pay nothing beyond the ~0.4 ms resident overhead.
3. Two or more concurrent long requests, plus preemption/resume tests. Move the scheduler monkeypatch into a real
   manager or spec (gap e).
4. Cheaper prefill staging: stage only the spilled blocks into reserved pool blocks, and drop the `.cpu()` sync.
5. Fuse the resolve step into stream A's select kernel (~0.25 ms of the 0.41 ms fixed cost).
6. Hot buffer sizing: real traces miss more than the synthetic ones (64K: 16% at 10% hot). The live run measured
   13.6% at 177K. Consider 15–20% hot on x4.
7. Track B steps 1–4 above, if HiSparse's scheduler side is preferred over growing Track A.
