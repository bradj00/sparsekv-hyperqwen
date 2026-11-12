# Sparse-KV release: sparse decode + dense/sparse switch + system-RAM KV tier for HyperQwen

Design, knobs, measurements and known limitations of the release. Per-component design notes are in `STREAM_*.md`.
G-numbers (G-1, G-16, ...) refer to findings of the internal code review; each fix is described where it matters.

## What's included

| component | what |
|---|---|
| sparse decode in pthfa | `pth_flash_attn/sparse_kv.py` plus the SPARSE_KV hook in `vllm_backend.py`. Sketch write, then select → gather → stock pthfa over a staging cache. Prefill and the MTP drafter stay dense. Static dense shortcut when the block table holds ≤ C tokens |
| host-RAM tier | `sparsekv/tier.py`, `sparsekv/tier_connector.py`. HostTierConnector: pinned host mirror, device hot buffer (windowed CLOCK), real spill, pressure policy (`HostTierSpillPolicy`), host prefix cache, **GDN/mamba checkpoints in host RAM**, multi-user/preemption support, host-pool sizing knobs with a RAM check. One vLLM patch (`sparse-tier-spill-kvcheck.patch`) |
| dense/sparse switch | per-step host decision from max_seq_len and a per-GPU cutoff table. Two FULL graph families via `patches/sparse-switch-dual-graphs.patch` |
| kernels | `sparsekv/triton_ops.py`: sketch write (int8/int4), deterministic radix top-k select, gather, `select_suffix` |
| packaging | `patches/series` + `patches/apply.sh --sparse`, `sparsekv/patches/pthfa-sparse-kv.patch`, `sparsekv/install.sh`, `serve-sparse.sh`, `run_release/` |
| basis | `sparsekv/basis/basis27.pt` (mixed_rope fit, `sparsekv/tools/fit_basis27.py`), sha256 `dee8b28281d5837a…`, installed to `venv/share/sparsekv/basis27.pt` |

`pthfa-sparse-kv.patch` targets the pth_flash_attn package, not vllm, so it lives in `sparsekv/patches/` rather than
`patches/` (every `*.patch` there must be listed in `patches/series`).

## Install

On an existing HyperQwen venv (the series applied, KVarN optional, stock pthfa from this repo's base):
```bash
bash sparsekv/install.sh <venv> [--basis sparsekv/basis/basis27.pt]   # idempotent; upgrades an older install
bash sparsekv/install.sh <venv> --check                                  # report only (serve-sparse.sh runs this)
bash sparsekv/install.sh <venv> --uninstall                              # exact reverse
```
The install steps:
1. `patches/apply.sh --sparse <sp>/vllm`: the `sparse-*.patch` series entries, idempotent. In v1.2 these are
   `sparse-tier-spill-kvcheck.patch`, `sparse-switch-dual-graphs.patch`, `sparse-prefill-yield-bypass.patch`,
   `sparse-parser-qwen3-backports.patch`, `sparse-gdn-recover.patch`, `sparse-int8-scale-nofill.patch` and
   `sparse-frontend-backports.patch`. Patches may stack on one file (the two parser patches do): applied-ness is
   decided for the whole set, last first, on a scratch copy (`apply.sh --sparse-state <sp>/vllm` prints it).
2. The pthfa patch on `<sp>/pth_flash_attn`.
3. Copy the runtime `sparsekv/` package (ref, triton_ops, tier, tier_connector, sched) into site-packages, so it imports from
   any cwd, including the KV connector in EngineCore.
4. Copy the basis with its sha256 and check the shape.
5. Write a stamp (`<sp>/sparsekv/RELEASE`) and copies of the applied patches (`<venv>/share/sparsekv/applied/`). The
   copies are what makes upgrade and uninstall exact. Then an import check.

**Fresh build:** `bash patches/apply.sh <sp>/vllm` applies the whole series (58 patches in v1.2, the seven sparse ones
included), and the pthfa
sources in this repo already carry the change. Then `install.sh` adds steps 3-5.

**The installed changes are inert unless their knob is set:**
- kvcheck patch: only with `SPARSE_TIER_SPILL=1`;
- switch patch: only when `pth_flash_attn.sparse_kv.dual_graphs()` is true, which needs `SPARSE_KV=1`;
- pthfa hook: only with `SPARSE_KV=1`.

The dense control below proves this on the GPU.

**Tests:** `sparsekv/test_fixes_cpu.py` and `sparsekv/test_s4_cpu.py` (CPU; need vLLM importable),
`test_fixes_gpu.py`, `test_tier.py`, `test_s4_gpu.py`, `test_backend_sparse.py` (GPU). After `--uninstall`, `vllm/`
and `pth_flash_attn/` were byte-identical to the pristine venv; on a copy of pristine vLLM 0.31.0, `apply.sh DIR`
applies the full series (53 patches) at `--fuzz 0`.

## Launch: `serve-sparse.sh`

`HQ_GPU=<card> ./serve-sparse.sh`:
- Base settings as the dense HyperQwen baseline: PTH_FA=1, SPEC=mtp, DRAFT_TOKENS=7, CTX=long, PREFIX_CACHE,
  EMBED_UVA, INT8_ACT/PREFILL_ATTN int8.
- Defaults to **127.0.0.1:8093**.
- Refuses 8080/8082 unless `SPARSE_PRODUCTION=1`.
- Refuses to start if the basis is missing or `install.sh --check` fails. That guards against a silently dense server
  (G-7) and a random basis (G-2).

| knob | default | notes |
|---|---|---|
| `SPARSE_PROFILE` | `tier` | `tier`: MAX_LEN 200000, **MAX_SEQS 2**, GPU_UTIL 0.90, tier + pressure spill; GPU pool 71K tokens at 2 slots (111K at 1). `sparse`: MAX_LEN 153600, MAX_SEQS 2, GPU_UTIL 0.92, no tier, with the switch |
| `GPU_UTIL` | 0.90 / 0.92 | the highest validated with the G-1 reservation . Pools: tier 111,196 tokens, sparse 184,320, against dense 215,789 at 0.93 |
| `SPARSE_MEM_RESERVE` | 1 | subtracts sketches + staging + tier GPU buffers from the KV budget before sizing the pool. `0` = old behaviour, which OOMs at high util |
| `SPARSE_SWITCH` / `SPARSE_CUTOFF` | 1 / per-GPU table | 3090 Ti `{1:44000, 2:29000}`, 3090 `{1:20000, 2:19000}`. `SPARSE_CUTOFF=0` = always sparse. **Spill forces always-sparse** |
| `SPARSE_IMPL` | triton | `ref` = test oracle only. A failed triton import is fatal (G-5) |
| `SPARSE_BASIS` | `venv/share/sparsekv/basis27.pt` | a missing file is fatal unless `SPARSE_ALLOW_RANDOM_BASIS=1` |
| `SPARSE_KSEL` / `SINK` / `WIN` | 2048 / 64 / 1024 | C = 3136 tokens per layer-step. KSEL 4096 gave no gain at 176K (harness) |
| `MAX_LEN` | 200000 / 153600 | with the tier, MAX_LEN may exceed the GPU pool (RoPE cap 262144). Host pool = MAX_LEN × MAX_SEQS tokens, 8.4 GB pinned per 250K |
| `SPARSE_TIER_SPILL_POLICY` | pressure | spill only below the watermark (`SPARSE_TIER_WATERMARK_BLOCKS`, default pool/10) |
| host pool | MAX_LEN × MAX_SEQS tokens, ~14 GB per server for 2 × 200K | the KV mirror is 33.3 KB/token (13.37 GB for 2 × 200K), plus `SPARSE_TIER_HOST_MAMBA_SLOTS` 8 GDN checkpoints × 85.3 MB (0.68 GB). Capped at `SPARSE_TIER_HOST_GB_CAP` 16; override with `SPARSE_TIER_HOST_GB`. The worker refuses to start if less than 8 GB `MemAvailable` would be left. Host-indexed sketch rows cost 4.2 KB of GPU per host-pool token (1.6 GB here, reserved by G-1) |
| `SPARSE_TIER_HOT_FRAC` | 0.10 | hot buffer per slot as a fraction of MAX_LEN (1.3 GB at 200K). Use 0.15-0.20 on the x4 card for fewer misses (costs more GPU memory, smaller pool) |
| `SPARSE_TIER_HOST_PREFIX` | 1 | spilled contexts stay prefix-cacheable |
| `HQ_GPU`, `PORT`, `BIND`, `HQ_TMP`, `HF_HOME`, `HQ_NAMES` | required, 8093, 127.0.0.1, run_release/tmp, ~/.cache/huggingface, as the dense launcher | |

## Smoke tests (installed release, one GPU per job, temperature 0, thinking off)

Client: `run_release/smoke.py`. Results: `run_release/results.<tag>.json`; logs: `run_release/logs/`.

The client sends, in order, on one ~126K-token code document:
- `short`: a short prompt with no document;
- `short_speed`: 256 tokens on a short prompt;
- `needle`: the needle at 30% depth, a cold prefill;
- two follow-ups on the same document: the line after a unique line at 60% depth, and a second needle at 75%;
- `speed`: 256 tokens of "explain the last file".

Every server was stopped by its process group with explicit PIDs, and every GPU was back to its idle baseline afterwards.

**Release build:**

| run | GPU | pool | all 6 checks | needle TTFT (cold) | follow-up TTFT | short_speed tok/s | 126K speed tok/s |
|---|---|---|---|---|---|---|---|
| dense control: same installed venv, no SPARSE_* | 3090 (GPU1) | 215,789 | PASS | 99.1 s | 2.13 s | 147.9 | 70.7 |
| `sparse` profile (switch on, util 0.92) | 3090 (GPU1) | 184,320 | PASS | 99.1 s | 2.13 s | **146.9** (switch → dense) | **92.1 (+30%)** |
| `tier` profile (spill, util 0.90): 126K doc on a 111K pool | 3090 (GPU1) | 111,196 | PASS | 105.4 s | 2.5 s | 146.9 (always sparse) | **100.1 (+42%)** |
| `sparse` profile (switch on, util 0.92) | **3090 Ti (GPU0)** | 182,072 | PASS | 119.8 s | 2.4-2.5 s | **171.2** (switch → dense) | **110.9** (dense ref. on this card: 89.9 explain @127K) |
| `tier` profile (spill), 126K doc on a 111K pool | **3090 Ti (GPU0)** | 111,196 | PASS | 118.5 s | 2.4-2.5 s | 147.7 (always sparse) | **113.9** |


No dense control ran on GPU0 for this build. Same-card dense references: 4K explain 157.7, 127K explain 89.9 tok/s.
GPU0's cold prefill (~119 s) was slower than GPU1's (~99 s); another job was running on the other card at the time.
No log contains an OOM or an "allocated > reserved" warning.

What the logs show:
- **Dense control:** no `sparse_kv` lines, and the pool equals the dense server's 215,789. The installed release does not
  change the dense path.
- **`sparse` profile:**
  - `FULL graphs captured per family {'dense': 2, 'sparse': 2}`;
  - at runtime `switch -> dense (max_seq_len=24)` for the short prompt, and `switch -> SPARSE (max_seq_len=20736)`
    once the document prefill crossed the 20K cutoff;
  - `reserving 0.91 GiB` (sketch 914 MiB).
- **`tier` profile:**
  - reserve 3.32 GiB (sketch 859, hot 1301, host sketch rows 811, prefill staging 427 MiB);
  - the spill really ran: 145 blocks mirrored, host-hit blocks served the follow-ups (432 host hits), and
    0 host-cache evictions;
  - the G-16 fix (free only blocks mirrored in an earlier step) was exercised by a real chunked prefill under
    pressure, with the needle, both follow-ups and the code line all correct.
- Single-run decode numbers include MTP-acceptance effects. Here the sparse outputs matched dense in content. Controlled
  numbers: 127K +23-34% on GPU0, +24-25% on GPU1; short prompts equal to dense within 1%.

**Earlier builds:**
- `smoke1`, 3090, tier at util 0.78 without the G-1 reservation: 126K on a 125K pool, PASS,
  99.8 tok/s.
- `smoke2`, 3090 Ti, sparse at util 0.88 without the switch: PASS, 108.4 tok/s at 126K, short 168 tok/s.

## Two-user tier test (RTX 3090 / GPU1, 3 runs)

`run_release/conc_job.sh` + `run_release/conc.py`. Setup:
- the tier profile with `MAX_SEQS=2`, `GPU_UTIL=0.90`, `SPARSE_TIER_WATERMARK_BLOCKS=1` (forced spill);
- GPU pool 71,042 tokens (92 sparse-group blocks), host pool 13.37 GB plus 0.68 GB of GDN checkpoints, G-1 reserve
  4.75 GiB.

Workload:
- User A: 176K tokens (vLLM sources, forward order). User B: 120K tokens on a different document (reverse order,
  its own needles). B starts 60 s after A.
- Phase 1: the needle (cold prefill).
- `POST /reset_prefix_cache` (success each time), then phase 2: quote the line after a code line at 60%.
- Reset again, then phase 3: the second needle.

| run | A needle / code / needle2 | B needle / code / needle2 | cold TTFT A / B | follow-up TTFT A / B | preemptions | host GDN restores |
|---|---|---|---|---|---|---|
| conc1 | PASS / PASS / PASS | PASS / PASS / PASS | 188.9 / 227.4 s | 4.5, 4.4 / 4.0, 3.9 s | 1 (re-admitted with 51,840 tokens from cache) | 4 |
| conc2 | PASS / PASS / PASS | PASS / PASS / PASS | 189.6 / 228.7 s | 4.5, 4.5 / 4.1, 4.0 s | 1 | 4 |
| conc3 | PASS / PASS / PASS | PASS / PASS / PASS | 189.7 / 228.9 s | 4.6, 4.5 / 4.1, 4.0 s | 1 | 4 |

- **A's 177K code-line question passed 3/3.** An earlier build failed it in 2 of 4 runs on this card
  (nondeterministic top-k, G-14/G-15).
- The release runs the refreshed deterministic kernel, and the answers are byte-identical across the three runs. So
  the repeats show the result is stable and reproducible; they are not three independent samples of a noisy process.
- **Host GDN restore is proven.** Every follow-up came after a GPU prefix-cache wipe, yet answered in 4-4.6 s instead
  of a ~190 s re-prefill. The tier stats show `mamba_restored 4` per run (2 users × 2 follow-ups), host-cache
  `evict 0`, and 739 host-hit blocks.
- No OOM and no "allocated > reserved" warning. conc2's log has one traceback at 08:58:47, raised by the SIGTERM
  shutdown after all requests had finished (engine abort), not during serving.
- Not covered: decode speed while the other user prefills (2.8-48 tok/s, from batching with a long prefill), and
  the 3090 Ti.

### After the G-21/G-23/G-24 fixes (GPU1 3090)

Same job (`run_release/conc_job.sh`, forced spill, 2 slots), staggered so that A's answer lands while B is still in
its cold prefill (B_DELAY=120 s: A done at 189 s, B done at 287 s). Before each run: `install.sh venv` (upgrade in
place, all patches "already applied", modules match the repo) and `test_fixes_cpu.py` with the release venv: ALL OK.

| run | A needle / code / needle2 | B needle / code / needle2 | follow-up TTFT A / B | GDN store (saved / restored / pinned / unpinned / flushed at finish) |
|---|---|---|---|---|
| g23 | PASS / PASS / PASS | PASS / PASS / PASS | 4.5, 4.4 / 4.0, 4.0 s | 2 / 4 / 2 / 2 / 0 |
| g23b (`A_MAXTOK1=3`: A's first answer capped at 3 tokens) | (capped: '741') / PASS / PASS | PASS / PASS / PASS | 4.7, 4.6 / 4.2, 4.2 s | 2 / 4 / 2 / 2 / 0 |

- Every host GDN restore after the prefix-cache wipe answered correctly, and the new pins were released (2/2), so no
  block stays pinned and `reset_prefix_cache` kept succeeding.
- **The exact G-23 race was not reproduced on the server.** In both runs each offered checkpoint's delayed save ran
  before its request finished (`flushed_at_finish 0`), because the last prompt boundary is offered one or more
  prefill chunks before the end of the prompt. The fix is verified by the fake-block-pool unit test, which fails on
  the old code. The server runs show that the fix does not break the restore path.
- One preemption per run, no OOM, no "allocated > reserved" warning.

## Other measurements (design notes)

- **Quality:**
  - Needle, code-next and essay at 32/64/128K: sparse = dense (internal eval harness).
  - 150K/176K (49 questions): sparse is within one question of dense per task.
- **Tier:**
  - 177K prompt on a 111K pool at util 0.90 with the fixes: needle, code and needle2 PASS, 127 tok/s (3090).
  - 177K on 139K and 238K on 82K (pre-fixes): PASS.
  - Host prefix cache: follow-up TTFT 197 s → 2.3 s.
- **Switch:** grow-across-cutoff copy is exact. A 3-turn 36K→57K conversation passes.

## Two-user speed investigation (`sparse` profile vs dense control)

**Question.** An earlier speed grid showed sparse 2-user per-request decode far below dense (for example 128K explain 53 vs 92 tok/s).

**Answer.** There is no sparse decode regression, and no CUDA-graph fallback. Batch-2 decode replays the FULL graphs.
- The low numbers are decode steps sharing the GPU with re-prefills of the other user's document.
- The re-prefills are prefix-cache misses caused by KV-pool capacity, and they hit dense the same way at the same pool size.

**Evidence: graph dispatch.** `SPARSE_CG_STATS=1` (new, inert by default) logs dispatch and replay counters. On a 2×48K sparse run: `('ModelCudaGraphManager','FULL',16,2): 648`, with decode at B=2 replaying the sparse family.

**Evidence: true concurrent decode, all cache hits.** 2×48K on GPU1, `run_release/grid.g1{sp,de}48.jsonl`, 3 reps (the `run_release/grid.*.jsonl` files were produced by an internal eval harness, not included):

| | per-request tok/s copy / explain | total tok/s copy / explain | TTFT |
|---|---|---|---|
| dense | 169.8 / 94.2 | 230.9 / 149.0 | 1.2–1.5 s |
| **sparse** | **183.5 / 106.0** | **242.7 / 165.8** | 1.3–1.5 s |

**Root cause of the 2-user collapse: prefix-cache capacity cliff (hybrid model).** a prefix-cache probe (sequential requests, GPU1):
- With a **184,320-token pool, dense and sparse behave identically**: only one 64K document stays cached.
  - After the second document's cold prefill, every follow-up on the first document misses completely (`cached=0`, 38 s re-prefill).
  - Dense at GPU_UTIL 0.88 was probed to give exactly the sparse profile's pool size.
- With **215,789 tokens (dense at 0.93 on GPU1) both stay cached**.
- **GPU0's dense pool, 212,792 tokens, is just below the cliff**: the cold-first 64K grid misses there too.
- Likely mechanism (not instrumented further):
  - The pool is shared by the attention group and the three GDN groups.
  - A 64K document plus the GDN state pages of a running request needs about 100 of the roughly 213 pages.
  - Evicting a cached document's tail blocks drops its hit below the only retained GDN checkpoints (align mode keeps the last two boundaries at or below the prompt end), so the hit falls to 0 rather than shrinking.
- The sparse profile's pool is about 15% smaller because the sparse buffers (sketches ≈ 4.2 KB/token across 16 layers, plus staging) are budgeted out of it (G-1). That moves the 2-user cliff from about 2×66K to about 2×57K tokens.
- The earlier grid's dense 64K cells looked clean only because their 64K documents extended already-cached 32K documents. With that ordering, `run_release/grid.g1dnord.jsonl` hits as well.

**128K and 150K with 2 users.** Two documents never fit either pool, so both servers serialise (one user waits for a full re-prefill). Per-request decode is then the 1-user decode, and sparse wins it.
- Total throughput is decided by which document happened to survive in the cache each turn, not by decode speed.

**GPU0 (3090 Ti) grid**, cold-first per length (cold-first protocol), 3 reps. Columns: per-request decode tok/s, mean TTFT for user 1 / user 2, total throughput.

| len | U | prompt | dense req tok/s | sparse req tok/s | dense TTFT | sparse TTFT | dense total | sparse total |
|---|---|---|---|---|---|---|---|---|
| 64K | 1 | explain / copy | 115.6 / 184.4 | 127.8 / 152.4 | 16.7 / 1.5 | 16.5 / 1.4 | 61.6 / 120.7 | 66.6 / 107.0 |
| 64K | 2 | explain | 47.9 | 58.4 | 32.9 / 81.0 | 32.7 / 80.2 | 12.7 | 13.1 |
| 64K | 2 | copy | 71.7 | 86.2 | 48.3 / 96.3 | 47.8 / 95.5 | 10.2 | 10.4 |
| 128K | 1 | explain / copy | 90.1 / 141.8 | 112.2 / 137.0 | 42.6 / 43.2 | 42.1 / 43.2 | 44.7 / 59.9 | 52.1 / 59.4 |
| 128K | 2 | explain | 92.5 | 117.5 | 84.6 / 129.0 | 44.3 / 88.4 | 6.3 | 7.6 |
| 128K | 2 | copy | 152.8 | 163.3 | 43.4 / 87.3 | 82.6 / 212.2 | 7.8 | 5.2 |
| 150K | 1 | explain / copy | 89.1 / 85.0 | 115.4 / 156.2 | 54.0 / 53.8 | 53.3 / 53.8 | 46.2 / 44.6 | 55.5 / 68.3 |
| 150K | 2 | explain | 88.6 | 112.8 | 105.6 / 269.3 | 160.4 / 214.4 | 4.1 | 4.1 |
| 150K | 2 | copy | 134.2 | 194.2 | 105.8 / 269.4 | 105.9 / 267.4 | 4.1 | 4.2 |

- Both servers re-prefill in every 2-user cell, the 1-user cells include one re-prefill (the TTFT means), and 2 documents of 128K or more never overlap.
- Per-request decode is higher for sparse in every 2-user cell.
- Total throughput is prefill-bound and decided by cache luck: which document survived each turn.
- Copy-prompt tok/s swings with MTP acceptance (for example 64K U=1 copy: acceptance length 4.27 sparse vs 5.87 dense).

**Tier profile.**
- At GPU_UTIL 0.93 (G-1 reservation on) it boots and runs 2×150K and 2×64K with no OOM, with an 88K-token GPU pool against 71K at 0.90.
- A 2×64K first turn re-prefilled both documents at 0.90 even though the host cache held them. Lookups counted host-block hits, but `cached=0`.
  - Likely cause: the GDN checkpoint restore did not cover the needed boundary (`gdn_store restored 1` for two requests). Not resolved.

**Sparse profile at GPU_UTIL 0.94.** It boots, 150K prefill and decode run without OOM, and the pool is 195,559 tokens. That is one run, not soaked, and still below the 2×64K cliff.

**Recommendations.**
1. Benchmark 2 users only with documents that fit the pool together, or report TTFT and total throughput next to per-request decode.
2. Fix the cliff itself for both dense and sparse, for example with an eviction policy that evicts a whole cached sequence, or with intermediate GDN checkpoint retention so that partial eviction degrades gracefully.
3. Consider GPU_UTIL 0.93–0.94 for the sparse profile after a soak.
## 250K tier profile, 2 users per card

**Goal:** the `tier` profile serves 2 concurrent ~250K-token chats per card.

### G-25: follow-ups on a host-cached document re-prefilled fully (cached=0)

**Root cause (instrumented, `SPARSE_TIER_DEBUG=1` lookup logs, GPU1, 2 × ~246K documents, max_tokens=1 first turn):**
- Host mirroring is planned at scheduling time from `num_computed_tokens - MARGIN`. A request that ends right after
  its prompt (max_tokens 1, an immediate EOS, an abort) never has its last prefill chunk's blocks mirrored: the
  connector dropped them at `request_finished`.
- When the GPU prefix cache later evicted those 2 tail blocks, the host chain stopped 2 blocks short of the prompt
  end, while the host GDN checkpoint sits exactly at the prompt-end boundary:
  `FA lookup max 245936: hit 282 blocks (282 host, 0 GPU) of 284` then
  `mamba lookup max 242784: GPU hit 0, no host checkpoint in (0, 242784]` -> hit 0 -> full re-prefill (336 s).
- This is the "1 restore for 2 requests" of the earlier 2×64K grid: its prefill turns used max_tokens 1, so only the
  request whose tail happened to survive on the GPU restored.

**Fix (`sparsekv/tier_connector.py`):**
- `request_finished` mirrors the remaining full blocks up to the processed end (the whole prompt once a token was
  sampled): host blocks are allocated under their prefix keys, the copy goes out in the next connector meta
  (`fin_copies`, done in the worker's `start_step` before the forward), and the GPU blocks are held (ref+1) until the
  worker reports the copy id (`fin_done`), so a re-handed-out block cannot be zeroed first (same reasoning as
  G-16/G-23). `SPARSE_TIER_MIRROR_AT_FINISH=0` restores the old behaviour (used for the "before" run).
- Host LRU: a finished request's host blocks are released deepest-first, so when the host pool is full, eviction
  trims a cached document from its tail instead of evicting block 0 (which killed the whole prefix hit).
- The host GDN restore's free-block margin is a knob (`SPARSE_TIER_MAMBA_RESTORE_MARGIN`, default 2, was a fixed 8);
  a deferred restore is logged.
- `SPARSE_TIER_DEBUG=1` logs every full-attention / mamba prefix lookup (host vs GPU blocks, restore decisions).
- Unit test `test_g25_finish_mirrors_tail_blocks_and_lru_trims_from_tail` (`test_fixes_cpu.py`: ALL OK): tail
  blocks keyed and copied, held until the copy id is reported, the off switch reproduces the old 7-of-10 chain, LRU
  evicts the deepest block.

**Before / after (GPU1, same documents, `PRIME_MT1=1`: first turn "Reply with OK." with max_tokens 1):**

| | A follow-up (code @60%) | log |
|---|---|---|
| before (`SPARSE_TIER_MIRROR_AT_FINISH=0`) | `cached=0`, **TTFT 336.4 s** (full re-prefill) | host chain 282/284, no host checkpoint below it |
| after | `cached=245376`, **TTFT 3.3 s** | `finish-time tail mirror ... 2 blocks [283..284]`, `host restore at 245376 -> hit 245376` |

Every later follow-up in every run (both cards, both users, soak) hit the host cache.

### Second finding: at the default sizing the two users are serialised (GPU pool too small)

- At `GPU_UTIL=0.92` with `MAX_LEN=262144` the G-1 reserve is 6.0 GiB (hot buffer 2.56 GB at HOT_FRAC 0.10, host
  sketch rows 2.12 GB for the 17.5 GB host pool) and the shared GPU pool is only 64-65 blocks (51K tokens).
- One running request holds 26-42 of them (GDN state + speculative + checkpoint blocks of the 3 mamba groups, plus
  the attention tail), so the second user's admission fails (`Running: 1, Waiting: 1`) until the first finishes:
  decode overlap 0 s, combined explain 66-77 tok/s (lower than one user).
- **Fix: `GPU_UTIL=0.93` + `SPARSE_TIER_HOT_FRAC=0.075`** (19,660 hot rows/slot: about the validated 200K
  config's 20,000). Reserve 5.42 GiB, pool **93-94 blocks (74K tokens)**; both users now decode together.

### Validated 2-user 250K configuration

`SPARSE_PROFILE=tier MAX_LEN=262144 MAX_SEQS=2 GPU_UTIL=0.93 SPARSE_TIER_HOT_FRAC=0.075 SPARSE_TIER_HOST_GB_CAP=20`
- Boot (GPU1): `reserving 5.42 GiB of the 8.10 GiB KV budget`, `GPU KV cache size: 74,445 tokens`,
  `tier: runtime mode=SPILL ... host pool 17.51 GB (609 blocks), GPU pool 94 blocks, hot 19660 rows/slot`.
- Host RAM: ~21 GB per tier server (17.5 GB pinned KV mirror + 0.68 GB GDN checkpoints + process).

### Results (temperature 0, thinking off, documents = vLLM sources ~246K tokens, harness `run_250k/chat250.py`)

**1 user** (follow-ups hit the host cache; recall = needle, needle2, code line @60%, code line @45%):

| card | recall | cold TTFT | follow-up TTFT | explain tok/s (384) | copy tok/s (384) |
|---|---|---|---|---|---|
| GPU0 3090 Ti, final config | 4/4 | 342 s | 2.6 s | 104.6 | 170.3 |
| GPU1 3090, util 0.92 run | needles + 45% PASS (60% question was ambiguous, harness fixed) | 337 s | 3.2-3.3 s | 95.4 | 170.3 |

**2 users concurrently, different ~246K documents, final config** (B cold-starts 20 s after A; per-user tok/s A / B):

| card | recall A / B | cold TTFT A / B | follow-up TTFT | explain tok/s | copy tok/s | combined explain / copy |
|---|---|---|---|---|---|---|
| GPU1 | 4/4 / 3/4 | 333 / 644 s | 6.4-7.0 s | 76.5 / 81.1 | 125.3 / 122.4 | **153.0 / 234.1** |
| GPU0 | 4/4 / 3/4 | 344 / 667 s | 5.4-5.8 s | 75.9 / 81.3 | 138.1 / 137.3 | **151.7 / 258.0** |

- B's 45% code line fails identically on both cards and every run: the model quotes the line from a near-identical
  block of the same file (approximate attention on a near-duplicate, not a tier fault).
- At util 0.92 / HOT_FRAC 0.10 (64-block pool) the same 2-user run serialised: overlap 0 s, combined 66 / 91 tok/s.
- While one user cold-prefills 246K, the other's decode drops to a few tok/s (chunked-prefill batching, known).

**Short context** (explain / copy tok/s, 384 tokens): GPU0 tier 4K 151 / 232, 16K 124 / 141 vs the `sparse` profile on
the same card and harness 144 / 258, 123 / 146 (explain equal, copy -3..-10%). GPU1 tier: 4K 126 / 224, 16K 108 / 129.

### 60-minute soak

Two long documents per card (~246K each, the same A/B), turns concurrent (2 users) or alternating, random question
from needle / code@60% / needle2 / explain / copy, plus a ~3K-token short request every 45 s
(`run_250k/chat250.py soak`, watchdog `run_250k/watchdog.sh`).

| run | duration | turns GPU0 / GPU1 | wrong | errors | full re-prefills after warm-up | follow-up TTFT | OOM / allocated>reserved |
|---|---|---|---|---|---|---|---|
| soak 1, both cards (GPU ~272/257 W) | 15.5 min, cut by a host power reset | 163 / 231 | 0 / 2 (near-duplicate code line) | 0 | 0 | 2.6-7.6 s | none |
| soak 2, both cards, capped at 250 W | 20 min | 88 / 110 | 0 / 0 | 0 | 0 (2 cold starts per card) | 2.9-9.6 s (median 6.5 / 6.8) | none |

- **Host reset during soak 1:** The host hard-reset (instant power-off, no panic). It was traced to
  the host's UPS running at 80-88 % with both cards loaded and cutting output on a transient; both cards were then capped
  at 250 W. The guest logs show nothing abnormal: steady cached follow-ups on both cards, no prefill, no
  error. Note the tier's steady host->GPU staging traffic: every follow-up stages the whole document's KV from pinned
  host RAM (about 5.8K block-layers per turn).
- **Soak tok/s (mixed workload, not a controlled A/B):** soak 1 explain / copy 83.2 / 135.4 (GPU0), 75.4 / 133.9 (GPU1);
  soak 2 at 250 W 69.6 / 110.9 (GPU0), 80.5 / 125.6 (GPU1).
- No preemption, no host-cache eviction, MemAvailable >= 10 GB with two tier servers running.

## Streams-per-card sweep, tier profile, 128K and 250K (both cards capped at 250 W)

Harness `run_250k/sweep.py PORT OUT NTOK KS`: one server per config; for each k, users 0..k-1 are active (a new user
cold-prefills its own document first), then 2 rounds of [code line, explain 384 tok, copy 384 tok] with all k users at
once, then needle2. Distinct documents per user (vLLM sources, per-user shuffled file order). Per-stream tok/s =
decode rate of each request; combined = all users' tokens over the span of the decode windows (includes the other
users' suffix prefills, which is what a real batch sees). Summary: `run_250k/swsum.py sw.*.jsonl`. Results:
`run_250k/sw.{128n2,128n2g1,128n3,128n4,250n3,250n3b}.jsonl`, server logs `run_250k/logs/server.sw*.log`.

**All points:** 0 errors, 0 OOM / "allocated > reserved", 0 preemptions, **0 re-prefills** (every follow-up hit the host
cache), needle recall 100 % (needle + needle2, every user, every k). Lowest MemAvailable over the sweep: 11.45 GB (250K
×3 server on GPU1 plus 128K ×2 server on GPU0 at once). The one Traceback (sw128n4, 18:11:47) is the SIGTERM shutdown after the sweep.

### 250K (documents ~246K tokens, MAX_LEN 262144)

| config (card) | GPU pool | host pool | k | per-stream explain / copy | combined explain / copy | follow-up TTFT | code-line recall |
|---|---|---|---|---|---|---|---|
| MAX_SEQS 2, util 0.93, HOT 0.075 (GPU1, earlier run at ~270 W) | 94 blk (74K) | 17.5 GB | 2 | 76.5-81.1 / 122-125 | **153.0 / 234.1** | 6.4-7.0 s | 3/4 (one near-duplicate miss) |
| MAX_SEQS 3, util 0.94, HOT 0.04 (GPU1) | 82 blk (65K) | 26.3 GB | 1 | 95.5 / 157.7 | 95.5 / 157.7 | 4.3 s | 2/2 |
| same | | | 2 | 87.0 / 148.3 | 59.6 / 83.9 (serialised: Running 1, Waiting 1) | 4.5-13.2 s | 4/4 |
| | | | 3 | stopped: already serialised at k=2 | | | |
| MAX_SEQS 3, util 0.95, HOT 0.025 (GPU0) | 106 blk (84K) | 26.3 GB | 1 | 89.9 / 158.8 | 89.9 / 158.8 | 4.1 s | 2/2 |
| same | | | 2 | 79.6 / 113.9 | 137.8 / 205.1 | 7.4-7.6 s | 4/4 |
| same | | | 3 | 77.3 / 104.7 | 100.7 / 129.9 (only 2 decode at once, the 3rd waits) | 7.3-14.8 s | 6/6 |

Cold TTFT per new 246K user at 250 W: 356-386 s (prefills are serial, so the k-th concurrent cold user waits k × that).

### 128K (documents ~122K tokens, MAX_LEN 131072)

| config (card) | GPU pool | host pool | k | per-stream explain / copy | combined explain / copy | follow-up TTFT | code-line recall |
|---|---|---|---|---|---|---|---|
| MAX_SEQS 2, util 0.93, HOT 0.075 (GPU0) | 172 blk (126K) | 8.8 GB | 1 | 108.9 / 107.5 | 108.9 / 107.5 | 1.7 s | 0/2 |
| same | | | 2 | 82.4 / 139.7 | **144.0 / 263.4** | 1.6-4.2 s | 0/4 |
| same, cross-check on GPU1 | 172 blk | 8.8 GB | 1 | 106.2 / 141.9 | 106.2 / 141.9 | 1.5 s | 2/2 |
| same | | | 2 | 82.5 / 127.3 | 142.5 / 214.6 | 1.8-3.8 s | 2/4 |
| MAX_SEQS 3, util 0.93, HOT 0.075 (GPU0) | 142 blk (104K) | 13.2 GB | 1 | 123.2 / 172.1 | 123.2 / 172.1 | 1.6 s | 2/2 |
| same | | | 2 | 102.8 / 138.7 | **199.8 / 231.6** | 3.8-4.1 s | 2/4 |
| same | | | 3 | 60.8 / 103.0 | 142.8 / 212.3 (all 3 running) | 3.8-6.2 s | 2/6 |
| MAX_SEQS 4, util 0.93, HOT 0.05 (GPU1) | 130 blk (95K) | 17.6 GB | 1 | 123.2 / 120.5 | 123.2 / 120.5 | 1.6 s | 2/2 |
| same | | | 2 | 86.8 / 129.3 | 168.5 / 238.4 | 3.7-3.9 s | 2/4 |
| same | | | 3 | 80.9 / 118.6 | 112.5 / 154.3 | 3.9-10.0 s | 2/6 |
| same | | | 4 | 65.0 / 80.2 | 112.4 / 139.9 | 4.0-13.5 s | 4/8 |

Cold TTFT per new 122K user: 113-135 s. Code-line misses are per document and repeat at every k: user 1's line (and user
2's) sits in near-identical CuTe/asm blocks, and user 0's line was missed on GPU0 only. Needles were always found.

### Knee and recommendation

- **The knee is 2 streams per card at both lengths.** Going from 2 to 3 or 4 users lowers combined decode at 128K:
  ~200 → 143 → 112 explain tok/s, with all 3 or 4 requests running. At 250K the third user cannot even run
  concurrently: the shared GPU pool (attention tail + GDN state/speculative blocks, ~26 blocks per running request)
  is too small once the third slot's hot buffer and 8.7 GB more host-indexed sketch rows (4.2 KB GPU per host token)
  are reserved. Follow-up TTFT also grows with k: each user's host-staged suffix prefill delays the others' decode.
  Not investigated: CUDA-graph coverage at B = 3/4 (FULL graphs were profiled for B ≤ 2 in earlier runs). It may be
  a second reason the batch-3/4 decode is slower.
- **250K: 2 slots per card** (MAX_SEQS 2, util 0.93, HOT 0.075, 17.5 GB host pool): 153 / 234 tok/s combined
  for 2 users. A third 250K slot costs 8.7 GB of RAM and GPU pool, and gains nothing.
- **128K:** 2 streams is also the throughput optimum. The best 2-stream number came from the MAX_SEQS 3 build
  (199.8 / 231.6 vs 144.0 / 263.4 for MAX_SEQS 2). The
  configs differ only in slots and pool size, and single runs vary with MTP acceptance, so treat 144-200 explain as one band. A dedicated 128K card would use `MAX_LEN=131072 MAX_SEQS=2`
  (8.8 GB host, 172-block pool). MAX_SEQS 3-4 only helps when more users must be admitted at once (latency), not
  for throughput.

## 4 streams per card (S4, both cards capped at 250 W)

Goal: 4 concurrent ~250K chats on one 24 GB card at 300+ tok/s combined decode, 4 x 128K per card, recall intact,
follow-ups from cache. Harness `run_s4/dec.py` (as `run_250k/sweep.py`, but 1024-token answers and a **steady** metric:
tokens of all k streams inside the window where all k decode, so follow-up/prefill time is not hidden in decode numbers),
summary `run_s4/sum.py`. Results `run_s4/d.*.jsonl`, logs `run_s4/logs/`.

### Root causes (evidence)
1. **Not CUDA graphs.** `SPARSE_CG_STATS=1` on the 1843b7e build, 128K x4: FULL `(24,3)` / `(32,4)` replayed for target and
   drafter, no eager uniform-decode step. The 2-sample "FULL capture 2/2" is vLLM's memory-profiling pass.
2. **Follow-up re-staging serialised the streams.** Each follow-up recomputes 1.2-1.7K tokens (prefix hits end at the last
   full 864-token block) as a dense chunk, and `stage_prefill` copied every spilled block of all 16 layers (per-block
   Python loop after a `.cpu()` sync): 2.4-15 GB per round at 128K, last user's TTFT 16-20 s, so at k >= 3 the streams
   barely decoded together (no steady window at all). Decode itself already scaled: steady copy 185/298/451 tok/s k1/2/4.
3. **Per-request GPU footprint:** a running request holds (2 + 7) GDN pages x 3 groups (28.5 MB each, ~0.77 GB at MTP-7)
   plus the attention tail; the reserve also funded a hot buffer for the CUDA-graph padding slot and sized the sketch
   region from the whole KV budget (~0.5 GiB too much), and the int8 host-indexed sketch costs 4.2 KB per host token.
4. **Selection cost is linear in context x streams:** select+gather = 2.5 ms (122K) / 4.9 ms (246K) per request per step
   over 16 layers on the 3090; the normaliser (stats) pass is 52 % of it.

### Fixes
(Build labels such as `1843b7e` or `210c029` in the tables below name development builds; the shipped code is the final one.)
- 561cb1e: no hot-buffer KV rows for the padding slot (guarded: such a slot stages from host and caches nothing).
- 1548c20: `SPARSE_SKETCH_BITS=4` int4 rank-64 sketch (half the sketch bytes; ref + triton, graph-safe); sketch reserve
  from the budget left after the other reserves. Pool 128K x4: 130 -> 148-150 blocks; 250K x4 int4: 157 blocks.
- e557eb9: **follow-ups no longer re-stage the document.** A follow-up suffix (uncached remainder <= 2048 tokens of a
  request with spilled blocks) attends sparsely: `triton_ops.select_suffix` (pseudo-requests of 16 rows, exact causal
  stats, summed scores per segment) + tiered gather of only the picked rows (resident from the GPU cache, the rest
  zero-copy from pinned host RAM via UVA; the padding slot is used so suffixes neither evict nor fill decode hot rows),
  one causal pthfa launch per request with segments as sub-requests. Cold chunks stay dense (by the `uncached` remainder at
  admission), staged with one batched `copy_blocks` launch from host block lists (no `.cpu()`, no per-block loop).
  Per-step host sync in `start_step` (pageable `batch_slots` upload) replaced by a pinned ring (measured neutral at k=2).
- 210c029: segments = 16-row question segments over the last 128 rows + 256-row tail segments, suffix KSEL 4096
  (tuned on a warm server, `run_s4/tune.out`: code lines 6/8 vs 2/8 unsegmented and 4/8 for the old dense staging);
  `SPARSE_STATS_SUB=4` (normaliser pass samples every 4th tile of the unmasked bulk; 4x246K select 18.9 -> 12.0 ms/step)
  and `SPARSE_SELECT_EVERY=2` (a decoding request reuses its top-k on alternate steps: per-slot store, device per-row
  flags, sinks/window rebuilt; ~50 % of decode-row selections reused).
- Tests: `sparsekv/test_s4_gpu.py` (int4 vs ref, graph capture, padding slot, select_suffix vs ref, segmented causal
  attention vs fp32, copy_blocks, selection reuse incl. graph replay), `sparsekv/test_s4_cpu.py` (prefill planner),
  existing suites pass (test_tier, test_fixes_gpu, test_backend_sparse triton, test_fixes_cpu).
- a3066da: tuned launch config for int4 sketches (4x246K select+gather 0.98 -> 0.83 ms per layer-step).
- 5bfabb9: **`SPARSE_TIER_HOST_BITS=4` int4 host KV tier** (64-dim groups, fp16 group scales: 272 B instead of 520 B per
  token-head; quantised on the GPU straight into pinned memory via UVA, dequantised to pthfa int8 rows on fetch, so
  attention and the GPU cache are unchanged): pinned host RAM per 250K agent 8.4 -> 4.4 GB. **Prefix lookups prefer the
  host copy for mid-prefix blocks**: a returning agent no longer re-acquires its GPU-cached document blocks, which kept
  the 4th of 4 x 250K Waiting at ~51 % pool (follow-up TTFT 15-18 s, no concurrent decode).
- Not usable as a knob: MTP-3/5 change the hybrid block size to 816 (kernels are built for 864) -> boot refused.

### Before / after (steady = all k agents decoding at once; 1024-token answers; GPU0 3090 Ti unless noted)

128K (documents ~120K tokens, MAX_SEQS 4, util 0.93, HOT 0.05):

| build | k | steady explain / copy | per-agent explain | follow-up TTFT med / max | recall | staged GB / round | miss MB/step | pool |
|---|---|---|---|---|---|---|---|---|
| 1843b7e | 1 | 131 / 185 | 131 | 2.7 / 2.7 | 3/3 | 2.4 | 3.6 | 130 blk |
| 1843b7e | 2 | 208 / 298 | 104 | 4.8 / 4.8 | 4/5 | 6.3 | 10.4 | |
| 1843b7e | 3 | no common window | 103 | 4.8 / 15.9 | 6/7 | 10.3 | 10.6 | |
| 1843b7e | 4 | - / 451 | 79 | 6.4 / 20.3 | 8/9 | 15.2 | 29.4 | |
| 210c029 | 1 | 137 / 166 | 137 | 1.5 / 3.3 | 4/4 | 0 | 1.3 | 150 blk |
| 210c029 | 2 | 217 / 332 | 108 | 2.5 / 2.7 | 5/7 | 0 | 7.7 | |
| 210c029 | 3 | 262 / 420 | 83 | 3.4 / 5.1 | 8/10 | 0 | 15.8 | |
| 210c029 | 4 | **312 / 501** | 73 | 4.4 / 5.6 | 11/13 | 0 | 24.3 | |
(Recall misses are user 1's code line, a near-duplicate CuTe block, missed by every build.)

250K (documents ~244K tokens, MAX_LEN 262144):

| build / card | k | steady explain / copy | per-agent explain | follow-up TTFT med / max | recall | pool / host |
|---|---|---|---|---|---|---|
| 1843b7e sweep, MAX_SEQS 3 (GPU0) | 3 | only 2 decode at once: 101 / 130 combined | 77 | 7.3 / 14.8 | 6/6 | 106 blk / 26 GB |
| a3066da- int4 sketch (GPU0) | 1 | 110 / 167 | 110 | 2.0 / 4.9 | 4/4 | 157 blk / 33.7 GB |
| | 2 | 188 / 250 | 94 | 3.4 / 4.5 | 7/7 | |
| | 3 | 219-226 / 256-365 | 72 | 4.3 / 5.6 | 12/12 | |
| | 4 | 4th agent Waiting (no window) | | 4.3 / 18.5 | 13/13 | |
| 5bfabb9 prefer-host (GPU0) | 4 | **270-277 / 368-439** | 60-72 | 5.5 / 8.9 | 16/16 | 157 blk / 33.7 GB |
| e557eb9 (GPU1 3090 x4, int4 sketch) | 1 / 2 | 93 / 150, 154 / 250 | 93, 70 | 2.6 / 5.7 | 5/8 | 181 blk / 22 GB |

Cold prefill of one 244K document: 390-396 s (3090 Ti), ~350 s on the 3090 when alone; prefills are serial.

### Footprint per 250K agent (measured boots) and capacity projection
- GPU, per running agent: 27 GDN state pages (MTP-7, 3 groups x (2+7) x 28.5 MB) = 0.77 GB + sink/tail ~0.1 GB + hot share
  0.22 GB (HOT 0.025) + int4 sketch of its host tokens 0.53 GB = **~1.6 GB**; plus per server: prefill staging 0.55 GB,
  graph/activation headroom. A 24 GB card at util 0.94 holds **4** agents (pool 157-158 blocks vs ~124 needed); a 5th does
  not fit at MTP-7 (would need GDN state outside the pool or a kernel build for the MTP-3/5 block size 816).
- Host, per agent: 244K tokens x 33.3 KB = 8.4 GB pinned with the int8 host tier, **4.4 GB with int4** (`SPARSE_TIER_HOST_BITS=4`),
  plus a share of 8 x 85 MB GDN checkpoints. Per server ~13 GB non-pool (process, pinned embeddings for EMBED_UVA, GDN store,
  CUDA context); OS ~4 GB; floor 8 GB MemAvailable.
- Projection (agents limited by min(GPU: 4 per 24 GB card, RAM)):

| host RAM | int8 host tier | int4 host tier | this box (2 cards) |
|---|---|---|---|
| 78 GB (today) | 4 + 2 is over the floor; 4 + 1 = 5 | **8** (4 + 4; MemAvailable ~12 GB measured) | 8 |
| 96 GB | 6 (4 + 2) | 8 | 8 (GPU-bound) |
| 128 GB | 8 | 8 (room for a 3rd card: 12) | 8 |
| 192 GB | 8 (room for 3 cards: 12) | 8 (room for 5 cards: 20) | 8 |
| 256 GB | 8 (room for 4-5 cards: 16-20) | 8 (room for 7 cards: 28) | 8 |
  RAM per 4-agent card: int4 17.7 + 13 = ~31 GB, int8 33.7 + 13 = ~47 GB; box budget = RAM - 4 (OS) - 8 (floor).

### Headline: 8 concurrent 250K agents on the box (`run_s4/g1part.sh`, results `run_s4/d.joint_g{0,1}.jsonl`)
GPU0 (3090 Ti) 4 x 250K + GPU1 (3090, x4 link) 4 x 250K, both servers at once, int4 host tier, the joint rounds of the two
servers aligned within ~4 s (8 agents decoding simultaneously):

| server | steady explain | steady copy | per agent explain / copy | follow-up TTFT med / max | recall |
|---|---|---|---|---|---|
| GPU0 3090 Ti | 268-271 | 394-415 | 58-74 / 87-103 | 5.2 / 5.5 s | 15/16 |
| GPU1 3090 x4 | 268-285 | 375-395 | 61-74 / 76-104 | 5.3 / 5.9 s | 16/16 |
| **box** | **~540-556** | **~770-810** | | | 31/32 |
MemAvailable >= 11.6 GB throughout (two 17.66 GB int4 host pools), 0 errors, 0 re-prefills, 0 GB staged. Hot miss 14-16 %,
30-39 MB/step per server. Int4 vs int8 host tier at 128K x1 (5 rounds): recall 7/7 vs 7/7, 134/178 vs 140/181 tok/s; at 4 x
250K on GPU0: 269/405 vs 270-277/368-439, recall 15/16 vs 16/16.
**Not reached:** 300+ explain tok/s per card at 4 x 250K (268-285). Copy-style decode (high MTP acceptance) is 375-415 per card.
What is left: per-agent selection (3.3 ms/agent/step at 246K int4 even with STATS_SUB 4 + reuse 2) and host misses (35 MB/step)
grow with context x agents; next levers: a fused stats+score scan, larger hot buffer (needs GPU bytes), GDN spec states outside
the pool (would also allow a 5th agent per card).

### Recommended config: 4 x 250K agents per 24 GB card
```
SPARSE_PROFILE=tier MAX_LEN=262144 MAX_SEQS=4 GPU_UTIL=0.94 SPARSE_TIER_HOT_FRAC=0.025 SPARSE_SKETCH_BITS=4
SPARSE_TIER_HOST_BITS=4 SPARSE_TIER_HOST_GB=17.8 SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2
```
(suffix defaults are built in: sparse follow-ups, 16-row question segments, suffix KSEL 4096; prefer-host lookups on.)
Check the boot log for `host pool 17.7 GB (1174 blocks)`, `GPU pool 157 blocks` (GPU1: 158), no `allocated > reserved`.
For 4 x 128K on a card: `MAX_LEN=131072 MAX_SEQS=4 GPU_UTIL=0.93 SPARSE_TIER_HOT_FRAC=0.05 SPARSE_STATS_SUB=4
SPARSE_SELECT_EVERY=2` (+ `SPARSE_TIER_HOST_BITS=4 SPARSE_TIER_HOST_GB=8.8`).

## Known limitations and open findings

1. **Tier with 2 slots: validated on the 3090 only, 3 runs.** Preemption and resume are exercised (1 per run). Not
   validated: the 3090 Ti, more than 2 slots, and long soak. While one user prefills a long prompt, the other user's
   decode drops to a few tok/s; that is chunked-prefill batching, not the tier.
2. **Spill forces always-sparse.** In the `tier` profile short contexts pay the sparse floor: 2-6% slower decode on
   short prompts (always-sparse 4K 150-153 vs dense 154-158 tok/s). The `sparse` profile has no such cost.
3. **Smaller GPU KV pool than dense:**
   - `sparse` at 0.92: 184K tokens, against dense 216K at 0.93. Still above the 153,600 window, at 1.20× concurrency
     instead of 1.40×.
   - `tier` at 0.90: 71K tokens at 2 slots (111K at 1), extended by ~14 GB of host RAM per server.
4. **The first prefill of a context larger than the GPU pool is slow** (177K: ~200 s TTFT). Prefill staging copies all
   blocks per layer per chunk, with a host sync on that path. Follow-ups are fast.
5. **Greedy outputs can differ from dense** after some tokens. That is approximate attention, and MTP acceptance
   shifts with it. The triton select itself is deterministic.
6. **Switch gaps:**
   - two concurrent requests across the B=2 cutoff are untested (only warm-up exercises it);
   - the tier in shadow mode together with the switch is untested.
7. **Fragile scheduler hooks:** the tier's hook is a monkeypatch (wraps `remove_skipped_blocks`, swaps the hit-manager
   class), and so is G-1's (wraps `Worker.determine_available_memory`). Both are fragile across vLLM upgrades.
8. **MTP drafter under spill** sees only the sink block plus the non-spilled tail. Drafts at long range may get worse;
   target outputs stay exact.
9. **Hot-buffer sizing:** real traces miss more often than the synthetic ones (64K: 16% at 10% hot). The first step
   after a new question is a cold burst of ~68 MB.

## v1.1 (2026-11-04)

v1.1 changes the defaults. Knobs, measurements and the rejected list are in `docs/CHANGELOG.md`; the prefill test
tables are in `eval/prefill/README.md`. In short:

| knob | v1.1 default | v1.0 behaviour |
|---|---|---|
| `SPARSE_TIER_SUFFIX_SPARSE` | 0 (exact follow-up prefill) | 1 |
| `SPARSE_TIER_STAGE_SERIAL` | 1 (crash-safe staging) | 0 |
| `SPARSE_STATS_SUB` / `SPARSE_SELECT_EVERY` | 1 / 1 | 4 / 2 in the recommended run line |
| `SPARSE_TIER_HOST_MAMBA_SLOTS` | 32 | 8 |
| `SPARSE_TIER_MAMBA_SAVE_AT_FINISH` | 1 | 0 |
| `SPEC_NO_BLOCK_DROP` (start_qwen.sh, set by serve-sparse.sh) | 1 | 0 |
| `PTHFA_PREFILL_FA2` (pthfa, set by serve-sparse.sh; needs pthfa rebuilt) | 1 | 0 |
| `PTHFA_FLAT_GQA` | 0 (not bit-exact) | n/a |
| `SPARSE_PREFILL_YIELD_STEPS_PER_1K` / `SPARSE_PREFILL_SHARE_CHUNK` | 12 / 864 | n/a (no yield) |
| `SPARSE_PREFILL_YIELD_SHORT_BYPASS` | 2600 | n/a |
| `SPARSE_TIER_PIN_PREFIX_MAX` | 2 (`vllm_xargs {"sparse_pin_prefix": 1}`) | n/a |

The earlier sections that recommend `SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2` and sparse follow-up suffixes describe v1.0.
Those shortcuts cost call-site accuracy (`eval/bench/README.md`).

## v1.2 (2026-11-11)

v1.2 adds the knobs below. Their **code defaults are all off**; `serve-sparse.sh` turns on the ones marked "launcher 1".
Measurements, costs and the list of what was tried and not adopted are in `docs/CHANGELOG.md` and
`docs/OPTIMIZATION_NOTES.md`; method and tables in `eval/v12/README.md`. The full build with all
switches on passed the accuracy gate (call-site F1 0.852 -> 0.859, paired difference +0.007 [-0.016, +0.031],
n = 120). In the 8-agent mix, switches on against off, the long-turn TTFT median went 55.2 -> 41.3 s and
39.8 -> 36.9 s (two seeds), the worst case 130 -> 97 s and 135 -> 96 s, with 0 failed requests.

| knob | code default | `serve-sparse.sh` | notes |
|---|---|---|---|
| `SPARSE_GDN_RECOVER` (vLLM, `sparse-gdn-recover.patch`) | 0 | launcher 1 (when `SPEC=mtp` and `PREFIX_CACHE=1`) | one GDN state block per request under speculative decoding: 27 → 6 pinned blocks per request, 13-17 resident requests per card instead of 3-4. Bit-identical output. Needs `--mamba-cache-mode align` and the Triton GDN decode path (refuses to start otherwise) |
| `SPARSE_TIER_HOT_GLOBAL` | 0 | launcher 1 (when `SPARSE_TIER_SPILL=1`) | one hot pool per sparse layer, keyed by host token and shared by all requests, instead of one hot buffer per request slot. Needs `SPARSE_TIER_SPILL=1` |
| `SPARSE_TIER_HOT_GLOBAL_ROWS` | min(`MAX_SEQS`, 4) × the per-slot rows | not set | rows of the shared pool. At 8 or more concurrent spilled streams the default costs 5-8% decode; the 11 × 90K configuration uses 50176 (double) |
| `SPARSE_TIER_SHARED_PLAN` | 0 | launcher 1 | the block-level residency plan of a decode step is computed by the first sparse layer and read by the others. Spill only. Speed effect not separable from noise |
| `SPARSE_TIER_SHARED_PLAN_CHECK` | 0 | not set | debug: every reader also computes the plan and counts differing entries |
| `SPARSE_SCHEDULER_CLS` (launcher) | n/a | launcher 1 (tier profile) | passes `--scheduler-cls sparsekv.sched.SparseScheduler`: the prefill-yield policy (`PrefillYield.step`) runs from a scheduler class instead of the `Scheduler.schedule` monkeypatch. Same policy object and knobs; follows the server's async-scheduling setting. The short-request bypass still needs `sparse-prefill-yield-bypass.patch` |
| `INDUCTOR_DETERMINISTIC` (launcher; `start_qwen.sh` reads `INDUCTOR_DET`) | n/a | 1 | deterministic Inductor compile: every compile chooses the same reduction-kernel variants, so two builds of the same code give the same output. 0 = the old behaviour. No speed cost measured |
| `SPARSE_PREFILL_BUDGET_SHARE` | 0 | not set | 1: pending prefills share each prefill step's token budget shortest-remaining-first (2: a fairness mode that reserves one block for the oldest prefill; it moves along the same trade-off). Needs `SPARSE_PREFILL_SHARE_CHUNK > 0` and the yield-bypass vLLM hook. Median long-turn wait −55-70%, worst case more than doubled, total output unchanged: use it with the deadline below |
| `SPARSE_PREFILL_BUDGET_AGE` | 256 | not set | age credit of the ordering, in tokens per prefill step waited |
| `SPARSE_PREFILL_BUDGET_MAX_WAIT` | 0 (off) | not set | deadline in seconds: prefills waiting longer go first, oldest first. 45 keeps the tail where it was with a modest median gain; 30 collapses to first-come-first-served; at 90 the tail is back |
| `SPARSE_TIER_MAMBA_EVICT_SUPERSEDED` | 0 | not set | evict superseded host GDN checkpoints first. Measured and **not adopted**: output 98 → 60 tok/s, long TTFT 50 → 91 s (the "superseded" checkpoints are still hit) |
| `SPARSE_PREFILL_DEEP_CHUNK_DEPTH` | 0 (off) | not set | cap prefill chunks beyond this depth while other requests decode. Measured and **not adopted**: within noise |
| `SPARSE_TIER_S5_DMA` (existing) | 0 | not set | copy-engine prefetch for staging. v1.2 measurement: −8% follow-up prefill at an 80K prefix and −2-3% at 30K on the x4 card, 0-1% on the x16 card, output identical; not resolvable in the 8-agent mix |

Without a knob (active once installed): `sparse-parser-qwen3-backports.patch` and `sparse-frontend-backports.patch`
(Qwen3 tool-call parser and detokenizer fixes from upstream vLLM; a repeated tool parameter keeps its first value),
`sparse-int8-scale-nofill.patch` (vLLM #64141).

The larger MTP draft vocabulary (29,670 → 63,510 tokens) is a model directory, not a knob: build an id list with
`sparsekv/tools/mk_draft_vocab_ids.py` from your own text, then a variant directory with
`sparsekv/tools/mk_draft_vocab_variant.sh`, and serve it with `MODEL=<variant dir>`. No id list is shipped.

Configurations (also in the README):

```
MAX_LEN=262144 MAX_SEQS=4  SPARSE_TIER_HOST_GB=17.8 SPARSE_TIER_HOST_MAMBA_SLOTS=8
MAX_LEN=262144 MAX_SEQS=5  SPARSE_TIER_HOST_GB=22   SPARSE_TIER_HOST_MAMBA_SLOTS=10
MAX_LEN=90112  MAX_SEQS=11 SPARSE_TIER_HOST_GB=17.8 SPARSE_TIER_HOST_MAMBA_SLOTS=32 SPARSE_TIER_HOT_GLOBAL_ROWS=50176
```

**Tests (CPU, no GPU; `PYTHONPATH=<tree>/pthfa/python:<tree>`, vLLM importable):** `sparsekv/test_s13_cpu.py`,
`test_s14_sched_cpu.py`, `test_s14_hot_cpu.py`, `test_s15_budget_cpu.py`, `test_s17_combo_cpu.py`, and
`bash bench/test_sparse_apply_stacked.sh` in the HyperQwen tree. GPU: `test_s13_gdn_recover_gpu.py`,
`test_s14_hot_gpu.py`. `test_s15_budget_cpu.py` can also replay its scripted runs against an older revision of
`tier_connector.py` (`SPARSE_TEST_BASE_REV`); that comparison is skipped by default.
