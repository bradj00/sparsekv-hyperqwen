# Changelog

All numbers are from Qwen3.8-27B W4A16 with MTP-7 on the test hardware described in the README: two 24 GB Ampere
cards (RTX 3090 Ti on PCIe x16, RTX 3090 on x4), each serving 4 × 250K agents. All were measured at a fixed power
limit. Both cards are power-bound in prefill, so absolute prefill times depend on the power limit. Per-test method and
tables: [`eval/prefill/README.md`](../eval/prefill/README.md).

## v1.2 (2026-11-11): more agents per card, a larger draft vocabulary, reproducible builds

v1.2 comes out of a two-day pass over everything that might make this stack faster on two 24 GB cards. The working
notes of that pass, with everything that was ruled out and the open items, are in
[`OPTIMIZATION_NOTES.md`](OPTIMIZATION_NOTES.md); method, tables and raw records are in
[`eval/v12/README.md`](../eval/v12/README.md). For these measurements both cards also had the GeForce tensor-core
rate unlock applied (see the notes), and every comparison is base against variant on the same card.

Two results for the full v1.2 build (all switches on) were still open when this was written:
- accuracy gate: call-site F1 0.852 -> 0.859, paired difference +0.007 [-0.016, +0.031], n = 120, 75/120 outputs identical: pass
- 8-agent mix: long-turn TTFT median 55.2 -> 41.3 s and 39.8 -> 36.9 s (two seeds), max 130 -> 97 s and 135 -> 96 s, output tokens +3% and +16%, 0 failed requests

### Default configuration

`serve-sparse.sh` now also sets:

```
SPARSE_GDN_RECOVER=1  SPARSE_TIER_HOT_GLOBAL=1  SPARSE_TIER_SHARED_PLAN=1  SPARSE_SCHEDULER_CLS=1
INDUCTOR_DETERMINISTIC=1
```

- Unlike v1.1, the **code defaults of every new knob are off**. Only the launcher turns these on, so a server started
  another way, or with a knob set to 0, behaves as v1.1.
- `SPARSE_GDN_RECOVER` is a vLLM-side knob (`patches/sparse-gdn-recover.patch`). The launcher turns it on only for
  its own baseline (`SPEC=mtp`, `PREFIX_CACHE=1`), which is what recover mode needs.
- Not set by the launcher: budget sharing (`SPARSE_PREFILL_BUDGET_SHARE`, optional and small), the copy-engine
  prefetch (`SPARSE_TIER_S5_DMA`, optional), the pool size for many slots (`SPARSE_TIER_HOT_GLOBAL_ROWS`), and the two
  knobs that were measured and not adopted (`SPARSE_TIER_MAMBA_EVICT_SUPERSEDED`, `SPARSE_PREFILL_DEEP_CHUNK_DEPTH`).
- The larger draft vocabulary is not a knob: it is a model directory you build
  (`sparsekv/tools/mk_draft_vocab_ids.py`, `mk_draft_vocab_variant.sh`) and pass as `MODEL`.
- The parser and detokenizer backports and the int8 scale-view fix are vLLM patches without a knob. They are active
  once installed.
- The two configurations to run (4 or 5 × 250K, and 11 × 90K) are in the README's Run section.

### Headline results

| change | knob | measured effect |
|---|---|---|
| One GDN state block per request under MTP | `SPARSE_GDN_RECOVER=1` | pinned blocks per request **27 → 6**; seat budget **3-4 → 13-17** resident requests per card; output bit-identical |
| The same, 8 agents on one card | | short follow-up TTFT **17.1 → 1.5-1.9 s**; medium follow-ups **25.2 → 2.3-2.8 s** |
| A fifth 250K agent per card | `MAX_SEQS=5` | fits, recall 20/20; capacity, not throughput: combined output 232/278 tok/s (explain/copy) at five against 241/332 at four |
| Larger MTP draft vocabulary, 29,670 → 63,510 tokens | `MODEL=<variant dir>` | tokens per step at 4 streams **3.6-3.7 → 4.0** (78K documents), **3.6 → 4.3** at 4 × 250K; 4 × 250K combined output 230/257 → **241/332 tok/s** (+5% / +29%) |
| Accuracy gate for the draft vocabulary | | call-site F1 0.837 → 0.852, **+0.015 [−0.007, +0.038]**, n = 120: no loss |
| One shared hot buffer per layer | `SPARSE_TIER_HOT_GLOBAL=1` | reserve **2,275 → 862 MB** at the default pool size; GPU KV pool **108 → 154 blocks (+43%)**, or 124 blocks (+15%) with the doubled pool of the 11-slot configuration |
| The same, decode at 4 streams | | steady 266/390 → 282/491 tok/s; combined 207/255 → 228/310; follow-up TTFT median 2.2 → 1.4 s; level at 8 streams |
| Accuracy gate for the shared hot buffer (with the smaller pieces, at 4 × 262K) | | call-site F1 0.852 → 0.862, **+0.010 [−0.011, +0.033]**, n = 120: no loss |
| Deterministic compile | `INDUCTOR_DETERMINISTIC=1` | two fresh compiles and a run with poisoned free memory bit-identical; no speed cost (cold prefill 67 s both ways) |
| Full v1.2 build, all switches on: accuracy gate | | call-site F1 0.852 -> 0.859, paired difference +0.007 [-0.016, +0.031], n = 120, 75/120 outputs identical: pass |
| Full v1.2 build, all switches on: 8-agent mix | | long-turn TTFT median 55.2 -> 41.3 s and 39.8 -> 36.9 s (two seeds), max 130 -> 97 s and 135 -> 96 s, output tokens +3% and +16%, 0 failed requests |

Costs, which you should weigh:
- **The fifth 250K agent is capacity only.** Each agent is 12-27% slower at five than at four, and 5 × 250K pins
  about 22 GB of host RAM per card (the int4 host tier costs about 17.4 KB per token).
- **The shared hot buffer at its default size costs 5-8% decode at 8 or more concurrent spilled streams.** The
  11-slot configuration therefore doubles the rows (`SPARSE_TIER_HOT_GLOBAL_ROWS=50176`), which gives back part of
  the pool gain (124 blocks instead of 154).
- **A different draft vocabulary changes the output text**, because draft acceptance is probabilistic in this build.
  That is why it was gated. The full head accepts slightly more but is slower at one stream; 64K is the best point
  measured.
- **Tool-call parsing changes in one case:** a repeated tool parameter now keeps its first value in both streamed
  and non-streamed calls.
- The 250K figures were measured with `SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2`. The gated defaults are 1 / 1, so
  expect slightly lower decode there. v1.2 speed at 250K has not been measured.

### Details

**`SPARSE_GDN_RECOVER=1` (`patches/sparse-gdn-recover.patch`; new vLLM file
`model_executor/layers/mamba/gdn/recover.py`).**
- Stock align-mode speculative decoding stores the recurrent state after each of the K draft tokens in its own block.
  Every resident request therefore holds K extra state blocks per mamba group: with MTP-7 and three groups, 21 of
  the 27 blocks a request pins. On a pool of about 110 blocks that is 3-4 resident requests, and production logs
  showed requests waiting for a seat 43% of active time.
- Recover mode keeps one state block per group. The verify step reads the running state and stores nothing; the
  per-token inputs of the window are kept in a small buffer; after sampling, a kernel replays the accepted tokens
  onto the running state and writes the block-aligned checkpoint when one is crossed.
- Output is bit-identical to the baseline: kernel test on bf16/fp16/fp32 states
  (`sparsekv/test_s13_gdn_recover_gpu.py`), and a character-identical greedy A/B.
- 4 × 250K: a fifth 250K agent now fits on a card.

**Larger MTP draft vocabulary (`sparsekv/tools/mk_draft_vocab_ids.py`, `mk_draft_vocab_variant.sh`).**
- The draft head is sliced from the quantized lm_head. A draft token outside the list ends the draft chain, so a
  larger list accepts more per step.
- The list was built from token-frequency coverage: the ids already in use are kept, and the most frequent tokens
  of a workload-like corpus are added. No list ships with this repository, because part of that corpus was private
  agent traffic. The tools build one from your own text and make a model-directory variant from it without touching
  the source directory.

**`SPARSE_TIER_HOT_GLOBAL=1` (`sparsekv/tier.py`, `sparsekv/tier_connector.py`).**
- The hot buffer that caches host-tier rows on the GPU was one buffer per request slot. With 11 slots that reserved
  2.3 GB of a 7.9 GB KV budget, most of it idle. A single pool per layer, keyed by host token and shared by all requests,
  serves the same traffic from less memory.
- Pool size: `SPARSE_TIER_HOT_GLOBAL_ROWS`, default 4 × the per-slot rows.
- Hot-buffer miss rate with the doubled pool, 78K documents on the x4 card: 15-19% → 2.8% at one stream,
  13% → 6.7% at four, 14.5% → 12% at eight.
- Part of the copy-task gain at 4 streams is higher draft acceptance in those runs. The explain figures and the
  miss rates are the steadier evidence.
- It needs `SPARSE_TIER_SPILL=1`.

**Smaller pieces.**
- **Shared residency plan** (`SPARSE_TIER_SHARED_PLAN=1`): the block-level residency tables of a decode step are
  computed once instead of once per sparse layer and gather. 0 mismatches in a checked run
  (`SPARSE_TIER_SHARED_PLAN_CHECK=1` is the debug check); the speed effect is not separable from noise.
- **Prefill yield as a scheduler class** (`SPARSE_SCHEDULER_CLS=1`, `sparsekv/sched.py`): the duty-cycle policy runs
  as a vLLM `--scheduler-cls` subclass instead of an import-time monkeypatch. Identical decisions on 6,720 scripted
  steps and identical output; robustness only.
- **Deterministic compile** (`INDUCTOR_DETERMINISTIC=1`, launcher and `start_qwen.sh`). Two builds of the same code
  produced different logits (up to 0.3 nats on log-probabilities), each bit-reproducible run to run. Inductor times
  two block-size variants of each fused RMS-norm reduction kernel at compile time and bakes the winner into vLLM's
  cached compile artifact; the variants sum in a different order. The launcher now passes
  `"inductor_compile_config":{"deterministic":true,"combo_kernels":true,"benchmark_combo_kernel":false}`, so every
  compile chooses the same variants. Neither variant is the "correct" one; what changes is that a greedy A/B between
  two trees now means something. Steady decode was 266/421 against 256/405 tok/s without it.
- **int8 scale-view fix** (`patches/sparse-int8-scale-nofill.patch`, vLLM #64141): creating the per-token-head scale
  views no longer fills them, which wrote 1.0 over shared KV pages at first use. No output change on the greedy A/B.
- **Qwen3 tool-call parser and detokenizer backports** (`patches/sparse-parser-qwen3-backports.patch`,
  `patches/sparse-frontend-backports.patch`; vLLM #62027, #61217, #59882, #61253, #64675, #65775, #63668): tool
  arguments stream incrementally, and malformed UTF-8 output no longer costs quadratic time. Parser suite and a live
  tool-call test pass.
- **Budget sharing with a deadline** (`SPARSE_PREFILL_BUDGET_SHARE=1 SPARSE_PREFILL_BUDGET_MAX_WAIT=45`, off by
  default). Pending prefills are served shortest-remaining-first inside each prefill step; requests waiting longer
  than the deadline go first, oldest first. It reorders, it does not add capacity:

  | 8 agents, Ti | off | shortest-first | + 45 s deadline | + 90 s deadline |
  |---|---|---|---|---|
  | long-turn TTFT, median | 52.4 s | 22.0 s | 45.8 s | 29.5 s |
  | long-turn TTFT, max | 104 s | 280 s | 107 s | 201 s |
  | output tokens in 12 min | 77,235 | 82,390 | 74,458 | 76,886 |

  Without a deadline the median falls 55-70%, but the worst case more than doubles and total output does not
  change. A 45 s deadline keeps the tail where it was and keeps a modest median gain (−13% and −35% on two seeds).
  `SPARSE_PREFILL_BUDGET_AGE` is the age credit of the ordering.
- **Copy-engine prefetch** (`SPARSE_TIER_S5_DMA=1`, existing switch, still off by default): −8% follow-up prefill
  at an 80K prefix and −2-3% at 30K on the x4 card, 0-1% on the x16 card, output identical. Not resolvable in the
  8-agent mix.
- **Installer:** `sparsekv/install.sh` and `patches/apply.sh --sparse` now handle sparse patches that stack on one
  file (the two parser patches do). Re-installing no longer leaves reject files. `apply.sh --sparse-state DIR`
  reports what is applied. Test: `bench/test_sparse_apply_stacked.sh`.

**New tests.**
- CPU: `sparsekv/test_s13_cpu.py` (superseded-first eviction), `test_s14_sched_cpu.py` (scheduler class, async
  scheduling), `test_s14_hot_cpu.py` (shared hot buffer and residency plan against the reference),
  `test_s15_budget_cpu.py` (budget sharing), `test_s17_combo_cpu.py` (all knobs together),
  `bench/test_sparse_apply_stacked.sh` (installer).
- GPU: `sparsekv/test_s13_gdn_recover_gpu.py`, `test_s14_hot_gpu.py`.
- Simulations (CPU): `sparsekv/tier_sim_s14.py` (per-slot buffers against one pool), `sim_s15_budget.py`.

### Tried and not adopted

Two of these ship as knobs, off: `SPARSE_TIER_MAMBA_EVICT_SUPERSEDED` and `SPARSE_PREFILL_DEEP_CHUNK_DEPTH`. The
full list with the clean checks is in [`OPTIMIZATION_NOTES.md`](OPTIMIZATION_NOTES.md).

| idea | result |
|---|---|
| Evicting superseded GDN checkpoints first (`SPARSE_TIER_MAMBA_EVICT_SUPERSEDED=1`) | output 98 → 60 tok/s, long TTFT 50 → 91 s: the "superseded" checkpoints are still hit |
| Capping prefill chunks at depth (`SPARSE_PREFILL_DEEP_CHUNK_DEPTH`) | within noise |
| Depth-weighted prefill yield (`SPARSE_PREFILL_YIELD_CTX_UNIT`) | long-turn median 52 → 64 s, output −12% |
| Prefill yield 12 → 6 → 3 steps per 1K | long TTFT 54 → 40 → 25 s, per-stream decode 55 → 30 → 21 tok/s; a preference, left as configured |
| Int8 activations for prefill (W4A8 Marlin path) | MLP layers only: prefill 7-9% slower; all layers: no change. With the tensor-core unlock fp16 already runs at full rate |
| Padding int8 KV slots to 16 bytes (vLLM #62113) | bit-identical, 1.6-2.3× on the Triton attention kernels, which are not on this stack's path: pthfa decode flat, pthfa 8-token verify about 20% slower |
| Rewriting the GDN prefill kernels | GDN kernels plus conv are 3-5% of prefill here |
| Shorter drafts (K = 5, 3) | per-stream decode 60 → 43 → 35 tok/s |
| Dequantising host blocks once into borrowed free pool blocks | −1 to −4% on an idle server, less under load; the copy-engine prefetch does better alone |
| Sparse attention in prefill at 90K windows | estimated 0-0.7% on follow-ups; an earlier sparse suffix prefill cost 9-14 points of call-site F1 |

### Where follow-up prefill time goes

Prefill of a cache-hit follow-up turn, per 864-token block of new tokens (3090, one request): quantized GEMM is
83% / 56% / 36% at a 0-5K / 30K / 80K prefix, attention 4% / 25% / 36%, host-tier staging 0 / 9% / 20%. At 8 agents
per card, long follow-ups (more than 2,600 new tokens, about three quarters of turns) wait a median of 40-50 s.
That wait is queueing behind the prefill/decode duty cycle and behind other prefills. Nothing in v1.2 adds prefill
capacity at that load.

### Upgrading from v1.1

1. In the HyperQwen tree, restore the files the v1.1 diff touched: `git checkout -- .gitignore launcher_common.sh
   patches/apply.sh patches/series single-user/start_qwen.sh pthfa/install.sh pthfa/csrc/pth_attn
   pthfa/python/pth_flash_attn/vllm_backend.py`.
2. Then `git apply` the new `patches/hyperqwen-base.diff`. Against v1.1 it changes `patches/apply.sh` (stacked sparse
   patches, `--sparse-state`), `patches/series` (four new entries) and `single-user/start_qwen.sh` (the compile
   option).
3. Copy `sparsekv/`, `pthfa/`, `serve-sparse.sh`, `patches/sparse-*.patch` and `bench/test_sparse_apply_stacked.sh`
   as in the README install. The pthfa sources did not change, so pthfa needs no rebuild.
4. Re-run `bash sparsekv/install.sh venv`. It applies the four new vLLM patches and installs `sparsekv/sched.py`.
5. The compile artifact is keyed per source tree and configuration, so expect a fresh compile at the first start.

To get the v1.1 behaviour back from the environment (the patches without a knob stay active):

```
SPARSE_GDN_RECOVER=0 SPARSE_TIER_HOT_GLOBAL=0 SPARSE_TIER_SHARED_PLAN=0 SPARSE_SCHEDULER_CLS=0
INDUCTOR_DETERMINISTIC=0
```

## v1.1 (2026-11-04): prefill and multi-agent cache reuse

v1.1 is about the time agents wait for prefill: re-prefills after tool calls, follow-up turns, cold starts, and short
turns queued behind a long prefill. Every change below was validated on GPU on 2026-11-04, and the configuration
below is the one now in production use.

### Default configuration

`serve-sparse.sh` and the code defaults are now:

```
SPARSE_TIER_SUFFIX_SPARSE=0  SPARSE_TIER_STAGE_SERIAL=1  SPARSE_STATS_SUB=1  SPARSE_SELECT_EVERY=1
SPARSE_TIER_HOST_MAMBA_SLOTS=32  SPEC_NO_BLOCK_DROP=1  SPARSE_TIER_MAMBA_SAVE_AT_FINISH=1  PTHFA_PREFILL_FA2=1
SPARSE_PREFILL_YIELD_SHORT_BYPASS=2600  SPARSE_TIER_PIN_PREFIX_MAX=2
```

- `SPEC_NO_BLOCK_DROP` and `PTHFA_PREFILL_FA2` are HyperQwen-side knobs (start_qwen.sh and pthfa). Their code default
  stays off, so a dense HyperQwen server is unchanged; `serve-sparse.sh` turns them on.
- The sparse-tier knobs default to the values above in code (`sparsekv/tier_connector.py`, `sparsekv/triton_ops.py`)
  and in the launcher.
- For 4 × 250K per card, use the README's Run line (it sets the slot sizing).

### Headline results

| change | knob (v1.1 default) | measured effect |
|---|---|---|
| More host GDN checkpoint slots | `SPARSE_TIER_HOST_MAMBA_SLOTS` 8 → **32** | full re-prefills after tool waits: **43% of prefill → 0**; per-step prefill **6.4 → 4.0 s** |
| Keep the last prefix-cache block on a hit | `SPEC_NO_BLOCK_DROP=1` | **+864 cached tokens** per follow-up; follow-up TTFT **2.56 → 1.80 s** (3090), **2.82 → 1.93 s** (Ti) |
| Host-save the decode-boundary GDN state at finish | `SPARSE_TIER_MAMBA_SAVE_AT_FINISH=1` | turn-2 TTFT after eviction **5.75 → 2.06 s** (Ti), **4.84 → 2.02 s** (3090) |
| Accuracy gate for the two rows above, together | | call-site F1 pooled **+0.015 [−0.010, +0.039]**, n = 180 (F1 on a 0–1 scale): no loss |
| FlashAttention-2 prefill over the int8 cache | `PTHFA_PREFILL_FA2=1` | cold ~240K on the Ti **397 → 337 s**; 128K **−9%**; 64K **−7%** |
| Accuracy gate for FA2 prefill | | shard 0 **+0.9 [−4.0, +6.4]**, shard 1 **−3.6 [−8.3, +0.7]**, pooled ≈ **−1.4 F1**: within noise |
| Short requests bypass the prefill yield | `SPARSE_PREFILL_YIELD_SHORT_BYPASS=2600` | 4-agent realistic mix, median TTFT over all follow-ups: Ti **9.8 → 7.1 s**, 3090 **8.3 → 6.7 s** |
| Shared stable prefix (harness side) + pin knob | `SPARSE_TIER_PIN_PREFIX_MAX=2` | first-request common prefix **7,262 → 16,026 tokens**; **5/8 builds hit 15,552 cached tokens** on their first step |

Costs of the yield bypass, which you should weigh:
- decode throughput during the mix is **4-10% lower** (Ti 82.7 → 74.8, 3090 82.1 → 79.0 tok/s per request);
- in one replay, the Ti's **64K cold prefill took 35% longer** under contention (114.8 → 155.1 s). The bypass serves
  short requests first and pushes the long prefill back, by design. On the 3090 the same replay showed no change
  (112.8 → 110.7 s).

### Details

**Host GDN checkpoint slots: 8 → 32.**
- In this hybrid model a prefix-cache hit also needs the recurrent (GDN) state at the hit boundary. With 8 host slots,
  the LRU dropped an agent's checkpoint while the agent waited on a tool. Its next turn then re-prefilled the whole
  context, even though the KV was still cached.
- In ~16 h of agent traffic on 8 slots (8,102 steps), these full re-prefills were 43.3% of prefill GPU time. In a
  51-minute window on 32 slots (731 steps, larger median prompts) there were none, and the per-step prefill time
  fell from 6.41 to 4.02 s.
- Cost: 85.3 MB of pinned host RAM per slot, so 2.7 GB per server at 32 slots (0.68 GB at 8). The 32-slot window
  is short. As contexts grow toward the window limit, host KV eviction could bring full misses back.

**`SPEC_NO_BLOCK_DROP=1` (start_qwen.sh: `"disable_eagle_block_drop": true`).**
- On every prefix hit, vLLM dropped the trailing 864-token block for the MTP drafter. So each follow-up stopped one
  block short of the previous prompt and re-prefilled it.
- With the knob on, every follow-up hit exactly `floor(P_prev/864)·864` tokens, 864 more than before.
- Target outputs stay exact, because speculative decoding verifies every draft. The knob does move the prefill
  chunk and checkpoint split, so greedy text differs from the first token, through numerics, even on identical cold
  prompts. That is why it was accuracy-gated. MTP acceptance on 35K-token replays was 3.61 with the drop on and
  3.65 with it off, so there was no measurable loss.

**`SPARSE_TIER_MAMBA_SAVE_AT_FINISH=1`.**
- Before, host GDN saves kept only prompt-end boundaries. The previous turn's generated tokens (reasoning and tool
  call) were therefore re-prefilled whenever the GPU copy had been evicted.
- Now, at request finish, the newest decode-boundary state is host-saved. The connector keeps one per request, or
  two when the eagle block drop is on. Saves that may not be final yet are cancelled, and their holds released.
- In the test, turn 2 after eviction hit the last decode boundary (+2.6-3.5K tokens). The host-saved states were
  bit-identical to the GPU copies at the same boundary (4/4).

**`PTHFA_PREFILL_FA2=1` (`pthfa/python/pth_flash_attn/fa2_prefill.py`).**
- The prefill launches of the 16 full-attention layers (D = 256, causal) run FlashAttention-2 varlen. The KV comes
  from the int8 cache, dequantised per KV piece into a 16-bit workspace (128 MiB at the default 32K piece), and the
  pieces are merged by log-sum-exp.
- Decode, MTP verify and the sparse decode path are unchanged.
- Numerics: in the fp16 rounding class (K/V ~2^-12 relative), not bit-identical to pthfa. Hence the gate.
- FA2 is called through vLLM's `vllm.vllm_flash_attn`; no FlashAttention code is vendored.
- Requires rebuilding pthfa from the patched tree (see Upgrading below).
- Cold TTFT, Ti / 3090: 64K −7.4% / −5.2%, 128K −9.3% / −6.5%, ~240K −15.2% / −13.7%.

**`PTHFA_FLAT_GQA=1` (E1, shipped off).**
- Flat (token, head) row packing in pthfa prefill: 64 of 64 MMA rows used instead of 60.
- Kernel time −7% to −10% (Ti) and −5% to −8% (3090); cold TTFT −2% to −6%.
- **Off by default: not bit-exact.** Kernel check: up to ~5e-4 max abs difference on prefill shapes. It was not
  accuracy-gated, and FA2 replaces this path anyway.

**`SPARSE_PREFILL_YIELD_SHORT_BYPASS=2600` (vLLM hook: `patches/sparse-prefill-yield-bypass.patch`).**
- The prefill yield (see below) makes a long prefill share the GPU with running decodes through decode-only steps.
  Before v1.1, a short follow-up arriving during those steps also waited.
- Now a request whose uncached prompt is ≤ N tokens is still scheduled on a decode-only step, up to N such tokens
  per step.
- The bypassed tokens are added to the decoders' debt. So the decode duty cycle is kept, and the long prefill is
  pushed back (shortest remaining first).
- There is no bypass while the debt is above `SPARSE_PREFILL_YIELD_SHORT_MAX_DEBT` (240 steps), so a stream of short
  requests cannot hold a long prefill forever.
- N = 1024 was inconclusive. N = 1728 cut short-follow-up TTFT the most but made long follow-ups' p90 worse. N = 2600
  was best on the realistic mix.
- Without the vLLM hook the knob is ignored, with a log line.

**`SPARSE_TIER_PIN_PREFIX_MAX=2` (pinned warm prefix).**
- A request sent with `vllm_xargs {"sparse_pin_prefix": 1}` pins, at finish, its host GDN checkpoint and the host KV
  blocks below it. They are kept out of LRU eviction, so every later request with the same prefix hits it however
  much traffic passes. `{"sparse_pin_prefix": "clear"}` releases all pins, and the oldest pin is released beyond the
  maximum.
- Cost per pin: one GDN slot (85 MB) plus the prefix's host KV blocks (~0.3 GB at 16.5K tokens, int4).
- Test: 8 agents' real first requests (common prefix 16,026 tokens), warmed with one 15,552-token request. After 20
  minutes of unrelated traffic (187 requests, 6.9M prompt tokens) all 8 hit 15,552 cached tokens, with TTFT ≤ 1.11 s.
- How to make your agents share a prefix in the first place: [`AGENT_HARNESS_TIPS.md`](AGENT_HARNESS_TIPS.md).

**Also new in this repo with v1.1.** These were found and validated during the full benchmark
([`eval/bench/README.md`](../eval/bench/README.md#found-and-fixed-during-the-run)), but the v1.0 code did not
carry them:
- **Fixed configuration, accuracy level with dense.** Follow-up prefill is exact (`SPARSE_TIER_SUFFIX_SPARSE=0`) and
  the two selection shortcuts are off (`SPARSE_STATS_SUB=1`, `SPARSE_SELECT_EVERY=1`). Call-site F1 vs dense goes
  from −13.7 [−21.6, −6.9] (v1.0 as shipped) to −2.1 [−7.1, +2.3] (1 agent) and −1.3 [−3.9, +1.3] (4 agents).
- **Crash-safe staging** (`SPARSE_TIER_STAGE_SERIAL=1`). Each dense prefill request is staged alone, so staging
  memory stays within its reserve. v1.0 could run out of memory when several cold prefills or follow-ups landed in
  one step at 4 agents. Test: `sparsekv/test_s5_cpu.py`. About 5% time cost at 4 agents.
  - Experimental, off: `SPARSE_TIER_SUFFIX_EXACT`, `SPARSE_TIER_S5_DMA`, `SPARSE_TIER_SUFFIX_STATS_SUB`.
- **Prefill yield.** After a step with n prefill tokens while others decode, the next `ceil(n/1024 × 12)` steps are
  decode-only, and prefill chunks are capped at 864 tokens while requests wait. This stops one agent's long prefill
  from starving the others' decode (2-5 tok/s for 30-60 s before).
  - Knobs: `SPARSE_PREFILL_YIELD_STEPS_PER_1K` (12, 0 = off), `SPARSE_PREFILL_SHARE_CHUNK` (864, 0 = off).
- **Small fix:** `sparsekv/tier_connector.py` imports again without vLLM (docs and tests).

**New tests (CPU, no GPU):**
- `sparsekv/test_s5_cpu.py`: staging;
- `test_s7_cpu.py`: decode-boundary saves;
- `test_s9_cpu.py`: bypass and pinned prefix;
- `test_fa2_prefill_cpu.py`: FA2 piece decomposition and LSE merge against an fp64 reference;
- `test_s5_gpu.py` (GPU).

### Rejected or not shipped

| idea | result | why not |
|---|---|---|
| Sage-style int8-QK prefill (hd256 port) | cold −9.3% vs FA2 (232K: Ti 311 → 282 s, 3090 271 → 245 s) | call-site F1 vs FA2 **−4.8 [−8.8, −1.3]**, n = 120 (192K bin −9.3) |
| Hierarchical (two-level) select | 4 × 246K: 17.8 vs 16.0 ms/step; 64K: 2× slower | not faster: the select kernels are launch-bound (~1 ms per layer fixed) |
| Decode prefetch of the next layer's picks | covers ~15% of misses (73% of prefetched rows unused) | worth ≤ 1-2% of decode |
| Disk session tier | read-back after a restart is broken (resume returned 0 cached tokens) | not shipped |

### Where cold prefill time goes

Measured at a fixed power limit, cold 244K prefill on the Ti:
- full attention (16 layers) is **~63%**;
- the Marlin W4A8 linears are **~29%**;
- everything else is under 10%.

The attention share grows with length: 31% at 64K, 47% at 128K, 63% at 244K. Both cards are power-bound in prefill:
attention reaches about half of the FP16 tensor peak at the clocks the power limit allows. That is why FA2's more
efficient prefill helped, while ideas that only add FLOP throughput help less than their kernel speedups suggest.

### Upgrading from v1.0

1. In the HyperQwen tree, restore the five files the v1.0 diff touched: `git checkout -- .gitignore patches/apply.sh
   patches/series pthfa/python/pth_flash_attn/vllm_backend.py single-user/start_qwen.sh`.
2. Then `git apply` the new `patches/hyperqwen-base.diff`. It now also touches `launcher_common.sh`, `pthfa/install.sh`
   and `pthfa/csrc/pth_attn/*`.
3. Copy `sparsekv/`, `pthfa/` (now with `fa2_prefill.py`), `serve-sparse.sh` and `patches/sparse-*.patch` as in the
   README install.
4. Rebuild pthfa from the tree (`PY=venv/bin/python bash pthfa/install.sh`). Its stamp now hashes the sources, so
   the rebuild is not skipped.
5. Re-run `bash sparsekv/install.sh venv`. It reverses the v1.0 patches it applied, then applies the new series,
   including `sparse-prefill-yield-bypass.patch`.

To get the v1.0 behaviour back from the environment:

```
SPARSE_TIER_SUFFIX_SPARSE=1 SPARSE_TIER_STAGE_SERIAL=0 SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2
SPARSE_TIER_HOST_MAMBA_SLOTS=8 SPARSE_TIER_MAMBA_SAVE_AT_FINISH=0 SPEC_NO_BLOCK_DROP=0 PTHFA_PREFILL_FA2=0
SPARSE_PREFILL_YIELD_SHORT_BYPASS=0 SPARSE_TIER_PIN_PREFIX_MAX=0 SPARSE_PREFILL_YIELD_STEPS_PER_1K=0
SPARSE_PREFILL_SHARE_CHUNK=0
```

## v1.0

The initial release:
- query-aware sparse decode;
- the host-RAM int4 KV tier with zero-copy misses;
- per-step dense/sparse switch;
- multi-agent scheduling fixes.

The README's throughput headline was measured with its shortcuts on (`SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2`,
sparse follow-up prefill). See `sparsekv/RELEASE.md`.
