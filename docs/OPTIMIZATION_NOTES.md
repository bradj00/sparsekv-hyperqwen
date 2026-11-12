# Optimization notes, 2026-11-10 to 2026-11-11

Working notes from a two-day pass over everything that might make this stack faster on two 24 GB Ampere cards: what
was tried, what it measured, what was kept, and what was ruled out. The changes that were kept are released as v1.2
([`CHANGELOG.md`](CHANGELOG.md)). Work is paused here; the open items at the end are where to pick it up.

## Test conditions

- Model: Qwen3.8-27B W4A16 with MTP-7, pthfa int8 KV, sparse decode, int4 host-RAM tier.
- Cards: RTX 3090 Ti on PCIe x16 ("Ti") and RTX 3090 on PCIe x4 ("3090"), 24 GB each, one server per card.
- Both cards ran at a fixed power limit and with the GeForce tensor-core rate unlock
  ([gpu-unlock](https://github.com/Zagitta/gpu-unlock)) applied, so fp16/bf16 matrix math runs at full rate. Several
  results below depend on that (see "Int8 activations").
- Every comparison is base against variant on the same card, one server at a time. At identical settings the Ti was
  about 20% slower than the 3090 at cold prefill (78K-token document: 67 s against 54 s) and level or slightly ahead
  at decode, so numbers are never compared across cards.
- Harnesses: a decode harness (N streams each holding its own 78K or 246K-token document; explain and copy tasks;
  needle and code-line recall), a 12-minute realistic agent mix (8 agents per card, measured turn sizes and tool
  gaps), a greedy A/B over 8 turns for output identity, and the cross-file call-site accuracy gate of
  [`eval/bench/`](../eval/bench/README.md) (120 items, paired bootstrap).
- Noise: decode harness 3-5%; agent-mix medians about 10-15% run to run (controls for one card ranged from 41.6 to
  52.4 s on the long-turn median). Copy-task speed is bimodal because it follows draft acceptance.

## What was kept

### 1. One GDN state block per request under speculative decoding (`SPARSE_GDN_RECOVER=1`)

Stock align-mode speculative decoding stores the recurrent state after each of the K draft tokens in its own block,
so every resident request holds K extra state blocks per mamba group: with MTP-7 and three groups, 21 of the 27
blocks a request pins. On a pool of about 110 blocks that is 3-4 resident requests, and production logs showed
requests waiting for a seat 43% of active time.

Recover mode keeps one state block per group. The verify step reads the running state and stores nothing; the
per-token inputs of the window are kept in a small buffer; after sampling, a kernel replays the accepted tokens onto
the running state and writes the block-aligned checkpoint when one is crossed.

- Pinned blocks per request: 27 -> 6. Seat budget: 3-4 -> 13-17 resident requests per card.
- Output: bit-identical to the baseline (kernel test on bf16/fp16/fp32 states, and a character-identical greedy A/B).
- 8 agents on one card: short follow-up TTFT 17.1 -> 1.5-1.9 s; medium follow-ups 25.2 -> 2.3-2.8 s.
- 4 x 250K: a fifth 250K agent now fits on a card (recall 20/20). It is capacity, not throughput: combined output
  232/278 tok/s (explain/copy) at five against 241/332 at four, each agent 12-27% slower.

### 2. Larger MTP draft vocabulary (29,670 -> 63,510 tokens)

The draft head is sliced from the quantized lm_head. A draft token outside the list ends the draft chain, so a
larger list accepts more per step.

- Tokens per step at 4 streams: 3.6-3.7 -> 4.0 (78K documents); 3.6 -> 4.3 at 4 x 250K.
- 4 x 250K, same card: combined output 230/257 -> 241/332 tok/s (explain/copy), +5% / +29%.
- The full head accepts slightly more but is slower at one stream; 64K is the best point measured.
- This changes output text (draft acceptance is probabilistic in this build), so it went through the accuracy gate:
  call-site F1 0.837 -> 0.852, paired difference +0.015 [-0.007, +0.038], n = 120.

### 3. One shared hot buffer per layer (`SPARSE_TIER_HOT_GLOBAL=1`)

The hot buffer that caches host-tier rows on the GPU was one buffer per request slot. With 11 slots that reserved
2.3 GB of an 7.9 GB KV budget, most of it idle. A single pool per layer, keyed by host token and shared by all
requests, serves the same traffic from less memory.

- Reserve 2,275 -> 862 MB at the default pool size (4 x the per-slot rows): GPU KV pool 108 -> 154 blocks (+43%).
  At 8 or more concurrent spilled streams the default size costs 5-8% decode, so the 11-slot configuration uses
  double the rows (`SPARSE_TIER_HOT_GLOBAL_ROWS=50176`): 124 blocks (+15%).
- Hot-buffer miss rate with the doubled pool, 78K documents on the x4 card: 15-19% -> 2.8% at one stream,
  13% -> 6.7% at four, 14.5% -> 12% at eight.
- Decode at 4 streams: steady 266/390 -> 282/491 tok/s, combined 207/255 -> 228/310; follow-up TTFT median
  2.2 -> 1.4 s. Level at 8 streams. Part of the copy gain is higher draft acceptance in those runs; the explain
  figures and the miss rates are the steadier evidence.
- Accuracy gate (with the shared plan, scheduler class and scale-view fix of item 4, at 4 x 262K): F1 0.852 -> 0.862, +0.010 [-0.011, +0.033], n = 120.

### 4. Smaller pieces

- **Shared residency plan** (`SPARSE_TIER_SHARED_PLAN=1`): the block-level residency tables of a decode step are
  computed once instead of once per sparse layer and gather. 0 mismatches in a checked run; speed effect not
  separable from noise.
- **Prefill yield as a scheduler class** (`SPARSE_SCHEDULER_CLS=1`): the duty-cycle policy runs as a vLLM
  `--scheduler-cls` subclass instead of an import-time monkeypatch. Identical decisions on 6,720 scripted steps and
  identical output; robustness only.
- **Deterministic compile** (on by default in the launcher): see "Output differs between builds" below. No speed
  cost (cold prefill 67 s both ways; steady decode 266/421 against 256/405 tok/s).
- **int8 scale-view fix** (vLLM #64141): creating the per-token-head scale views no longer fills them, which wrote
  1.0 over shared KV pages at first use. No output change on the greedy A/B.
- **Qwen3 tool-call parser and detokenizer backports** (vLLM #62027, #61217, #59882, #61253, #64675, #65775,
  #63668): tool arguments stream incrementally; a repeated parameter keeps its first value in both streamed and
  non-streamed calls; malformed UTF-8 output no longer costs quadratic time. Parser suite and a live tool-call test
  pass.
- **Budget sharing with a deadline** (`SPARSE_PREFILL_BUDGET_SHARE=1 SPARSE_PREFILL_BUDGET_MAX_WAIT=45`, off by
  default): see the next section.
- **Copy-engine prefetch** (`SPARSE_TIER_S5_DMA=1`, existing switch, off by default): -8% follow-up prefill at an
  80K prefix and -2-3% at 30K on the x4 card, 0-1% on the x16 card, output identical. Not resolvable in the
  8-agent mix.

## The v1.2 build as a whole

All of the above in one tree, with the recommended switches on (hot pool at the doubled size, shared plan,
scheduler class, budget sharing with a 45 s deadline, copy-engine prefetch, deterministic compile), against the
previous live build (v1.1 code plus items 1 and 2):

- **Accuracy gate** (4 x 262K, 120 items, 3090): F1 0.852 -> 0.859, paired difference +0.007 [-0.016, +0.031],
  75/120 outputs identical, 0 errors. Pass.
- **Decode, 78K documents, 3090:** 1 stream 109/174 -> 115/213 tok/s (explain/copy); 4 streams steady
  266/390 -> 282/491, combined 207/255 -> 228/310; 8 streams steady 334/545 -> 331/548; recall 4/4, 15/15, 28/28.
- **8-agent mix, Ti, same tree with the switches off and on:**

| | seed A, off | seed A, on | seed B, off | seed B, on |
|---|---|---|---|---|
| long-turn TTFT, median | 55.2 s | 41.3 s | 39.8 s | 36.9 s |
| long-turn TTFT, 90th percentile | 90.1 s | 77.5 s | 101.6 s | 80.9 s |
| long-turn TTFT, max | 130 s | 97 s | 135 s | 96 s |
| medium-turn TTFT, median | 2.8 s | 2.3 s | 2.8 s | 2.6 s |
| output tokens in 12 min | 78,392 | 80,519 | 62,478 | 72,403 |
| failed requests | 0 | 0 | 0 | 0 |

- Live tool-call test (streamed arguments valid and equal to the non-streamed call, two calls in one turn) passes.

## Where the time goes

Prefill of a cache-hit follow-up turn, per 864-token block of new tokens (3090, one request):

| | prefix 0-5K | prefix 30K | prefix 80K |
|---|---|---|---|
| quantized GEMM | 373 ms (83%) | 336 ms (56%) | 321 ms (36%) |
| attention | 20 ms (4%) | 149 ms (25%) | 315 ms (36%) |
| host-tier staging (int4 dequantise per layer and chunk) | 0 | 54 ms (9%) | 172 ms (20%) |
| GDN chunk kernels + causal conv | 24 ms (5%) | 23 ms (4%) | 22 ms (3%) |
| everything else | 34 ms | 37 ms | 51 ms |
| total | 451 ms | 599 ms | 881 ms |

At 8 agents per card, long follow-ups (more than 2,600 new tokens, about three quarters of turns) wait a median of
40-50 s. That wait is queueing behind the prefill/decode duty cycle and behind other prefills, not prefill compute
for the turn itself. Nothing tried here adds prefill capacity at that load.

### Budget sharing: a reordering knob, not capacity

Serving pending prefills shortest-remaining-first inside each prefill step:

| 8 agents, Ti | off | shortest-first | + 45 s deadline | + 90 s deadline |
|---|---|---|---|---|
| long-turn TTFT, median | 52.4 s | 22.0 s | 45.8 s | 29.5 s |
| long-turn TTFT, max | 104 s | 280 s | 107 s | 201 s |
| output tokens in 12 min | 77,235 | 82,390 | 74,458 | 76,886 |

The median falls 55-70% on both cards and two seeds, but the worst case more than doubles and total output does not
change (+21%, +7%, -3% over three pairs). A stronger age credit or a fairness mode moves along the same trade-off.
A deadline (requests waiting longer than N seconds go first, oldest first) at 45 s keeps the tail where it was and
keeps a modest median gain (-13% and -35% on two seeds); at 30 s the order collapses to first-come-first-served; at
90 s the tail is back (170-200 s). Shipped off by default.

## Ruled out

| Tried | Result |
|---|---|
| Int8 activations for prefill (W4A8 Marlin path) | MLP layers only: prefill 7-9% slower; all layers: no change. The gain reported upstream is against half-rate fp16; with the tensor-core unlock fp16 already runs at full rate |
| Padding int8 KV slots to 16 bytes (vLLM #62113) | Bit-identical, 1.6-2.3x on the Triton attention kernels, but those are not on this stack's path: pthfa decode flat, pthfa 8-token verify about 20% slower |
| Rewriting the GDN prefill kernels (fused or two-kernel chunk forms, tiled conv) | GDN kernels plus conv are 3-5% of prefill here |
| GDN prefill launch-grid reorder (vLLM #66946) | Bit-identical, about 2 ms per chunk |
| Padding 2-4-row GEMM batches (HyperQwen #396) | The slow cuBLAS path does not exist on this card: plain 25 us, padded 62 us |
| Depth-weighted prefill yield (`SPARSE_PREFILL_YIELD_CTX_UNIT`) | Long-turn median 52 -> 64 s, output -12% |
| Prefill yield 12 -> 6 -> 3 steps per 1K | Long TTFT 54 -> 40 -> 25 s, per-stream decode 55 -> 30 -> 21 tok/s; a preference, left as configured |
| Evicting superseded GDN checkpoints first | Output 98 -> 60 tok/s, long TTFT 50 -> 91 s: the "superseded" checkpoints are still hit |
| Capping prefill chunks at depth | Within noise |
| Shorter drafts (K = 5, 3) | Per-stream decode 60 -> 43 -> 35 tok/s |
| `--renderer-num-workers=4` | Within noise |
| Dequantising host blocks once into borrowed free pool blocks | -1 to -4% on an idle server, less under load; the copy-engine prefetch does better alone |
| Sparse attention in prefill at 90K windows | Estimated 0-0.7% on follow-ups (attention is 14-33% of a chunk at 16-50K depth and the method has a fixed overhead); an earlier sparse suffix prefill cost 9-14 points of call-site F1 |
| Persisting GDN state to disk | Logs show about 1% slow follow-ups when uncongested, flat across idle gaps from 30 s to 60 min: no evidence of lost state |
| A separate draft model, tree drafting, other quantization formats, newer Triton | Memory cost, no multi-stream evidence, or regressions on this architecture |

Checks that came back clean: int8 KV under pool pressure (vLLM #50702; 0/30 corrupt idle and under an 8-worker
flood, though the tier spills instead of pinning the pool full), the offload-connector stall (vLLM #65938; this
connector never parks a request on an asynchronous load), the 8-token final-chunk report (vLLM #55766; no garbage at
tail lengths 1, 7, 8, 9, 16), known Triton 3.7.1 miscompile patterns (none of 824 kernels matches), the int32 offset
overflow in the GDN prefill kernels (lowest trigger is 131,072 tokens in one step), checkpoint restore ordering (232
compares equal), and drafter acceptance under the tier (no loss against a GPU-only server).

## Output differs between builds: compile-time kernel choice

Two builds of the same code produced different logits (up to 0.3 nats on log-probabilities, from the first token
after prefill), each bit-reproducible run to run. The cause is not memory corruption: Inductor times two block-size
variants of each fused RMS-norm reduction kernel at compile time and bakes the winner into vLLM's cached compile
artifact, which is keyed per source tree and configuration. The variants sum in a different order. Swapping one
tree's artifact into another reproduced the first tree's output exactly.

Consequences: a greedy A/B between two trees is only meaningful if both compiled the same variants, and neither
variant is the "correct" one. The launcher now passes
`"inductor_compile_config":{"deterministic":true,"combo_kernels":true,"benchmark_combo_kernel":false}`, which makes
every compile choose the same variants (two fresh compiles and a run with poisoned free memory were bit-identical).
`INDUCTOR_DETERMINISTIC=0` restores the old behaviour. The `TORCHINDUCTOR_DETERMINISTIC` environment variable alone
does not survive to vLLM's standalone compile on this PyTorch.

## Configurations

Both use the v1.2 build. Common settings:

```
SPARSE_GDN_RECOVER=1                    one GDN state block per request under MTP
MODEL=<model dir with the 63,510-token draft list>
SPARSE_TIER_HOT_GLOBAL=1                one shared hot pool per layer
SPARSE_TIER_SHARED_PLAN=1
SPARSE_SCHEDULER_CLS=1
SPARSE_PREFILL_BUDGET_SHARE=1 SPARSE_PREFILL_BUDGET_MAX_WAIT=45     optional, small
SPARSE_TIER_S5_DMA=1                                               optional, helps a narrow PCIe link
```

**Maximum context: 4 x 250K per card** (5 if capacity matters more than per-agent speed):

```
MAX_LEN=262144 MAX_SEQS=4 SPARSE_TIER_HOST_GB=17.8 SPARSE_TIER_HOST_MAMBA_SLOTS=8
MAX_LEN=262144 MAX_SEQS=5 SPARSE_TIER_HOST_GB=22   SPARSE_TIER_HOST_MAMBA_SLOTS=10
```

**At least 90K, most agents: 11 slots per card:**

```
MAX_LEN=90112 MAX_SEQS=11 SPARSE_TIER_HOST_GB=17.8 SPARSE_TIER_HOST_MAMBA_SLOTS=32 SPARSE_TIER_HOT_GLOBAL_ROWS=50176
```

Host RAM is not the limit for a fifth 250K agent: the int4 host tier costs about 17.4 KB per token, so 5 x 250K is
about 22 GB pinned per card.

| | 4 x 250K | 5 x 250K | 11 x 90K |
|---|---|---|---|
| Recall (needle and code line) | 16/16 | 20/20 | 28/28 at 8 streams |
| Combined output, all streams typing (explain / copy) | 241 / 332 tok/s | 232 / 278 tok/s | 228 / 310 tok/s at 4 streams |
| Follow-up TTFT, median | 6.2 s | 8.1 s | 1.4 s at 4 streams |
| Measured on | v1.1 code + items 1 and 2 (3090) | same | v1.2, all switches on (3090) |

The 250K rows were measured with `SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2`; the gated defaults are
`SPARSE_STATS_SUB=1 SPARSE_SELECT_EVERY=1`, so expect slightly lower decode there. v1.2 speed at 250K has not been
measured.

## Open items

- **Prefill capacity at 8+ agents per card.** GEMM (36-83% of a prefill block) and attention (up to 36%) have no
  cheap lever left on this hardware. The remaining idea with real headroom is routing each turn to whichever card
  has less prefill pending, with sessions sticky by default and one host-RAM KV and state store both servers read;
  queueing arithmetic suggests roughly half the wait. It needs a correct hand-off of GDN state between servers.
  First check how often one card queues while the other is idle.
- **Lookup-first drafting with MTP as fallback** (vLLM #60615): +15-17% decode on code edits reported for a single
  stream on a related model; the tree already has a lookup drafter.
- **A decode row scheduled with zero draft tokens** is built as a non-speculative row and reads the GDN conv state
  without the accepted-token shift. Found by reading the code; a live probe was too noisy to confirm it (one 9.8-nat
  outlier under a prefill flood). Needs a deterministic kernel-level test.
- **Cache-hit against cold mismatch** in one probe family (prefix 2,592 of about 4,690 tokens cached: answer differs
  from the cold answer in 8 of 10 cases, not garbage). Re-run under the deterministic compile before reading
  anything into it.
- **Pool sizing for in-flight GDN states** under async scheduling (HyperQwen #382): ported and CPU-tested, lowers
  the seat budget, not validated under pool pressure.
- **Per-request cache accounting on a live server** for a day (`SPARSE_PREFILL_YIELD_LOG=1`), and a check for
  chat-template differences that break the cached prefix inside an assistant turn (reasoning blocks, tool-call
  whitespace).
- **Rebase to vLLM 0.33**: 50 of 56 patches ported in a dry run; recover mode, the pooled speculative scratch
  sizing, lookup drafting hand-off and n-gram chains need redesign against upstream's changes.
