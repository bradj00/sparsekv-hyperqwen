# Prefill results (v1.1)

This is an addendum to the full benchmark in [`eval/bench/`](../bench/README.md). It holds the measurements behind the
v1.1 prefill and cache-reuse changes ([`docs/CHANGELOG.md`](../../docs/CHANGELOG.md)). All runs are from 2026-11-04.

## Setup common to all tests

**Model and hardware.**
- Qwen3.8-27B W4A16, MTP-7, pthfa int8 KV, sparse decode + int4 host-RAM tier.
- Production slot sizing: 4 × 250K agents per card (`MAX_LEN=262144 MAX_SEQS=4 GPU_UTIL=0.94 SPARSE_TIER_HOT_FRAC=0.025
  SPARSE_TIER_HOST_GB=17.8`), plus the fixed configuration.
- Ti = RTX 3090 Ti (PCIe x16), 3090 = RTX 3090 (PCIe x4), both 24 GB.
- Every number was measured at a fixed power limit. Both cards are power-bound in prefill, so absolute times move
  with the power limit; the A/B ratios are what carry over.

**Same-card A/B.** Every comparison is base vs variant on the same card, one dev server at a time, with the production
environment and the variant's knob as the only difference. Results across cards are not mixed.

**Accuracy gate.** Cross-file call-site retrieval, the call-site suite of [`eval/bench/`](../bench/README.md):
- the task is "list every caller of function X" over 64-192K tokens of real code;
- 60 items per shard, concurrency 1, temperature 0;
- metric: F1 of the variant minus F1 of the base on the same items, with a paired 95% bootstrap CI;
- a change passes if the CI includes 0 (no measurable loss).

For scale, two identical base servers on different cards differ by −0.003 [−0.053, +0.047] F1 (0–1 scale; 28/60
outputs identical). So single-shard intervals of about ±0.05 (±5 points) are the noise floor.

**TTFT.** Time to the first streamed token, client side, through the OpenAI-compatible API.

## 1. Cold prefill: FA2 prefill and flat GQA rows

Method:
- one cold prompt per length (unique code text, so there is no prefix hit), `max_tokens` small;
- arms base / `PTHFA_PREFILL_FA2=1` / `PTHFA_FLAT_GQA=1`, run one after another on the same card;
- prompt sizes are 64,896 / 129,778 / 241,576 tokens.

Cold TTFT in seconds (change vs base):

| card | prompt | base | FA2 prefill | flat GQA (E1) |
|---|---|---|---|---|
| Ti | 64K | 57.1 | 52.9 (**−7.4%**) | 55.1 (−3.6%) |
| Ti | 128K | 150.8 | 136.8 (**−9.3%**) | 145.6 (−3.4%) |
| Ti | ~240K | 397.2 | 336.9 (**−15.2%**) | 371.9 (−6.4%) |
| 3090 | 64K | 46.2 | 43.8 (−5.2%) | 45.3 (−1.9%) |
| 3090 | 128K | 122.7 | 114.7 (−6.5%) | 120.2 (−2.1%) |
| 3090 | ~240K | 327.9 | 282.9 (−13.7%) | not run |

Attention kernel alone (pthfa prefill launch, 1728-token chunk, flat vs padded rows):

| context | Ti | 3090 |
|---|---|---|
| 64K | −10.2% | −5.1% |
| 128K | −8.6% | −5.8% |
| 244K | −7.2% | −7.6% |

**Numerics.**
- FA2 prefill dequantises K/V to fp16 (~2^-12 relative) and is in pthfa's rounding class, but not bit-identical.
- Flat GQA rows are not bit-exact either. On prefill shapes the kernel check found up to ~5e-4 max absolute
  difference. Decode, MTP verify and short slices (< 64 rows) are bit-identical.
- E1 was not accuracy-gated and ships off.

**FA2 accuracy gate (Ti, 60 items per shard):**

| shard | F1 vs base (points) |
|---|---|
| 0 | +0.9 [−4.0, +6.4] |
| 1 | −3.6 [−8.3, +0.7] |
| pooled | ≈ −1.4: within noise |

Three real retrieval prompts (16-125K tokens), greedy:
- on the two with a findable answer, FA2 gave the same answer as base on both cards;
- on the third, the target was cut out, and the wording differs between runs. Base itself differs between the two
  cards there.

Free-form text diverges after some tokens, as with any change of rounding.

## 2. Follow-up turns: block drop and decode-boundary checkpoints

**Test 1: `SPEC_NO_BLOCK_DROP`.**
- Method: 4 sessions × 3 turns; a 70K-token first prompt, then +1K tokens per turn; legs alternate off/on/off (Ti)
  and on/off/on (3090).
- Result: with the knob on, every follow-up hit `floor(P_prev/864)·864` tokens, against 864 fewer with it off.

| card | follow-up TTFT, drop on (v1.0) | drop off (`SPEC_NO_BLOCK_DROP=1`) |
|---|---|---|
| 3090 | 2.56 s | **1.80 s** |
| Ti | 2.82 s | **1.93 s** |

**Test 2: `SPARSE_TIER_MAMBA_SAVE_AT_FINISH`, with `SPEC_NO_BLOCK_DROP=1`.**
- Method: 4 sessions round robin, 60K prompts, 2.5-3.5K-token answers. The other sessions evict a session's GPU
  cache between its turns.
- Result: turn 2 hit +2.6-3.5K tokens more, exactly up to the last decode boundary.

| card | turn-2 TTFT after eviction, off | on |
|---|---|---|
| Ti | 5.75 s | **2.06 s** |
| 3090 | 4.84 s | **2.02 s** |

- Host-saved decode-boundary GDN states were bit-identical to the GPU copies at the same boundary (4/4 states, 16/16
  pages × 3 groups).
- Decode tok/s unchanged; no errors.
- MTP acceptance on 35K generated tokens: 3.61 (drop on) vs 3.65 (drop off).

**Gate for both knobs** (F1, 0–1 scale, same card):

| comparison | n | F1 vs base |
|---|---|---|
| block drop off only (Ti, shard 1) | 60 | −0.030 [−0.091, +0.026] |
| both knobs (3090, shard 1) | 60 | −0.008 [−0.055, +0.036] |
| both knobs (Ti, shard 0) | 60 | +0.042 [+0.004, +0.085] |
| both knobs (3090, shard 0) | 60 | +0.012 [−0.029, +0.053] |
| **both knobs, pooled** | **180** | **+0.015 [−0.010, +0.039]** |

Call-site job wall time fell 7-11% with both knobs.

## 3. Host GDN checkpoint slots (agent workload)

**Method.**
- Data: per-step token records of a real coding-agent workload, 4 agents per card on two cards. For each step, the
  records give prompt tokens and cached tokens.
- Prefill GPU time per step is modelled as 0.596 ms + 8.24 ns × position per token, fitted to measured cold
  prefills.
- A step counts as a "full re-prefill" if its hit is under 20% of the previous prompt.

| window | GDN slots | steps | uncached tokens | prefill per step | full re-prefills (share of prefill time) |
|---|---|---|---|---|---|
| ~16 h | 8 | 8,102 | 15.3% | 6.41 s | **43.3%** |
| 51 min | 32 | 731 | 5.7% | **4.02 s** | **0%** (0 of 227 steps after a > 30 s tool gap; the 8-slot rate predicts ~36) |

- The full misses tracked the tool-wait length. With 8 slots they hit 0.1% of steps at < 30 s gaps, 12.6% at
  30-120 s and 37% at 120-600 s.
- The 32-slot window is short, and its prompts were larger (median 68K vs 47K).
- Cost: 85.3 MB pinned per slot, so 2.7 GB per server at 32.

## 4. Short-request bypass of the prefill yield

Two workloads, each run with bypass off / 1728 / 2600 on both cards.

**Replay (stress).**
- 3 agents with warm ~30K contexts keep decoding, and send follow-ups of 300-3000 new tokens after 1-2 s of tool
  time.
- Meanwhile a fourth request runs a 64K cold prefill. 3 rounds.

**Agent mix (realistic), 15 minutes, 4 agents.** Every size follows the measured agent workload:
- session starts are real ~16K-token first requests, salted so they are cold;
- follow-ups' uncached size: q50 2.8K, q90 6.6K;
- decode length: q50 470, q90 2.7K tokens;
- tool gap: q50 19 s;
- sessions end at a peak prompt of q50 62K;
- one full 30-110K cold re-prefill every 4 minutes.

Agent mix, median / p90 TTFT in seconds:

| card | bypass | short follow-ups (≤ 1728) | 1729-2600 | long follow-ups | **all follow-ups** | decode tok/s |
|---|---|---|---|---|---|---|
| Ti | off | 1.7 / 8.0 | 4.6 / 18.7 | 10.1 / 24.2 | **9.8** / 23.1 | 82.7 |
| Ti | 1728 | 1.7 / 4.0 | 3.6 / 33.1 | 10.1 / 33.1 | 9.3 / 28.8 | 79.5 |
| Ti | **2600** | 1.7 / 3.1 | 2.6 / 5.3 | 7.7 / 28.4 | **7.1** / 25.1 | 74.8 |
| 3090 | off | 2.6 / 8.7 | 8.3 / 14.7 | 9.5 / 23.8 | **8.3** / 22.8 | 82.1 |
| 3090 | 1728 | 1.8 / 6.1 | 8.1 / 20.5 | 10.7 / 28.5 | 9.7 / 25.6 | 84.2 |
| 3090 | **2600** | 2.0 / 3.6 | 2.7 / 5.2 | 8.2 / 25.2 | **6.7** / 21.7 | 79.0 |

Replay, median TTFT of all follow-ups and the long prefill:

| card | bypass | all follow-ups med / p90 | 64K cold prefill | decode tok/s during it |
|---|---|---|---|---|
| Ti | off | 8.6 / 16.3 | 114.8 s | 54.7 |
| Ti | 2600 | 4.3 / 13.1 | **155.1 s (+35%)** | 69.0 |
| 3090 | off | 9.1 / 14.6 | 112.8 s | 68.3 |
| 3090 | 2600 | 6.9 / 16.3 | 110.7 s | 68.9 |

Costs:
- per-request decode throughput in the mix is 4-10% lower;
- under contention a long prefill can be pushed back, as on the Ti replay (+35%).

N = 2600 covers about half of real follow-ups (47% have ≤ 2600 uncached tokens). Every request succeeded in all mix
runs.

## 5. Shared stable prefix and the pin knob

**Harness side.**
- Method: the first requests of 8 parallel agent builds, rendered with the production chat template and tokenizer.
  Each build gets the same system prompt, tools and brief, and its own task.
- Metric: the shortest common token prefix across all 8.
- What each step changes: [`docs/AGENT_HARNESS_TIPS.md`](../../docs/AGENT_HARNESS_TIPS.md).

| setup | common prefix, all 8 |
|---|---|
| as launched | 7,262 tokens (split by a per-server model line) |
| + stable-prefix plugin + identical working directory + per-agent text last | **16,026 tokens** (15,552 aligned to 864-token blocks) |

**Server side (`SPARSE_TIER_PIN_PREFIX_MAX=2`).**
- Method: one warm request of the 15,552 aligned tokens with `vllm_xargs {"sparse_pin_prefix": 1}`, `max_tokens 1`
  (8.5 s). Then 4 agents' first requests. Then 20 minutes of unrelated traffic (187 requests, 6.9M prompt tokens).
  Then 4 more agents.
- Result: all 8 first requests hit **15,552 cached tokens**, with TTFT 0.48-1.11 s (phase 1) and 0.50-0.55 s
  (after the traffic). The cold warm request itself took 8.5 s.

**Live.** In a production round after the harness change, **5 of 8 builds hit 15,552 cached tokens** on their first
step.

## 6. Rejected: Sage-style int8-QK prefill

- What it is: an hd256 port of SageAttention-style int8 Q·K with fp16 P·V for prefill.
- Method: compared against FA2 prefill on the same card, cold TTFT and the call-site gate on both shards and both
  cards (n = 120).

| card | prompt | FA2 prefill | int8-QK |
|---|---|---|---|
| Ti | 64K | 52.6 s | 50.8 s |
| Ti | 128K | 130.2 s | 122.9 s |
| Ti | ~232K | 310.9 s | 282.0 s (−9.3%) |
| 3090 | 64K | 43.3 s | 40.8 s |
| 3090 | 128K | 109.7 s | 101.2 s |
| 3090 | ~232K | 271.3 s | 244.9 s (−9.7%) |

- Call-site F1 vs FA2: **−4.8 [−8.8, −1.3]** points, n = 120. The 192K bin alone is −9.3. Rejected.

## 7. Where cold prefill time goes

Cold 244K prefill on the Ti, from a cost model fitted to measured cold prefills and kernel timings:

| part | time | share |
|---|---|---|
| full attention, 16 layers (pthfa, FP16 MMA on int8 KV) | ~250 s | **~63%** |
| Marlin W4A8 linears (QKV, out, MLP) | ~115 s | **~29%** |
| MTP drafter prefill | ~4-5 s | ~1% |
| everything else | | < 10% |

- The attention share by length is 31% at 64K, 47% at 128K and 63% at 244K.
- At the clocks the power limit allows, attention reaches about half of the FP16 tensor peak.
- The cards are power-bound, so end-to-end gains are smaller than isolated kernel gains. Every number here is at a
  fixed power limit.

## Other ideas measured and not shipped

- **Hierarchical select:** 4 × 246K: 17.8 vs 16.0 ms/step; 2× slower at 64K. The select kernels are launch-bound
  (~1 ms per layer fixed).
- **Decode prefetch of the next layer's picks:** on recorded picks it covers 15.5% of misses, with 73% of
  prefetched rows unused, so it is worth ≤ 1-2% of decode.
- **Disk session tier:** the resume after a restart returned 0 cached tokens. Read-back is broken; not shipped.
