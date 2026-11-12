# v1.2 results

This is an addendum to the full benchmark in [`eval/bench/`](../bench/README.md) and to the v1.1 prefill results in
[`eval/prefill/`](../prefill/README.md). It holds the method and the tables behind the v1.2 changes
([`docs/CHANGELOG.md`](../../docs/CHANGELOG.md)). The numbers come from the working notes of 2026-11-10 and
2026-11-11, [`docs/OPTIMIZATION_NOTES.md`](../../docs/OPTIMIZATION_NOTES.md), which also list everything that was
ruled out and the open items.

Two results for the full v1.2 build (all switches on) were still open when this was written:

| result | status |
|---|---|
| accuracy gate, full v1.2 build against the build before it | call-site F1 0.852 -> 0.859, paired difference +0.007 [-0.016, +0.031], n = 120, 75/120 outputs identical: pass |
| 8-agent mix, full v1.2 build, switches on against off | long-turn TTFT median 55.2 -> 41.3 s and 39.8 -> 36.9 s (two seeds), max 130 -> 97 s and 135 -> 96 s, output tokens +3% and +16%, 0 failed requests |

## Setup common to all tests

**Model and hardware.**
- Qwen3.8-27B W4A16 with MTP-7, pthfa int8 KV, sparse decode, int4 host-RAM tier.
- Ti = RTX 3090 Ti on PCIe x16, 3090 = RTX 3090 on PCIe x4, 24 GB each, one server per card.
- Both cards ran at a fixed power limit and with the GeForce tensor-core rate unlock
  ([gpu-unlock](https://github.com/Zagitta/gpu-unlock)) applied, so fp16/bf16 matrix math runs at full rate. Several
  results depend on that, most of all the int8-activation result under "Ruled out" in the notes.

**Same-card A/B.** Every comparison is base against variant on the same card, one server at a time. At identical
settings the Ti was about 20% slower than the 3090 at cold prefill (78K-token document: 67 s against 54 s) and level
or slightly ahead at decode, so numbers are never compared across cards.

**Harnesses.**
- *Decode harness:* N streams, each holding its own 78K or 246K-token document; explain and copy tasks (all streams
  typing at once); needle and code-line recall. "Combined" is the output rate over the whole task, "steady" the rate
  while all streams decode.
- *Agent mix:* 12 minutes, 8 agents per card, with measured turn sizes and tool gaps. Follow-up turns are grouped by
  their uncached prompt tokens: short (up to 1,728), medium (1,729-2,600) and long (more than 2,600, about three
  quarters of turns).
- *Greedy A/B:* 8 turns at temperature 0, compared character by character, for output identity.
- *Accuracy gate:* the cross-file call-site suite of [`eval/bench/`](../bench/README.md), 120 items, concurrency 1,
  temperature 0. Metric: F1 of the variant minus F1 of the base on the same items, with a paired 95% bootstrap CI.
  A change passes if the CI includes 0.

**Noise.** Decode harness 3-5%. Agent-mix medians about 10-15% run to run (controls for one card ranged from 41.6 to
52.4 s on the long-turn median). Copy-task speed is bimodal because it follows draft acceptance.

## 1. One GDN state block per request (`SPARSE_GDN_RECOVER=1`)

Stock align-mode speculative decoding keeps the recurrent state after each of the K draft tokens in its own block:
with MTP-7 and three mamba groups, 21 of the 27 blocks a request pins. Recover mode keeps one state block per group
and replays the accepted tokens after sampling.

| | before | recover mode |
|---|---|---|
| pinned blocks per request | 27 | **6** |
| seat budget, resident requests per card | 3-4 | **13-17** |
| 8 agents on one card: short follow-up TTFT | 17.1 s | **1.5-1.9 s** |
| 8 agents on one card: medium follow-up TTFT | 25.2 s | **2.3-2.8 s** |

- **Output:** bit-identical to the baseline. Kernel test on bf16/fp16/fp32 states
  (`sparsekv/test_s13_gdn_recover_gpu.py`), and a character-identical greedy A/B.
- **A fifth 250K agent** fits on a card (recall 20/20). It is capacity, not throughput:

| 250K agents on one card (3090) | 4 | 5 |
|---|---|---|
| recall (needle and code line) | 16/16 | 20/20 |
| combined output, explain / copy | 241 / 332 tok/s | 232 / 278 tok/s |
| follow-up TTFT, median | 6.2 s | 8.1 s |

  Each agent is 12-27% slower at five. Host RAM is not the limit: the int4 host tier costs about 17.4 KB per token,
  so 5 × 250K is about 22 GB pinned per card. These rows were measured with `SPARSE_STATS_SUB=4
  SPARSE_SELECT_EVERY=2`.

## 2. Larger MTP draft vocabulary (29,670 → 63,510 tokens)

The draft head is sliced from the quantized lm_head. A draft token outside the list ends the draft chain.

| | 29,670 tokens | 63,510 tokens |
|---|---|---|
| tokens per step, 4 streams, 78K documents | 3.6-3.7 | **4.0** |
| tokens per step, 4 × 250K | 3.6 | **4.3** |
| 4 × 250K combined output, explain / copy (same card) | 230 / 257 tok/s | **241 / 332 tok/s** (+5% / +29%) |

- The full head accepts slightly more but is slower at one stream; 64K is the best point measured.
- **Gate.** A different list changes the output text (draft acceptance is probabilistic in this build):

| comparison | n | call-site F1 | difference |
|---|---|---|---|
| 63,510-token list against the 29,670-token list | 120 | 0.837 → 0.852 | **+0.015 [−0.007, +0.038]** |

- **How the list was built:** `sparsekv/tools/mk_draft_vocab_ids.py` (token-frequency coverage; the ids in use are
  kept, the most frequent tokens of a workload-like corpus are added). The list itself is not published, because
  part of its corpus was private agent traffic.

## 3. One shared hot buffer per layer (`SPARSE_TIER_HOT_GLOBAL=1`)

The hot buffer that caches host-tier rows on the GPU was one buffer per request slot. A single pool per layer, keyed
by host token, serves the same traffic from less memory.

| 11 slots × 90K | per-slot buffers | shared pool, default rows (4 × per-slot) | shared pool, `SPARSE_TIER_HOT_GLOBAL_ROWS=50176` |
|---|---|---|---|
| hot-buffer reserve | 2,275 MB | **862 MB** | |
| GPU KV pool | 108 blocks | **154 blocks (+43%)** | 124 blocks (+15%) |

At 8 or more concurrent spilled streams the default size costs 5-8% decode, so the 11-slot configuration uses the
doubled pool.

Hot-buffer miss rate with the doubled pool, 78K documents on the x4 card:

| streams | per-slot buffers | shared pool |
|---|---|---|
| 1 | 15-19% | **2.8%** |
| 4 | 13% | **6.7%** |
| 8 | 14.5% | **12%** |

Decode at 4 streams (explain / copy):

| | before | after |
|---|---|---|
| steady | 266 / 390 tok/s | 282 / 491 tok/s |
| combined | 207 / 255 tok/s | 228 / 310 tok/s |
| follow-up TTFT, median | 2.2 s | 1.4 s |

Level at 8 streams. Part of the copy gain is higher draft acceptance in those runs; the explain figures and the miss
rates are the steadier evidence.

**Gate** (with the smaller pieces of section 4, at 4 × 262K):

| comparison | n | call-site F1 | difference |
|---|---|---|---|
| shared hot buffer build against the build before it | 120 | 0.852 → 0.862 | **+0.010 [−0.011, +0.033]** |

## 4. Smaller pieces

| piece | check | result |
|---|---|---|
| Shared residency plan (`SPARSE_TIER_SHARED_PLAN=1`) | checked run (`SPARSE_TIER_SHARED_PLAN_CHECK=1`) | 0 mismatches; speed effect not separable from noise |
| Prefill yield as a scheduler class (`SPARSE_SCHEDULER_CLS=1`) | scripted schedules, greedy A/B | identical decisions on 6,720 scripted steps, identical output |
| Deterministic compile (`INDUCTOR_DETERMINISTIC=1`) | two fresh compiles, and a run with poisoned free memory | bit-identical; cold prefill 67 s both ways; steady decode 266/421 against 256/405 tok/s |
| int8 scale-view fix (vLLM #64141) | greedy A/B | no output change |
| Qwen3 parser and detokenizer backports | parser suite, live tool-call test | pass |
| Copy-engine prefetch (`SPARSE_TIER_S5_DMA=1`, off) | follow-up prefill, idle server | −8% at an 80K prefix and −2-3% at 30K on the x4 card, 0-1% on the x16 card, output identical; not resolvable in the 8-agent mix |

**Why the deterministic compile.** Two builds of the same code produced different logits (up to 0.3 nats on
log-probabilities, from the first token after prefill), each bit-reproducible run to run. Inductor times two
block-size variants of each fused RMS-norm reduction kernel at compile time and bakes the winner into vLLM's cached
compile artifact, which is keyed per source tree and configuration. The variants sum in a different order. Swapping
one tree's artifact into another reproduced the first tree's output exactly. So a greedy A/B between two trees is
only meaningful if both compiled the same variants, and neither variant is the "correct" one.

## 5. Budget sharing among pending prefills (off by default)

`SPARSE_PREFILL_BUDGET_SHARE=1` serves pending prefills shortest-remaining-first inside each prefill step.
`SPARSE_PREFILL_BUDGET_MAX_WAIT=N` adds a deadline: requests waiting longer than N seconds go first, oldest first.

| 8 agents, Ti | off | shortest-first | + 45 s deadline | + 90 s deadline |
|---|---|---|---|---|
| long-turn TTFT, median | 52.4 s | 22.0 s | 45.8 s | 29.5 s |
| long-turn TTFT, max | 104 s | 280 s | 107 s | 201 s |
| output tokens in 12 min | 77,235 | 82,390 | 74,458 | 76,886 |

- Without a deadline the median falls 55-70% on both cards and two seeds, but the worst case more than doubles and
  total output does not change (+21%, +7%, −3% over three pairs).
- A stronger age credit or a fairness mode moves along the same trade-off.
- A 45 s deadline keeps the tail where it was and keeps a modest median gain (−13% and −35% on two seeds). At 30 s
  the order collapses to first-come-first-served; at 90 s the tail is back (170-200 s).
- It is a reordering knob, not capacity.

## 6. Where follow-up prefill time goes

Prefill of a cache-hit follow-up turn, per 864-token block of new tokens (3090, one request):

| | prefix 0-5K | prefix 30K | prefix 80K |
|---|---|---|---|
| quantized GEMM | 373 ms (83%) | 336 ms (56%) | 321 ms (36%) |
| attention | 20 ms (4%) | 149 ms (25%) | 315 ms (36%) |
| host-tier staging (int4 dequantise per layer and chunk) | 0 | 54 ms (9%) | 172 ms (20%) |
| GDN chunk kernels + causal conv | 24 ms (5%) | 23 ms (4%) | 22 ms (3%) |
| everything else | 34 ms | 37 ms | 51 ms |
| total | 451 ms | 599 ms | 881 ms |

At 8 agents per card, long follow-ups wait a median of 40-50 s. That wait is queueing behind the prefill/decode duty
cycle and behind other prefills, not prefill compute for the turn itself. Nothing tried here adds prefill capacity
at that load.

## 7. Configurations

| | 4 × 250K | 5 × 250K | 11 × 90K |
|---|---|---|---|
| Recall (needle and code line) | 16/16 | 20/20 | 28/28 at 8 streams |
| Combined output, all streams typing (explain / copy) | 241 / 332 tok/s | 232 / 278 tok/s | 228 / 310 tok/s at 4 streams |
| Follow-up TTFT, median | 6.2 s | 8.1 s | 1.4 s at 4 streams |
| Measured on | v1.1 code + sections 1-2 (3090) | same | v1.2, all switches on (3090) |

The settings are in the README's Run section. v1.2 speed at 250K has not been measured.

## Raw records

`results/` holds the records behind most of the tables above. They are timing, count and score records only:

- **No prompt or answer text.** The answer field of the decode harness and the model outputs of the gate were
  removed; the gate records keep a hash of each answer (`out_sha`) so identical answers can still be counted.
- The agent mix replays requests of real coding sessions. Its records never held their content, only sizes and
  timings; the two harness arguments that named local directories were removed.
- Not published: the greedy A/B transcripts, the log-probability probes, the follow-up prefill timings (they hold
  generated text), and the result files of the two open items above.

| files | what | tables |
|---|---|---|
| `results/mix/mix8_k7.json`, `mix8_rec.json` | 8-agent mix on one card, before and with recover mode (one of several recover runs) | section 1 |
| `results/mix/mix8_g0_bud0.json`, `mix8_g0_bud1.json`, `mix8_g0_bud1w45.json`, `mix8_g0_bud1w90.json` | budget sharing on the Ti, seed 11: off, shortest-first, 45 s and 90 s deadline | section 5 |
| `results/mix/mix8_g0_bud1w30.json`, `mix8_g0_*s7.json` | 30 s deadline; the same arms with seed 7 | section 5 |
| `results/decode/d.big250_before4.jsonl`, `d.big250_after4.jsonl`, `d.big250_after5.jsonl` | 4 × 250K before (29,670-token list, no recover mode), 4 × 250K and 5 × 250K after | sections 1, 2, 7 |
| `results/decode/d.base.jsonl`, `d.dv64.jsonl`, `d.fullhead.jsonl` | 78K documents, 1 and 4 streams: 29,670-token list, 63,510-token list, full head | section 2 |
| `results/decode/d.g1_w4a16.jsonl`, `d.s17_on.jsonl` | 78K documents on the 3090: before the shared hot buffer, and the v1.2 build with all switches on (1, 4, 8 streams) | sections 3, 7 |
| `results/decode/d.s14_off8.jsonl`, `d.s14_hot8.jsonl`, `d.s14_hot8big.jsonl` | 8 streams: per-slot buffers, shared pool at the default rows, shared pool at 50176 rows | section 3 |
| `results/decode/d.g0_base.jsonl`, `d.g0_det.jsonl` | Ti, without and with the deterministic compile | section 4 |
| `results/decode/d.g1_w4a8_mlp.jsonl`, `d.g1_w4a8_all.jsonl` | int8 activations for prefill, MLP layers and all layers (ruled out; baseline `d.g1_w4a16.jsonl`) | notes |
| `results/gate/vocab/{base,cand}.jsonl` | gate of section 2, per-item scores | section 2 |
| `results/gate/hotpool/{base,cand}.jsonl` | gate of section 3, per-item scores | section 3 |

To print the tables from the records:

```bash
python run_s4/sum.py eval/v12/results/decode/d.*.jsonl          # decode harness, one line per stream count
python eval/v12/mix_sum.py eval/v12/results/mix/*.json          # agent mix
python eval/v12/gate_an.py eval/v12/results/gate/vocab eval/v12/results/gate/hotpool   # needs numpy
```

`mix_sum.py` takes the plain median over the records. The `summary` block inside each mix file was written by the
harness with a nearest-rank percentile, so it can differ in the first decimal (the 17.1 s and 25.2 s of section 1
are from that block).
