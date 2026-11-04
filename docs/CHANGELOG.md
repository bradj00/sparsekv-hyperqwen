# Changelog

All numbers are from Qwen3.8-27B W4A16 with MTP-7 on the test hardware described in the README: two 24 GB Ampere
cards (RTX 3090 Ti on PCIe x16, RTX 3090 on x4), each serving 4 × 250K agents. All were measured at a fixed power
limit. Both cards are power-bound in prefill, so absolute prefill times depend on the power limit. Per-test method and
tables: [`eval/prefill/README.md`](../eval/prefill/README.md).

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
