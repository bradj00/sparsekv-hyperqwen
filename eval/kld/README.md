# KL divergence of sparse-KV decode vs dense attention

This directory holds a teacher-forced KL-divergence test of the sparse decode path against dense attention, run on the same model and the same token paths. It replaces anecdotal recall checks with a per-token number. Everything needed to rerun it is here. The full tables are in `results/results.md`, and the per-sample numbers are in `results/results.json`.

## Result

KLD is per-token KL(dense ‖ run) in nats, over decode steps only. Top-1 is the share of steps where the run's argmax token equals dense's.

| comparison | 32K | 64K | 128K | 192K | all (mean KLD / top-1) | p99 (all) |
|---|---|---|---|---|---|---|
| **noise floor A**: dense vs dense, same server, rerun | | | 0.0000 / 100% | 0.0000 / 100% | **0.0000 / 100%** (6 samples) | 0 |
| **noise floor B**: dense on the RTX 3090 Ti vs dense on the RTX 3090 (same arch, other server) | 0.0230 / 96.4% | 0.0167 / 96.8% | 0.0198 / 95.9% | 0.0206 / 95.9% | **0.0200 / 96.2%** | 0.241 |
| sparse, GPU-only (`SPARSE_PROFILE=sparse`), vs dense on the same server | 0.0185 / 96.9% | 0.0152 / 96.9% | 0.0187 / 96.1% | 0.0228 / 95.8% | **0.0188 / 96.4%** | 0.225 |
| sparse + RAM tier, int8 host copy, vs dense on the same card (3 docs only, see caveats) | 0.0180 / 96.4% | 0.0242 / 96.3% | 0.0285 / 94.9% | 0.0290 / 95.7% | **0.0249 / 95.8%** | 0.318 |
| sparse + RAM tier, int4 host copy (**production config**), vs dense on the same card | 0.0241 / 96.1% | 0.0221 / 96.1% | 0.0279 / 95.1% | 0.0333 / 95.2% | **0.0269 / 95.6%** | 0.333 |

- Unless marked otherwise, each cell covers 5 documents × 3 tasks = 15 samples and 3,825 decode tokens, so 60 samples and 15,300 tokens per row.
- Medians are tiny for every config (0.0003-0.0009 nats). The mean comes from a minority of steps where dense itself is uncertain, such as the choice between two phrasings. `results/results.md` has the median, p95 and max for each length. It also has the same numbers against the other card's dense reference, and slices by task and document type.
- **The int4 and int8 host copies score the same.** On the 36 samples both tier runs share, the int8 tier scores 0.0249 / 95.8% and the int4 tier 0.0248 / 95.9%. On that subset the GPU-only sparse run scores 0.0167 / 96.9%, and the cross-card dense floor 0.0180 / 96.5%. The tier configs' extra divergence over GPU-only sparse therefore comes from the production speed knobs, not from int4 host storage. Those knobs are the int4 sketch, `SPARSE_STATS_SUB=4`, `SPARSE_SELECT_EVERY=2` and sparse attention for the uncached prompt suffix. We did not ablate them one by one.
- **By task:**
  - *Recall*, which quotes 30 lines after a line at about 35% depth, is almost unaffected: 0.0016 KLD and 99.9% top-1 for GPU-only sparse, and 0.007 / 99.6% for the production tier.
  - *Continue* and *explain* carry most of the divergence. They are open-ended text where dense itself is often near a tie.
  - Prose diverges about 2× more than code in every config, the dense-vs-dense floor included.
- **Context length:** divergence grows slowly with context. Production tier: 0.024 at 32K to 0.033 at 192K.

**In plain words:** with GPU-only sparse decode, the next-token distribution moves away from dense by less than it moves when you run the identical dense model on a different card of the same architecture. The production config (RAM tier + int4) adds about 35% on top of that floor, at about 0.6 points lower top-1 agreement. These are teacher-forced per-token numbers: free-running greedy output still drifts apart after the first differing token, as it does between the two dense cards.

## Method

- **Model / stack:** Qwen3.8-27B W4A16 (AutoRound) on the release in this repo (vLLM 0.31 HyperQwen + pthfa int8 KV), Model Runner V2, temperature 0, thinking off.
  - **Speculative decoding (MTP) is off in every config.** vLLM rejects custom logits processors under spec decode, and the question is about the base model's distribution.
  - All servers use `--block-size 864`, the production page size. Without MTP vLLM would pick 800, and the sparse kernels are built for 864.
  - `--no-async-scheduling`, one request at a time.
- **Teacher forcing:** `kld_proc.py` is a Model Runner V2 custom logits processor. At every output step it:
  - takes the log-softmax of the model's logits over the full vocabulary (248,320);
  - records the top-1024 for the reference, or, for forced runs, the run's log-probs at the reference's top-1024 ids;
  - then masks the logits so that the sampler emits the reference token.

  Every measured step is therefore a real decode forward on the path under test, and none of the forced tokens become prefill. The processor also records:
  - the token fed at each step. The analysis checks that it equals the reference token for every step of every forced run, and it does;
  - `sparse_kv.step_sparse()`, the per-step decision of the attention backend. It was sparse on every decode step of every sparse run and dense on every step of the dense runs. The server logs show the matching `switch -> SPARSE` / `switch -> dense` lines.
- **Reference:** a dense server (no `SPARSE_*`, same base config) generates up to 256 greedy tokens per sample. Every config is then forced along exactly those tokens.
- **KL:** KL(P‖Q) is computed on P's top-1024 support plus one tail bucket for the remaining mass. This is a lower bound on the full-vocabulary KL, because merging outcomes cannot increase KL. The reference's tail mass averages 2.5e-4, so the bound is tight.
  - **Step 0 is excluded.** The first generated token is computed by the dense prefill in every config.
  - Steps 1-255 are decode.
- **Configs:**
  - GPU1, an RTX 3090:
    - the **dense reference** server;
    - the **int4 tier**: the production 4 × 250K command from the top-level README, `SPARSE_PROFILE=tier MAX_LEN=262144 MAX_SEQS=4 GPU_UTIL=0.94 SPARSE_TIER_HOT_FRAC=0.025 SPARSE_SKETCH_BITS=4 SPARSE_TIER_HOST_BITS=4 SPARSE_TIER_HOST_GB=17.8 SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2`.
  - GPU0, an RTX 3090 Ti:
    - one `SPARSE_PROFILE=sparse` server (int8 sketch, exact statistics, selection every step, no tier) that decodes each sample twice from the same prefix-cached prefill. The first pass forces the per-step switch to dense, the second to sparse (`KLD_TOGGLE=1`; `SPARSE_CUTOFF` table set per request). This gives a same-server, same-prefill dense-vs-sparse comparison;
    - the **int8 tier**: the production command with `SPARSE_TIER_HOST_BITS=8`.
  - "Same card" rows compare each run with the dense run on its own card:
    - GPU0: the toggle server's dense pass, for sparse GPU-only and the int8 tier;
    - GPU1: the reference, for the int4 tier.
- **Is the RAM tier really used?** Yes. In the tier configs every decode step reads the selected rows not resident in the GPU hot buffer from the host copy. Over the runs that was 13.6% of selected rows (int4) and 13.7% (int8): `hot[hit, miss]` in the server's `tier: stats` lines. At 192K the GPU pool (154K tokens) also forces real spill.
- **Data** (`build_samples.py`): 5 documents.
  - Code:
    - vLLM's `v1/` sources;
    - PyTorch's `torch/_inductor`;
    - transformers' `generation` / `integrations` / `pipelines`.
  - Prose, from Project Gutenberg:
    - *War and Peace* (#2600);
    - *The Count of Monte Cristo* (#1184).
  - Each document is cut at 4 lengths, so that header + document = L + 16 tokens for L = 32,832, 65,664, 131,328 and 196,128 (multiples of the 864-token block). The table calls them 32K/64K/128K/192K.
  - Each cut gets 3 tasks, every one ending in a chat turn:
    - **continue** the document (next 60 lines);
    - **explain** its final part;
    - **recall**: quote the 30 lines after a unique line at about 35% depth.
  - Prompts of one document share their prefix, so prefix caching reuses the prefill, which is dense in every config.
  - 60 samples per config, up to 256 tokens each. Some recall answers end early at EOS.

## Caveats

- **No 250K row.** The dense GPU-only comparison server held 240K tokens, so the longest length every config could run is 192K. Without MTP, the dense server's pool would have fit 250K, but we kept one length set for all configs.
- **Noise floor A is exactly zero.** At temperature 0 a server is bit-for-bit deterministic, even after a cold re-prefill of the same prompt (6 samples at 128K/192K).
- **Noise floor B is the honest yardstick:** the same dense model, weights and kernels on a different card of the same architecture (3090 Ti vs 3090, 84 vs 82 SMs), served by a different process. Kernel launch and split choices differ, and int8 KV / int8 activation rounding then amplifies tiny differences. Any change to the stack moves the distribution by about this much.
- **Fewer samples for the int8 tier:** 3 documents (2 code + 1 prose, 36 samples) instead of 5, to keep the production downtime near 1.5 h. Its "all" figure is therefore code-heavier. The same-subset comparison above is the fair one.
- **Tier configs are not purely "sparse decode".** They are the production config, so they include:
  - selection reuse on alternate steps;
  - the sampled normaliser;
  - the int4 sketch;
  - sparse attention for the uncached prompt suffix (the few dozen question tokens after a cache hit).

  The GPU-only row is the clean sparse-decode-only measurement.
- **Not covered:**
  - MTP on;
  - sampling at temperature > 0;
  - batches with several concurrent requests;
  - long free-running generations. Teacher forcing measures per-step fidelity on a fixed path, not how far a free-running generation drifts.
- **Basis fit:** the sketch basis was fit offline on a fixed corpus that includes coding-agent traffic over a vLLM checkout. The vLLM code document is therefore closest to in-distribution. Results for the other code and the prose are similar, apart from prose's higher divergence, which the dense-vs-dense floor shows too.

## Files / rerun

| file | what |
|---|---|
| `kld_proc.py` | the recorder / forcing logits processor (Model Runner V2 interface), loaded with `--logits-processors kld_proc:KLDRecorder` |
| `make_launch.sh` | builds `launch/`, a symlink mirror of your HyperQwen checkout. Its `single-user/start_qwen.sh` drops `--speculative-config` and adds the logits processor |
| `kld_serve.sh` | `ref` / `toggle` / `tier8` / `tier4` server configs |
| `build_samples.py` | documents → `data/*.npy` + `data/samples.json` |
| `kld_client.py` | drives a server: reference generation, toggle (dense + sparse) or forced runs |
| `run_gpu.sh` | boot → client → stop, per GPU |
| `kld_analyze.py` | KL / top-1 tables (`results/results.md`, `results/results.json`) |
| `results/client.*.jsonl` | per-request timing, cached tokens and the first 400 characters of each output |

```bash
export HQ_REPO=/path/to/HyperQwen   # checkout + venv with this release installed (top-level README)
export KLD_DIR=$PWD API_KEY_FILE=/path/to/api_key.txt KLD_API_KEY_FILE=/path/to/api_key.txt
bash make_launch.sh
$HQ_REPO/venv/bin/python build_samples.py data $HQ_REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
./run_gpu.sh 1 8099 ref:ref tier4:forced:tier4 &      # reference first; the other GPU waits for each reference file
./run_gpu.sh 0 8097 toggle:toggle tier8:forced:tier8 &
wait; $HQ_REPO/venv/bin/python kld_analyze.py data out --md results/results.md --json results/results.json
```
About 80 minutes on the two cards (dominated by the 192K prefills). The `.npz` step records (about 0.4 GB) are not committed.
