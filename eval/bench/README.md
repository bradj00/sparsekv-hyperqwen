# Accuracy benchmark: sparse-KV decode and the RAM tier vs dense attention

This folder holds the complete benchmark behind the accuracy claims of this repo: every per-item result, the scorers,
the job runtimes and the event log, packaged so that every number can be traced back to a model output.

- **Question.** Does query-aware sparse decode (top-2048 + sinks + recent tail per step), alone or with the host-RAM
  KV tier at int8 or int4, cost accuracy against dense attention on the same model, stack and cards? We tested on
  long-context retrieval and reasoning (32K-192K), multi-turn reuse of a cached context, and short "thinking" tasks.
- **Why.** The KL-divergence test in [`eval/kld`](../kld/README.md) measures per-token drift. These suites measure end
  task accuracy, including follow-up turns over a cached prefix. Those turns run through code paths that the KL test
  does not exercise.
- **What happened.** The first release lost accuracy on cross-file retrieval with the RAM tier on (-10.9 F1 points vs
  dense). The benchmark located the cause (sparse prefill of the uncached question suffix, plus the speed knobs). The
  fixed configuration is within noise of dense on every suite. A concurrency-4 out-of-memory crash was also found and
  fixed along the way. The details are in [Found and fixed](#found-and-fixed-during-the-run) below and in [LOG.md](LOG.md).

| file | contents |
|---|---|
| [RESULTS.md](RESULTS.md) | every scorer table: per length, per task, per turn, ablations, 4-agent runs, q4 2x2, MTP speed, KLD q4 |
| [CONFIGS.md](CONFIGS.md) | what each config is, and the internal result dir -> public name mapping |
| [RUNTIME.md](RUNTIME.md), [runtime.csv](runtime.csv) | every job (incl. crashed, re-run and KLD stages): GPU, items, wall time, UTC start/end, server boot |
| [LOG.md](LOG.md) | the event log written during the run (anonymized) |
| `results/` | per-item records for every suite and config (JSONL, gzipped when > 256 KB) + `MANIFEST.json` + `kld/` |
| `gt/` | redistributable ground truth (RULER, cross-file, LongBench v2 answers, SCBench, MMLU-Pro answers) |
| `harness/` | generators, client, queue runner, serve script, scorers, packaging and reproduction scripts |

## Setup

**Hardware.** 1x RTX 3090 Ti (PCIe Gen4 x16) + 1x RTX 3090 (PCIe Gen4 slot at x4), 24 GB each, both capped at 250 W,
Intel i9-12900K host, serving in a VM with GPU passthrough. GPU0 = 3090 Ti, GPU1 = 3090. One server per GPU; the two
GPUs ran independent job queues in parallel. **Model/stack:** Qwen3.8-27B W4A16 (AutoRound), HyperQwen (vLLM 0.31
fork) @ `ce9615c` + pthfa int8 KV, int8 activations/prefill, prefix caching, block size 864, MTP-7 speculative
decoding on, with the sparse-KV release of this repo installed (S5/S6 rows: development branches; q4 rows: a debug
build with a fake-int4 KV hook). Full rig details: [top-level README](../../README.md#the-rig).

**Configs** ([CONFIGS.md](CONFIGS.md) has the exact environment of every file):

- **dense**: no `SPARSE_*` at all.
- **sparse GPU-only**: sparse decode on every step (int8 sketch, exact statistics, selection every step), no tier.
- **first release, int4 / int8**: the production 4 x 250K tier command as first shipped.
- **fixed, int4 / int8**: the same with `SPARSE_TIER_SUFFIX_SPARSE=0 SPARSE_STATS_SUB=1 SPARSE_SELECT_EVERY=1`.
- Ablations, staging variants (S5), fake-int4 KV (q4) and sketch-basis variants (S6) are listed in CONFIGS.md.

**Suites.**

| suite | items (shard 0 / 1) | context | decoding | max tokens | conc. | metric | source data (license) |
|---|---|---|---|---|---|---|---|
| RULER, 13 tasks | 114 / 107 run, of 520 generated (10 per task x length) | 32K, 64K, 128K, 192K | greedy, thinking off | 30-128 (per task, RULER defaults) | 1 | RULER string match (all / part) | NVIDIA RULER generator (Apache-2.0); haystack = Paul Graham essays (not redistributed); qa_1/qa_2 = SQuAD / HotpotQA (CC BY-SA 4.0) |
| Cross-file call sites (`xfile`) | 60 / 60 | 64K, 128K, 192K | greedy | 1024 | 1 | F1 over (file, enclosing function) pairs; also precision, recall, exact | packed source of vLLM (Apache-2.0) and PyTorch `torch/_inductor` (BSD-3); ground truth from the Python AST |
| LongBench v2 subset | 24 / 24 (12 per length bin) | <=32K ... 128-192K | greedy, 0-shot, no CoT | 128 | 1 | exact answer letter | THUDM/LongBench-v2 (Apache-2.0) |
| SCBench multi-turn | 16 / 16 (8 per task: kv, repoqa, prefix_suffix, qa_eng) | up to ~190K, 5 turns, own answers kept | greedy | official per-task budgets (40-1024) | 1 | official SCBench metric, mean over turns | microsoft/SCBench (MIT) |
| MMLU-Pro after 150K (`mmlupro_hs`) | 24 / 24 | ~150K haystack (LongBench v2 docs) + question | thinking on, T=1.0, top-p 0.95, top-k 20, per-item seed | 32768 | 1 | answer letter | TIGER-Lab/MMLU-Pro (MIT); haystack from LongBench v2 (Apache-2.0) |
| GPQA Diamond | 99 / 99 | short | thinking on, same sampling | 32768 | 4 | answer letter (simple-evals pattern) | GPQA (CC BY 4.0; gated, authors ask that examples not be published in plain text) |
| LiveCodeBench | 40 / 40 (2025-02-01..05-31 window; 19 easy / 24 medium / 37 hard) | short | thinking on, same sampling | 32768 | 4 | pass@1, all public + private tests | livecodebench/code_generation_lite (harness MIT; problems from AtCoder / LeetCode / Codeforces, not redistributed) |
| IFBench | 150 / 150 | short | thinking on, same sampling | 32768 | 4 | loose prompt-level accuracy (strict kept too) | allenai/IFBench_test (ODC-BY-1.0) |
| MMLU-Pro | 140 / 140 | short | thinking on, same sampling | 32768 | 4 | answer letter (official regexes) | TIGER-Lab/MMLU-Pro (MIT) |
| MTP speed (`mtp`) | 10 / 5 | 32K-240K, 3 documents | greedy "explain" | 512 | 1 | decode tok/s, accepted tokens per step | vLLM / PyTorch source, *War and Peace* (Project Gutenberg, public domain) |
| KLD q4 2x2 | 60 samples | 32K-192K | teacher-forced, MTP off | 256 | 1 | KL(dense ‖ run), top-1 agreement | see [`eval/kld`](../kld/README.md) |

## Methodology

- **Shards.** Every suite's items are split into two halves by index parity. Shard 0 always ran on GPU0 and shard 1
  on GPU1, except the 4-agent S5 runs (shard 1 on GPU0) and the sparse fake-int4 RULER run (shard 1 on GPU0).
  For most suites a config was run on both shards. Some fixed-config runs exist for one shard only; the tables
  label this.
- **Paired comparison.** Every config is compared with dense **on the same items** (same prompts, same seeds for
  sampled suites). For shard-matched runs this also means the same card. Tables report the per-config mean with a 95%
  bootstrap CI over items, and the **paired difference vs dense** with a 95% paired-bootstrap CI (5,000 resamples of
  the per-item differences, percentile interval, RNG seed 0). Items that errored are excluded from both sides.
  Scores are in points (0-100).
- **Decoding.** Long-context suites are greedy with thinking off. Short suites use thinking on and the
  model's recommended sampling (T=1.0, top-p 0.95, top-k 20) with a fixed per-item seed (`sha256(id)`). Dense and the
  tier therefore sample with identical seeds, but per-token logits differ, so sampled traces diverge. Treat
  short-suite deltas as noisy.
- **Prompts** use the Qwen chat template and are rendered offline into request files (`harness/*_gen.py`,
  `harness/mkreqs.py`). `harness/client.py` sends them to `/v1/completions` and stores the raw text, finish
  reasons, token usage, TTFT and wall time. When concurrency is 1 it also stores the spec-decode and prefix-cache
  counter deltas.
- **Scoring** reuses the official extraction and metrics where they exist: RULER string match, simple-evals GPQA
  pattern, MMLU-Pro regexes, the IFBench evaluation library, LiveCodeBench `testing_util.run_test` in a sandbox, and
  SCBench `compute_scores` copied verbatim. Details are in each scorer's docstring.

## Headline results

Paired difference vs dense in points, 95% CI, n = paired items (blank = not run). Generated from `results/` by
`harness/reproduce.py`; full tables in [RESULTS.md](RESULTS.md).

<!-- HEADLINE:BEGIN (generated by harness/reproduce.py) -->
| suite | dense | sparse GPU-only | fixed, RAM tier int8 | **fixed, RAM tier int4** | first release, int4 (before fix) |
|---|---|---|---|---|---|
| RULER, 13 tasks, 32K-192K (score) | 86.2 (n=221) | +0.6 [-0.3, +1.8] (n=221) | +0.3 [-1.6, +2.3] (n=221) | **-0.3 [-2.6, +1.4] (n=107)** | -1.0 [-3.0, +0.9] (n=221) |
| Cross-file call sites, 64K-192K (F1) | 82.2 (n=120) | -0.9 [-3.8, +1.8] (n=120) | -2.0 [-5.4, +1.1] (n=120) | **-2.1 [-7.2, +2.2] (n=60)** | -10.9 [-16.3, -5.9] (n=120) |
| LongBench v2, no CoT (acc.) | 16.7 (n=48) | +0.0 [+0.0, +0.0] (n=48) | +2.1 [+0.0, +6.2] (n=48) | **-4.2 [-12.5, +0.0] (n=24)** | +4.2 [+0.0, +12.5] (n=24) |
| SCBench multi-turn, 4 tasks (official metric) | 78.8 (n=32) | -3.1 [-6.9, +0.0] (n=32) | -1.9 [-5.0, +1.2] (n=32) | **-1.2 [-4.4, +1.9] (n=32)** |  |
| MMLU-Pro thinking after 150K haystack (acc.) | 91.7 (n=48) | +0.0 [-6.2, +6.2] (n=48) |  | **+0.0 [-6.2, +6.2] (n=48)** |  |
| GPQA Diamond, thinking (acc.) | 75.8 (n=198) |  |  | **+0.0 [-3.5, +4.0] (n=198)** |  |
| LiveCodeBench v5/v6 window, thinking (pass@1) | 50.0 (n=80) |  |  | **-2.5 [-10.0, +5.0] (n=80)** |  |
| IFBench, thinking (loose prompt acc.) | 80.3 (n=300) |  |  | **+1.3 [-2.3, +5.0] (n=300)** |  |
| MMLU-Pro, thinking (acc.) | 82.5 (n=280) |  |  | **-2.1 [-5.7, +1.1] (n=280)** |  |
<!-- HEADLINE:END -->

How to read it:

- **Fixed config.** No paired delta is significant: every 95% CI includes 0 (two LongBench v2 intervals end exactly
  at 0). The two RAM-tier storage widths (int8, int4) behave alike.
- **Fixed int4, coverage.** Its RULER and cross-file numbers come from shard 1 only. Its LongBench v2 number comes from
  shard 0 only (shard 1 ran the first release).
- **First release.** Cross-file F1 lost 10.9 points (CI excludes 0); that loss is the bug described below.
- **LongBench v2 is uninformative here.** Every config scores 12-21%, below the 25% chance level for 4 options. Most
  responses (28 of 48 for dense) start reasoning and hit the 128-token limit before the required "The correct answer
  is (X)" line, so they score as wrong.
- **Short thinking suites.** Contexts are short, so they mostly exercise the dense path (a sanity check that the tier
  build does no harm). They are sampled, so their CIs are wide.
- **KL divergence.** The teacher-forced per-token result is in [`eval/kld`](../kld/README.md). The fake-int4 KV 2x2
  (does int4 KV hurt sparse decode less than dense?) found no support for that hypothesis on RULER, cross-file or KLD
  (RESULTS.md sections 8 and 10).

## Found and fixed during the run

Times are UTC (2026-11-02/03); the evidence is in [LOG.md](LOG.md) and RESULTS.md section 4.

1. **18:05Z: sparse GPU-only failed to boot on GPU1.** Its KV need (7.34 GiB) exceeded the budget at
   `GPU_UTIL=0.93`. The run was re-queued at 0.945 (3 boot-failure rows in runtime.csv).
2. **20:17Z: the tier lagged dense on cross-file and RULER**, while sparse GPU-only matched dense. That pointed at
   tier-only code, and an ablation ladder followed on cross-file shard 1:
   - Knobs off (`STATS_SUB=1`, `SELECT_EVERY=1`) recovered about a third of the loss.
   - An int8 sketch changed nothing.
   - An int8 host copy with knobs off was no better than int4, so host storage width is not the cause.
3. **00:15Z: found the cause.** Cross-file prompts share 99.8% of their prefix, so the question is an uncached suffix
   after a cached document. In the tier profile that suffix was prefilled with *sparse* attention. Turning it off
   (`SPARSE_TIER_SUFFIX_SPARSE=0`) brought F1 to -2.1 [-7.1, +2.3] vs dense, against -13.7 for the as-shipped
   config on this shard. A wider sparse suffix (KSEL 16384) crashed the server (HTTP 500).
4. **00:40Z: release candidate.** Suffix sparse off with the knobs left on still lost 5.1 F1, so the fix also turns
   the knobs off. From 00:26Z all remaining tier jobs ran with the fixed settings.
5. **01:34Z: the fixed int4 server crashed out of memory** at concurrency 4 (mmlupro_hs). Concurrent host-to-GPU
   staging overflowed. The 6 affected jobs were discarded and re-run (`*-rq`).
   - The S5 branch serialises staging (`SPARSE_TIER_STAGE_SERIAL=1`). It passed the crash repro: 24/24, no staging
     growth.
   - With 4 agents on cross-file it scored -1.3 F1 vs dense, against -6.9 for as-shipped + serial staging.
6. **Audit note.** RULER shard 0 for `tier4` (job 041) is listed as **first release**. Its job file was edited to the
   fixed settings while the job was already running on a first-release server. The server's own environment line is
   authoritative (see [CONFIGS.md](CONFIGS.md#note-job-file-vs-server-environment)).

## Runtime

The full benchmark (plus KLD stages) is 105 job instances and **68.6 GPU-hours** (34.3 h per GPU), plus 2.1 h of
server boots. The benchmark queue ran 2026-11-02 14:24Z to 2026-11-04 01:24Z (35.0 h wall clock; the main benchmark
finished at 00:27Z, followed by S6 tests). Per suite, config and job: [RUNTIME.md](RUNTIME.md).

## What is published, and what is not

- **Per-item records** (`results/<config dir>/<suite>.g<shard>.jsonl[.gz]`) contain one JSON object per request:
  - `id`, `suite`, `variant`, `config_dir`, `shard`, `gpu`, `job`;
  - `outputs` (the model's text, one entry per turn), `finish`, `usage` (prompt / completion / cached tokens);
  - `ttft` and `secs` per turn, and `spec` (spec-decode accepted / drafted tokens and prefix-cache hits, concurrency-1
    jobs only);
  - `error`, then `score` (the number the tables use) and `score_detail` (e.g. extracted answer, precision / recall,
    per-turn scores, IFBench strict and loose, LCB tests passed).
- **No prompts or contexts are published.** That covers documents, haystacks, questions and few-shot text. Datasets
  are referenced by item id; the generators in `harness/` rebuild the prompts from the original sources.
  - RULER's Paul Graham essay haystack is deliberately excluded for licensing reasons.
  - PyTorch and vLLM source packs, LongBench v2 and SCBench contexts are not re-hosted.
- **GPQA:** model output text is withheld for every item, because thinking traces routinely restate the question.
  Only the extracted answer, correctness and token counts are published. No GPQA ground truth is published.
- **IFBench and LiveCodeBench ground truth** (instruction kwargs including prompts; hidden test cases) is not
  redistributed. Their per-item scores were computed with the full ground truth by `harness/package_results.py` and
  are stored in the records. `analyze.py` uses the stored scores when the ground-truth file is absent.
- **`gt/`** holds what is safe to redistribute:
  - RULER needle values and answers (QA answers are SQuAD / HotpotQA spans, CC BY-SA 4.0);
  - cross-file (file, function) pairs;
  - LongBench v2 answer letters and metadata (Apache-2.0);
  - SCBench answers (MIT);
  - MMLU-Pro answer letters (MIT);
  - the short-suite sampling manifest.
- **KLD:** the raw per-step distributions (588 MB of `.npz`) are not included. `results/kld/` has the per-sample
  results, the q4 2x2 results and the client logs (generated text and timings).

## Reproduce

**Re-score (CPU only, about 10 s).** You need Python 3.10+ with `numpy` and `nltk`, plus `tree_sitter_languages` for
SCBench repoqa:

```bash
cd eval/bench
python harness/reproduce.py      # rewrites RESULTS.md, results/RESULTS.json and the headline table above
python harness/runtime_tables.py # rewrites RUNTIME.md from runtime.csv
```

`reproduce.py` builds each comparison as a temporary directory of renamed copies of the published files, then runs
the unmodified table code (`analyze.py`, `q4_analyze.py`). The KLD q4 section needs the raw `.npz` files: run
`harness/kld_q4/kld_q4_analyze.py DATA OUT --md f` and pass the result with `--kld-md f`.

**Re-run.** You need the release installed per the [top-level README](../../README.md), the model, and the datasets.

1. Build the requests and ground truth, with a work dir `$BENCH` whose `src/` is this `harness/`:
   - `ruler_gen.sh` + `mkreqs.py ruler`;
   - `xfile_gen.py` (corpora `vllm/v1`, `torch/_inductor`);
   - `dl_hf.py` + `mkreqs.py lbv2`;
   - `scbench_gen.py`;
   - `short_gen.py` (GPQA CSV, LiveCodeBench release files, IFBench, MMLU-Pro);
   - `hs_gen.py`;
   - `mtp_gen.py`.
2. Queue jobs with `mkjob.py`. Start one `run_gpu.sh GPU PORT` per card. The runner boots
   `serve_bench.sh <dense|sparse|tier8|tier4>` with each job's env overrides; the environments are in
   `results/MANIFEST.json`.
3. Package and score with `package_results.py OUT_DIR GT_DIR MANIFEST.json results/` (it needs the full ground truth),
   then run `reproduce.py`.

## Known issues and caveats

- **Duplicate RULER ids.** Two RULER ids collide (`niah_single_2` at 128K and 192K: the generator reused an index).
  Only the full-shard fixed int4 run (`fix/ruler.g1`) contains both copies. The one matching the ground truth is
  scored and the other is marked in `score_detail`, so that file has 239 scored items.
- **LiveCodeBench re-execution.** LCB pass/fail was re-executed on the packaging machine (6 s per-test timeout, same
  sandbox). Two re-scoring runs agreed exactly. Results on other hardware could differ for time-limit edge cases.
- **Interim log numbers.** Numbers quoted in `director` lines of LOG.md were computed on partial data at the time;
  RESULTS.md is authoritative.
- **MTP speed cells.** These compare different documents and cards per cell. They are indicative only, not paired.
- **4-agent comparison.** The 4-agent S5 rows are compared against a concurrency-1 dense run on the other card.
