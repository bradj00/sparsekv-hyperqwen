# Configurations

Every result file lives in `results/<config dir>/<suite>.g<shard>.jsonl[.gz]`. The config dir names are the internal
ones used while the benchmark ran (so they match [LOG.md](LOG.md) and [runtime.csv](runtime.csv)). Each record also
carries a `variant` field with the public name, and [`results/MANIFEST.json`](results/MANIFEST.json) lists, per file,
the variant, the server environment actually used, the job(s), the GPU, the shard, the concurrency, the record counts
and a SHA-256 of the uncompressed content.

## Server configurations

All servers run the same weights (Qwen3.8-27B W4A16 AutoRound), the same tokenizer and the same base stack:
HyperQwen (vLLM 0.31) @ `ce9615c` + pthfa int8 KV + int8 activations/prefill, prefix caching, block size 864,
MTP-7 speculative decoding on (the served configuration), and the sparse-KV release from this repo installed. The
exact launch logic is [`harness/serve_bench.sh`](harness/serve_bench.sh).

| public name | server | what differs |
|---|---|---|
| **dense** | `serve_bench.sh dense` | No `SPARSE_*` variable at all: the plain base stack. `MAX_LEN=199000`, 1 sequence, `GPU_UTIL=0.93`. Short thinking suites: `MAX_LEN=65536`, 4 sequences. |
| **sparse GPU-only** (`sparse-gpu`) | `serve_bench.sh sparse` | `SPARSE_PROFILE=sparse` with `SPARSE_CUTOFF=0` (every decode step sparse): int8 sketch, exact statistics, selection every step, no RAM tier. `MAX_LEN=199000`, 1 sequence, `GPU_UTIL=0.945` (0.93 failed to boot on GPU1, see LOG.md 2026-11-02 18:05Z). |
| **first release, int4** (`release-int4`) | `serve_bench.sh tier4`, no overrides | The production 4 x 250K command of the first release: `SPARSE_PROFILE=tier MAX_LEN=262144 MAX_SEQS=4 GPU_UTIL=0.94 SPARSE_TIER_HOT_FRAC=0.025 SPARSE_SKETCH_BITS=4 SPARSE_TIER_HOST_GB=17.8 SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2 SPARSE_TIER_HOST_BITS=4`. Sparse prefill of an uncached follow-up suffix is on (the release default). |
| **first release, int8** (`release-int8`) | `serve_bench.sh tier8`, no overrides | Same, with `SPARSE_TIER_HOST_BITS=8` (int8 host copy). |
| **fixed, int4** (`fixed-int4`) | `tier4` + `SPARSE_TIER_SUFFIX_SPARSE=0 SPARSE_STATS_SUB=1 SPARSE_SELECT_EVERY=1` | The fix found during this benchmark: dense attention for the uncached prompt suffix, exact statistics, selection every step. |
| **fixed, int8** (`fixed-int8`) | `tier8` + the same three overrides | |

## Config dir -> public name

| config dir | variant (public name) | files | notes |
|---|---|---|---|
| `dense` | dense | all suites, shards 0 and 1 | |
| `sparse` | sparse GPU-only | xfile, ruler, lbv2, scbench, mtp; shards 0 and 1 | Short thinking suites were not run for sparse GPU-only (short prompts take the dense path). |
| `tier4` | **first release, int4**: `xfile.g0`, `xfile.g1`, `ruler.g0`, `ruler.g1`, `lbv2.g1`. **fixed, int4**: everything else (`lbv2.g0`, `scbench.*`, `gpqa.*`, `lcb.*`, `ifbench.*`, `mmlupro.*`, `mtp.*`) | | Classified from the environment the server was *started with* (the first line of each server log), not from the job file; see the note below. |
| `tier8` | **first release, int8**: `xfile.g1`. **fixed, int8**: `xfile.g0`, `ruler.*`, `lbv2.*`, `scbench.*` | | |
| `abl_knobs` | abl-knobsoff-int4 | xfile.g1 | first release int4 + `SPARSE_STATS_SUB=1 SPARSE_SELECT_EVERY=1` |
| `abl_knobs_sk8` | abl-knobsoff-sketch8-int4 | xfile.g1 | + `SPARSE_SKETCH_BITS=8` |
| `abl_knobs_tier8` | abl-knobsoff-int8 | xfile.g1 | first release int8 + knobs off |
| `abl_sfx16k` | abl-knobsoff-ksel16k-int4 | xfile.g1 | knobs off + `SPARSE_TIER_SUFFIX_KSEL=16384`; the server answered 2/60, the other 58 records are HTTP 500 errors (kept, `error` set) |
| `abl_sfxoff` | **fixed, int4** | xfile.g1 | knobs off + `SPARSE_TIER_SUFFIX_SPARSE=0`: the run that found the fix; identical environment to fixed int4, and the fixed int4 cross-file result |
| `fix` | **fixed, int4** | ruler.g1 | all 241 shard-1 RULER items (other configs ran a 107-item subset) |
| `fix8` | **fixed, int8** | xfile.g1 | |
| `fix_knobson` | fixed-knobson-int4 | xfile.g1 | `SPARSE_TIER_SUFFIX_SPARSE=0` only (knobs stay at 4/2) |
| `s5_serial`, `s5_serial_hs`, `s5t_serial` | s5-serial | xfile.g1 (conc 1), mmlupro_hs g0 (conc 4), xfile.g1 (conc 4) | S5 development branch, crash-safe serial staging: fixed int4 + `SPARSE_TIER_STAGE_SERIAL=1` |
| `s5_exact`, `s5_exact_hs`, `s5t_exact` | s5-exact | same | + `SPARSE_TIER_SUFFIX_EXACT=1 SPARSE_TIER_S5_DMA=1` |
| `s5_hyb128`, `s5t_hyb128` | s5-hyb128 | xfile.g1 (conc 1 / conc 4) | serial staging, knobs off, suffix sparse ON with 128 exact rows (`SPARSE_TIER_SUFFIX_EXACT_ROWS=128`, `SPARSE_TIER_SUFFIX_STATS_SUB=1`, DMA) |
| `s5t_s4ref` | s5-release+serial | xfile.g1 (conc 4) | first release int4 + `SPARSE_TIER_STAGE_SERIAL=1` only (the "S4 shipped" reference at 4 agents) |
| `s5t_*` | | | `t` = the 4-agent timing runs: concurrency 4 on GPU0 (RTX 3090 Ti), shard-1 items |
| `dense_q4` | dense-fakeq4 | xfile.g1, ruler.g1 | dense + `SPARSE_DEBUG_KV_FAKEQ4=1` (full-attention KV rounded to int4 in place, 64-dim groups, fp16 scales), q4test debug build |
| `sparse_q4` | sparse-fakeq4 | xfile.g1, ruler.g1 | sparse GPU-only + the same knob; the RULER file ran on GPU0 (RTX 3090 Ti), the rest of the q4 2x2 on GPU1 |
| `dense_hs`, `sparse_hs`, `tier4_hs` | dense / sparse GPU-only / **fixed, int4** | mmlupro (suite `mmlupro_hs`), shards 0 and 1 | MMLU-Pro with thinking after a fixed ~150K-token haystack |
| `s6_basis/old` | s6-basis-release | `old.g1` (= xfile.g1) | S6 branch control: fixed int4 + serial staging, release sketch basis |
| `s6_basis/lmse` | s6-basis-lmse | `lmse.g1` (= xfile.g1) | same, with a query-aware logit-MSE sketch basis (`basis27_lmse.pt`) |
| `s6_resume` | s6-disk-resume | `disk_prime`, `disk_resume` | S6 branch smoke test: prime two long prompts, persist the KV tier to local NVMe, restart the server, answer follow-ups from disk. Unscored. |

## Not included

- The outputs of the 6 jobs that were running when the fixed int4 server OOM-crashed at 2026-11-03 01:34Z
  (0232 mmlupro_hs at concurrency 4 with 4/24 answered, then 042, 510-513 with 0 answered). They were moved aside and
  all 6 jobs were re-run from scratch (`*-rq` jobs); the re-runs are the published `tier4` / `tier4_hs` files. Both the
  crashed runs and the re-runs are in [runtime.csv](runtime.csv).
- A scorer self-test directory (synthetic outputs, not model results).
- S6 disk-resume tests 0703-0705, still running when the data was packaged.

## Note: job file vs server environment

The queue runner restarts the server only when a job's `(cfg, env)` differs from the running server's. At 2026-11-03
00:26Z the operator rewrote the `env` of every queued tier job to the fixed settings. Job `041-tier4-ruler` (RULER
shard 0, GPU0) was already running at that time on a server started at 2026-11-02 21:55Z with an empty environment.
Its job file therefore says "fixed", but it ran entirely on the first-release server. `results/MANIFEST.json` marks
it `job_file_env_differs: true`, and it is classified as **first release, int4**. Every other file's job environment
matches its server log.

## Sketch of what each fix changed

- **Suffix sparse off** (`SPARSE_TIER_SUFFIX_SPARSE=0`): the uncached part of a prompt that shares a long cached
  prefix, such as the question after a document that is already in the prefix cache, is prefilled with dense
  attention instead of sparse attention. In the first release this path was only active in the tier profile. That is
  why the GPU-only sparse config never showed the loss.
- **Knobs off** (`SPARSE_STATS_SUB=1 SPARSE_SELECT_EVERY=1`): exact normaliser statistics and a fresh top-k selection
  at every decode step. The first release sampled every 4th tile and reused the selection on alternate steps.
- **Serial staging** (S5, `SPARSE_TIER_STAGE_SERIAL=1`): host-to-GPU staging for concurrent requests is serialised,
  so 4 agents cannot overflow the staging buffer. This removed the 01:34Z out-of-memory crash.
