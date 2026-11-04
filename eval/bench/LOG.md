# Benchmark event log

Anonymized copy of the live progress log written during the run (UTC, minute resolution). Lines tagged `gpu0`/`gpu1`
were appended automatically by the queue runner when a job finished (GPU0 = RTX 3090 Ti, GPU1 = RTX 3090);
`director` lines are the operator's notes and decisions, written as results came in. Only host names, internal paths
and service names were rewritten; timestamps, numbers and wording are otherwise unchanged.

How to read it:
- Job labels: `xfile` = cross-file call-site retrieval, `ruler-A` = RULER subset run, `ruler` = full-shard RULER
  (241 items), `lbv2` = LongBench v2, `*-short` = GPQA / LCB / IFBench / MMLU-Pro, `mmlupro_hs` = MMLU-Pro after
  150K, `mtp` = speed run, `s5 ...` / `s6 ...` = development-branch tests. `cfg=` is the server kind (dense, sparse,
  tier4 = int4 RAM tier, tier8 = int8 RAM tier). Which tier runs were first-release and which were fixed is listed per
  file in [CONFIGS.md](CONFIGS.md); several `cfg=tier4` lines are ablations stored in their own result dirs.
- Interim numbers in `director` lines were computed on whatever had finished at that moment (often one shard or a
  partial run). [RESULTS.md](RESULTS.md) supersedes them. For example, the 13:44Z short-suite line used shard 0 only
  for the tier (its shard-1 jobs finished 14:19-16:46Z), and LiveCodeBench pass/fail was re-executed for publication
  on a different machine (6 s per-test timeout).
- "knobs" = the speed settings `SPARSE_STATS_SUB=4` + `SPARSE_SELECT_EVERY=2` ("knobs off" = 1 + 1). "suffix sparse"
  = sparse attention over a follow-up turn's uncached prompt suffix (`SPARSE_TIER_SUFFIX_SPARSE`).
- The S6 disk-resume tests still running when the data was packaged (0703-0705) are not included.

- 2026-11-02T14:23Z: production serving paused (both GPUs freed); round A (xfile + RULER) queued on both GPUs
- 2026-11-02 14:39Z gpu1: xfile cfg=tier4 shard=1 done 60/60 in 10 min
- 2026-11-02 14:41Z gpu0: xfile cfg=dense shard=0 done 60/60 in 12 min
- 2026-11-02T14:48Z plan: A=xfile+RULER(n per task 6/6/4/2 per config @32/64/128/192K)+LongBench v2 (48) for dense/sparse/tier4; B=GPQA/LCB/IFBench/MMLU-Pro dense vs tier4; C=SCBench; D=MTP speed + KLD 240K; tier8 runs last (cut first if time runs out)
- 2026-11-02 16:53Z gpu1: ruler-A cfg=tier4 shard=1 done 107/107 in 135 min
- 2026-11-02 17:23Z gpu1: lbv2 cfg=tier4 shard=1 done 24/24 in 29 min
- 2026-11-02 17:28Z gpu1: FAILED to boot sparse for 030-sparse-xfile.json
- 2026-11-02 17:30Z gpu1: FAILED to boot sparse for 031-sparse-ruler.json
- 2026-11-02 17:32Z gpu1: FAILED to boot sparse for 032-sparse-lbv2.json
- 2026-11-02 17:37Z gpu0: ruler-A cfg=dense shard=0 done 114/114 in 176 min
- 2026-11-02 17:44Z gpu1: xfile cfg=dense shard=1 done 60/60 in 10 min
- 2026-11-02 18:05Z director: gpu1 sparse boot fix (KV 7.34>7.2 GiB) -> GPU_UTIL=0.945, re-queued 030-032
- 2026-11-02 18:12Z gpu0: lbv2 cfg=dense shard=0 done 24/24 in 35 min
- 2026-11-02 18:26Z gpu0: xfile cfg=sparse shard=0 done 60/60 in 12 min
- 2026-11-02 19:57Z gpu1: ruler-A cfg=dense shard=1 done 107/107 in 133 min
- 2026-11-02 20:09Z gpu1: xfile cfg=sparse shard=1 done 60/60 in 10 min
- 2026-11-02 20:17Z director: tier4 down vs dense on xfile/RULER while sparse GPU-only = dense; queued ablations 0311 (tier4, exact stats + select every step) and 0312 (+int8 sketch) on gpu1 xfile
- 2026-11-02 21:20Z gpu0: ruler-A cfg=sparse shard=0 done 114/114 in 174 min
- 2026-11-02 21:55Z gpu0: lbv2 cfg=sparse shard=0 done 24/24 in 35 min
- 2026-11-02 22:07Z gpu0: xfile cfg=tier4 shard=0 done 60/60 in 11 min
- 2026-11-02 22:22Z gpu1: ruler-A cfg=sparse shard=1 done 107/107 in 133 min
- 2026-11-02 22:34Z gpu1: xfile cfg=tier4 shard=1 done 60/60 in 11 min
- 2026-11-02 22:46Z gpu1: xfile cfg=tier4 shard=1 done 60/60 in 11 min
- 2026-11-02 22:49Z director: ablation result: knobs off recovers ~1/3 of the xfile loss, int8 sketch no change; queued 0313 tier8+knobs-off and moved tier8 xfile up (0314) to test the int4 host copy
- 2026-11-02 23:05Z director: added mmlupro_hs (48 MMLU-Pro, thinking ON, sampled, after a fixed 150K-token haystack) dense/sparse/tier4 on both GPUs, ahead of the short suites
- 2026-11-02 23:16Z gpu1: lbv2 cfg=sparse shard=1 done 24/24 in 29 min
- 2026-11-02 23:28Z gpu1: xfile cfg=tier8 shard=1 done 60/60 in 11 min
- 2026-11-02 23:40Z gpu1: xfile cfg=tier8 shard=1 done 60/60 in 11 min
- 2026-11-02 23:45Z director: int8 host copy is NOT the cause (tier8 knobs-off -6.8 F1 ~ tier4 knobs-off -9.2). Suspect: S4 sparse-suffix prefill (xfile items share 99.8% prefix -> question prefilled sparsely in tier only). Queued 03145 (suffix KSEL 16384) + 03146 (suffix sparse off) after the running hs job
- 2026-11-02 23:59Z gpu1: mmlupro_hs cfg=dense shard=1 done 24/24 in 18 min
- 2026-11-03 00:02Z gpu1: xfile cfg=tier4 shard=1 done 2/60 in 1 min errors=58
- 2026-11-03 00:14Z gpu1: xfile cfg=tier4 shard=1 done 60/60 in 11 min
- 2026-11-03 00:15Z director: FOUND IT: S4 sparse-suffix prefill. tier4 knobs-off + SUFFIX_SPARSE=0 -> xfile F1 -2.1 [-7.1,+2.3] vs dense (was -13.7 prod), same wall time at conc1. KSEL16K crashes (HTTP 500). Queued 03147 (suffix off, knobs on) + 03148 (fix on RULER)
- 2026-11-03 00:24Z gpu1: mmlupro_hs cfg=sparse shard=1 done 24/24 in 10 min
- 2026-11-03 00:26Z director: all remaining tier4/tier8 jobs now run with SPARSE_TIER_SUFFIX_SPARSE=0 (release-candidate fix); prod-as-shipped results kept as the before record
- 2026-11-03 00:36Z gpu1: xfile cfg=tier4 shard=1 done 60/60 in 11 min
- 2026-11-03 00:40Z director: fix + knobs ON = F1 -5.1 [-10.4,-0.2] vs knobs off -2.1 -> release candidate = suffix sparse off + STATS_SUB=1 + SELECT_EVERY=1; all remaining tier jobs updated
- 2026-11-03 00:50Z gpu1: xfile cfg=tier8 shard=1 done 60/60 in 12 min
- 2026-11-03 00:52Z director: fixed config xfile g1, paired vs dense: F1 int4 -2.1, int8 -2.3, sparse GPU-only -3.6 (this shard); exact int4 -3.3, int8 0.0, sparse +1.7 -> all within noise, int4 = int8
- 2026-11-03 00:55Z director: queued q4 interaction test (fake-int4 GPU KV; dense_q4/sparse_q4): xfile pair right after 03148 RULER, RULER pair at the end of gpu1; KLD on gpu0 later (needs gpu0 runner STOP)
- 2026-11-03 01:00Z gpu0: ruler-A cfg=tier4 shard=0 done 114/114 in 173 min
- 2026-11-03 01:12Z gpu0: mmlupro_hs cfg=dense shard=0 done 24/24 in 11 min
- 2026-11-03 01:27Z gpu0: mmlupro_hs cfg=sparse shard=0 done 24/24 in 14 min
- 2026-11-03 01:34Z gpu0: mmlupro_hs cfg=tier4 shard=0 done 4/24 in 6 min errors=20
- 2026-11-03 01:34Z gpu0: lbv2 cfg=tier4 shard=0 done 0/24 in 0 min errors=24
- 2026-11-03 01:34Z gpu0: gpqa-short cfg=tier4 shard=0 done 0/99 in 0 min errors=99
- 2026-11-03 01:34Z gpu0: lcb-short cfg=tier4 shard=0 done 0/40 in 0 min errors=40
- 2026-11-03 01:34Z gpu0: ifbench-short cfg=tier4 shard=0 done 0/150 in 0 min errors=150
- 2026-11-03 01:34Z gpu0: mmlupro-short cfg=tier4 shard=0 done 0/140 in 0 min errors=140
- 2026-11-03 02:01Z director: gpu0 tier4 server (fix: SUFFIX_SPARSE=0) OOM-crashed at 01:34 in mmlupro_hs conc=4 (15x "staging grows", aten::empty OOM in marlin) = predicted concurrent-staging overflow. Requeued 0232(conc1)/042/510-513 as *-rq; gpu1 hs tier4 conc->1. Bad outputs moved aside (not published; the 6 jobs were re-run, see RUNTIME.md)
- 2026-11-03 02:09Z director: S5 staged jobs queued: gpu0 0200 a1 (hs conc4 repro, STAGE_SERIAL) / 0201 a2 (SUFFIX_EXACT+DMA) / 0202 c1-c4 timing; gpu1 031476 b1-b3 xfile accuracy after RULER
- 2026-11-03 03:05Z gpu0: gpqa-short cfg=dense shard=0 done 99/99 in 87 min
- 2026-11-03 03:20Z gpu0: mmlupro_hs s5 serial conc4 (0232 crash repro) cfg=tier4 shard=0 done 24/24 in 11 min
- 2026-11-03 03:22Z director: S5 a1 PASS: STAGE_SERIAL + fix at conc4 on the 0232 crash repro: 24/24, 0 staging grows, 0 tracebacks, 11 min; MMLU-Pro-after-150K 22/24 (dense 22/24, sparse 21/24, shard 0)
- 2026-11-03 03:31Z gpu0: mmlupro_hs s5 exact conc4 (0232 crash repro) cfg=tier4 shard=0 done 24/24 in 9 min
- 2026-11-03 03:53Z gpu0: xfile s5 timing s4ref conc4 cfg=tier4 shard=1 done 60/60 in 20 min
- 2026-11-03 04:16Z gpu0: xfile s5 timing serial conc4 cfg=tier4 shard=1 done 60/60 in 21 min
- 2026-11-03 04:38Z gpu0: xfile s5 timing exact conc4 cfg=tier4 shard=1 done 60/60 in 21 min
- 2026-11-03 04:48Z director: 4-agent xfile (conc4): time S4 shipped 20 min / serial 21 / exact 21; F1 vs dense: shipped -6.9 [-13.6,-0.8], serial -1.3 [-3.9,+1.3], exact -4.8 [-11.4,+0.9] -> release candidate v2 = STAGE_SERIAL=1 + SUFFIX_SPARSE=0 + knobs off
- 2026-11-03 05:00Z gpu0: xfile s5 timing hyb128 conc4 cfg=tier4 shard=1 done 60/60 in 20 min
- 2026-11-03 05:13Z gpu0: mmlupro_hs cfg=tier4 shard=0 done 24/24 in 12 min
- 2026-11-03 05:47Z gpu0: lbv2 cfg=tier4 shard=0 done 24/24 in 35 min
- 2026-11-03 07:18Z gpu0: gpqa-short cfg=tier4 shard=0 done 99/99 in 91 min
- 2026-11-03 07:33Z gpu1: ruler cfg=tier4 shard=1 done 241/241 in 402 min
- 2026-11-03 07:50Z gpu1: xfile s5 serial cfg=tier4 shard=1 done 60/60 in 11 min
- 2026-11-03 08:02Z gpu1: xfile s5 exact cfg=tier4 shard=1 done 60/60 in 10 min
- 2026-11-03 08:08Z gpu0: lcb-short cfg=tier4 shard=0 done 40/40 in 50 min
- 2026-11-03 08:13Z gpu1: xfile s5 hyb128 cfg=tier4 shard=1 done 60/60 in 10 min
- 2026-11-03 08:28Z gpu1: xfile dense_q4 cfg=dense shard=1 done 60/60 in 11 min
- 2026-11-03 08:40Z gpu1: xfile sparse_q4 cfg=sparse shard=1 done 60/60 in 10 min
- 2026-11-03 08:42Z director: RULER g1 fixed -0.3 [-2.6,+1.4] vs dense (shipped -2.6). S5 xfile conc1 F1: serial -1.9 [-5.2,+1.4] = exact (identical), question-only -5.6 [-10.8,-1.0] (reject). Q4 2x2 xfile: int4 damage dense -1.5, sparse -1.2 F1; interaction +0.3 [-7.6,+8.4] -> hypothesis not supported on xfile; int4 KV ~harmless for both. RULER q4 pair pending
- 2026-11-03 08:53Z gpu1: mmlupro_hs cfg=tier4 shard=1 done 24/24 in 12 min
- 2026-11-03 09:23Z gpu1: lbv2 cfg=dense shard=1 done 24/24 in 29 min
- 2026-11-03 09:26Z gpu0: ifbench-short cfg=tier4 shard=0 done 150/150 in 77 min
- 2026-11-03 09:58Z gpu0: mmlupro-short cfg=tier4 shard=0 done 140/140 in 32 min
- 2026-11-03 10:41Z gpu1: gpqa-short cfg=dense shard=1 done 99/99 in 74 min
- 2026-11-03 10:52Z gpu0: lcb-short cfg=dense shard=0 done 40/40 in 53 min
- 2026-11-03 11:27Z gpu1: lcb-short cfg=dense shard=1 done 40/40 in 47 min
- 2026-11-03 12:09Z gpu0: ifbench-short cfg=dense shard=0 done 150/150 in 77 min
- 2026-11-03 12:28Z gpu1: ifbench-short cfg=dense shard=1 done 150/150 in 60 min
- 2026-11-03 12:37Z gpu0: mmlupro-short cfg=dense shard=0 done 140/140 in 28 min
- 2026-11-03 13:04Z gpu1: mmlupro-short cfg=dense shard=1 done 140/140 in 37 min
- 2026-11-03 13:20Z gpu0: scbench cfg=dense shard=0 done 16/16 in 42 min
- 2026-11-03 13:44Z director: overnight: short thinking suites fixed-tier vs dense (paired): GPQA +0.0, LCB +0.0, IFBench +0.7, MMLU-Pro -0.7 (all within noise; short ctx mostly dense path). MMLU-Pro thinking after 150K doc: dense 44/48, sparse 44/48, fixed tier 44/48. LBv2 no-CoT: all 16.7% (below chance; uninformative)
- 2026-11-03 14:03Z gpu0: scbench cfg=sparse shard=0 done 16/16 in 42 min
- 2026-11-03 14:19Z gpu1: gpqa-short cfg=tier4 shard=1 done 99/99 in 73 min
- 2026-11-03 14:45Z gpu0: scbench cfg=tier4 shard=0 done 16/16 in 41 min
- 2026-11-03 14:57Z gpu0: mtp cfg=dense shard=0 done 8/8 in 11 min
- 2026-11-03 15:08Z gpu1: lcb-short cfg=tier4 shard=1 done 40/40 in 49 min
- 2026-11-03 15:09Z gpu0: mtp cfg=sparse shard=0 done 8/8 in 10 min
- 2026-11-03 15:24Z gpu0: mtp cfg=tier4 shard=0 done 10/10 in 14 min
- 2026-11-03 15:38Z gpu0: xfile cfg=tier8 shard=0 done 60/60 in 12 min
- 2026-11-03 16:08Z gpu1: ifbench-short cfg=tier4 shard=1 done 150/150 in 60 min
- 2026-11-03 16:46Z gpu1: mmlupro-short cfg=tier4 shard=1 done 140/140 in 38 min
- 2026-11-03 17:17Z gpu1: scbench cfg=tier4 shard=1 done 16/16 in 32 min
- 2026-11-03 17:50Z gpu1: scbench cfg=sparse shard=1 done 16/16 in 31 min
- 2026-11-03 18:22Z gpu1: scbench cfg=dense shard=1 done 16/16 in 31 min
- 2026-11-03 18:29Z gpu1: mtp cfg=tier4 shard=1 done 5/5 in 6 min
- 2026-11-03 18:35Z gpu1: mtp cfg=sparse shard=1 done 4/4 in 4 min
- 2026-11-03 18:36Z gpu0: ruler-A cfg=tier8 shard=0 done 114/114 in 177 min
- 2026-11-03 18:40Z gpu1: mtp cfg=dense shard=1 done 4/4 in 4 min
- 2026-11-03 19:11Z gpu0: lbv2 cfg=tier8 shard=0 done 24/24 in 35 min
- 2026-11-03 19:53Z gpu0: scbench cfg=tier8 shard=0 done 16/16 in 42 min
- 2026-11-03 19:53Z gpu0: queue empty, waiting for jobs
- 2026-11-03 20:12Z director: gpu0 queue empty -> running q4 KLD (toggle_ctl + toggle_q4 + dense_q4 forced) then tier8 lbv2/scbench shard1
- 2026-11-03 20:58Z gpu1: ruler-A cfg=tier8 shard=1 done 107/107 in 136 min
- 2026-11-03 21:28Z gpu1: lbv2 cfg=tier8 shard=1 done 24/24 in 30 min
- 2026-11-03 21:34Z gpu0: q4 KLD finished rc=0
- 2026-11-03 21:35Z director: moved 961-sparse_q4-ruler to gpu0 (runs after q4 KLD) to halve the tail; cross-card noise noted for the q4 RULER interaction
- 2026-11-03 21:38Z director: chain bug: moved running 940 to gpu0 -> duplicate started 21:35; killed gpu0 copy (no records written), restored file to gpu1, restarted gpu0 runner -> 961 sparse_q4 RULER
- 2026-11-03 22:00Z gpu1: scbench cfg=tier8 shard=1 done 16/16 in 32 min
- 2026-11-04 00:13Z gpu1: ruler-A dense_q4 cfg=dense shard=1 done 107/107 in 132 min
- 2026-11-04 00:13Z gpu1: queue empty, waiting for jobs
- 2026-11-04 00:23Z director: benchmark tail; gpu1 -> S6 GPU unit tests + microbench (test_s6_gpu.py)
- 2026-11-04 00:24Z director: S6 GPU tests: T3/T4 OK (hier == flat at M=all, graph-deterministic); T1 prefetch FAILS (pos2row mismatch; item shelved). T5 microbench: hier select NOT faster as drafted: 246Kx4 flat 17.8 vs hier 16.0 ms/step (-10%), 64K hier 2x slower, 1M ~same -> kernels latency-bound (~1 ms/layer fixed). Item 4 demoted pending kernel fusion. Queued 0721/0722 (basis old vs lmse, xfile) on gpu1
- 2026-11-04 00:27Z gpu0: ruler-A sparse_q4 cfg=sparse shard=1 done 107/107 in 164 min
- 2026-11-04 00:27Z gpu0: queue empty, waiting for jobs
- 2026-11-04 00:38Z director: BENCHMARK COMPLETE (last job 00:27). gpu0 idle -> S6 disk-resume tests 0701-0705
- 2026-11-04 00:40Z gpu1: s6 call-site F1, release basis (control for b1/b2) cfg=tier4 shard=1 done 60/60 in 11 min
- 2026-11-04 00:51Z gpu0: s6 resume: prime A(128K)+B(192K), persist to disk cfg=tier4 shard=None done 2/2 in 7 min
- 2026-11-04 00:53Z gpu1: s6 call-site F1, query-aware logit-MSE basis (Gram fit, CPU) cfg=tier4 shard=1 done 60/60 in 11 min
- 2026-11-04 00:53Z gpu1: queue empty, waiting for jobs
- 2026-11-04 00:55Z director: FINAL: SCBench (both shards) vs dense: fixed int4 -1.2 [-4.4,+1.9], int8 -1.9, sparse GPU-only -3.1 (prefix_suffix -12.5). q4 interaction: RULER +0.6 [-2.5,+3.9], xfile F1 +3.1 [-3.9,+10.6], KLD sensitivity +0.0006 [-0.0003,+0.0016] -> not supported; int4 damage small for both (RULER dense -1.2, sparse -0.6). Basis lmse vs old (xfile F1 vs dense): -1.5 vs -2.5, exact -1.7 vs -5.0 (n.s.)
- 2026-11-04 00:59Z director: stopping for host maintenance (storage configuration change). Runners stopped (queue/STOP). S6 disk-resume 0702-0705 interrupted (0701 prime done; rerun 0701-0705 after boot). Production server autostart disabled until the fix is shipped
- 2026-11-04 01:17Z director: host cold-booted (both GPUs unlocked). Disk after the storage change: NVMe data volume 1.24 GB/s @4 streams (raw NVMe ~1.06 GB/s/stream -> drive-bound); SATA scratch volume ~0.45-0.5 GB/s. gpu0: S6 disk resume 0701-0705; gpu1: S6 capture->distill->eval_recall chain
- 2026-11-04 01:18Z gpu0: s6 resume: prime A(128K)+B(192K), persist to disk cfg=tier4 shard=None done 2/2 in 0 min
- 2026-11-04 01:24Z gpu0: s6 resume: RESTARTED server, follow-ups from disk cfg=tier4 shard=None done 2/2 in 4 min
