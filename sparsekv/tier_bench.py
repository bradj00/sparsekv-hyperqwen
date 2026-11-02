# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Standalone benchmark of the host tier (stream E): ms per decode step for 16 sparse layers.

  python sparsekv/tier_bench.py gen   --ctx 131072 262144                # CPU: synthesize traces (outside the GPU lock)
  python sparsekv/tier_bench.py run   --ctx 131072 --hot 0.10 --preset lab   # GPU

Setup per request: context N, ALL blocks spilled except block 0 (sinks) and the tail blocks covering the 1024-token
window (+1 for the verify tokens) -- the worst case for the tier. Host mirror = pinned [16, nblk, 4, 864, 520] int8
read through a UVA view. Selection per layer per step: [topk 2048 from a synthetic trace | 64 sinks | 1024 window].
Timed: hot.next_step + 16 x gather_rows_tiered (resolve + copy into staging), eager and as one CUDA graph.
Baselines: same kernels with every block resident (pure GPU gather, the compaction path without a tier) and the
raw UVA read bandwidth on this card.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv.tier_trace import gen_trace, overlap_stats  # noqa: E402

HKV, BS, W, L = 4, 864, 520, 16
KSEL, SINK, WIN = 2048, 64, 1024
C = KSEL + SINK + WIN
TRACE_DIR = os.environ.get("TIER_TRACES", "traces")


def trace_path(N, preset, layer, steps):
    return f"{TRACE_DIR}/{preset}_{N}_{steps}_L{layer}.npy"


def cmd_gen(a):
    os.makedirs(TRACE_DIR, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(N, p, l) for N in a.ctx for p in a.preset for l in range(L)]
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count())) as ex:
        futs = [ex.submit(_gen_one, N, p, l, a.steps) for (N, p, l) in jobs]
        for f in futs:
            print(f.result(), flush=True)


def _gen_one(N, preset, l, steps):
    path = trace_path(N, preset, l, steps)
    if not os.path.exists(path):
        tr = gen_trace(N, steps, K=KSEL, sink=SINK, win=WIN, seed=1000 * l + N % 997, preset=preset)
        np.save(path, tr)
    else:
        tr = np.load(path)
    return path, {k: round(v, 3) for k, v in overlap_stats(tr[64:], N).items()}


def uva_bw(host_dev, nrows_pick=20000):
    """index_select of scattered 520-B head rows from the UVA view (lab method), GB/s."""
    flat = host_dev.view(torch.int32).view(-1, W // 4)
    idx = torch.randint(0, flat.shape[0], (nrows_pick * HKV,), device="cuda")
    out = torch.empty((idx.numel(), W // 4), dtype=torch.int32, device="cuda")
    for _ in range(3):
        torch.index_select(flat, 0, idx, out=out)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10):
        torch.index_select(flat, 0, idx, out=out)
    torch.cuda.synchronize()
    s = (time.perf_counter() - t) / 10
    return out.numel() * 4 / s / 1e9


def dma_bw(mirror, kv, nblocks=32):
    """Block mirroring (D2H, contiguous 1.8 MB blocks via copy_) and staging-style H2D, GB/s."""
    nb = min(nblocks, kv.shape[0], mirror.HB)
    gb = list(range(nb)); hb = list(range(nb))
    for _ in range(2):
        mirror.mirror_blocks(0, kv, gb, hb)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(5):
        mirror.mirror_blocks(0, kv, gb, hb)
    torch.cuda.synchronize(); d2h = 5 * nb * kv[0].numel() / (time.perf_counter() - t) / 1e9
    t = time.perf_counter()
    for _ in range(5):
        for i in range(nb):
            kv[i].copy_(mirror.host[0, i], non_blocking=True)
    torch.cuda.synchronize(); h2d = 5 * nb * kv[0].numel() / (time.perf_counter() - t) / 1e9
    return d2h, h2d


def cmd_run(a):
    from sparsekv.tier import HostKVMirror, HotBuffer, TierScratch, gather_rows_tiered
    dev = "cuda"
    res_all = []
    gpu = torch.cuda.get_device_name(0)
    jobs = [(N, None) for N in a.ctx] if not a.real else [(None, p) for p in a.real]
    for N, real in jobs:
        steps = a.steps
        if real is not None:
            # stream F's real 27B picks: <tag>_picks.npy [MAXNEW, 16, 2048] + <tag>.json (n_ctx, q_len, n_gen)
            meta = json.load(open(real.replace("_picks.npy", ".json")))
            pk = np.load(real)
            T = int(meta["n_gen"])
            N = int(meta["n_ctx"]) + int(meta["q_len"])
            steps = min(steps, T)
            a.warm = min(a.warm, max(1, steps // 4))
            traces = {"real:" + os.path.basename(real): [np.ascontiguousarray(pk[:T, l]) for l in range(L)]}
            presets = list(traces)
        else:
            traces = {p: [np.load(trace_path(N, p, l, a.trace_steps)) for l in range(L)] for p in a.preset}
            presets = a.preset
        nblk = (N + BS - 1) // BS
        t0 = time.time()
        mirror = HostKVMirror(L, nblk * a.nreq, HKV, BS, W, device=dev)
        # touch the host pages so first-use faults are not timed (content irrelevant for timing)
        mirror.host.view(-1)[:: 1 << 20].zero_()
        t_alloc = time.time() - t0
        # GPU cache: resident blocks only (sink block + tail), or everything for the resident baseline
        n_res_tail = (WIN + 8 + BS - 1) // BS + 1
        kvs_full = None
        if a.baseline:
            kvs_full = [torch.zeros((nblk * a.nreq + 1, HKV, BS, W), dtype=torch.int8, device=dev) for _ in range(L)]
        kvs = [torch.zeros((a.nreq * (1 + n_res_tail) + 1, HKV, BS, W), dtype=torch.int8, device=dev) for _ in range(L)]
        gpu_bt = torch.full((a.nreq, nblk), -1, dtype=torch.int32, device=dev)
        host_bt = torch.zeros((a.nreq, nblk), dtype=torch.int32, device=dev)
        full_bt = torch.zeros((a.nreq, nblk), dtype=torch.int32, device=dev)
        for r in range(a.nreq):
            host_bt[r] = torch.arange(r * nblk, (r + 1) * nblk, dtype=torch.int32)
            full_bt[r] = torch.arange(1 + r * nblk, 1 + (r + 1) * nblk, dtype=torch.int32)
            base = 1 + r * (1 + n_res_tail)
            gpu_bt[r, 0] = base
            gpu_bt[r, nblk - n_res_tail:] = torch.arange(base + 1, base + 1 + n_res_tail, dtype=torch.int32)
        NBS = (C + BS - 1) // BS
        sbt = torch.arange(a.nreq * NBS, dtype=torch.int32, device=dev).view(a.nreq, NBS)
        stg = torch.zeros((a.nreq * NBS, HKV, BS, W), dtype=torch.int8, device=dev)
        scratch = TierScratch(a.nreq, C, dev)
        slot = torch.arange(a.nreq, dtype=torch.int32, device=dev)
        nv = torch.full((a.nreq,), C, dtype=torch.int32, device=dev)
        bw = uva_bw(mirror.dev)
        d2h, h2d = dma_bw(mirror, kvs_full[0] if kvs_full is not None else kvs[0])
        for preset in presets:
            # idx per (step, layer): [steps, L, nreq, C] on GPU
            sinks = np.arange(SINK)
            winr = np.arange(N - WIN, N)
            allidx = np.empty((steps, L, a.nreq, C), dtype=np.int32)
            for l in range(L):
                tr = traces[preset][l]
                for r in range(a.nreq):
                    off = 37 * r  # different requests see shifted steps of the trace
                    for t in range(steps):
                        allidx[t, l, r] = np.concatenate([np.clip(tr[(t + off) % len(tr)], 0, N - 1), sinks, winr])
            allidx_d = torch.from_numpy(allidx).to(dev)
            for hf in a.hot:
                H = max(2 * C, int(hf * N))
                hot = HotBuffer(L, a.nreq, H, HKV, W, max_ctx=nblk * BS, device=dev, age=a.age)
                idx_s = torch.empty((L, a.nreq, C), dtype=torch.int32, device=dev)

                def step_fn(kv_list, gbt):
                    hot.next_step()
                    for l in range(L):
                        gather_rows_tiered(kv_list[l], gbt, mirror.dev[l], host_bt, idx_s[l], nv, stg, sbt, hot, l,
                                           slot, scratch)

                # warm-up: fill the hot buffer
                for t in range(a.warm):
                    idx_s.copy_(allidx_d[t]); step_fn(kvs, gpu_bt)
                torch.cuda.synchronize()
                hot.stats.zero_()
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev0.record()
                for t in range(a.warm, steps):
                    idx_s.copy_(allidx_d[t]); step_fn(kvs, gpu_bt)
                ev1.record(); torch.cuda.synchronize()
                eager_ms = ev0.elapsed_time(ev1) / (steps - a.warm)
                st = hot.stats.tolist()
                # CUDA graph of one step: reset the tier and re-warm eagerly so the graph pass sees the same miss
                # sequence as the eager pass (not a second pass over already-cached picks)
                for sl in range(a.nreq):
                    hot.reset_slot(sl)
                for t in range(a.warm):
                    idx_s.copy_(allidx_d[t]); step_fn(kvs, gpu_bt)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    step_fn(kvs, gpu_bt)
                hot.stats.zero_()
                ev0.record()
                for t in range(a.warm, steps):
                    idx_s.copy_(allidx_d[t]); g.replay()
                ev1.record(); torch.cuda.synchronize()
                graph_ms = ev0.elapsed_time(ev1) / (steps - a.warm)
                st_g = hot.stats.tolist()
                picks = st[0] + st[1] + st[2]
                miss_mb = st[1] * HKV * W / 1e6 / (steps - a.warm)
                row = dict(gpu=gpu, ctx=N, nreq=a.nreq, preset=preset, hot_frac=hf, H=H, age=a.age,
                           eager_ms=round(eager_ms, 3), graph_ms=round(graph_ms, 3),
                           miss_pct_of_topk=round(100 * st[1] / max(1, st[0] + st[1]), 2),
                           miss_pct_of_picks=round(100 * st[1] / max(1, picks), 2),
                           miss_MB_per_step=round(miss_mb, 2),
                           graph_miss_pct_of_topk=round(100 * st_g[1] / max(1, st_g[0] + st_g[1]), 2),
                           uva_GBps=round(bw, 2), dma_d2h_GBps=round(d2h, 2), dma_h2d_GBps=round(h2d, 2),
                           hot_MB=round(hot.nbytes / 1e6), host_GB=round(mirror.nbytes / 1e9, 2),
                           host_alloc_s=round(t_alloc, 1))
                if kvs_full is not None:
                    # resident baseline: every block on GPU (pure compaction gather; resolve kernel still runs)
                    hot.stats.zero_()
                    gb = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gb):
                        step_fn(kvs_full, full_bt)
                    ev0.record()
                    for t in range(a.warm, steps):
                        idx_s.copy_(allidx_d[t]); gb.replay()
                    ev1.record(); torch.cuda.synchronize()
                    row["resident_graph_ms"] = round(ev0.elapsed_time(ev1) / (steps - a.warm), 3)
                    del gb
                del g
                print("RESULT " + json.dumps(row), flush=True)
                res_all.append(row)
                del hot
                torch.cuda.empty_cache()
        mirror.close()
        del mirror, kvs, kvs_full
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "a") as f:
            for r in res_all:
                f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "run"])
    ap.add_argument("--ctx", type=int, nargs="+", default=[131072, 262144])
    ap.add_argument("--preset", nargs="+", default=["lab"])
    ap.add_argument("--hot", type=float, nargs="+", default=[0.10])
    ap.add_argument("--nreq", type=int, default=1)
    ap.add_argument("--age", type=int, default=8)
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--trace-steps", type=int, default=384)
    ap.add_argument("--warm", type=int, default=64)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--real", nargs="*", default=[], help="stream F *_picks.npy files (real 27B selections)")
    a = ap.parse_args()
    if a.mode == "gen":
        a.steps = a.trace_steps
        cmd_gen(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
