# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""S14 GPU test of the global hot pool kernels.

    sparsekv/gpurun.sh python sparsekv/test_s14_hot_gpu.py        (needs one free card; a few hundred MB)

1. Triton gather_rows_tiered_global vs the torch reference gather_rows_global_ref, 30 steps x 2 layers x 3 requests
   (disjoint host blocks, mixed residency, padded columns, one uncached row), for a roomy pool and for a pool smaller
   than one step's misses (uncached fallback): staged rows == reference == ground truth; tag / last / pos2row / hand
   identical; every reachable pool row holds its key's host bytes.
2. CUDA graph: capture one step, replay 8 steps, same checks.
3. Requests that share host blocks (racy duplicate inserts): staged rows == ground truth, pool invariants hold.
4. invalidate_host_blocks on the device: a rewritten host block is re-read.
5. int4 host tier (HOST4 kernels): for one request and equal rows, the pool path stages the same bytes as the
   existing per-slot path (gather_rows_tiered).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv.tier import (HostKVMirror, HotBuffer, TierScratch, W4H, gather_rows_global_ref,  # noqa: E402
                           gather_rows_tiered, gather_rows_tiered_global, quant_blocks_to_host4)
from sparsekv.tier_trace import gen_trace  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
HKV, BS, W = 4, 864, 520
L = 2
KSEL, SINK, WIN = 512, 64, 1024
C = KSEL + SINK + WIN
CACHED = 8  # slots below this use the pool (TierRuntime passes max_reqs)


def build(ctxs, share=False, host_w=W):
    nreq = len(ctxs)
    maxblk = max(ctxs) // BS + 1
    NB = nreq * maxblk + 1
    mirror = HostKVMirror(L, nreq * maxblk, HKV, BS, host_w, device=dev)
    PAD = 4096
    kvs = []
    for _ in range(L):
        raw = torch.randint(-128, 127, (NB, HKV * BS * W + PAD), dtype=torch.int8, device=dev)
        kvs.append(raw[:, : HKV * BS * W].view(NB, HKV, BS, W))  # strided, like vLLM's cache views
    gpu_bt = torch.full((nreq, maxblk), -1, dtype=torch.int32, device=dev)
    host_bt = torch.full((nreq, maxblk), -1, dtype=torch.int32, device=dev)
    full_bt = torch.zeros((nreq, maxblk), dtype=torch.int32, device=dev)
    for r in range(nreq):
        nblk = (ctxs[r] + BS - 1) // BS
        src = 0 if share else r
        gbl = list(range(1 + src * maxblk, 1 + src * maxblk + nblk))
        hbl = list(range(src * maxblk, src * maxblk + nblk))
        full_bt[r, :nblk] = torch.tensor(gbl, dtype=torch.int32)
        host_bt[r, :nblk] = torch.tensor(hbl, dtype=torch.int32)
        if r == src:
            for l in range(L):
                if host_w == W:
                    mirror.mirror_blocks(l, kvs[l], gbl, hbl)
                else:
                    quant_blocks_to_host4(kvs[l], torch.tensor(gbl, dtype=torch.int32, device=dev), mirror.dev[l],
                                          torch.tensor(hbl, dtype=torch.int32, device=dev))
        for b in {nblk - 1, nblk - 2, nblk // 2} | ({0} if r == 0 else set()):
            gpu_bt[r, b] = gbl[b]
    torch.cuda.synchronize()
    return mirror, kvs, gpu_bt, host_bt, full_bt


def make_idx(ctxs, traces, step, pad=(1,)):
    nreq = len(ctxs)
    idx = torch.zeros((nreq, C), dtype=torch.int32)
    nv = torch.zeros(nreq, dtype=torch.int32)
    for r in range(nreq):
        row = np.concatenate([traces[r][step], np.arange(SINK), np.arange(ctxs[r] - WIN, ctxs[r])])
        idx[r, :row.size] = torch.from_numpy(row.astype(np.int32))
        nv[r] = row.size - (7 if r in pad else 0)
    return idx.to(dev), nv.to(dev)


def staged(stg, sbt, r, n):
    j = torch.arange(n, device=dev)
    return stg[sbt[r, j // BS].long(), :, j % BS]


def truth(kv, full_bt, idx, r, n):
    pos = idx[r, :n].long()
    return kv[full_bt[r, pos // BS].long(), :, pos % BS]


def check_pool(hot, l, host):
    p2r, tag = hot.pos2row[l, 0].long(), hot.tag[l, 0].long()
    keys = torch.nonzero(p2r >= 0).flatten()
    rows = p2r[keys]
    assert torch.equal(tag[rows], keys), "pos2row -> tag round trip broken"
    assert torch.equal(hot.kv[l, 0][rows], host[(keys // BS).cpu(), :, (keys % BS).cpu()].to(dev)), "pool row holds wrong bytes"


def run_vs_ref(H, tag):
    ctxs = [20000, 13000, 9000]
    nreq = len(ctxs)
    mirror, kvs, gpu_bt, host_bt, full_bt = build(ctxs)
    NK = mirror.HB * BS
    hot1 = HotBuffer(L, 1, H, HKV, W, max_ctx=NK, device=dev, age=2, kv_slots=1)
    hot2 = HotBuffer(L, 1, H, HKV, W, max_ctx=NK, device=dev, age=2, kv_slots=1)
    slot = torch.tensor([2, 0, CACHED], dtype=torch.int32, device=dev)  # third request: uncached row
    traces = [gen_trace(ctxs[r], 40, K=KSEL, sink=SINK, win=WIN, seed=r) for r in range(nreq)]
    NBS = (C + BS - 1) // BS
    sbt = torch.arange(nreq * NBS, dtype=torch.int32, device=dev).view(nreq, NBS)
    stg1 = torch.zeros((nreq * NBS, HKV, BS, W), dtype=torch.int8, device=dev)
    stg2 = torch.zeros_like(stg1)
    scratch = TierScratch(nreq, C, dev, glob=True)
    tot = np.zeros(4, dtype=np.int64)

    def compare(step, idx, nv, where):
        torch.cuda.synchronize()
        for l in range(L):
            for name in ("tag", "last", "pos2row", "hand"):
                assert torch.equal(getattr(hot1, name)[l], getattr(hot2, name)[l]), f"{where}: {name} {step} {l}"
            check_pool(hot1, l, mirror.host[l])

    for step in range(30):
        idx, nv = make_idx(ctxs, traces, step)
        hot1.next_step(); hot2.next_step()
        st = int(hot2.step.item())
        for l in range(L):
            gather_rows_tiered_global(kvs[l], gpu_bt, mirror.dev[l], host_bt, idx, nv, stg1, sbt, hot1, l, slot,
                                      scratch, cached_slots=CACHED)
            tot += gather_rows_global_ref(kvs[l], gpu_bt, mirror.dev[l], host_bt, idx, nv, stg2, sbt, hot2.layer(l),
                                          slot, st, hot2.age, cached_slots=CACHED)
            torch.cuda.synchronize()
            for r in range(nreq):
                n = int(nv[r])
                a = staged(stg1, sbt, r, n)
                assert torch.equal(a, staged(stg2, sbt, r, n)), f"{tag}: staging != ref s{step} l{l} r{r}"
                assert torch.equal(a, truth(kvs[l], full_bt, idx, r, n)), f"{tag}: staging != truth s{step} l{l} r{r}"
        compare(step, idx, nv, tag)
    assert hot1.stats.tolist() == tot[:3].tolist(), (hot1.stats.tolist(), tot.tolist())
    print(f"{tag}: eager OK, hit/miss/resident/uncached = {tot.tolist()}")

    idx_s, nv_s = make_idx(ctxs, traces, 30)
    g = torch.cuda.CUDAGraph()
    snap = {k: getattr(hot1, k).clone() for k in ("kv", "tag", "last", "pos2row", "hand", "step", "stats")}
    with torch.cuda.graph(g):
        hot1.next_step()
        for l in range(L):
            gather_rows_tiered_global(kvs[l], gpu_bt, mirror.dev[l], host_bt, idx_s, nv_s, stg1, sbt, hot1, l, slot,
                                      scratch, cached_slots=CACHED)
    for k, v in snap.items():
        getattr(hot1, k).copy_(v)
    for step in range(30, 38):
        idx, nv = make_idx(ctxs, traces, step)
        idx_s.copy_(idx); nv_s.copy_(nv)
        g.replay()
        hot2.next_step()
        st = int(hot2.step.item())
        for l in range(L):
            gather_rows_global_ref(kvs[l], gpu_bt, mirror.dev[l], host_bt, idx, nv, stg2, sbt, hot2.layer(l), slot,
                                   st, hot2.age, cached_slots=CACHED)
        compare(step, idx, nv, tag + " graph")
        l = L - 1  # staging holds the last layer's rows
        for r in range(nreq):
            assert torch.equal(staged(stg1, sbt, r, int(nv[r])), truth(kvs[l], full_bt, idx, r, int(nv[r])))
    print(f"{tag}: CUDA graph replay OK (8 steps)")
    mirror.close()


def run_shared_and_invalidate():
    ctxs = [20000, 20000]
    mirror, kvs, gpu_bt, host_bt, full_bt = build(ctxs, share=True)
    hot = HotBuffer(L, 1, 3 * C, HKV, W, max_ctx=mirror.HB * BS, device=dev, age=2, kv_slots=1)
    slot = torch.zeros(2, dtype=torch.int32, device=dev)
    traces = [gen_trace(20000, 24, K=KSEL, sink=SINK, win=WIN, seed=7)] * 2  # identical picks: duplicate inserts
    NBS = (C + BS - 1) // BS
    sbt = torch.arange(2 * NBS, dtype=torch.int32, device=dev).view(2, NBS)
    stg = torch.zeros((2 * NBS, HKV, BS, W), dtype=torch.int8, device=dev)
    scratch = TierScratch(2, C, dev, glob=True)

    def step_once(step):
        idx, nv = make_idx(ctxs, traces, step, pad=())
        hot.next_step()
        for l in range(L):
            gather_rows_tiered_global(kvs[l], gpu_bt, mirror.dev[l], host_bt, idx, nv, stg, sbt, hot, l, slot,
                                      scratch, cached_slots=CACHED)
            torch.cuda.synchronize()
            for r in range(2):
                assert torch.equal(staged(stg, sbt, r, int(nv[r])), truth(kvs[l], full_bt, idx, r, int(nv[r]))), \
                    f"shared: staging != truth s{step} l{l} r{r}"
            check_pool(hot, l, mirror.host[l])

    for step in range(20):
        step_once(step)
    print("shared host blocks (duplicate inserts in one step): OK")
    # rewrite 4 host blocks that hold pool rows, invalidate, read again
    blocks = torch.unique(torch.nonzero(hot.pos2row[0, 0] >= 0).flatten() // BS)[:4]
    for hb in blocks.tolist():
        gb = int(full_bt[0, hb])
        for l in range(L):
            kvs[l][gb] = torch.randint(-128, 127, (HKV, BS, W), dtype=torch.int8, device=dev)
            mirror.mirror_blocks(l, kvs[l], [gb], [hb])
    hot.invalidate_host_blocks(blocks.to(dev), BS)
    torch.cuda.current_stream().wait_stream(mirror.copy_stream)
    step_once(19)
    step_once(20)
    print("invalidate_host_blocks on device: rewritten blocks re-read: OK")
    mirror.close()


def run_host4():
    os.environ["SPARSE_TIER_HOST_BITS"] = "4"
    ctxs = [20000]
    mirror, kvs, gpu_bt, host_bt, full_bt = build(ctxs, host_w=W4H)
    H = 2 * C
    hg = HotBuffer(L, 1, H, HKV, W, max_ctx=mirror.HB * BS, device=dev, age=2, kv_slots=1)
    hs = HotBuffer(L, 2, H, HKV, W, max_ctx=(ctxs[0] // BS + 1) * BS, device=dev, age=2, kv_slots=1)
    slot = torch.zeros(1, dtype=torch.int32, device=dev)
    traces = [gen_trace(20000, 24, K=KSEL, sink=SINK, win=WIN, seed=11)]
    NBS = (C + BS - 1) // BS
    sbt = torch.arange(NBS, dtype=torch.int32, device=dev).view(1, NBS)
    s1 = torch.zeros((NBS, HKV, BS, W), dtype=torch.int8, device=dev)
    s2 = torch.zeros_like(s1)
    sc1, sc2 = TierScratch(1, C, dev, glob=True), TierScratch(1, C, dev)
    for step in range(24):
        idx, nv = make_idx(ctxs, traces, step, pad=())
        hg.next_step(); hs.next_step()
        for l in range(L):
            gather_rows_tiered_global(kvs[l], gpu_bt, mirror.dev[l], host_bt, idx, nv, s1, sbt, hg, l, slot, sc1,
                                      cached_slots=CACHED)
            gather_rows_tiered(kvs[l], gpu_bt, mirror.dev[l], host_bt, idx, nv, s2, sbt, hs, l, slot, sc2)
            torch.cuda.synchronize()
            n = int(nv[0])
            assert torch.equal(staged(s1, sbt, 0, n), staged(s2, sbt, 0, n)), f"host4: pool != per-slot s{step} l{l}"
        assert hg.stats.tolist() == hs.stats.tolist(), (hg.stats.tolist(), hs.stats.tolist())
    # tiny pool: misses beyond the victims are dequantised straight into staging (same bytes as the per-slot path)
    ht = HotBuffer(L, 1, 64, HKV, W, max_ctx=mirror.HB * BS, device=dev, age=2, kv_slots=1)
    for step in range(4):
        idx, nv = make_idx(ctxs, traces, step, pad=())
        ht.next_step(); hs.next_step()
        gather_rows_tiered_global(kvs[0], gpu_bt, mirror.dev[0], host_bt, idx, nv, s1, sbt, ht, 0, slot, sc1,
                                  cached_slots=CACHED)
        gather_rows_tiered(kvs[0], gpu_bt, mirror.dev[0], host_bt, idx, nv, s2, sbt, hs, 0, slot, sc2)
        torch.cuda.synchronize()
        assert torch.equal(staged(s1, sbt, 0, int(nv[0])), staged(s2, sbt, 0, int(nv[0]))), f"host4 tiny pool s{step}"
    print("int4 host tier: pool path == per-slot path (roomy and 64-row pool): OK")
    mirror.close()


if __name__ == "__main__":
    run_vs_ref(6 * C, "roomy pool")
    run_vs_ref(300, "pool < misses per step")
    run_shared_and_invalidate()
    run_host4()
    print("ALL OK")
