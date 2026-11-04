# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""GPU tests for S5 (exact follow-up suffix streamed from the host tier). DRAFT: not yet run.

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> venv/bin/python sparsekv/test_s5_gpu.py
  1. landing path == UVA path: copy-engine DMA of int4 host runs into the landing buffer + GPU dequant gives the
     same pf_kv bytes as copy_blocks_from_host4 straight from the UVA view (the old dense-mode staging).
  2. pipeline ordering: TierRuntime._s5_stage over 16 layers with the one-layer-ahead prefetch on the copy stream;
     after each layer's stage, pf_kv equals that layer's reference (catches a prefetch overwriting the landing buffer
     before the dequant consumed it, or a missing wait). Run with a slow compute stream (a sleep kernel) too.
  3. KSEL 16K regression: gather_rows_tiered on the padding slot with C > hot.H (cache_rows=False) stages the right
     rows and leaves the hot buffer untouched; the old assert still fires for a real slot.
  4. timing: per-layer DMA vs UVA staging time at 250K (prints GB/s; x16 expect ~20+ vs ~10, x4 ~6.5 vs ~6.2).
"""
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv.tier import (W4H, HotBuffer, TierScratch, alloc_pinned, copy_blocks, copy_blocks_from_host4,  # noqa
                           gather_rows_tiered, quant_blocks_to_host4, uva_view)
from sparsekv.tier_connector import TierRuntime, _runs  # noqa: E402

dev = "cuda"
Hkv, BS, W = 4, 864, 520


def _real_rows(nb, seed=0):
    """Valid pthfa int8 rows (values + positive fp32 scales) for nb blocks."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randint(-127, 128, (nb, Hkv, BS, W), dtype=torch.int8, generator=g)
    sc = (torch.rand(nb, Hkv, BS, 2, generator=g) * 0.02 + 1e-3).to(torch.float32)
    kv[..., 256:260] = sc[..., 0:1].contiguous().view(torch.int8)
    kv[..., 516:520] = sc[..., 1:2].contiguous().view(torch.int8)
    return kv.to(dev)


def _host_mirror(L, HB, nb_fill, seed=1):
    host = alloc_pinned((L, HB, Hkv, BS, W4H))
    hdev = uva_view(host)
    for l in range(L):
        kv = _real_rows(nb_fill, seed + l)
        quant_blocks_to_host4(kv, torch.arange(nb_fill, dtype=torch.int32, device=dev), hdev[l],
                              torch.arange(nb_fill, dtype=torch.int32, device=dev))
    torch.cuda.synchronize()
    return host, hdev


def test_landing_equals_uva():
    host, hdev = _host_mirror(1, 64, 64)
    hs = [5, 6, 7, 8, 20, 21, 40, 3]                       # three runs + a singleton
    n = len(hs)
    land = torch.empty((n + 1, Hkv, BS, W4H), dtype=torch.int8, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for h0, i0, k in _runs(hs):
            land[i0:i0 + k].copy_(host[0, h0:h0 + k], non_blocking=True)
    ev = torch.cuda.Event()
    ev.record(s)
    torch.cuda.current_stream().wait_event(ev)
    dst = torch.tensor([1 + i for i in range(n)], dtype=torch.int32, device=dev)
    a = torch.zeros((n + 1, Hkv, BS, W), dtype=torch.int8, device=dev)
    b = torch.zeros_like(a)
    copy_blocks_from_host4(a, dst, land, torch.arange(n, dtype=torch.int32, device=dev))
    copy_blocks_from_host4(b, dst, hdev[0], torch.tensor(hs, dtype=torch.int32, device=dev))
    torch.cuda.synchronize()
    assert torch.equal(a, b), "landing-buffer dequant differs from the UVA dequant"
    print("S5 landing buffer (DMA runs + GPU dequant) == UVA dequant, bit for bit: OK")


def _fake_rt(L, host, hdev, nb, land_rows):
    rt = TierRuntime.__new__(TierRuntime)
    rt.L, rt.dev, rt.host4 = L, dev, True
    rt.mirror = types.SimpleNamespace(host=host, dev=hdev)
    rt.pf_kv = torch.zeros((nb + 1, Hkv, BS, W), dtype=torch.int8, device=dev)
    rt.s5_land = torch.empty((land_rows, Hkv, BS, W4H), dtype=torch.int8, device=dev)
    rt.s5_stream = torch.cuda.Stream()
    rt._s5_pf = None
    rt.stats = {"steps": 1, "prefill_staged_blocks": 0, "s5_dma_blocks": 0, "s5_uva_blocks": 0}
    return rt


def _sleep_kernel(ms):
    torch.cuda._sleep(int(ms * 1e6 / 1.4))  # ~cycles at 1.4 GHz; only meant to delay the compute stream


def test_pipeline_ordering(slow=False):
    L, nb_host, nb_res = 16, 40, 3
    host, hdev = _host_mirror(L, 64, 64)
    hs = list(range(10, 10 + nb_host))
    kv_res = _real_rows(8, 99)
    rt = _fake_rt(L, host, hdev, nb_host + nb_res, nb_host + 1)
    d = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)  # noqa: E731
    p = {"rid": "r", "x_n": (nb_host, nb_res), "x_runs": _runs(hs),
         "x_h": (d([1 + i for i in range(nb_host)]), d(hs)), "x_land": d(list(range(nb_host))),
         "x_g": (d([1 + nb_host + i for i in range(nb_res)]), d([2, 5, 7]))}
    rt._s5_kick(0, p)
    for l in range(L):
        if slow:
            _sleep_kernel(20)  # compute stream lags the copy stream: the wait/event pairing must still hold
        rt._s5_stage(l, p, kv_res)
        got = rt.pf_kv.clone()
        ref = torch.zeros_like(rt.pf_kv)
        copy_blocks_from_host4(ref, p["x_h"][0], hdev[l], p["x_h"][1])
        copy_blocks(ref, p["x_g"][0], kv_res, p["x_g"][1])
        torch.cuda.synchronize()
        assert torch.equal(got, ref), f"layer {l}: staged pf_kv differs (slow={slow})"
    assert rt.stats["s5_dma_blocks"] == L * nb_host and rt.stats["s5_uva_blocks"] == 0
    print(f"S5 prefetch pipeline over {L} layers (slow compute={slow}) == per-layer reference: OK")


def test_ksel16k_padding_slot():
    torch.manual_seed(3)
    ctx, ksel, sink, win = 30 * BS, 16384, 64, 1024
    C = ksel + sink + win
    maxblk = ctx // BS
    host, hdev = _host_mirror(1, maxblk, maxblk)
    kv = _real_rows(4)
    gpu_bt = torch.full((1, maxblk), -1, dtype=torch.int32, device=dev)
    host_bt = torch.arange(maxblk, dtype=torch.int32, device=dev)[None].contiguous()
    H = 6553                                                    # HOT_FRAC 0.025 x 262144: C = 17472 > H
    hot = HotBuffer(1, 2, H, Hkv, W, max_ctx=maxblk * BS, device=dev, kv_slots=1)
    pad = torch.tensor([1], dtype=torch.int32, device=dev)
    pos = torch.cat([torch.randperm(ctx - sink - win, device=dev)[:ksel] + sink, torch.arange(sink, device=dev),
                     torch.arange(ctx - win, ctx, device=dev)]).to(torch.int32)[None].contiguous()
    nbs = -(-C // BS)
    st = torch.zeros(nbs, Hkv, BS, W, dtype=torch.int8, device=dev)
    sbt = torch.arange(nbs, dtype=torch.int32, device=dev)[None].contiguous()
    tag0 = hot.tag.clone()
    hot.next_step()
    gather_rows_tiered(kv, gpu_bt, hdev[0], host_bt, pos, torch.tensor([C], dtype=torch.int32, device=dev), st, sbt,
                       hot, 0, pad, TierScratch(1, C, dev), cache_rows=False)
    full = torch.zeros(maxblk, Hkv, BS, W, dtype=torch.int8, device=dev)
    copy_blocks_from_host4(full, torch.arange(maxblk, dtype=torch.int32, device=dev), hdev[0],
                           torch.arange(maxblk, dtype=torch.int32, device=dev))
    torch.cuda.synchronize()
    p = pos[0].long()
    exp = full[p // BS, :, p % BS]
    j = torch.arange(C, device=dev)
    got = st[sbt[0, j // BS].long(), :, j % BS]
    assert torch.equal(got, exp), "padding-slot gather with C > H staged wrong rows"
    assert torch.equal(hot.tag, tag0) and int((hot.pos2row[0, 1] >= 0).sum()) == 0, "hot buffer touched"
    try:
        gather_rows_tiered(kv, gpu_bt, hdev[0], host_bt, pos, torch.tensor([C], dtype=torch.int32, device=dev), st,
                           sbt, hot, 0, torch.tensor([0], dtype=torch.int32, device=dev), TierScratch(1, C, dev))
        raise RuntimeError("expected the C <= H assert for a caching slot")
    except AssertionError:
        pass
    print(f"S5 KSEL 16K fix: padding-slot gather with C={C} > H={H} stages exact rows, hot buffer untouched: OK")


def test_timing_250k():
    nblk = 290                                                  # ~250K tokens
    host = alloc_pinned((1, nblk, Hkv, BS, W4H))
    hdev = uva_view(host)
    land = torch.empty((nblk, Hkv, BS, W4H), dtype=torch.int8, device=dev)
    pf = torch.empty((nblk + 1, Hkv, BS, W), dtype=torch.int8, device=dev)
    ids = torch.arange(nblk, dtype=torch.int32, device=dev)
    s = torch.cuda.Stream()
    gb = nblk * Hkv * BS * W4H / 1e9
    for name, fn in (("DMA (copy engine, 1 run)", lambda: land.copy_(host[0], non_blocking=True)),
                     ("UVA dequant kernel", lambda: copy_blocks_from_host4(pf, ids + 1, hdev[0], ids))):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 5
        print(f"S5 timing: {name}: {dt * 1e3:.1f} ms per layer ({gb / dt:.1f} GB/s, x16 layers = {16 * dt:.2f} s)")
    t = time.perf_counter()
    copy_blocks_from_host4(pf, ids + 1, land, ids)
    torch.cuda.synchronize()
    print(f"S5 timing: GPU->GPU dequant of one layer: {(time.perf_counter() - t) * 1e3:.1f} ms")
    del s


if __name__ == "__main__":
    test_landing_equals_uva()
    test_pipeline_ordering(False)
    test_pipeline_ordering(True)
    test_ksel16k_padding_slot()
    test_timing_250k()
    print("ALL OK")
