# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""S14 CPU tests (no GPU). Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=<tree> venv/bin/python sparsekv/test_s14_hot_cpu.py

SPARSE_TIER_HOT_GLOBAL (global hot pool keyed by host token) and SPARSE_TIER_SHARED_PLAN (one residency plan per decode
step for all sparse layers). Both are off by default. What runs here is the torch REFERENCE of the pool
(tier.gather_rows_global_ref) and the connector's Python wiring; the Triton kernels of the pool are not executed on
CPU (sparsekv/test_s14_hot_gpu.py compares them with this reference on a GPU).
"""
import importlib
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv import tier  # noqa: E402
from sparsekv.tier import (HotBuffer, NEG_AGE, gather_rows_global_ref, gather_rows_tiered_ref)  # noqa: E402
from sparsekv.tier_trace import gen_trace  # noqa: E402

HKV, BS, W = 2, 32, 8  # small rows: the logic does not depend on the row size
KSEL, SINK, WIN = 96, 8, 64
C = KSEL + SINK + WIN


def _world(ctxs, share=None, seed=0):
    """Requests with contexts `ctxs` tokens. share = (r_a, r_b, nblk): r_b maps r_a's first nblk host blocks (common
    prefix). Returns kv (GPU truth), host_kv (same bytes, host block layout), full_bt, gpu_bt (residency), host_bt."""
    g = torch.Generator().manual_seed(seed)
    nreq = len(ctxs)
    maxblk = max(ctxs) // BS + 1
    NB = nreq * maxblk + 1
    kv = torch.randint(-128, 127, (NB, HKV, BS, W), dtype=torch.int8, generator=g)
    host_kv = torch.zeros((NB, HKV, BS, W), dtype=torch.int8)
    full_bt = torch.zeros((nreq, maxblk), dtype=torch.int32)
    gpu_bt = torch.full((nreq, maxblk), -1, dtype=torch.int32)
    host_bt = torch.full((nreq, maxblk), -1, dtype=torch.int32)
    for r in range(nreq):
        nblk = (ctxs[r] + BS - 1) // BS
        for b in range(nblk):
            gb = 1 + r * maxblk + b
            hb = r * maxblk + b
            if share is not None and r == share[1] and b < share[2]:
                gb = 1 + share[0] * maxblk + b  # same content and same host block as r_a
                hb = share[0] * maxblk + b
            full_bt[r, b] = gb
            host_bt[r, b] = hb
            host_kv[hb] = kv[gb]
            if b in (0, nblk - 1, nblk - 2):  # sinks + tail stay resident
                gpu_bt[r, b] = gb
    return kv, host_kv, full_bt, gpu_bt, host_bt


def _idx(ctxs, traces, step, pad=()):
    nreq = len(ctxs)
    idx = torch.zeros((nreq, C), dtype=torch.int32)
    nv = torch.zeros(nreq, dtype=torch.int32)
    for r in range(nreq):
        N = ctxs[r]
        row = np.concatenate([traces[r][step], np.arange(SINK), np.arange(N - WIN, N)])
        idx[r, :row.size] = torch.from_numpy(row.astype(np.int32))
        nv[r] = row.size - (5 if r in pad else 0)
    return idx, nv


def _pool(H, host_tokens, age=2):
    return HotBuffer(1, 1, H, HKV, W, max_ctx=host_tokens, device="cpu", age=age, kv_slots=1)


def _check_pool(hot, host_kv):
    """Invariants: key -> row -> key round trip, and every reachable row holds the host bytes of its key."""
    p2r, tag = hot.pos2row[0, 0].long(), hot.tag[0, 0].long()
    keys = torch.nonzero(p2r >= 0).flatten()
    rows = p2r[keys]
    assert torch.equal(tag[rows], keys), "pos2row -> tag round trip broken"
    assert rows.unique().numel() == rows.numel(), "two keys share one pool row"
    assert torch.equal(hot.kv[0, 0][rows], host_kv[keys // BS, :, keys % BS]), "pool row holds wrong bytes"
    return keys.numel()


def _run(ctxs, H, steps=24, share=None, slots=None, pad=(), check_truth=True, seed=0):
    kv, host_kv, full_bt, gpu_bt, host_bt = _world(ctxs, share, seed)
    nreq = len(ctxs)
    hot = _pool(H, host_kv.shape[0] * BS)
    traces = [gen_trace(ctxs[r], steps, K=KSEL, sink=SINK, win=WIN, seed=seed + (0 if share and r == share[1] else r))
              for r in range(nreq)]
    nbs = (C + BS - 1) // BS
    sbt = torch.arange(nreq * nbs, dtype=torch.int32).view(nreq, nbs)
    stg = torch.zeros((nreq * nbs, HKV, BS, W), dtype=torch.int8)
    slot = torch.zeros(nreq, dtype=torch.int32) if slots is None else torch.tensor(slots, dtype=torch.int32)
    tot = np.zeros(4, dtype=np.int64)
    for step in range(steps):
        idx, nv = _idx(ctxs, traces, step, pad)
        hot.next_step()
        tot += gather_rows_global_ref(kv, gpu_bt, host_kv, host_bt, idx, nv, stg, sbt, hot.layer(0), slot,
                                      int(hot.step), hot.age, cached_slots=1)
        if check_truth:
            for r in range(nreq):
                n = int(nv[r])
                j = torch.arange(n)
                pos = idx[r, :n].long()
                truth = kv[full_bt[r, pos // BS].long(), :, pos % BS]
                assert torch.equal(stg[sbt[r, j // BS].long(), :, j % BS], truth), f"staging != truth s{step} r{r}"
        _check_pool(hot, host_kv)
    return tot, hot


def test_global_ref_exact_any_pool_size():
    """Staged rows equal the ground truth at every step, for a roomy pool, a pool smaller than one step's picks
    (uncached fallback) and a 1-row pool; mixed residency, padded columns, three requests."""
    ctxs = [3000, 2200, 1700]
    for H in (2 * C * 3, 200, 60, 1):
        tot, hot = _run(ctxs, H, pad=(1,))
        hit, miss, res, unc = tot.tolist()
        assert hit + miss + res == 24 * (3 * C - 5), tot
        assert res > 0 and miss > 0
        if H >= 2 * C * 3:
            assert unc == 0 and hit > miss, tot
        if H <= 60:
            assert unc > 0, tot  # more cacheable misses than rows: the rest is staged from host, uncached
    print("S14 global pool reference: exact staging for pools of 6C, 200, 60 and 1 rows (uncached fallback): OK")


def test_global_matches_per_slot_for_one_request():
    """One request, pool rows == slot rows: same windowed CLOCK, so the same hits and misses as today's layout."""
    ctxs = [4000]
    H = 2 * C
    kv, host_kv, full_bt, gpu_bt, host_bt = _world(ctxs)
    hg = _pool(H, host_kv.shape[0] * BS, age=2)
    hs = HotBuffer(1, 2, H, HKV, W, max_ctx=(ctxs[0] // BS + 1) * BS, device="cpu", age=2, kv_slots=1)
    tr = [gen_trace(ctxs[0], 40, K=KSEL, sink=SINK, win=WIN, seed=3)]
    nbs = (C + BS - 1) // BS
    sbt = torch.arange(nbs, dtype=torch.int32).view(1, nbs)
    s1, s2 = torch.zeros((nbs, HKV, BS, W), dtype=torch.int8), torch.zeros((nbs, HKV, BS, W), dtype=torch.int8)
    slot = torch.zeros(1, dtype=torch.int32)
    for step in range(40):
        idx, nv = _idx(ctxs, tr, step)
        hg.next_step(); hs.next_step()
        a = gather_rows_global_ref(kv, gpu_bt, host_kv, host_bt, idx, nv, s1, sbt, hg.layer(0), slot, int(hg.step), 2)
        b = gather_rows_tiered_ref(kv, gpu_bt, host_kv, host_bt, idx, nv, s2, sbt, hs.layer(0), slot, int(hs.step), 2)
        assert a[:3] == b and a[3] == 0, (step, a, b)
        assert torch.equal(s1, s2)
        assert int(hg.hand[0, 0]) == int(hs.hand[0, 0])
    print("S14 global pool == per-slot buffer for one request (hits, misses, hand, staged bytes, 40 steps): OK")


def test_uncached_rows_never_touch_the_pool():
    """Rows whose slot is >= cached_slots (graph padding, prefill suffixes) read host directly and leave no trace."""
    ctxs = [3000, 2200]
    tot, hot = _run(ctxs, 4 * C, slots=[5, 5])
    assert tot[0] == 0 and tot[3] == 0 and tot[1] > 0
    assert int((hot.tag >= 0).sum()) == 0 and int((hot.pos2row >= 0).sum()) == 0 and int(hot.hand.sum()) == 0
    assert int((hot.last != NEG_AGE).sum()) == 0
    print("S14 uncached rows (slot >= cached_slots): exact staging, pool untouched: OK")


def test_shared_host_blocks_share_rows_and_rows_survive_requests():
    """Two requests mapping the same host blocks hit each other's rows; a 'new request' on the same host blocks starts
    warm (the per-slot layout resets the slot: all misses)."""
    ctxs = [3000, 3000]
    kv, host_kv, full_bt, gpu_bt, host_bt = _world(ctxs, share=(0, 1, 3000 // BS + 1))
    assert torch.equal(host_bt[0], host_bt[1])
    hot = _pool(4 * C, host_kv.shape[0] * BS)
    tr = gen_trace(3000, 12, K=KSEL, sink=SINK, win=WIN, seed=5)
    nbs = (C + BS - 1) // BS
    sbt = torch.arange(2 * nbs, dtype=torch.int32).view(2, nbs)
    stg = torch.zeros((2 * nbs, HKV, BS, W), dtype=torch.int8)
    one, two = slice(0, 1), slice(0, 2)
    # steps 0..5: request 0 alone
    for step in range(6):
        idx, nv = _idx(ctxs[:1], [tr], step)
        hot.next_step()
        gather_rows_global_ref(kv, gpu_bt[one], host_kv, host_bt[one], idx, nv, stg, sbt[one], hot.layer(0),
                               torch.zeros(1, dtype=torch.int32), int(hot.step), hot.age)
    # step 6: request 1 ("a follow-up turn", other slot) asks for request 0's last picks: all non-resident picks hit
    idx, nv = _idx(ctxs[:1], [tr], 5)
    hot.next_step()
    h, m, r, u = gather_rows_global_ref(kv, gpu_bt[1:2], host_kv, host_bt[1:2], idx, nv, stg, sbt[one], hot.layer(0),
                                        torch.zeros(1, dtype=torch.int32), int(hot.step), hot.age)
    assert m == 0 and h > 0, (h, m, r)
    # both in one batch with identical NEW picks: both miss (duplicate keys in one step), data still exact
    idx1, nv1 = _idx(ctxs[:1], [tr], 11)
    idx, nv = idx1.repeat(2, 1), nv1.repeat(2)
    hot.next_step()
    gather_rows_global_ref(kv, gpu_bt, host_kv, host_bt, idx, nv, stg, sbt, hot.layer(0),
                           torch.zeros(2, dtype=torch.int32), int(hot.step), hot.age)
    for rr in range(2):
        n = int(nv[rr]); j = torch.arange(n); pos = idx[rr, :n].long()
        assert torch.equal(stg[sbt[rr, j // BS].long(), :, j % BS], kv[full_bt[rr, pos // BS].long(), :, pos % BS])
    _check_pool(hot, host_kv)
    print("S14 rows keyed by host token: shared by requests on the same host blocks, kept across requests: OK")


def test_invalidate_on_host_block_rewrite():
    """A rewritten host block must lose its pool rows: with invalidate_host_blocks the next read is the new content;
    without it the stale row would be served (the test has teeth)."""
    ctxs = [3000]
    for invalidate in (True, False):
        kv, host_kv, full_bt, gpu_bt, host_bt = _world(ctxs)
        hot = _pool(4 * C, host_kv.shape[0] * BS)
        tr = [gen_trace(3000, 8, K=KSEL, sink=SINK, win=WIN, seed=9)]
        nbs = (C + BS - 1) // BS
        sbt = torch.arange(nbs, dtype=torch.int32).view(1, nbs)
        stg = torch.zeros((nbs, HKV, BS, W), dtype=torch.int8)
        slot = torch.zeros(1, dtype=torch.int32)
        for step in range(4):
            idx, nv = _idx(ctxs, tr, step)
            hot.next_step()
            gather_rows_global_ref(kv, gpu_bt, host_kv, host_bt, idx, nv, stg, sbt, hot.layer(0), slot,
                                   int(hot.step), hot.age)
        # rewrite the host blocks that hold cached rows (new content under the same host block ids)
        cached_blocks = torch.unique(torch.nonzero(hot.pos2row[0, 0] >= 0).flatten() // BS)
        assert cached_blocks.numel() > 3
        victims = cached_blocks[:3]
        before = int((hot.pos2row[0, 0] >= 0).sum())
        for hb in victims.tolist():
            gb = int(full_bt[0, hb])
            kv[gb] = torch.randint(-128, 127, kv[gb].shape, dtype=torch.int8)
            host_kv[hb] = kv[gb]
        if invalidate:
            hot.invalidate_host_blocks(victims, BS)
            after = int((hot.pos2row[0, 0] >= 0).sum())
            assert after < before
            gone = (hot.tag[0, 0] < 0) & (hot.last[0, 0] == NEG_AGE)
            assert int(gone.sum()) >= before - after  # freed rows are empty and first in line as victims
            _check_pool(hot, host_kv)
        idx, nv = _idx(ctxs, tr, 3)  # the same picks again
        hot.next_step()
        gather_rows_global_ref(kv, gpu_bt, host_kv, host_bt, idx, nv, stg, sbt, hot.layer(0), slot, int(hot.step),
                               hot.age)
        n = int(nv[0]); j = torch.arange(n); pos = idx[0, :n].long()
        exact = torch.equal(stg[sbt[0, j // BS].long(), :, j % BS], kv[full_bt[0, pos // BS].long(), :, pos % BS])
        assert exact == invalidate, f"invalidate={invalidate}: exact={exact}"
    # absent keys / duplicates / several layers: no error, nothing else disturbed
    hot = HotBuffer(3, 1, 16, HKV, W, max_ctx=10 * BS, device="cpu", kv_slots=1)
    hot.tag[1, 0, 4] = 2 * BS + 7; hot.pos2row[1, 0, 2 * BS + 7] = 4; hot.last[1, 0, 4] = 9
    hot.tag[1, 0, 0] = 5 * BS; hot.pos2row[1, 0, 5 * BS] = 0; hot.last[1, 0, 0] = 9
    hot.invalidate_host_blocks(torch.tensor([2, 2, 7]), BS)
    assert int(hot.tag[1, 0, 4]) == -1 and int(hot.pos2row[1, 0, 2 * BS + 7]) == -1 and int(hot.last[1, 0, 4]) == NEG_AGE
    assert int(hot.tag[1, 0, 0]) == 5 * BS and int(hot.pos2row[1, 0, 5 * BS]) == 0 and int(hot.last[1, 0, 0]) == 9
    assert int((hot.tag[0] >= 0).sum()) == 0 and int((hot.tag[2] >= 0).sum()) == 0
    print("S14 invalidate_host_blocks: rewritten host blocks drop their rows (stale without it), others untouched: OK")


# ------------------------------------------------------------------------------------------------ connector wiring
def _tc(**env):
    for k in ("SPARSE_TIER_HOT_GLOBAL", "SPARSE_TIER_HOT_GLOBAL_ROWS", "SPARSE_TIER_SHARED_PLAN",
              "SPARSE_TIER_SHARED_PLAN_CHECK", "SPARSE_TIER_HOT_FRAC", "SPARSE_TIER_SPILL", "SPARSE_TIER_HOST_GB",
              "SPARSE_TIER_HOST_BITS"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    from sparsekv import tier_connector as tc
    return importlib.reload(tc)


def _vcfg(max_seqs=11, max_len=90112):
    mc = types.SimpleNamespace(max_model_len=max_len, get_num_kv_heads=lambda pc: 4, get_head_size=lambda: 256)
    return types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_seqs=max_seqs), model_config=mc,
                                 parallel_config=None)


def test_knobs_default_off_and_memory():
    Cp, bs, nl, hkv, w, r = 3136, 864, 16, 4, 520, 64
    tc = _tc(SPARSE_TIER_SPILL=1, SPARSE_TIER_HOST_GB=4, SPARSE_TIER_HOT_FRAC=0.025)
    assert not tc.hot_global_enabled() and not tc.shared_plan_enabled()
    d0 = tc.tier_dims(_vcfg(), bs, Cp, True)
    assert "Hg" not in d0 and d0["H"] == 6272
    off = tc.gpu_extra_bytes(_vcfg(), nl, hkv, w, r, bs, Cp)
    # today's formula, unchanged: 11 slots x 6272 rows x 2080 B x 16 layers (+ bookkeeping)
    assert off["tier_hot"] == nl * (11 * 6272 * hkv * w + 12 * (6272 * 8 + d0["maxblk"] * bs * 4))
    assert "tier_shared_plan" not in off
    tc = _tc(SPARSE_TIER_SPILL=1, SPARSE_TIER_HOST_GB=4, SPARSE_TIER_HOT_FRAC=0.025, SPARSE_TIER_HOT_GLOBAL=1)
    d1 = tc.tier_dims(_vcfg(), bs, Cp, True)
    assert d1["Hg"] == 4 * 6272 and d1["H"] == 6272
    on = tc.gpu_extra_bytes(_vcfg(), nl, hkv, w, r, bs, Cp)
    assert on["tier_hot"] == nl * (d1["Hg"] * hkv * w + d1["Hg"] * 8 + d1["HB"] * bs * 4 + 4)
    assert {k: v for k, v in on.items() if k != "tier_hot"} == {k: v for k, v in off.items() if k != "tier_hot"}
    tc = _tc(SPARSE_TIER_SPILL=1, SPARSE_TIER_HOST_GB=4, SPARSE_TIER_HOT_GLOBAL=1, SPARSE_TIER_HOT_GLOBAL_ROWS=9000)
    assert tc.tier_dims(_vcfg(), bs, Cp, True)["Hg"] == 9000
    assert tc.tier_dims(_vcfg(max_seqs=2), bs, Cp, True)["Hg"] == 9000
    tc = _tc(SPARSE_TIER_SPILL=1, SPARSE_TIER_HOST_GB=4, SPARSE_TIER_HOT_GLOBAL=1)
    assert tc.tier_dims(_vcfg(max_seqs=2), bs, Cp, True)["Hg"] == 2 * max(2 * Cp, int(0.10 * 90112))  # never above today
    try:
        tc.tier_dims(_vcfg(), bs, Cp, False)
        raise AssertionError("global pool without spill must be refused")
    except ValueError:
        pass
    tc = _tc(SPARSE_TIER_SPILL=1, SPARSE_TIER_HOST_GB=4, SPARSE_TIER_HOT_FRAC=0.025, SPARSE_TIER_SHARED_PLAN=1)
    sp = tc.gpu_extra_bytes(_vcfg(), nl, hkv, w, r, bs, Cp)
    assert sp["tier_shared_plan"] == 3 * 19 * d0["maxblk"] * 4 and sp["tier_hot"] == off["tier_hot"]
    print("S14 knobs: off by default (sizes and reserve unchanged: %.0f MB hot), global pool reserve %.0f MB at the "
          "default 4 x 6272 rows, needs SPILL: OK" % (off["tier_hot"] / 1e6, on["tier_hot"] / 1e6))
    _tc()


def _rt(tc, shared, glob, check=False, nlayers=16, max_reqs=3, maxblk=12, HB=40, NB=50):
    rt = tc.TierRuntime.__new__(tc.TierRuntime)
    rt.spill, rt.groups, rt.BS, rt.sink_blocks, rt.tail, rt.NB, rt.HB = True, [3], BS, 1, 2 * BS, NB, HB
    rt.max_reqs, rt.maxblk, rt.L, rt.dev, rt.C = max_reqs, maxblk, 0, "cpu", C
    rt.layers = {f"model.layers.{3 + 4 * i}.self_attn.attn": (i, 3) for i in range(nlayers)}
    rt.batch_slots = torch.full((max_reqs + 8,), max_reqs, dtype=torch.int32)
    rt.host_bt = torch.full((max_reqs + 1, maxblk), -1, dtype=torch.int32)
    rt.freed_upto = torch.zeros(max_reqs + 1, dtype=torch.int32)
    rt.barange = torch.arange(maxblk, dtype=torch.int32)
    rt.mirror = types.SimpleNamespace(dev=[None] * nlayers)
    rt.verify, rt.stats, rt.sketch, rt.kv_by_slot = False, {"mirrored_blocks": 0}, {}, []
    rt.h2d = lambda xs, dtype=torch.int32: torch.tensor(list(xs), dtype=torch.int64).to(dtype)
    rt.hot_global, rt.shared_plan, rt._plan_check = glob, shared, check
    rt._plan, rt._plan_counts, rt.plan_diff = {}, [0, 0], torch.zeros((), dtype=torch.int64)
    rt._plan_leader = min(rt.layers.items(), key=lambda kv: int(tc._LAYER_RE.search(kv[0]).group(1)))[1][0]
    rt.hot = (HotBuffer(nlayers, 1, 64, HKV, W, max_ctx=HB * BS, device="cpu", kv_slots=1) if glob
              else HotBuffer(nlayers, max_reqs + 1, 64, HKV, W, max_ctx=maxblk * BS, device="cpu", kv_slots=max_reqs))
    rt.scratch = None
    rt.free_slots = list(range(max_reqs - 1, -1, -1))
    rt.mirrored = torch.zeros((4, max_reqs + 1), dtype=torch.int32)
    rt.reqs = {}
    return rt


def _batch(rt, width=10):
    """3 real rows (slots 2, 0, 1) + 1 padding row; request rows have host-mapped, partly freed / null blocks."""
    rt.batch_slots[:4] = torch.tensor([2, 0, 1, rt.max_reqs], dtype=torch.int32)
    bt = torch.zeros((4, width), dtype=torch.int32)
    seq = torch.tensor([9 * BS + 5, 6 * BS, 3 * BS + 1, 1], dtype=torch.int32)
    for row, (slot, nblk, freed, null) in enumerate([(2, 10, 5, ()), (0, 6, 0, (1, 2)), (1, 4, 0, ())]):
        bt[row, :nblk] = torch.arange(100 * (row + 1), 100 * (row + 1) + nblk, dtype=torch.int32)
        for b in null:
            bt[row, b] = 0
        rt.host_bt[slot, : nblk - 1] = torch.arange(10 * slot, 10 * slot + nblk - 1, dtype=torch.int32)
        rt.freed_upto[slot] = freed
    return bt, seq


def test_shared_plan_same_tables_one_computation():
    tc = _tc(SPARSE_TIER_SPILL=1)
    calls = []

    def run(shared, check=False, glob=False):
        rt = _rt(tc, shared, glob, check)
        bt, seq = _batch(rt)
        n_sp = [0]
        orig = rt._spilled
        rt._spilled = lambda *a: (n_sp.__setitem__(0, n_sp[0] + 1), orig(*a))[1]
        out = []
        calls.clear()
        for name, (l, g) in rt.layers.items():  # forward order
            sk = rt.sketch_block_table(name, bt, seq)
            rt.gather(l, g, "kv", bt, seq, torch.zeros((4, C), dtype=torch.int32), "nv", "stkv", "stbt")
            out.append((sk.clone(), calls[-1]["gbt"].clone(), calls[-1]["hbt"].clone()))
            assert calls[-1]["gbt"].stride(0) == calls[-1]["gbt"].shape[1], "gpu_bt rows must stay dense"
            assert sk.stride(1) == 1 and calls[-1]["hbt"].shape[1] >= calls[-1]["gbt"].shape[1]
        return rt, out, n_sp[0], [c["fn"] for c in calls]

    def rec(fn):
        def f(kv, gbt, host_dev, hbt, idx, n_valid, st_kv, st_bt, hot, layer, slots, scratch, **kw):
            calls.append({"fn": fn, "gbt": gbt, "hbt": hbt, "slots": slots, "kw": kw, "layer": layer})
        return f

    tc.gather_rows_tiered = rec("slot")
    tc.gather_rows_tiered_global = rec("global")
    rt0, base, n0, fns0 = run(False)
    assert n0 == 32 and set(fns0) == {"slot"}  # today: the spilled mask is built twice per layer
    assert bool((base[0][1] == -1).any()) and bool((base[0][0] >= rt0.NB * 1).any())  # the batch does have spilled blocks
    rt1, sh, n1, _ = run(True)
    assert n1 == 1 and rt1._plan_counts == [1, 31], (n1, rt1._plan_counts)
    for a, b in zip(base, sh):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    # a second forward with other values: the leader refreshes the buffers, followers see the new plan
    bt, seq = _batch(rt1)
    rt1.freed_upto[2] = 8
    bt[1, 4] = 0
    names = list(rt1.layers)
    want_sk = _rt_unshared(tc, rt1).sketch_block_table(names[5], bt, seq)
    rt1.sketch_block_table(names[0], bt, seq)
    assert torch.equal(rt1.sketch_block_table(names[5], bt, seq), want_sk)
    # another block-table width gets its own buffers (dense rows), created by the leader
    bt7 = bt[:, :7].contiguous()
    rt1.sketch_block_table(names[0], bt7, seq)
    got = rt1.sketch_block_table(names[9], bt7, seq)
    assert got.shape == (4, 7) and torch.equal(got, _rt_unshared(tc, rt1).sketch_block_table(names[9], bt7, seq))
    assert sorted(rt1._plan) == [7, 10]
    # no buffer yet for a width and a follower asks first: it computes its own tables (same values)
    rt2 = _rt(tc, True, False)
    bt, seq = _batch(rt2)
    assert torch.equal(rt2.sketch_block_table(names[3], bt, seq), base[3][0]) and rt2._plan == {}
    # check mode: everyone computes, differing entries are counted (0 here; > 0 once the buffers are made stale)
    rt3, chk, n3, _ = run(True, check=True)
    assert n3 == 32 and int(rt3.plan_diff) == 0
    for a, b in zip(base, chk):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    bt, seq = _batch(rt3)
    rt3.freed_upto[2] = 9  # plan changes but the leader does not run: a follower must notice
    rt3.sketch_block_table(names[4], bt, seq)
    assert int(rt3.plan_diff) > 0
    # both knobs together: the pool gather gets the shared tables and cached_slots = max_reqs
    rt4, both, n4, fns4 = run(True, glob=True)
    assert n4 == 1 and set(fns4) == {"global"} and calls[-1]["kw"] == {"cached_slots": rt4.max_reqs}
    for a, b in zip(base, both):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    rt5, g_only, n5, fns5 = run(False, glob=True)
    assert n5 == 32 and set(fns5) == {"global"}
    print("S14 shared residency plan: 16 layers get the unshared tables with 1 mask computation instead of 32; "
          "refresh, widths, fallback, check mode, with and without the global pool: OK")
    _tc()


def _rt_unshared(tc, rt):
    u = _rt(tc, False, False)
    u.batch_slots, u.host_bt, u.freed_upto = rt.batch_slots, rt.host_bt, rt.freed_upto
    return u


def test_connector_pool_lifecycle():
    """Global pool: a new request does not reset rows; every GPU->host block copy invalidates the written host blocks."""
    tc = _tc(SPARSE_TIER_SPILL=1)
    for glob in (False, True):
        rt = _rt(tc, False, glob)
        if glob:
            rt.hot.tag[:, 0, 3] = 7 * BS + 1; rt.hot.pos2row[:, 0, 7 * BS + 1] = 3
            rt.hot.tag[:, 0, 5] = 9 * BS; rt.hot.pos2row[:, 0, 9 * BS] = 5
        else:
            rt.hot.tag[:, 2, 3] = 17; rt.hot.pos2row[:, 2, 17] = 3
        rt._new("a", {3: [1, 2, 3]}, 0)
        slot = rt.reqs["a"]["slot"]
        if glob:
            assert int((rt.hot.tag >= 0).sum()) == 2 * rt.hot.L  # rows kept
            rt._copy_blocks(3, [11, 12], [7, 8])
            assert int(rt.hot.tag[0, 0, 3]) == -1 and int(rt.hot.pos2row[0, 0, 7 * BS + 1]) == -1
            assert int(rt.hot.tag[0, 0, 5]) == 9 * BS and rt.stats["hot_invalidated_blocks"] == 2
        else:
            rt.hot.tag[:, slot, 3] = 17
            rt._drop("a"); rt._new("b", {3: [1]}, 0)
            assert int((rt.hot.tag[:, rt.reqs["b"]["slot"]] >= 0).sum()) == 0  # today's behaviour: slot reset
            rt._copy_blocks(3, [11], [7])
            assert "hot_invalidated_blocks" not in rt.stats
    print("S14 connector: pool rows survive a new request, host block copies invalidate; per-slot path as before: OK")
    _tc()


if __name__ == "__main__":
    test_global_ref_exact_any_pool_size()
    test_global_matches_per_slot_for_one_request()
    test_uncached_rows_never_touch_the_pool()
    test_shared_host_blocks_share_rows_and_rows_survive_requests()
    test_invalidate_on_host_block_rewrite()
    test_knobs_default_off_and_memory()
    test_shared_plan_same_tables_one_computation()
    test_connector_pool_lifecycle()
    print("ALL OK")
