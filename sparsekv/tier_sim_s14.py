# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""S14 CPU simulation: per-slot hot buffers (today) vs one GLOBAL pool keyed by host token (SPARSE_TIER_HOT_GLOBAL).

    CUDA_VISIBLE_DEVICES= python sparsekv/tier_sim_s14.py [--ctx 90112] [--preset lab heavy] [--nreq 1 4 8 11]

Model (same worst case as tier_sim.py: every top-k pick is non-resident; K = 2048 picks per request and step):
  * `nreq` requests decode concurrently, each on its OWN document (disjoint host blocks), one synthetic selection
    trace per request (tier_trace.gen_trace, different seeds).
  * per-slot: MAX_SEQS buffers of H rows each are allocated whatever the load (GPU rows = MAX_SEQS * H); a request's
    miss rate does not depend on the other requests. Policies: exact LRU, and the device policy (windowed CLOCK,
    the real reference gather_rows_tiered_ref).
  * global: ONE pool of T rows (GPU rows = T). Policies: exact LRU over all requests, and the S14 device policy
    (the real reference gather_rows_global_ref: one victim scan per step, uncached fallback).
  * --turns: every request is a chain of turns of --turn-steps steps. A new turn is a new request: the per-slot buffer
    is reset (HotBuffer.reset_slot in TierRuntime._new), the global pool keeps its rows (same host blocks via the
    host prefix cache). "cont": the selection trace simply continues across the turn; "fresh": every turn draws a
    new trace over the same document (new focus, no carried working set) -- the two ends of what a real follow-up
    could look like; there are no recorded multi-turn traces on this box.
Output: JSON lines; miss = misses / non-resident picks after --warm steps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv.tier import HotLayer, NEG_AGE, gather_rows_global_ref, gather_rows_tiered_ref  # noqa: E402
from sparsekv.tier_trace import gen_trace  # noqa: E402

BS = 864


def lru_sim(picks, NK: int, H: int, warm: int, resets=()) -> float:
    """Exact LRU with capacity H over keys in [0, NK). picks[t] = int64 keys of step t (unique). A miss that finds no
    evictable row (everything was used in this step) is not cached. resets: steps at which the cache is dropped."""
    last = np.full(NK, -1, dtype=np.int64)
    cached = np.zeros(NK, dtype=bool)
    count = miss_tot = pick_tot = 0
    for t, P in enumerate(picks):
        if t in resets:
            cached[:] = False
            count = 0
        hit = cached[P]
        mp = P[~hit]
        last[P[hit]] = t
        need = mp.size - (H - count)
        if need > 0:
            cpos = np.flatnonzero(cached & (last < t))
            k = min(need, cpos.size)
            if k:
                ev = cpos[np.argpartition(last[cpos], k - 1)[:k]]
                cached[ev] = False
                count -= k
            mp_c = mp[: mp.size - (need - k)]
        else:
            mp_c = mp
        cached[mp_c] = True
        last[mp_c] = t
        count += mp_c.size
        if t >= warm:
            miss_tot += mp.size
            pick_tot += P.size
    return miss_tot / pick_tot


def _hot(S: int, H: int, NK: int) -> HotLayer:
    return HotLayer(kv=None, tag=torch.full((S, H), -1, dtype=torch.int32),
                    last=torch.full((S, H), NEG_AGE, dtype=torch.int32),
                    pos2row=torch.full((S, NK), -1, dtype=torch.int32), hand=torch.zeros(S, dtype=torch.int32))


def clock_slot_sim(tr: np.ndarray, N: int, H: int, warm: int, age: int, C: int, resets=()) -> float:
    """Today's device policy for ONE request (its own slot of H rows)."""
    hot = _hot(1, H, N)
    K = tr.shape[1]
    gpu_bt = torch.full((1, N // BS + 2), -1, dtype=torch.int32)
    dummy = torch.empty((1, 1, BS, 1), dtype=torch.int8)
    slot = torch.zeros(1, dtype=torch.int32)
    nv = torch.tensor([K], dtype=torch.int32)
    miss_tot = pick_tot = 0
    for t, P in enumerate(tr):
        if t in resets:
            hot.tag.fill_(-1); hot.last.fill_(NEG_AGE); hot.pos2row.fill_(-1); hot.hand.zero_()
        idx = torch.zeros((1, C), dtype=torch.int32)
        idx[0, :K] = torch.from_numpy(P)
        _, m, _ = gather_rows_tiered_ref(None, gpu_bt, dummy, None, idx, nv, None, None, hot, slot, t + 1, age,
                                         copy=False)
        if t >= warm:
            miss_tot += m
            pick_tot += K
    return miss_tot / pick_tot


def clock_global_sim(trs: list[np.ndarray], N: int, T: int, warm: int, age: int, C: int) -> tuple[float, float]:
    """S14 device policy: all requests share one pool of T rows. Returns (miss rate, uncached fraction of picks)."""
    n = len(trs)
    nblk = N // BS + 2
    hot = _hot(1, T, n * nblk * BS)
    K = trs[0].shape[1]
    gpu_bt = torch.full((n, nblk), -1, dtype=torch.int32)
    host_bt = (torch.arange(n, dtype=torch.int32)[:, None] * nblk + torch.arange(nblk, dtype=torch.int32)[None, :])
    dummy = torch.empty((1, 1, BS, 1), dtype=torch.int8)
    slot = torch.zeros(n, dtype=torch.int32)
    nv = torch.full((n,), K, dtype=torch.int32)
    miss_tot = pick_tot = unc_tot = 0
    for t in range(trs[0].shape[0]):
        idx = torch.zeros((n, C), dtype=torch.int32)
        for i in range(n):
            idx[i, :K] = torch.from_numpy(trs[i][t])
        _, m, _, u = gather_rows_global_ref(None, gpu_bt, dummy, host_bt, idx, nv, None, None, hot, slot, t + 1, age,
                                            copy=False)
        if t >= warm:
            miss_tot += m
            unc_tot += u
            pick_tot += n * K
    return miss_tot / pick_tot, unc_tot / pick_tot


def traces(N, n, steps, K, preset, turns, turn_steps, mode):
    """n request traces [steps_total, K]; with turns > 1 each request is a chain of turns."""
    if turns <= 1:
        return [gen_trace(N, steps, K=K, preset=preset, seed=1000 + i) for i in range(n)], ()
    if mode == "cont":
        trs = [gen_trace(N, turns * turn_steps, K=K, preset=preset, seed=1000 + i) for i in range(n)]
    else:
        trs = [np.concatenate([gen_trace(N, turn_steps, K=K, preset=preset, seed=1000 + i + 7919 * j)
                               for j in range(turns)]) for i in range(n)]
    return trs, frozenset(range(turn_steps, turns * turn_steps, turn_steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=90112)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--warm", type=int, default=64)
    ap.add_argument("--preset", nargs="+", default=["lab", "heavy"])
    ap.add_argument("--nreq", type=int, nargs="+", default=[1, 4, 8, 11])
    ap.add_argument("--max-seqs", type=int, default=11)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--C", type=int, default=2048 + 64 + 1024)
    ap.add_argument("--age", type=int, default=8)
    ap.add_argument("--slot-rows", type=int, nargs="+", default=[3136, 4096, 6272])
    ap.add_argument("--global-rows", type=int, nargs="+", default=[4096, 6272, 12544, 25088, 34496, 68992])
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--turn-steps", type=int, default=100)
    ap.add_argument("--turn-mode", choices=["cont", "fresh"], default="cont")
    ap.add_argument("--no-clock", action="store_true", help="exact LRU only (fast)")
    a = ap.parse_args()
    N, K = a.ctx, a.K
    nblk = N // BS + 2
    for preset in a.preset:
        steps = a.steps + a.warm
        warm = a.warm if a.turns <= 1 else 0  # multi-turn: count everything, cold starts are the point
        alltr, resets = traces(N, max(a.nreq), steps, K, preset, a.turns, a.turn_steps, a.turn_mode)
        base = {"preset": preset, "ctx": N, "turns": a.turns, "turn_mode": a.turn_mode if a.turns > 1 else None}
        # per-slot: one number per H (independent of the load); mean over the first min(4, n) request traces
        for H in a.slot_rows:
            ts = alltr[: min(4, len(alltr))]
            row = {**base, "layout": "slot", "rows_per_slot": H, "gpu_rows": a.max_seqs * H,
                   "lru": round(float(np.mean([lru_sim(list(tr.astype(np.int64)), N, H, warm, resets) for tr in ts])), 4)}
            if not a.no_clock:
                row["clock"] = round(float(np.mean([clock_slot_sim(tr, N, H, warm, a.age, a.C, resets) for tr in ts])), 4)
            print(json.dumps(row), flush=True)
        for n in a.nreq:
            trs = alltr[:n]
            merged = [np.concatenate([trs[i][t].astype(np.int64) + i * nblk * BS for i in range(n)])
                      for t in range(trs[0].shape[0])]
            for T in a.global_rows:
                row = {**base, "layout": "global", "nreq": n, "gpu_rows": T,
                       "lru": round(lru_sim(merged, n * nblk * BS, T, warm), 4)}
                if not a.no_clock:
                    m, u = clock_global_sim(trs, N, T, warm, a.age, a.C)
                    row["clock"], row["uncached"] = round(m, 4), round(u, 4)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
