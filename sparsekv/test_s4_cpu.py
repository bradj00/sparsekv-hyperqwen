# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""CPU tests for the S4 prefill planner (sparsekv.tier_connector.TierRuntime.prefill_plan), with fakes.

CUDA_VISIBLE_DEVICES= venv/bin/python sparsekv/test_s4_cpu.py
  - a follow-up (short uncached remainder, spilled blocks) -> 'sparse'; a cold chunk with spilled blocks -> 'dense'
    (never sparse, even when the chunk is short); no spilled block -> 'plain'
  - dense staging lists: spilled blocks from their host blocks, the rest from their GPU blocks, staged contiguously
  - token-count mismatch -> None (legacy path)
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv.tier_connector import TierRuntime  # noqa: E402

BS = 864


def _rt(reqs, rows):
    rt = TierRuntime.__new__(TierRuntime)
    rt.spill, rt.groups, rt.BS, rt.sink_blocks = True, [3], BS, 1
    rt.suffix_sparse, rt.suffix_max, rt.ksel, rt.sink, rt.win = True, 2048, 2048, 64, 1024
    rt.sfx, rt._sfx_file = {"sparse": True, "max": 2048, "ksel": 2048, "seg": 256, "seg_last": 128, "seg_q": 128}, None
    rt.C = 2048 + 64 + 1024
    rt.stats = {"steps": 5, "prefill_staged_blocks": 0, "suffix_sparse": 0, "suffix_tokens": 0}
    rt._pp_key, rt._pp, rt.L, rt.maxblk, rt.dev = None, None, 16, 400, "cpu"
    rt.pf_kv = torch.empty((401, 1, 1, 4), dtype=torch.int8)
    rt.kv_by_slot = [rt.pf_kv]
    rt.batch_rows, rt.reqs = rows, reqs
    rt.h2d = lambda xs, dtype=torch.int32: torch.tensor(xs, dtype=torch.int64).to(dtype)
    return rt


def _req(ncomp, nsched, nblk_alloc, host, freed, uncached=None, null=()):
    blocks = [0 if b in null else 1000 + b for b in range(nblk_alloc)]
    st = {"slot": 0, "blocks": {3: blocks}, "ncomp": ncomp, "nsched": nsched, "host": dict(host), "freed": freed}
    if uncached is not None:
        st["uncached"] = uncached
    return st


def _part(qls, width=320):
    n = len(qls)
    return types.SimpleNamespace(seq_lens=torch.zeros(n, dtype=torch.int32), token_start=32,
                                 token_end=32 + sum(qls), block_table=torch.zeros((n, width), dtype=torch.int32))


def test_prefill_plan_modes_and_staging():
    L_fu = 200_000
    nb_fu = -(-L_fu // BS)
    # follow-up: 1500 uncached tokens; host hits for blocks 0..200 (null GPU blocks) -> spilled
    fu = _req(L_fu - 1500, 1500, nb_fu, {b: 5000 + b for b in range(201)}, 0, uncached=1500, null=range(201))
    # cold chunk at 150K of a 250K prompt: blocks [1, 100) freed by the spill policy, host-mapped
    cold = _req(150_000, 2048, -(-152_048 // BS), {b: 7000 + b for b in range(120)}, 100, uncached=250_000)
    # short cold chunk (last chunk of a big prompt): still dense, never sparse
    cold2 = _req(248_000, 900, -(-248_900 // BS), {b: 9000 + b for b in range(200)}, 150, uncached=248_900)
    plain = _req(10_000, 300, -(-10_300 // BS), {}, 0, uncached=300)
    reqs = {"fu": fu, "cold": cold, "cold2": cold2, "plain": plain}
    rows = ["d0", "d1", "fu", "cold", "cold2", "plain"]  # two decode rows first (req_start = 2)
    rt = _rt(reqs, rows)
    qls = [1500, 2048, 900, 300]
    plan = rt.prefill_plan(_part(qls), 2)
    assert [p["mode"] for p in plan] == ["sparse", "dense", "dense", "plain"], [p["mode"] for p in plan]
    assert [p["qoff"] for p in plan] == [0, 1500, 3548, 4448]
    from sparsekv.tier_connector import suffix_segments
    coarse = {"seg": 256, "seg_last": 128, "seg_q": 128}
    assert suffix_segments(1500, coarse) == [92, 256, 256, 256, 256, 256, 128]
    assert suffix_segments(100, coarse) == [100] and suffix_segments(129, coarse) == [1, 128]
    fine = {"seg": 256, "seg_last": 128, "seg_q": 32}
    assert suffix_segments(1500, fine) == [92, 256, 256, 256, 256, 256, 32, 32, 32, 32]
    assert suffix_segments(70, fine) == [6, 32, 32]
    assert plan[0]["segs"] == suffix_segments(1500, rt.sfx) and plan[0]["L"] == L_fu
    assert plan[0]["seg_ends"][-1] == L_fu and plan[0]["seg_ends"][0] == L_fu - 1500 + 92
    # dense staging: cold's blocks [1,100) from host, block 0 and [100, nblk) from GPU; then cold2's
    hdst, hsrc = (t.tolist() for t in rt._pp_h)
    gdst, gsrc = (t.tolist() for t in rt._pp_g)
    c0 = plan[1]["pf0"]
    assert c0 == 1 and plan[2]["pf0"] == 1 + plan[1]["nblk"]
    hmap = dict(zip(hdst, hsrc))
    gmap = dict(zip(gdst, gsrc))
    for b in range(plan[1]["nblk"]):
        if 1 <= b < 100:
            assert hmap[c0 + b] == 7000 + b
        else:
            assert gmap[c0 + b] == 1000 + b
    assert len(hdst) == 99 + 149 and len(set(hdst) | set(gdst)) == plan[1]["nblk"] + plan[2]["nblk"]
    assert plan[1]["bt_d"][0, :plan[1]["nblk"]].tolist() == list(range(1, 1 + plan[1]["nblk"]))
    assert plan[0]["qsl_d"].tolist() == [0, 92, 348, 604, 860, 1116, 1372, 1500]
    assert plan[0]["L_d"].tolist() == plan[0]["seg_ends"]
    assert plan[1]["qsl_d"].tolist() == [0, 2048] and plan[1]["L_d"].tolist() == [152_048]
    # cached for the other 15 layers of the step
    assert rt.prefill_plan(_part(qls), 2) is plan
    print("S4 prefill plan: follow-up sparse, cold chunks dense (short last chunk too), plain; staging lists: OK")


def test_prefill_plan_no_uncached_info_is_dense_and_mismatch_is_none():
    L = 100_000
    st = _req(L - 500, 500, -(-L // BS), {b: b for b in range(50)}, 50)   # resumed request: no 'uncached'
    rt = _rt({"x": st}, ["x"])
    plan = rt.prefill_plan(_part([500]), 0)
    assert plan[0]["mode"] == "dense"
    rt2 = _rt({"x": st}, ["x"])
    assert rt2.prefill_plan(_part([499]), 0) is None
    print("S4 prefill plan: unknown remainder -> dense; token mismatch -> legacy (None): OK")


def test_prefill_yield_duty_cycle():
    from sparsekv.tier_connector import install_prefill_yield

    class R:
        def __init__(self, rid, pf):
            self.request_id, self.is_prefill_chunk = rid, pf

    class FakeSched:
        def __init__(self):
            self.running, self.prefill_capacity_bound, self.calls = [], True, []
            self.waiting, self.kv_holding_waiting = [], []
            self.scheduler_config = types.SimpleNamespace(long_prefill_token_threshold=0)
            self.thr = []

        def schedule(self, throttle_prefills=False):
            self.calls.append(throttle_prefills)
            self.thr.append(self.scheduler_config.long_prefill_token_threshold)
            chunk = self.scheduler_config.long_prefill_token_threshold or 2048
            sched = {r.request_id: 8 for r in self.running if not r.is_prefill_chunk}
            if not throttle_prefills:
                sched.update({r.request_id: chunk for r in self.running if r.is_prefill_chunk})
            return types.SimpleNamespace(num_scheduled_tokens=sched)

    os.environ["SPARSE_PREFILL_YIELD_STEPS_PER_1K"] = "8"
    os.environ["SPARSE_PREFILL_SHARE_CHUNK"] = "864"
    pol = install_prefill_yield(FakeSched)
    assert install_prefill_yield(FakeSched) is pol  # idempotent
    s = FakeSched()
    s.running = [R("d0", False), R("d1", False), R("p", True)]
    for _ in range(40):
        s.schedule()
    # pattern: one prefill step then 16 decode-only steps
    assert s.calls[:18] == [False] + [True] * 16 + [False], s.calls[:18]
    assert s.prefill_capacity_bound is False
    # no decodes running -> prefill every step, debt cleared
    s2 = FakeSched()
    s2.running = [R("p", True)]
    for _ in range(5):
        s2.schedule()
    assert s2.calls == [False] * 5
    # an external DP throttle request is passed through unchanged
    s3 = FakeSched()
    s3.running = [R("d", False)]
    s3.schedule(True)
    assert s3.calls == [True]
    # requests waiting behind a running prefill: the chunk is capped at 864 for that step (and restored), and the
    # yield owed is proportional (864 tokens -> 7 decode steps)
    s4 = FakeSched()
    s4.running, s4.waiting = [R("d", False), R("p", True)], ["w"]
    for _ in range(9):
        s4.schedule()
    assert s4.thr[0] == 864 and s4.scheduler_config.long_prefill_token_threshold == 0
    assert s4.calls[:9] == [False] + [True] * 7 + [False], s4.calls[:9]
    s5 = FakeSched()
    s5.running = [R("p", True)]
    s5.schedule()
    assert s5.thr == [0], "no waiting requests -> no cap"
    print("S4 prefill yield: 2048-token chunk -> 16 decode-only steps, prefill-only batches unthrottled, chunk capped at "
          "864 while requests wait: OK")


if __name__ == "__main__":
    test_prefill_plan_modes_and_staging()
    test_prefill_plan_no_uncached_info_is_dense_and_mismatch_is_none()
    test_prefill_yield_duty_cycle()
    print("ALL OK")
