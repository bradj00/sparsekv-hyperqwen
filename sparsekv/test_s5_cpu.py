# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""CPU tests for the S5 exact-suffix planner (no GPU: run with an empty CUDA_VISIBLE_DEVICES).

CUDA_VISIBLE_DEVICES= venv/bin/python sparsekv/test_s5_cpu.py
  - SPARSE_TIER_SUFFIX_EXACT on: a follow-up becomes 'exact' (all rows: one segment, nx = ql), cold chunks stay
    'dense', no spilled block stays 'plain'; precedence over SUFFIX_SPARSE
  - exact_rows N: S4 segments over the document tail + one N-row exact segment, causal ends intact
  - per-request staging lists at pf_kv[1:]: host blocks from their host ids, the rest from GPU ids, landing indices
  - _runs coalescing; PrefillYield context weighting (off by default)
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv.tier_connector import TierRuntime, _runs, suffix_segments  # noqa: E402
from sparsekv.test_s4_cpu import BS, _part, _req, _rt  # noqa: E402


def _rt5(reqs, rows, exact_rows=0, sparse=True):
    rt = _rt(reqs, rows)
    rt.sfx = dict(rt.sfx, exact=True, exact_rows=exact_rows, sparse=sparse, seg_q=16)
    rt.suffix_sparse = sparse
    rt.stats.update(suffix_exact=0, suffix_exact_rows=0, s5_dma_blocks=0, s5_uva_blocks=0)
    rt.s5_land, rt._s5_pf = None, None
    return rt


def _reqs():
    L_fu = 200_000
    nb_fu = -(-L_fu // BS)
    host = {b: 5000 + b for b in range(150)}
    host.update({b: 9000 + b for b in range(150, 201)})  # two host runs: [5000..5149], [9150..9200]
    fu = _req(L_fu - 1500, 1500, nb_fu, host, 0, uncached=1500, null=range(201))
    cold = _req(150_000, 2048, -(-152_048 // BS), {b: 7000 + b for b in range(120)}, 100, uncached=250_000)
    plain = _req(10_000, 300, -(-10_300 // BS), {}, 0, uncached=300)
    return {"fu": fu, "cold": cold, "plain": plain}, ["d0", "fu", "cold", "plain"], L_fu, nb_fu


def test_exact_all_rows():
    reqs, rows, L_fu, nb_fu = _reqs()
    for sparse in (True, False):  # exact takes precedence over SUFFIX_SPARSE either way
        rt = _rt5(reqs, rows, 0, sparse)
        plan = rt.prefill_plan(_part([1500, 2048, 300]), 1)
        assert [p["mode"] for p in plan] == ["exact", "dense", "plain"], [p["mode"] for p in plan]
        p = plan[0]
        assert p["segs"] == [1500] and p["nx"] == 1500 and p["L_d"].tolist() == [L_fu]
        assert p["x_qsl"].tolist() == [0, 1500]
        hd, hs = (t.tolist() for t in p["x_h"])
        gd, gs = (t.tolist() for t in p["x_g"])
        assert hd == [1 + b for b in range(201)] and hs == [5000 + b for b in range(150)] + [9000 + b for b in
                                                                                           range(150, 201)]
        assert gd == [1 + b for b in range(201, nb_fu)] and gs == [1000 + b for b in range(201, nb_fu)]
        assert p["x_land"].tolist() == list(range(201))
        assert p["x_runs"] == [(5000, 0, 150), (9150, 150, 51)]
        assert p["bt_d"][0, :nb_fu].tolist() == list(range(1, 1 + nb_fu)) and int(p["bt_d"][0, nb_fu]) == 0
        # the dense chunk keeps its own batched staging at pf0 (not disturbed by the exact lists)
        assert plan[1]["pf0"] == 1 and len(rt._pp_h[0]) == 99
        assert rt.stats["suffix_exact"] == 1 and rt.stats["suffix_exact_rows"] == 1500 and rt.stats["suffix_sparse"] == 0
    print("S5 plan: follow-up -> exact (all rows, one segment, causal end L), staging lists + host runs: OK")


def test_exact_question_rows_only():
    reqs, rows, L_fu, _ = _reqs()
    rt = _rt5(reqs, rows, exact_rows=128)
    plan = rt.prefill_plan(_part([1500, 2048, 300]), 1)
    p = plan[0]
    assert p["mode"] == "exact" and p["nx"] == 128
    tail = suffix_segments(1500 - 128, rt.sfx)
    assert p["segs"] == tail + [128], p["segs"]
    assert p["seg_ends"][-1] == L_fu and p["seg_ends"][-2] == L_fu - 128
    assert p["qsl_d"].tolist()[-2:] == [1372, 1500] and p["x_qsl"].tolist() == [0, 128]
    # exact_rows >= ql -> all rows
    rt2 = _rt5(reqs, rows, exact_rows=4096)
    assert rt2.prefill_plan(_part([1500, 2048, 300]), 1)[0]["segs"] == [1500]
    print("S5 plan: exact_rows=128 -> S4 tail segments + one 128-row exact segment; >= ql -> all rows: OK")


def test_serial_dense_staging_is_bounded():
    """Repro of the observed staging OOM (4 concurrent agents): 3 cold chunks of the same 150K prefix in one step (dense mode). Batched staging
    needs 1 + 3 x nblk blocks (> maxblk + 1); serial staging stages each at pf_kv[1:] -> need <= 1 + max nblk."""
    nb = -(-152_000 // BS)
    reqs = {f"c{i}": _req(150_000 + 600 * i, 600, -(-(150_600 + 600 * i) // BS), {b: 100 * i + b for b in range(150)},
                          150, uncached=160_000) for i in range(3)}
    rows = list(reqs)
    for serial in (False, True):
        rt = _rt5(reqs, rows, sparse=False)
        rt.sfx["exact"] = False
        rt.stage_serial = serial
        rt.pf_kv = torch.empty((200, 1, 1, 4), dtype=torch.int8)  # reserved: maxblk + 1 (here 200 >= 1 + nb)
        grew = []
        import sparsekv.tier_connector as tc
        old = tc._log
        tc._log = lambda m, *a: grew.append(m % a) if "grows" in m else None
        try:
            plan = rt.prefill_plan(_part([600, 600, 600]), 0)
        finally:
            tc._log = old
        assert [p["mode"] for p in plan] == ["dense"] * 3
        if serial:
            assert not grew and rt.pf_kv.shape[0] == 200, grew
            assert all(p["serial"] and p["pf0"] == 1 for p in plan)
            assert [p["bt_d"][0, 0].item() for p in plan] == [1, 1, 1]
            assert len(rt._pp_h[0]) == 0 and len(rt._pp_g[0]) == 0  # nothing in the batched lists
            for p in plan:
                assert p["x_qsl"].tolist() == [0, 600] and p["x_n"][0] == 149  # host blocks [1, 150)
        else:
            assert grew and rt.pf_kv.shape[0] >= 1 + 3 * (nb - 1), (grew, rt.pf_kv.shape)
    print("S5 serial staging: 3 concurrent 150K cold chunks stay within the reserved pf_kv (batched: grows): OK")


def test_runs_and_yield_ctx():
    assert _runs([]) == [] and _runs([7]) == [(7, 0, 1)]
    assert _runs([3, 4, 5, 9, 10, 2]) == [(3, 0, 3), (9, 3, 2), (2, 5, 1)]
    from sparsekv.tier_connector import PrefillYield

    class R:
        def __init__(self, rid, pf, nc=0):
            self.request_id, self.is_prefill_chunk, self.num_computed_tokens = rid, pf, nc

    def sched_cls():
        class S:
            def __init__(self):
                self.running = [R("d", False), R("p", True, 200_000)]
                self.requests = {r.request_id: r for r in self.running}
                self.waiting, self.kv_holding_waiting, self.prefill_capacity_bound = [], [], True
                self.scheduler_config = types.SimpleNamespace(long_prefill_token_threshold=0)
                self.calls = []

            def schedule(self, throttle_prefills=False):
                self.calls.append(throttle_prefills)
                d = {"d": 8}
                if not throttle_prefills:
                    d["p"] = 1024
                return types.SimpleNamespace(num_scheduled_tokens=d)
        return S

    for unit, owed in ((0, 12), (100_000, 36)):  # 1024 tokens at 200K context: 12 steps, x3 with ctx_unit 100K
        S = sched_cls()
        pol = PrefillYield(12, 0, unit)
        S.schedule = pol.wrap(S.schedule)
        s = S()
        for _ in range(owed + 2):
            s.schedule()
        assert s.calls[:owed + 2] == [False] + [True] * owed + [False], (unit, s.calls)
    print("S5 _runs coalescing; prefill yield context weighting (off: 12 steps, ctx_unit 100K at 200K: 36): OK")


if __name__ == "__main__":
    test_exact_all_rows()
    test_exact_question_rows_only()
    test_serial_dense_staging_is_bounded()
    test_runs_and_yield_ctx()
    print("ALL OK")
