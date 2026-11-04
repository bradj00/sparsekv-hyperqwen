# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""CPU tests for s7 SPARSE_TIER_MAMBA_SAVE_AT_FINISH (decode-boundary GDN checkpoint kept on host). No GPU.

PYTHONPATH=<tree>/pthfa/python:<tree> python sparsekv/test_s7_cpu.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_fixes_cpu as T  # noqa: E402  (sets sys.path / SPARSEKV_ROOT)


class _RTQ(T._FakeRTQ):
    pass


def _setup(drop=False):
    from sparsekv import tier_connector as tc
    bs = 864
    con, _, _ = T._fake_connector(bs=bs)
    pool = T._FakeMambaPool(60)
    con.hc, con._mamba_gids, con._block_pool, con._mholds = None, [1, 2], pool, {}
    con.save_at_finish, con._dec, con._dec_sid = True, {}, {}
    con._vllm_config = types.SimpleNamespace(speculative_config=types.SimpleNamespace(
        use_eagle_block_drop=lambda: drop))
    tc._MAMBA_HOST = tc.MambaHostStore(8)
    rt = tc._RT = _RTQ()
    rt.run_saves_now = types.MethodType(tc.TierRuntime.run_saves_now, rt)
    rt._run_mamba_saves = types.MethodType(tc.TierRuntime._run_mamba_saves, rt)
    real_copy = tc.TierRuntime._copy_save

    def copy_save(self, slot, blocks, sid):  # keep the real cancelled-skip logic, record real copies
        if sid in tc._DEC_CANCELLED:
            return real_copy(self, slot, blocks, sid)
        self.copied.append(sid)
        self.mamba_done.add(sid)
    rt._copy_save = types.MethodType(copy_save, rt)
    # request: prompt 4 blocks, then decodes; block_hashes grow with output
    req = types.SimpleNamespace(num_prompt_tokens=4 * bs + 100, block_hashes=[("h", i) for i in range(12)])
    con.req_obj["A"] = req
    return tc, con, pool, rt, bs


def _offer(tc, con, pool, rt, boundary, rid="A"):
    a = pool.get_new_blocks(2)
    offers = {rid: [(1, a[0].block_id, boundary), (2, a[1].block_id, boundary)]}
    so = types.SimpleNamespace(kv_connector_block_state=types.SimpleNamespace(boundary_state_offloads=offers),
                               scheduled_new_reqs=[], scheduled_cached_reqs=None)
    meta = con.build_connector_meta(so)
    rt._run_mamba_saves()  # worker start_step: age queued saves, then queue this meta's saves
    rt._msave_q = rt._msave_q + [(0, s, b, i) for s, b, i in meta.mamba_saves]
    pool.free_blocks(a)    # the request moves on; the hold keeps them
    return meta


def _step(tc, con, rt):
    empty = types.SimpleNamespace(kv_connector_block_state=None, scheduled_new_reqs=[], scheduled_cached_reqs=None)
    meta = con.build_connector_meta(empty)
    rt._run_mamba_saves()
    rt._msave_q = rt._msave_q + [(0, s, b, i) for s, b, i in meta.mamba_saves]


def test_off_ignores_decode_boundaries():
    tc, con, pool, rt, bs = _setup()
    try:
        con.save_at_finish = False
        meta = _offer(tc, con, pool, rt, 6 * bs)
        assert not meta.mamba_saves, "knob off must keep the prompt-only save window"
        meta = _offer(tc, con, pool, rt, 4 * bs)          # inside the prompt window: saved as before
        assert len(meta.mamba_saves) == 1
    finally:
        tc._MAMBA_HOST = tc._RT = None
    print("s7 knob off: decode boundaries ignored, prompt window unchanged: OK")


def test_newest_decode_save_kept_older_dropped():
    tc, con, pool, rt, bs = _setup()
    try:
        ms = tc._MAMBA_HOST
        _offer(tc, con, pool, rt, 6 * bs)
        for _ in range(3):
            _step(tc, con, rt)
        assert ("h", 5) in ms.h2s, "decode boundary 6 saved"
        _offer(tc, con, pool, rt, 8 * bs)
        _offer(tc, con, pool, rt, 7 * bs)                  # lower than the newest: ignored
        for _ in range(3):
            _step(tc, con, rt)
        assert ("h", 7) in ms.h2s and ("h", 5) not in ms.h2s and ("h", 6) not in ms.h2s, list(ms.h2s)
        assert ms.stats.get("dec_dropped") == 1
        con.request_finished(types.SimpleNamespace(request_id="A"), [])
        assert ("h", 7) in ms.h2s and not con._mholds and "A" not in con._dec
        assert len(ms.free) == ms.n - 1, (ms.free, ms.h2s)
    finally:
        tc._MAMBA_HOST = tc._RT = None
    print("s7 newest decode checkpoint kept, older dropped, one slot per request after finish: OK")


def test_young_decode_save_cancelled_at_finish():
    tc, con, pool, rt, bs = _setup()
    try:
        ms = tc._MAMBA_HOST
        _offer(tc, con, pool, rt, 6 * bs)
        for _ in range(3):
            _step(tc, con, rt)
        n_copied = len(rt.copied)
        _offer(tc, con, pool, rt, 7 * bs)                  # offered in the last step: copy not enqueued yet
        sid_young = max(con._dec_sid)
        con.request_finished(types.SimpleNamespace(request_id="A"), [])
        assert sid_young not in rt.copied, "a not-yet-final decode checkpoint must not be force-copied at finish"
        assert ("h", 6) not in ms.h2s and ("h", 5) in ms.h2s, list(ms.h2s)
        for _ in range(3):                                   # the worker later skips the cancelled copy
            _step(tc, con, rt)
        assert len(rt.copied) == n_copied and not con._mholds, (rt.copied, con._mholds)
        assert {b.block_id for b in pool.blocks[1:]} == {b.block_id for b in pool.free_q}, "holds leaked"
        # cancelled-but-never-shipped path: offer + finish in the same scheduler pass
        con.req_obj["B"] = types.SimpleNamespace(num_prompt_tokens=100, block_hashes=[("g", i) for i in range(4)])
        a = pool.get_new_blocks(2)
        offers = {"B": [(1, a[0].block_id, 2 * bs), (2, a[1].block_id, 2 * bs)]}
        con._save_mamba_offers(types.SimpleNamespace(kv_connector_block_state=types.SimpleNamespace(
            boundary_state_offloads=offers)))
        pool.free_blocks(a)
        con.request_finished(types.SimpleNamespace(request_id="B"), [])
        assert not ms.pending_saves and ("g", 1) not in ms.h2s and not con._mholds
        assert {b.block_id for b in a} <= {b.block_id for b in pool.free_q}
    finally:
        tc._MAMBA_HOST = tc._RT = None
    print("s7 decode checkpoint offered < SAVE_DELAY steps before finish is cancelled, holds released: OK")


def test_eagle_drop_keeps_two():
    tc, con, pool, rt, bs = _setup(drop=True)
    try:
        ms = tc._MAMBA_HOST
        for b in (6, 7, 8):
            _offer(tc, con, pool, rt, b * bs)
            for _ in range(3):
                _step(tc, con, rt)
        assert ("h", 6) in ms.h2s and ("h", 7) in ms.h2s and ("h", 5) not in ms.h2s, list(ms.h2s)
    finally:
        tc._MAMBA_HOST = tc._RT = None
    print("s7 with the eagle block drop on: the two newest decode checkpoints kept: OK")


if __name__ == "__main__":
    test_off_ignores_decode_boundaries()
    test_newest_decode_save_kept_older_dropped()
    test_young_decode_save_cancelled_at_finish()
    test_eagle_drop_keeps_two()
    print("ALL OK")
