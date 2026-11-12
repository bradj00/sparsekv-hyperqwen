# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""s13 CPU tests (no GPU). Run: CUDA_VISIBLE_DEVICES= venv/bin/python sparsekv/test_s13_cpu.py

1. Superseded-first eviction of host GDN checkpoints (SPARSE_TIER_MAMBA_EVICT_SUPERSEDED=1): a busy agent's stale
   turn-N states are recycled before an idle agent's only checkpoint; off (default) is plain LRU as before."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _store(on: bool, slots: int):
    os.environ["SPARSE_TIER_MAMBA_EVICT_SUPERSEDED"] = "1" if on else "0"
    from sparsekv import tier_connector as tc
    importlib.reload(tc)
    return tc.MambaHostStore(slots)


def _turns(ms, tag, nturns, start=0):
    """One conversation: turn t saves the state at chain depth t (its ancestors are depths < t)."""
    chain = [(tag, i) for i in range(start + nturns)]
    for t in range(start, start + nturns):
        ms.save(chain[t], {1: 1}, chain[:t])
    return chain


def test_superseded_goes_before_idle_agent():
    for on in (False, True):
        ms = _store(on, 4)
        idle = _turns(ms, "idle", 1)  # the idle agent's only checkpoint (oldest in LRU order)
        _turns(ms, "busy", 6)  # the busy agent keeps going: 6 chained states through the remaining 3 slots
        if on:
            assert idle[0] in ms.h2s, "idle agent's only checkpoint evicted with superseded-first on"
            assert ("busy", 5) in ms.h2s, "newest state of the busy agent must be stored"
            assert ms.stats.get("evict_superseded", 0) >= 1
        else:
            assert idle[0] not in ms.h2s, "plain LRU is expected to evict the idle agent here (the failure fixed)"
        assert len(ms.h2s) + len(ms.free) == 4
    print("s13 superseded-first: idle agent's checkpoint survives a busy agent's turns (LRU: evicted): OK")


def test_falls_back_to_lru_and_respects_pins():
    ms = _store(True, 3)
    a = _turns(ms, "a", 1)
    b = _turns(ms, "b", 1)
    c = _turns(ms, "c", 1)
    ms.kept.add(a[0])  # pinned warm prefix
    ms.save(("d", 0), {1: 1}, [])  # nothing is superseded: oldest UNPINNED goes (b), not the pin
    assert a[0] in ms.h2s and b[0] not in ms.h2s and c[0] in ms.h2s
    # a pinned state is never the victim even when it is superseded
    ms2 = _store(True, 2)
    ch = [("p", 0), ("p", 1), ("p", 2)]
    ms2.save(ch[0], {1: 1}, [])
    ms2.kept.add(ch[0])
    ms2.save(ch[1], {1: 1}, ch[:1])
    ms2.save(ch[2], {1: 1}, ch[:2])
    assert ch[0] in ms2.h2s and ch[2] in ms2.h2s and ch[1] not in ms2.h2s
    print("s13 superseded-first: LRU fallback and pins respected: OK")


def test_bookkeeping_after_evict_and_drop():
    ms = _store(True, 3)
    ch = _turns(ms, "x", 3)
    # the deepest state is dropped (e.g. a cancelled decode save): shallower ones are no longer superseded by it
    assert ms.drop(ch[2], ms.h2s[ch[2]])
    assert ch[2] not in ms.sup
    ms.save(("y", 0), {1: 1}, [])
    ms.save(("y", 1), {1: 1}, [("y", 0)])  # store full: x0 is superseded by live x1 -> x0 goes first
    assert ch[0] not in ms.h2s and ch[1] in ms.h2s
    for k, deeper in ms.sup.items():
        assert k in ms.h2s
    assert len(ms.h2s) + len(ms.free) == 3
    print("s13 superseded-first: bookkeeping after evict/drop: OK")


def test_off_is_unchanged():
    ms = _store(False, 2)
    ms.save(("a", 0), {1: 1}, [])
    ms.save(("a", 1), {1: 1}, [("a", 0)])
    assert ms.sup == {}
    ms.save(("b", 0), {1: 1})  # old call signature still works
    assert ("a", 0) not in ms.h2s
    print("s13 superseded-first off: plain LRU, no ancestry kept: OK")


# ------------------------------------------------------------------------------------- depth-aware prefill chunk cap
def _deep_run(depth_knob, with_decodes=True, start=0, total=20000):
    import types
    os.environ["SPARSE_TIER_MAMBA_EVICT_SUPERSEDED"] = "0"
    from sparsekv import tier_connector as tc
    importlib.reload(tc)
    import test_s9_cpu as S9

    class S(S9.FakeSched):
        pass
    pol = tc.PrefillYield(12, 864, 0.0, deep_chunk_depth=depth_knob)
    S.schedule = pol.wrap(S9.FakeSched.schedule)
    decs = [S9.Req(f"d{i}", 30000) for i in range(2)] if with_decodes else []
    long_ = S9.Req("L", total, computed=start)
    s = S(decs + [long_])
    chunks = []
    for _ in range(4000):
        before = long_.num_computed_tokens
        out = s.schedule()
        n = out.num_scheduled_tokens.get("L", 0)
        if n:
            chunks.append((before, n))
        if long_.num_computed_tokens >= long_.num_tokens:
            break
    return pol, chunks


def test_deep_chunk_cap():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    D = 8000
    pol, chunks = _deep_run(D)
    shallow = [n for at, n in chunks if at < D]
    deep = [n for at, n in chunks if at >= D]
    assert shallow and deep
    assert max(shallow) > 864, shallow[:4]  # below the depth: the normal budget-sized chunk
    assert max(deep) <= 864, deep[:4]  # at or past the depth, while others decode: capped
    assert pol.stats.get("deep_steps", 0) == len(deep)
    assert sum(n for _, n in chunks) == 20000  # every token still prefilled exactly once
    pol0, chunks0 = _deep_run(0)
    assert max(n for at, n in chunks0 if at >= D) > 864 and pol0.stats.get("deep_steps", 0) == 0  # off = unchanged
    pol1, chunks1 = _deep_run(D, with_decodes=False)
    assert max(n for at, n in chunks1 if at >= D) > 864 and pol1.stats.get("deep_steps", 0) == 0  # nobody to protect
    print("s13 deep-chunk cap: 864-token chunks past the depth while others decode; off / no decodes unchanged: OK")


if __name__ == "__main__":
    test_deep_chunk_cap()
    test_superseded_goes_before_idle_agent()
    test_falls_back_to_lru_and_respects_pins()
    test_bookkeeping_after_evict_and_drop()
    test_off_is_unchanged()
    os.environ["SPARSE_TIER_MAMBA_EVICT_SUPERSEDED"] = "0"
    print("ALL OK")
