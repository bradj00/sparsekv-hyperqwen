# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""CPU tests for s9: PrefillYield short-request bypass (SPARSE_PREFILL_YIELD_SHORT_BYPASS) and the pinned warm prefix
(SPARSE_TIER_PIN_PREFIX_MAX). No GPU.

PYTHONPATH=<tree>/pthfa/python:<tree> python sparsekv/test_s9_cpu.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_fixes_cpu as T  # noqa: E402  (sets sys.path / SPARSEKV_ROOT)

BUDGET = 2048


class Req:
    def __init__(self, rid, num_tokens, cached=0, computed=None):
        self.request_id, self.num_tokens, self.cached = rid, num_tokens, cached
        self.num_computed_tokens = num_tokens if computed is None else computed
        self.arrival_time = None

    @property
    def is_prefill_chunk(self):
        return self.num_computed_tokens < self.num_tokens


class FakeSched:
    """Mimics the two vLLM deferral points patched by sparse-prefill-yield-bypass.patch (running chunk / waiting
    admission) with a 2048-token budget, decodes 8 tokens each."""

    def __init__(self, running, waiting=(), thr=0):
        self.running, self.waiting, self.kv_holding_waiting = list(running), list(waiting), []
        self.requests = {r.request_id: r for r in self.running + self.waiting}
        self.prefill_capacity_bound = False
        self.scheduler_config = types.SimpleNamespace(long_prefill_token_threshold=thr)
        self._sparse_yield_bypass = None

    def add(self, r):
        self.waiting.append(r)
        self.requests[r.request_id] = r

    def schedule(self, throttle_prefills=False):
        defer = throttle_prefills and not self.prefill_capacity_bound and any(not r.is_prefill_chunk
                                                                             for r in self.running)
        byp = getattr(self, "_sparse_yield_bypass", None)
        thr = self.scheduler_config.long_prefill_token_threshold
        budget, nst, new = BUDGET, {}, []

        def take(r, nc):
            n = min(r.num_tokens - nc, budget)
            if 0 < thr < n:
                n = thr
            return n
        for r in self.running:
            if not r.is_prefill_chunk:
                nst[r.request_id] = 8
                budget -= 8
                continue
            if defer and not (byp is not None and byp(r, r.num_computed_tokens, True)):
                continue
            n = take(r, r.num_computed_tokens)
            if n > 0:
                nst[r.request_id] = n
                budget -= n
        skipped = []
        while self.waiting and budget > 0:
            r = self.waiting[0]
            nc = r.cached
            if defer and nc < r.num_tokens - 1 and not (byp is not None and byp(r, nc, False)):
                if byp is not None and getattr(byp, "scan", False):
                    skipped.append(self.waiting.pop(0))
                    continue
                break
            self.waiting.pop(0)
            r.num_computed_tokens = nc
            n = take(r, nc)
            new.append(types.SimpleNamespace(req_id=r.request_id, num_computed_tokens=nc))
            self.running.append(r)
            nst[r.request_id] = n
            budget -= n
        self.waiting = skipped + self.waiting
        for rid, n in nst.items():
            self.requests[rid].num_computed_tokens += n
        return types.SimpleNamespace(num_scheduled_tokens=nst, scheduled_new_reqs=new)


def _setup(short, scan=True, max_debt=240, thr=0):
    from sparsekv.tier_connector import PrefillYield

    class S(FakeSched):
        pass
    pol = PrefillYield(12, 864, 0.0, short_bypass=short, short_scan=scan, short_max_debt=max_debt)
    S.schedule = pol.wrap(FakeSched.schedule)
    decs = [Req(f"d{i}", 30000) for i in range(3)]
    long_ = Req("L", 64000, computed=0)
    return pol, S(decs + [long_], thr=thr), long_


def _run(pol, s, rid, arrive_at=3, mk=None, limit=3000):
    """steps from arrival to the first scheduled chunk and to the end of the prefill of request rid"""
    first = None
    for step in range(limit):
        if step == arrive_at:
            s.add(mk())
        out = s.schedule()
        if first is None and rid in out.num_scheduled_tokens:
            first = step - arrive_at
        r = s.requests.get(rid)
        if r is not None and r in s.running and r.num_computed_tokens >= r.num_tokens:
            return first, step - arrive_at
    raise AssertionError("never finished prefill")


def test_bypass_short_followup():
    for thr in (0, 864):  # 864: the share cap chunks the short request (as SPARSE_PREFILL_SHARE_CHUNK does live)
        mk = lambda: Req("S", 31000, cached=30000)  # noqa: E731  1000 uncached tokens after a prefix hit
        pol0, s0, L0 = _setup(0, thr=thr)
        f_off, d_off = _run(pol0, s0, "S", mk=mk)
        pol1, s1, L1 = _setup(1728, thr=thr)
        f_on, d_on = _run(pol1, s1, "S", mk=mk)
        assert f_off >= 5, f_off  # held by the yield debt of the long prefill's chunk
        assert f_on <= 1 and d_on <= 2, (f_on, d_on)
        assert pol1.stats["bypass_reqs"] == 1 and pol1.stats["bypass_tokens"] == 1000, pol1.stats
        assert pol0.stats["bypass_steps"] == 0
        print(f"bypass (chunk cap {thr}): short follow-up first chunk after {f_off} steps, prefill done after {d_off} "
              f"(off) vs {f_on} / {d_on} (on): OK")
    # the long prefill keeps its share: equal decode-only debt per chunk, plus the additive bypass charge
    assert pol1.stats["bypass_debt_added"] >= -(-1000 * 12 // 1024) // 2, pol1.stats


def test_bypass_long_not_bypassed_and_scan():
    pol, s, L = _setup(1728)
    for _ in range(3):
        s.schedule()  # L chunk -> debt
    assert pol.debt > 0
    s.add(Req("L2", 20000, cached=0))   # long new request at the head
    s.add(Req("S", 31000, cached=30200))  # short behind it
    out = s.schedule()
    assert "S" in out.num_scheduled_tokens and "L2" not in out.num_scheduled_tokens and "L" not in \
        out.num_scheduled_tokens, out.num_scheduled_tokens
    assert [r.request_id for r in s.waiting] == ["L2"]
    pol2, s2, _ = _setup(1728, scan=False)
    for _ in range(3):
        s2.schedule()
    s2.add(Req("L2", 20000, cached=0))
    s2.add(Req("S", 31000, cached=30200))
    out2 = s2.schedule()
    assert "S" not in out2.num_scheduled_tokens  # head-of-line without the scan
    print("bypass: long requests stay deferred; scan lets a short request pass a long head (scan off: blocked): OK")


def test_bypass_continuation_and_guards():
    pol, s, L = _setup(1728, thr=864)
    for _ in range(3):
        s.schedule()
    s.add(Req("S", 31600, cached=30000))  # 1600 uncached, chunk cap 864 -> 2 chunks, both bypassed
    o1 = s.schedule()
    o2 = s.schedule()
    assert o1.num_scheduled_tokens.get("S") == 864 and o2.num_scheduled_tokens.get("S") == 736, (o1, o2)
    assert "L" not in o1.num_scheduled_tokens and "L" not in o2.num_scheduled_tokens
    assert pol.stats["bypass_reqs"] == 1 and pol.stats["bypass_steps"] == 2
    # max-debt guard
    polg, sg, _ = _setup(1728, max_debt=0)
    for _ in range(3):
        sg.schedule()
    sg.add(Req("S", 31000, cached=30000))
    og = sg.schedule()
    assert "S" not in og.num_scheduled_tokens and polg.stats["bypass_denied_debt"] >= 1
    # per-step budget: two short requests of 1000 with N = 1728 -> only one per throttled step
    polb, sb, _ = _setup(1728)
    for _ in range(3):
        sb.schedule()
    sb.add(Req("S1", 31000, cached=30000))
    sb.add(Req("S2", 31000, cached=30000))
    ob = sb.schedule()
    assert ("S1" in ob.num_scheduled_tokens) != ("S2" in ob.num_scheduled_tokens), ob.num_scheduled_tokens
    assert polb.stats["bypass_denied_budget"] >= 1
    print("bypass: chunked short request continues via the running path; max-debt guard; per-step N budget: OK")


def test_short_behind_long_without_decodes():
    from sparsekv.tier_connector import PrefillYield
    for short, want_done in ((0, False), (1728, True)):
        class S(FakeSched):
            pass
        pol = PrefillYield(12, 864, 0.0, short_bypass=short)
        S.schedule = pol.wrap(FakeSched.schedule)
        L = Req("L", 64000, computed=0)
        s = S([L])
        s.schedule()
        s.add(Req("S", 31600, cached=30000))  # admitted with the share cap (864), 736 left
        s.schedule()
        for _ in range(3):
            s.schedule()
        done = s.requests["S"].num_computed_tokens >= 31600
        assert done == want_done, (short, s.requests["S"].num_computed_tokens)
    print("no decodes: a short request's leftover chunk no longer waits for the long prefill to end (s7: it does): OK")


def test_admission_stats():
    pol, s, L = _setup(1728)
    pol.log_admit = True
    for step in range(40):
        if step == 3:
            s.add(Req("S", 31000, cached=30000))
        s.schedule()
    assert pol.stats["admit_short"] == 1, pol.stats
    assert getattr(s.requests["S"], "_sparse_uncached") == 1000
    print("admission counters + per-request admit line: OK")


def test_install_refuses_without_hook():
    from sparsekv import tier_connector as tc

    class Sch:
        def schedule(self, throttle_prefills=False):
            return None
    os.environ["SPARSE_PREFILL_YIELD_SHORT_BYPASS"] = "1728"
    try:
        pol = tc.install_prefill_yield(Sch)
        assert pol.short_bypass == 0  # no hook in this schedule(): bypass forced off

        class Sch2:
            def schedule(self, throttle_prefills=False):
                return getattr(self, "_sparse_yield_bypass", None)
        pol2 = tc.install_prefill_yield(Sch2)
        assert pol2.short_bypass == 1728
    finally:
        del os.environ["SPARSE_PREFILL_YIELD_SHORT_BYPASS"]
    print("install: bypass only with the vLLM hook present: OK")


def test_vllm_hook_in_patch():
    p = os.path.join(os.path.dirname(HERE), "patches", "sparse-prefill-yield-bypass.patch")
    txt = open(p).read()
    assert txt.count('getattr(self, "_sparse_yield_bypass", None)') == 2
    print("patch: both deferral points carry the hook: OK")


# ------------------------------------------------------------------------------------------------ pinned prefix
def _pin_setup(nmax="2", slots=4, hblocks=24):
    from sparsekv import tier_connector as tc
    os.environ["SPARSE_TIER_PIN_PREFIX_MAX"] = nmax
    con, _, _ = T._fake_connector(bs=864)
    con.hc = tc.HostCache(hblocks)
    tc._MAMBA_HOST = tc.MambaHostStore(slots)
    return tc, con


def _warm(tc, con, rid, tag, nblocks, pin=1):
    bs = 864
    hashes = [(tag, i) for i in range(nblocks + 1)]
    req = types.SimpleNamespace(request_id=rid, num_prompt_tokens=nblocks * bs, block_hashes=hashes,
                                sampling_params=types.SimpleNamespace(extra_args={"sparse_pin_prefix": pin}))
    con.req_obj[rid] = req
    host = {}
    for b in range(nblocks):
        host[b] = con.hc.lookup(hashes[b]) if con.hc.lookup(hashes[b]) is not None else con.hc.alloc(hashes[b])
    con.sreq[rid] = {"host": host, "mirrored": nblocks}
    tc._MAMBA_HOST.save(hashes[nblocks - 1], {1: 1, 2: 2})
    con._maybe_pin_prefix(req)
    con._drop_req(rid)  # request_finished releases the request's own references
    return hashes


def test_pin_survives_churn():
    tc, con = _pin_setup()
    h = _warm(tc, con, "warm", "W", 5)
    ms = tc._MAMBA_HOST
    assert h[4] in ms.kept and con.hc.stats["pinned_blocks"] == 5
    # traffic: many GDN saves and host KV allocations (pool of 24 blocks, LRU)
    for i in range(50):
        ms.save(("T", i), {1: 1, 2: 2})
        hb = con.hc.alloc(("TK", i))
        con.hc.release(hb)
    assert h[4] in ms.h2s, "pinned GDN checkpoint evicted"
    assert all(con.hc.lookup(h[b]) is not None for b in range(5)), "pinned host KV evicted"
    assert ms.drop(h[4], ms.h2s[h[4]]) is False
    # without a pin the same churn evicts it
    tc2, con2 = _pin_setup(nmax="0")
    h2 = _warm(tc2, con2, "warm", "W", 5)
    for i in range(50):
        tc2._MAMBA_HOST.save(("T", i), {1: 1, 2: 2})
        con2.hc.release(con2.hc.alloc(("TK", i)))
    assert h2[4] not in tc2._MAMBA_HOST.h2s and con2.hc.lookup(h2[0]) is None
    print("pin: warm prefix GDN slot + host KV survive 50 saves/allocs of churn (unpinned: evicted): OK")


def test_pin_max_and_clear():
    tc, con = _pin_setup(nmax="2", slots=6, hblocks=40)
    a = _warm(tc, con, "w1", "A", 3)
    b = _warm(tc, con, "w2", "B", 3)
    c = _warm(tc, con, "w3", "C", 3)
    ms = tc._MAMBA_HOST
    assert a[2] not in ms.kept and b[2] in ms.kept and c[2] in ms.kept
    assert con.hc.stats["pinned_blocks"] == 6
    # re-warming the same prefix does not double-pin
    _warm(tc, con, "w4", "C", 3)
    assert con.hc.stats["pinned_blocks"] == 6 and len(con._pins) == 2
    assert max(con.hc.ref) == 1, con.hc.ref
    req = types.SimpleNamespace(request_id="clr", num_prompt_tokens=864, block_hashes=[("X", 0)],
                                sampling_params=types.SimpleNamespace(extra_args={"sparse_pin_prefix": "clear"}))
    con._maybe_pin_prefix(req)
    assert not ms.kept and con.hc.stats["pinned_blocks"] == 0 and max(con.hc.ref) == 0
    # all slots kept -> a new save is refused rather than evicting a pin
    tc3, con3 = _pin_setup(nmax="4", slots=2, hblocks=40)
    _warm(tc3, con3, "x1", "P", 2)
    _warm(tc3, con3, "x2", "Q", 2)
    assert tc3._MAMBA_HOST.save(("Z", 0), {1: 1, 2: 2}) is None and tc3._MAMBA_HOST.stats["refused_kept"] == 1
    # off by default: xarg ignored
    tc4, con4 = _pin_setup(nmax="0")
    _warm(tc4, con4, "y", "R", 2)
    assert not tc4._MAMBA_HOST.kept and not getattr(con4, "_pins", None)
    print("pin: max pins (oldest released), re-warm idempotent, clear, full store refuses saves, off by default: OK")


if __name__ == "__main__":
    test_bypass_short_followup()
    test_bypass_long_not_bypassed_and_scan()
    test_bypass_continuation_and_guards()
    test_short_behind_long_without_decodes()
    test_admission_stats()
    test_install_refuses_without_hook()
    test_vllm_hook_in_patch()
    test_pin_survives_churn()
    test_pin_max_and_clear()
    os.environ.pop("SPARSE_TIER_PIN_PREFIX_MAX", None)
    print("ALL OK")
