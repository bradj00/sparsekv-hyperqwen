# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""s15 CPU tests (no GPU): prefill-budget sharing among concurrent long prefills (SPARSE_PREFILL_BUDGET_SHARE).

PYTHONPATH=<tree>/pthfa/python:<tree> CUDA_VISIBLE_DEVICES= venv/bin/python sparsekv/test_s15_budget_cpu.py

AlignSched extends the s9 scripted scheduler with the parts of vLLM 0.31's schedule() that decide who gets prefill
tokens in mamba align mode (scheduler.py:648-748 running loop, :1070-1190 waiting loop, :414-527
_mamba_block_aligned_split): a chunk that does not end the prefill must end on an 864-token block boundary, else the
request gets zero tokens (running: next request; waiting: the loop stops); a chunk that starts mid-block stops at the
next boundary; with tail_split the last chunk stops at the prompt's last block boundary and the tail is a chunk of
its own. It is a model of those lines, not the real scheduler.
"""
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_fixes_cpu as T  # noqa: E402,F401  (sets sys.path / SPARSEKV_ROOT)
import test_s9_cpu as S9  # noqa: E402

ROOT = os.path.dirname(HERE)
# Optional: a git revision of this repository whose sparsekv/tier_connector.py carries the policy WITHOUT budget
# sharing but with the same constructor (deep_chunk_depth, log_admit). With it, test_off_is_identical_to_s13 replays
# 480 scripted runs against that revision; without it (the default: no such revision is published, the pre-budget
# policy only existed in the development tree, where this comparison passed) that comparison is skipped.
BASE_COMMIT = os.environ.get("SPARSE_TEST_BASE_REV", "")
BS = 864
Req = S9.Req


class AlignSched(S9.FakeSched):
    def __init__(self, running, waiting=(), thr=0, budget=S9.BUDGET, tail_split=True, max_active=None):
        super().__init__(running, waiting, thr)
        self.max_num_scheduled_tokens = budget
        self.scheduler_config.max_num_batched_tokens = budget
        self.num_spec_tokens = 7  # a decode takes 1 + 7 = 8 tokens of the step, as in the s9 fake
        self.tail_split, self.max_active = tail_split, max_active
        self.calls = []  # (throttle_prefills,) per step

    def _split(self, r, start, n):
        end, pe = start + n, r.num_tokens
        if end < pe:
            end = end // BS * BS
        stops = ((start // BS + 1) * BS if start % BS else 0, (pe - pe % BS) if self.tail_split else 0)
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)

    def schedule(self, throttle_prefills=False):
        self.calls.append(bool(throttle_prefills))
        defer = throttle_prefills and not self.prefill_capacity_bound and any(not r.is_prefill_chunk
                                                                             for r in self.running)
        byp = getattr(self, "_sparse_yield_bypass", None)
        thr = self.scheduler_config.long_prefill_token_threshold
        budget, nst, new = self.max_num_scheduled_tokens, {}, []

        def take(r, nc):
            n = r.num_tokens - nc
            if 0 < thr < n:
                n = thr
            return self._split(r, nc, min(n, budget))
        for r in self.running:
            if budget <= 0:
                break
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
            if self.max_active is not None and len(self.running) >= self.max_active:
                break
            r = self.waiting[0]
            nc = r.cached
            if defer and nc < r.num_tokens - 1 and not (byp is not None and byp(r, nc, False)):
                if byp is not None and getattr(byp, "scan", False):
                    skipped.append(self.waiting.pop(0))
                    continue
                break
            n = take(r, nc)
            if n == 0:
                break
            self.waiting.pop(0)
            r.num_computed_tokens = nc
            new.append(types.SimpleNamespace(req_id=r.request_id, num_computed_tokens=nc))
            self.running.append(r)
            nst[r.request_id] = n
            budget -= n
        self.waiting = skipped + self.waiting
        for rid, n in nst.items():
            self.requests[rid].num_computed_tokens += n
        return types.SimpleNamespace(num_scheduled_tokens=nst, scheduled_new_reqs=new)


def _policy_cls(base=False):
    if not base:
        from sparsekv.tier_connector import PrefillYield
        return PrefillYield
    src = subprocess.run(["git", "-C", ROOT, "show", f"{BASE_COMMIT}:sparsekv/tier_connector.py"], check=True,
                         capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="s15base")
    p = os.path.join(d, "tier_connector_base.py")
    open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location("tier_connector_base", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(d, ignore_errors=True)
    return mod.PrefillYield


def _mk(pol, fake, running, **kw):
    S = type("S", (fake,), {})
    S.schedule = pol.wrap(fake.schedule)
    return S(running, **kw)


# ------------------------------------------------------------------------------------------------ knob off
def _script(seed):
    """A scripted mix: decoders, a long prefill, then arrivals of random size at random steps."""
    rnd = random.Random(seed)
    arrivals = {}
    for i in range(rnd.randint(3, 9)):
        cached = rnd.choice((0, 17, 30, 60)) * BS
        unc = rnd.choice((300, 1000, 1600, 2600, 3000, 4500, 9000, 12000))
        arrivals.setdefault(rnd.randint(0, 120), []).append((f"a{i}", cached + unc, cached))
    return rnd.randint(0, 4), rnd.choice((0, 20000, 64000)), arrivals


def _trace(pol_cls, fake, seed, kw, steps=700):
    ndec, long_n, arrivals = _script(seed)
    pol = pol_cls(*kw["args"], **kw["kwargs"])
    running = [Req(f"d{i}", 30000) for i in range(ndec)] + ([Req("L", long_n, computed=0)] if long_n else [])
    s = _mk(pol, fake, running, thr=kw.get("thr", 0))
    tr = []
    for step in range(steps):
        for rid, n, cached in arrivals.get(step, ()):
            s.add(Req(rid, n, cached=cached))
        out = s.schedule()
        tr.append((tuple(sorted(out.num_scheduled_tokens.items())), tuple(n.req_id for n in out.scheduled_new_reqs),
                   pol.debt, tuple(r.request_id for r in s.waiting)))
    stats = {k: v for k, v in pol.stats.items() if not k.endswith("_sum_s")}  # wall-clock sums differ run to run
    return tr, stats


def test_off_is_identical_to_s13():
    new = _policy_cls(False)
    # the knob needs a block size: without SHARE_CHUNK it stays off, and the default constructor is off
    assert new(12, 0, 0.0, budget_share=1).budget_share == 0 and new(12, 864, 0.0).budget_share == 0
    if not BASE_COMMIT:
        print("knob off: constructor default off: OK (comparison with a pre-budget revision SKIPPED: "
              "SPARSE_TEST_BASE_REV not set)")
        return
    base = _policy_cls(True)
    variants = [dict(args=(12, 864, 0.0), kwargs={}),
                dict(args=(12, 864, 0.0), kwargs=dict(short_bypass=2600)),
                dict(args=(3, 864, 0.0), kwargs=dict(short_bypass=2600, short_max_debt=40)),
                dict(args=(12, 864, 0.0), kwargs=dict(short_bypass=1728, short_scan=False), thr=864),
                dict(args=(12, 864, 50000.0), kwargs=dict(short_bypass=2600, deep_chunk_depth=8000, log_admit=True)),
                dict(args=(12, 0, 0.0), kwargs=dict(short_bypass=2600))]
    n = 0
    for fake in (S9.FakeSched, AlignSched):
        for kw in variants:
            for seed in range(40):
                a = _trace(base, fake, seed, kw)
                b = _trace(new, fake, seed, kw)
                assert a == b, (fake.__name__, kw, seed)
                n += 1
    assert set(new(12, 864, 0.0).stats) == set(base(12, 864, 0.0).stats)
    print(f"knob off: {n} scripted runs x 700 steps, decisions / new requests / debt / waiting order / stats identical "
          f"to the base revision ({BASE_COMMIT}) on both fakes: OK")


# ------------------------------------------------------------------------------------------------- knob on
def _on(mode=1, age=0, ndec=4, running=(), k=12, short=2600, **kw):
    pol = _policy_cls()(k, 864, 0.0, short_bypass=short, budget_share=mode, budget_age=age,
                        deep_chunk_depth=kw.pop("deep", 0))
    s = _mk(pol, AlignSched, [Req(f"d{i}", 30000) for i in range(ndec)] + list(running), **kw)
    return pol, s


def _pf(out):
    return {rid: n for rid, n in out.num_scheduled_tokens.items() if not rid.startswith("d")}


def test_shortest_first_and_block_fill():
    C = 30 * BS
    # three prefills running, admitted long-first: the step goes to the one with the least left, with both blocks
    A, B, S = Req("A", C + 12000, computed=C), Req("B", C + 6000, computed=C), Req("S", C + 3000, computed=C)
    pol, s = _on(running=[A, B, S])
    assert _pf(s.schedule()) == {"S": 1728} and s.calls[-1] is True
    assert pol.debt == -(-1728 * 12 // 1024) and pol.stats["prefill_steps"] == 1 and pol.stats["yield_steps"] == 0
    for _ in range(pol.debt):
        assert _pf(s.schedule()) == {}
    # S has 1272 left, more than a block: it keeps the step to itself (vLLM stops it at the last block boundary
    # here; with an internal prefill checkpoint it would run through). The debt is per token, so nothing is lost.
    assert _pf(s.schedule()) == {"S": 864} and pol.debt == -(-864 * 12 // 1024)
    for _ in range(pol.debt):
        s.schedule()
    assert _pf(s.schedule()) == {"S": 408, "B": 864}  # final tail + a block for B
    assert not S.is_prefill_chunk
    # s13 on the same state: A (first in the running list) takes both blocks, B and S get nothing
    A, B, S = Req("A", C + 12000, computed=C), Req("B", C + 6000, computed=C), Req("S", C + 3000, computed=C)
    pol0, s0 = _on(mode=0, running=[A, B, S])
    assert _pf(s0.schedule()) == {"A": 1728} and s0.calls[-1] is False
    for _ in range(pol0.debt):
        s0.schedule()
    assert _pf(s0.schedule()) == {"A": 1728}
    print("on: shortest-remaining prefill takes the step (2 blocks), leftover block to the next; s13: first in the "
          "running list takes both blocks until it ends: OK")


def test_waiting_sizes_learned_and_ranked():
    C = 30 * BS
    A = Req("A", C + 12000, computed=C)
    pol, s = _on(running=[A], short=0)  # short bypass off: the recorder must still learn the sizes
    assert _pf(s.schedule()) == {"A": 1728} and s.calls[-1] is False  # one prefill: not touched
    s.add(Req("W1", C + 9000, cached=C))
    s.add(Req("W2", C + 3000, cached=C))
    assert _pf(s.schedule()) == {} and pol.stats["yield_steps"] == 1  # yield step: sizes noted, nobody let through
    assert s.requests["W1"]._sparse_unc_est == 9000 and s.requests["W2"]._sparse_unc_est == 3000
    assert [r.request_id for r in s.waiting] == ["W1", "W2"]
    while pol.debt:
        s.schedule()
    out = s.schedule()  # W2 is admitted past W1 and past the running A
    assert _pf(out) == {"W2": 1728} and [n.req_id for n in out.scheduled_new_reqs] == ["W2"]
    assert [r.request_id for r in s.waiting] == ["W1"]
    # an arrival of unknown size on the shared step itself ranks by its full length (last) and is noted for the next
    while pol.debt:
        s.schedule()
    s.add(Req("W3", C + 500, cached=C))
    out = s.schedule()
    assert _pf(out) == {"W2": 864, "W3": 500}, _pf(out)  # W3 fits the leftover once vLLM shows its size
    print("on: waiting sizes are learned on yield steps (also with the short bypass off); a short waiting follow-up "
          "is admitted past longer waiting and running prefills: OK")


def test_mode2_reserves_a_block_for_the_oldest():
    C = 30 * BS
    A, B, S = Req("A", C + 12000, computed=C), Req("B", C + 6000, computed=C), Req("S", C + 3000, computed=C)
    pol, s = _on(mode=2, running=[A, B, S])
    assert _pf(s.schedule()) == {"A": 864, "S": 864}
    print("on, mode 2: one block for the oldest running prefill, one for the shortest: OK")


def test_age_credit_bounds_starvation():
    C = 30 * BS

    def run(age):
        A = Req("A", C + 12 * BS, computed=C)
        pol, s = _on(age=age, running=[A])
        i, first = 0, None
        for step in range(4000):
            if not any(r.is_prefill_chunk for r in s.running if r is not A) and not s.waiting:
                s.add(Req(f"s{i}", C + 4 * BS, cached=C))  # an endless stream of shorter follow-ups
                i += 1
            out = s.schedule()
            if first is None and step > 0 and "A" in out.num_scheduled_tokens:
                first = pol.stats["prefill_steps"]
            if not A.is_prefill_chunk:
                return first, pol.stats["prefill_steps"]
        return first, None
    f0, d0 = run(0)
    f1, d1 = run(256)
    assert d0 is None, d0  # pure shortest-first: the long prefill never ends under a steady stream
    assert d1 is not None and d1 <= 60, (f1, d1)
    print(f"on: age credit 256/step lets a 10K prefill finish after {d1} prefill steps under an endless stream of 3.5K "
          f"follow-ups (credit 0: never): OK")


def test_missed_pick_does_not_hold_the_step():
    C = 30 * BS
    A = Req("A", C + 12000, computed=C)
    pol, s = _on(running=[A], max_active=5)  # 4 decodes + A: no slot for a new request
    s.schedule()
    s.add(Req("W", C + 3000, cached=C))
    s.requests["W"]._sparse_unc_est = 3000  # known shorter than A, but it cannot be admitted
    got = []
    for _ in range(200):
        got.append(_pf(s.schedule()).get("A", 0))
    assert pol.stats.get("budget_missed", 0) >= 1
    assert sum(1 for g in got if g) >= 4, got  # A keeps prefilling
    lost = pol.stats["budget_missed"]
    assert lost <= 6, lost  # 1, 2, ... 8 steps of hold: the misses thin out
    print(f"on: a picked request that cannot be admitted is held out of the next picks ({lost} empty steps in 200), "
          f"the running prefill goes on: OK")


def test_no_decodes_and_single_prefill_untouched():
    C = 30 * BS
    A, B = Req("A", C + 12000, computed=C), Req("B", C + 3000, computed=C)
    pol, s = _on(ndec=0, running=[A, B])
    out = s.schedule()
    assert s.calls[-1] is False and _pf(out)["A"] == 1728 and pol.stats.get("budget_steps", 0) == 0
    pol, s = _on(running=[Req("A", C + 12000, computed=C)])
    s.schedule()
    assert s.calls[-1] is False and pol.stats.get("budget_steps", 0) == 0
    print("on: steps without decodes, or with one prefill pending, are scheduled as before: OK")


def test_deep_chunk_cap_kept():
    C = 30 * BS
    A, S = Req("A", C + 12000, computed=C), Req("S", C + 3000, computed=C)
    pol, s = _on(running=[A, S], deep=20000)
    assert _pf(s.schedule()) == {"S": 864, "A": 864}  # S is past the depth: one block, the other block is not lost
    print("on: SPARSE_PREFILL_DEEP_CHUNK_DEPTH still caps a deep prefill at one block per step: OK")


# ------------------------------------------------------------------------------------- the scripted follow-up mix
def run_mix(mode, age=256, k=12, tail_split=True, sizes=(12000, 3000, 3000, 3000), ndec=4, cached=30 * BS):
    """4 streams decode; one 12K-uncached and three 3K-uncached follow-ups arrive together at step 2 (the 12K one
    first). Returns per-request (admission step, first-token step), and the step counts until all four are done."""
    pol = _policy_cls()(k, 864, 0.0, short_bypass=2600, budget_share=mode, budget_age=age)
    s = _mk(pol, AlignSched, [Req(f"d{i}", 30000) for i in range(ndec)], tail_split=tail_split)
    names = ["L"] + [f"F{i}" for i in range(1, len(sizes))]
    adm, ftok, dec_steps, ptok = {}, {}, {f"d{i}": 0 for i in range(ndec)}, 0
    for step in range(5000):
        if step == 2:
            for nm, u in zip(names, sizes):
                s.add(Req(nm, cached + u, cached=cached))
        out = s.schedule()
        for rid, n in out.num_scheduled_tokens.items():
            if rid in dec_steps:
                dec_steps[rid] += 1
            elif rid not in ftok:
                ptok += n
                adm.setdefault(rid, step - 2)
                if not s.requests[rid].is_prefill_chunk:
                    ftok[rid] = step - 2  # the step of its last prefill chunk samples its first token
        if len(ftok) == len(names):
            return dict(adm=adm, ftok=ftok, steps=step + 1, prefill_steps=pol.stats["prefill_steps"],
                        yield_steps=pol.stats["yield_steps"], dec_steps=dec_steps, prefill_tokens=ptok,
                        debt_left=pol.debt, stats=dict(pol.stats), names=names)
    raise AssertionError("mix did not finish")


def test_mix_off_vs_on():
    off, on, on2 = run_mix(0), run_mix(1), run_mix(2)
    print("mix: 4 decoders; L = 12000 and F1..F3 = 3000 uncached tokens arrive together (L first), K = 12")
    print("  %-22s %-34s %-34s %s" % ("", "admission step (L F1 F2 F3)", "first-token step (L F1 F2 F3)",
                                      "prefill / yield steps, decode steps per decoder"))
    for name, r in (("off (s13)", off), ("on, mode 1", on), ("on, mode 2", on2)):
        print("  %-22s %-34s %-34s %d / %d, %s" % (
            name, " ".join(str(r["adm"][n]) for n in r["names"]), " ".join(str(r["ftok"][n]) for n in r["names"]),
            r["prefill_steps"], r["yield_steps"], sorted(set(r["dec_steps"].values()))))
    for r in (off, on, on2):
        assert r["prefill_tokens"] == 21000  # every uncached token prefilled exactly once
        assert set(r["dec_steps"].values()) == {r["steps"]}  # every decoder decodes in every step
    # the duty cycle is per prefill token: the yield steps differ only by the per-step rounding of the debt
    for r in (on, on2):
        assert abs((r["yield_steps"] + r["debt_left"]) - (off["yield_steps"] + off["debt_left"])) <= \
            max(r["prefill_steps"], off["prefill_steps"]), (r, off)
        assert abs(r["steps"] - off["steps"]) <= 0.05 * off["steps"], (r["steps"], off["steps"])
    f_off = sorted(off["ftok"][n] for n in ("F1", "F2", "F3"))
    f_on = sorted(on["ftok"][n] for n in ("F1", "F2", "F3"))
    assert f_on[0] < 0.5 * f_off[0] and f_on[1] < 0.6 * f_off[1] and f_on[2] <= f_off[2], (f_on, f_off)
    assert on["ftok"]["L"] <= off["ftok"]["L"] * 1.05 + 30, (on["ftok"]["L"], off["ftok"]["L"])
    assert on["stats"]["budget_steps"] >= 4 and off["stats"].get("budget_steps", 0) == 0
    # the same with the last chunk allowed to run through (vLLM's internal prefill checkpoint) and at K = 3
    for kw in (dict(tail_split=False), dict(k=3)):
        a, b = run_mix(0, **kw), run_mix(1, **kw)
        assert a["prefill_tokens"] == b["prefill_tokens"] == 21000
        assert sum(b["ftok"][n] for n in ("F1", "F2", "F3")) < sum(a["ftok"][n] for n in ("F1", "F2", "F3")), kw
        assert abs(a["steps"] - b["steps"]) <= 0.05 * a["steps"] + 4, kw
    print("mix: the 3K follow-ups get their first token earlier, total steps / yield steps / decode steps of the "
          "decoders unchanged within the debt rounding: OK")


def test_install_knob():
    from sparsekv import tier_connector as tc

    class NoHook:
        def schedule(self, throttle_prefills=False):
            return None

    class Hook:
        def schedule(self, throttle_prefills=False):
            return getattr(self, "_sparse_yield_bypass", None)
    for k in ("SPARSE_PREFILL_BUDGET_SHARE", "SPARSE_PREFILL_BUDGET_AGE"):
        os.environ.pop(k, None)
    assert tc.install_prefill_yield(type("H0", (Hook,), {})).budget_share == 0  # default: off
    os.environ["SPARSE_PREFILL_BUDGET_SHARE"] = "1"
    try:
        assert tc.install_prefill_yield(type("N1", (NoHook,), {})).budget_share == 0  # no hook in vLLM: refused
        p = tc.install_prefill_yield(type("H1", (Hook,), {}))
        assert p.budget_share == 1 and p.budget_age == 256
        os.environ["SPARSE_PREFILL_SHARE_CHUNK"] = "0"
        assert tc.install_prefill_yield(type("H2", (Hook,), {})).budget_share == 0  # no block size: refused
    finally:
        os.environ.pop("SPARSE_PREFILL_BUDGET_SHARE", None)
        os.environ.pop("SPARSE_PREFILL_SHARE_CHUNK", None)
    print("install: SPARSE_PREFILL_BUDGET_SHARE off by default; needs the vLLM hook and SHARE_CHUNK: OK")


if __name__ == "__main__":
    import sparsekv.tier_connector as _tc
    assert os.path.realpath(_tc.__file__).startswith(os.path.realpath(ROOT) + os.sep), _tc.__file__
    print("testing", _tc.__file__)
    test_off_is_identical_to_s13()
    test_shortest_first_and_block_fill()
    test_waiting_sizes_learned_and_ranked()
    test_mode2_reserves_a_block_for_the_oldest()
    test_age_credit_bounds_starvation()
    test_missed_pick_does_not_hold_the_step()
    test_no_decodes_and_single_prefill_untouched()
    test_deep_chunk_cap_kept()
    test_mix_off_vs_on()
    test_install_knob()
    print("ALL OK")
