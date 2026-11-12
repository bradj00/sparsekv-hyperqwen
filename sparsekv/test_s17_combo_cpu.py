# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""s17 CPU tests (no GPU): the feature combinations no branch tested alone.

PYTHONPATH=<tree>/pthfa/python:<tree> CUDA_VISIBLE_DEVICES= venv/bin/python sparsekv/test_s17_combo_cpu.py

A. Scheduler class (sparsekv.sched) + prefill-budget sharing, with and without the deadline
   (SPARSE_PREFILL_BUDGET_MAX_WAIT): same decisions, base-call arguments, debt and stats as the Scheduler.schedule
   monkeypatch + budget sharing, on the scripted mixes of test_s14_sched_cpu and test_s15_budget_cpu. Both paths are
   built through install_prefill_yield from the environment, as the server does.
B. The deadline itself, through both paths: an overdue prefill outranks the shorter ones, oldest first.
C. The real vLLM classes: SparseScheduler / SparseAsyncScheduler carry the budget knobs and run a shared step.
D. All knobs on together (scheduler class, budget sharing + deadline, short bypass, deep chunk cap, ctx unit, global
   hot pool, shared residency plan): the scripted mixes run without error, every prompt token is prefilled exactly
   once, every decoder decodes in every step, and the tier runtime's residency tables stay equal to the unshared ones.

The schedulers are the scripted fakes of test_s9_cpu / test_s15_budget_cpu (models of vLLM's deferral points and
mamba block alignment), not the real scheduler; C drives the real classes' MRO over a stub base schedule.
"""
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_fixes_cpu as T  # noqa: E402,F401  (sets sys.path / SPARSEKV_ROOT)
import test_s9_cpu as S9  # noqa: E402
import test_s14_sched_cpu as S14  # noqa: E402
import test_s15_budget_cpu as S15  # noqa: E402

ROOT = os.path.dirname(HERE)
BS = S15.BS
Req = S9.Req
T0 = time.time()
OVERDUE, FRESH = T0 - 7200.0, T0 + 1.0e6  # with MAX_WAIT=3600: waited two hours / has not arrived yet
TIER_KNOBS = ("SPARSE_TIER_HOT_GLOBAL", "SPARSE_TIER_HOT_GLOBAL_ROWS", "SPARSE_TIER_SHARED_PLAN",
              "SPARSE_TIER_SHARED_PLAN_CHECK", "SPARSE_TIER_SPILL")


def _env(**kw):
    for k in [k for k in os.environ if k.startswith("SPARSE_PREFILL_")]:
        os.environ.pop(k)
    for k, v in kw.items():
        os.environ["SPARSE_PREFILL_" + k] = str(v)


def _tc():
    from sparsekv import tier_connector as tc
    tc._SCHED_CLS_ACTIVE = False
    return tc


def _rec(fake):
    """fake + a record of what the policy handed to the base schedule: throttle flag, chunk cap, which hook
    (bypass / recorder / budget), the budget plan and its cap."""
    class Rec(fake):
        def __init__(self, running, waiting=(), thr=0, **kw):
            super().__init__(running, waiting, thr, **kw)
            self.seen = []

        def schedule(self, throttle_prefills=False):
            byp = getattr(self, "_sparse_yield_bypass", None)
            self.seen.append((bool(throttle_prefills), self.scheduler_config.long_prefill_token_threshold,
                              getattr(byp, "__name__", None), tuple(sorted(getattr(byp, "planned", ()))),
                              getattr(byp, "cap", None), getattr(byp, "scan", None), self.prefill_capacity_bound))
            return super().schedule(throttle_prefills)
    return Rec


def _monkeypatch(fake):
    """install_prefill_yield wraps the scheduler class's schedule (production today)."""
    tc = _tc()
    Rec = _rec(fake)

    class M(Rec):
        schedule = Rec.schedule  # own attribute: the patch stays in this class

    pol = tc.install_prefill_yield(M)
    assert pol is not None and M.schedule._sparse_prefill_yield is pol
    return M, pol


def _subclass(fake):
    """SparseSchedulerMixin over an unpatched base; the connector builds the policy during __init__ (--scheduler-cls)."""
    tc = _tc()
    from sparsekv import sched as sp
    Rec = _rec(fake)

    class B(Rec):
        def __init__(self, running, waiting=(), thr=0, **kw):
            super().__init__(running, waiting, thr, **kw)
            self.connector = types.SimpleNamespace(prefill_yield=tc.install_prefill_yield(B))

    class Sub(sp.SparseSchedulerMixin, B):
        pass
    return Sub, B


def _build(path, fake, running, **kw):
    if path == "monkeypatch":
        M, pol = _monkeypatch(fake)
        return M(running, **kw), pol
    Sub, B = _subclass(fake)
    s = Sub(running, **kw)
    pol = s._sparse_prefill_yield
    assert pol is not None and pol is s.connector.prefill_yield
    assert getattr(B.schedule, "_sparse_prefill_yield", None) is None, "class mode must not monkeypatch"
    _tc()
    return s, pol


def _arr(i, deadline):
    """Arrival stamp of the i-th request of a script: with a deadline every third one is overdue (older = lower i)."""
    if not deadline:
        return None
    return OVERDUE - 10.0 * (50 - i) if i % 3 == 0 else FRESH


# ------------------------------------------------------------------------------------------------- the mixes
def mix_s15(s, pol, seed, deadline, steps=500):
    """test_s15_budget_cpu's scripted mix: decoders, a long prefill, arrivals of random size at random steps."""
    ndec, long_n, arrivals = S15._script(seed)
    for r in s.running:
        if r.request_id == "L":
            r.arrival_time = _arr(0, deadline)
    tr = []
    for step in range(steps):
        for rid, n, cached in arrivals.get(step, ()):
            r = Req(rid, n, cached=cached)
            r.arrival_time = _arr(int(rid[1:]) + 1, deadline)
            s.add(r)
        out = s.schedule()
        tr.append((tuple(sorted(out.num_scheduled_tokens.items())), tuple(n.req_id for n in out.scheduled_new_reqs),
                   pol.debt, tuple(r.request_id for r in s.waiting), s._sparse_yield_bypass is None))
    return tr


def _start_s15(seed):
    ndec, long_n, _ = S15._script(seed)
    return [Req(f"d{i}", 30000) for i in range(ndec)] + ([Req("L", long_n, computed=0)] if long_n else [])


def mix_s14(s, pol, explicit, deadline):
    """test_s14_sched_cpu's scripted workload (arrivals, all decoders finishing, explicit throttle calls)."""
    if deadline:
        add = s.add
        n = [0]

        def stamped(r):
            n[0] += 1
            r.arrival_time = _arr(n[0], deadline)
            add(r)
        s.add = stamped
        for r in s.running:
            if r.is_prefill_chunk:
                r.arrival_time = _arr(0, deadline)
    return S14._script(s, pol, explicit_throttle=explicit)


def mix_followups(s, pol, deadline, sizes=(12000, 3000, 3000, 3000), cached=30 * BS):
    """test_s15_budget_cpu.run_mix: 4 streams decode; one 12K and three 3K follow-ups arrive together at step 2."""
    names = ["L"] + [f"F{i}" for i in range(1, len(sizes))]
    dec = {r.request_id: 0 for r in s.running}
    ftok, got, tr = {}, {n: 0 for n in names}, []
    for step in range(5000):
        if step == 2:
            for i, (nm, u) in enumerate(zip(names, sizes)):
                r = Req(nm, cached + u, cached=cached)
                r.arrival_time = (OVERDUE if nm == "L" else FRESH) if deadline else None
                s.add(r)
        out = s.schedule()
        tr.append((tuple(sorted(out.num_scheduled_tokens.items())), pol.debt, tuple(r.request_id for r in s.waiting)))
        for rid, n in out.num_scheduled_tokens.items():
            if rid in dec:
                dec[rid] += 1
            elif rid not in ftok:
                got[rid] += n
                if not s.requests[rid].is_prefill_chunk:
                    ftok[rid] = step - 2
        if len(ftok) == len(names):
            return dict(tr=tr, ftok=ftok, got=got, dec=dec, steps=step + 1)
    raise AssertionError("mix did not finish")


# --------------------------------------------------------------------- A. scheduler class + budget == monkeypatch + budget
BUDGET_CONFIGS = (
    dict(YIELD_STEPS_PER_1K=12, BUDGET_SHARE=1),                                           # short bypass off
    dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600, BUDGET_SHARE=1),                  # K = 12 + mode 1  
    dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600, BUDGET_SHARE=2),                  # K = 12 + mode 2  
    dict(YIELD_STEPS_PER_1K=3, YIELD_SHORT_BYPASS=2600, BUDGET_SHARE=1, BUDGET_AGE=0),     # K = 3, no age credit 
    dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=1728, YIELD_SHORT_SCAN=0, BUDGET_SHARE=2, BUDGET_AGE=64),
    dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600, DEEP_CHUNK_DEPTH=8000, BUDGET_SHARE=1),
    dict(YIELD_STEPS_PER_1K=12, YIELD_CTX_UNIT=100000, YIELD_SHORT_BYPASS=2600, YIELD_LOG=1, BUDGET_SHARE=1),
    dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=1728, YIELD_SHORT_MAX_DEBT=10, BUDGET_SHARE=1),
)
SEEDS = range(24)


def _pair(cfg, fake, run, start, **kw):
    """Run one scripted mix through both paths with the same environment; assert equality; return (trace, stats)."""
    res = []
    for path in ("monkeypatch", "class"):
        _env(**cfg)
        s, pol = _build(path, fake, start(), **kw)
        want_share = cfg.get("BUDGET_SHARE", 0) if cfg.get("SHARE_CHUNK", 864) else 0
        assert pol.budget_share == want_share and pol.budget_max_wait == float(cfg.get("BUDGET_MAX_WAIT", 0)), cfg
        tr = run(s, pol)
        res.append((tr, pol.debt, S14._stats(pol), s.seen))
    (tr_m, debt_m, st_m, seen_m), (tr_s, debt_s, st_s, seen_s) = res
    if isinstance(tr_m, list):
        for i, (a, b) in enumerate(zip(tr_m, tr_s)):
            assert a == b, (cfg, fake.__name__, i, a, b)
    assert tr_m == tr_s, (cfg, fake.__name__)
    assert seen_m == seen_s, (cfg, fake.__name__)          # same throttle / chunk cap / hook / budget plan per step
    assert debt_m == debt_s and st_m == st_s, (cfg, st_m, st_s)
    return tr_m, st_m, seen_m


def test_class_equals_monkeypatch_with_budget(deadline):
    runs = steps = 0
    live = {"budget": 0, "overdue": 0}
    for base in BUDGET_CONFIGS:
        cfg = dict(base, BUDGET_MAX_WAIT=3600) if deadline else dict(base)
        tot = {"budget_steps": 0, "budget_overdue": 0, "budget_hook_steps": 0}

        def note(st, seen):
            tot["budget_steps"] += st.get("budget_steps", 0)
            tot["budget_overdue"] += st.get("budget_overdue", 0)
            tot["budget_hook_steps"] += sum(1 for x in seen if x[2] == "budget")
        for fake in (S9.FakeSched, S15.AlignSched):
            for seed in SEEDS:
                tr, st, seen = _pair(cfg, fake, lambda s, pol, seed=seed: mix_s15(s, pol, seed, deadline),
                                     lambda seed=seed: _start_s15(seed))
                note(st, seen)
                runs, steps = runs + 1, steps + len(tr)
            for explicit in ((), (5, 6, 50, 130)):
                tr, st, seen = _pair(cfg, fake, lambda s, pol, e=explicit: mix_s14(s, pol, e, deadline), S14._start)
                note(st, seen)
                runs, steps = runs + 1, steps + len(tr)
        for kw in (dict(), dict(tail_split=False)):
            r, st, seen = _pair(cfg, S15.AlignSched, lambda s, pol: mix_followups(s, pol, deadline),
                                lambda: [Req(f"d{i}", 30000) for i in range(4)], **kw)
            assert r["got"] == {"L": 12000, "F1": 3000, "F2": 3000, "F3": 3000}, r["got"]
            note(st, seen)
            runs, steps = runs + 1, steps + r["steps"]
        # the knob is live in these runs (through the scheduler class too: the two paths were just shown equal)
        assert tot["budget_steps"] > 0 and tot["budget_hook_steps"] == tot["budget_steps"], (cfg, tot)
        assert (tot["budget_overdue"] > 0) == bool(deadline), (cfg, tot)
        live["budget"] += tot["budget_steps"]
        live["overdue"] += tot["budget_overdue"]
    print(f"s17 scheduler class + budget sharing {'+ deadline ' if deadline else ''}== monkeypatch + budget sharing: "
          f"{len(BUDGET_CONFIGS)} knob sets, {runs} scripted runs, {steps} steps, identical decisions / base-call "
          f"arguments / debt / stats ({live['budget']} shared steps, {live['overdue']} overdue rankings): OK")


def test_knobs_change_decisions_through_the_class():
    """Guard against a vacuous A: through the scheduler class, budget sharing changes the schedule, and so does the
    deadline; with the budget knob off the class path ignores the deadline knob."""
    def run(cfg, deadline):
        _env(**cfg)
        s, pol = _build("class", S15.AlignSched, [Req(f"d{i}", 30000) for i in range(4)])
        return mix_followups(s, pol, deadline), pol
    base = dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600)
    off, p_off = run(base, False)
    on, p_on = run(dict(base, BUDGET_SHARE=1), False)
    dl, p_dl = run(dict(base, BUDGET_SHARE=1, BUDGET_MAX_WAIT=3600), True)
    off_dl, p_off_dl = run(dict(base, BUDGET_MAX_WAIT=3600), True)
    assert p_off.stats.get("budget_steps", 0) == 0 and "budget_overdue" not in p_off_dl.stats
    assert off_dl["tr"] == off["tr"], "MAX_WAIT without BUDGET_SHARE must change nothing"
    f = lambda r: sorted(r["ftok"][n] for n in ("F1", "F2", "F3"))  # noqa: E731
    assert f(on)[0] < 0.5 * f(off)[0], (f(on), f(off))                     # the 3K follow-ups start earlier
    assert dl["ftok"]["L"] < on["ftok"]["L"] and p_dl.stats["budget_overdue"] > 0, (dl["ftok"], on["ftok"])
    for r in (off, on, dl):
        assert set(r["dec"].values()) == {r["steps"]}                      # every decoder decodes in every step
        assert abs(r["steps"] - off["steps"]) <= 0.05 * off["steps"] + 4   # same duty cycle
    print("s17 scheduler class: budget sharing brings the 3K follow-ups' first token forward (%s -> %s steps); with the "
          "deadline the overdue 12K prefill ends at step %d instead of %d; MAX_WAIT alone changes nothing: OK"
          % (f(off), f(on), dl["ftok"]["L"], on["ftok"]["L"]))


# --------------------------------------------------------------------------------------------- B. the deadline
def test_deadline_ranks_overdue_first_on_both_paths():
    C = 30 * BS
    pf = lambda out: {rid: n for rid, n in out.num_scheduled_tokens.items() if not rid.startswith("d")}  # noqa: E731
    got = {}
    for path in ("monkeypatch", "class"):
        for name, wait, stamps in (("off", 0, (OVERDUE, FRESH, FRESH)),          # knob off: stamps ignored
                                   ("none overdue", 3600, (FRESH, FRESH, None)),
                                   ("A overdue", 3600, (OVERDUE, FRESH, FRESH)),
                                   ("A and B overdue, B older", 3600, (OVERDUE, OVERDUE - 60, FRESH))):
            _env(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600, BUDGET_SHARE=1, BUDGET_AGE=0, BUDGET_MAX_WAIT=wait)
            A, B, S = Req("A", C + 12000, computed=C), Req("B", C + 6000, computed=C), Req("S", C + 3000, computed=C)
            A.arrival_time, B.arrival_time, S.arrival_time = stamps
            s, pol = _build(path, S15.AlignSched, [Req(f"d{i}", 30000) for i in range(4)] + [A, B, S])
            out = pf(s.schedule())
            assert s.seen[-1][0] is True and s.seen[-1][2] == "budget"   # a shared step, picked by the hook
            got[path, name] = (out, pol.debt, pol.stats.get("budget_overdue", 0))
    for name in ("off", "none overdue", "A overdue", "A and B overdue, B older"):
        assert got["monkeypatch", name] == got["class", name], name
    g = lambda name: got["class", name]  # noqa: E731
    assert g("off")[0] == {"S": 1728} and g("off")[2] == 0               # shortest remaining first
    assert g("none overdue")[0] == {"S": 1728} and g("none overdue")[2] == 0
    assert g("A overdue")[0] == {"A": 1728} and g("A overdue")[2] == 1   # overdue beats shorter
    assert g("A and B overdue, B older")[0] == {"B": 1728} and g("A and B overdue, B older")[2] == 2  # oldest first
    assert all(v[1] == -(-1728 * 12 // 1024) for v in got.values())      # charged like any prefill step
    print("s17 deadline (SPARSE_PREFILL_BUDGET_MAX_WAIT): overdue prefill outranks the shorter ones, oldest first; "
          "no-op when nobody is overdue; same on the monkeypatch and the scheduler-class path: OK")


# ---------------------------------------------------------------------------------- C. the real vLLM classes
def test_real_vllm_classes_with_budget():
    from vllm.v1.core.sched.scheduler import Scheduler

    tc = _tc()
    from sparsekv import sched as sp
    assert tc._sched_has_bypass_hook(Scheduler), "installed vLLM lacks the sparse-prefill-yield-bypass hook"
    real_init, real_sched = Scheduler.__init__, Scheduler.schedule
    _env(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600, BUDGET_SHARE=1, BUDGET_AGE=128, BUDGET_MAX_WAIT=45)

    def fake_init(self, *a, **kw):
        self.connector = types.SimpleNamespace(prefill_yield=tc.install_prefill_yield())
        self.running, self.waiting, self.kv_holding_waiting, self.requests = [], [], [], {}
        self.prefill_capacity_bound = True
        self.scheduler_config = types.SimpleNamespace(long_prefill_token_threshold=0, max_num_batched_tokens=2048)
        self.max_num_scheduled_tokens = 2048
        self.num_spec_tokens = 7
        self.parallel_config = types.SimpleNamespace(pipeline_parallel_size=1)

    calls = []

    def fake_schedule(self, throttle_prefills=False):
        byp = getattr(self, "_sparse_yield_bypass", None)
        calls.append((bool(throttle_prefills), getattr(byp, "__name__", None), self.prefill_capacity_bound,
                      self.scheduler_config.long_prefill_token_threshold))
        nst = {r.request_id: 8 for r in self.running if not r.is_prefill_chunk}
        for r in self.running:
            if r.is_prefill_chunk and (not throttle_prefills or (byp is not None
                                                                 and byp(r, r.num_computed_tokens, True))):
                nst[r.request_id] = 864
        return types.SimpleNamespace(num_scheduled_tokens=nst, scheduled_new_reqs=[])

    def cfg(a):
        return types.SimpleNamespace(scheduler_config=types.SimpleNamespace(async_scheduling=a))
    Scheduler.__init__ = fake_init
    try:
        for async_, want in ((True, sp.SparseAsyncScheduler), (False, sp.SparseScheduler)):
            obj = sp.SparseScheduler(vllm_config=cfg(async_))
            assert type(obj) is want
            pol = obj._sparse_prefill_yield
            assert pol is not None and (pol.budget_share, pol.budget_age, pol.budget_max_wait) == (1, 128, 45.0)
            assert Scheduler.schedule is real_sched and getattr(real_sched, "_sparse_prefill_yield", None) is None
            Scheduler.schedule = fake_schedule
            try:
                del calls[:]
                R = types.SimpleNamespace
                C = 30 * BS
                obj.running = [R(request_id="d", is_prefill_chunk=False, num_computed_tokens=9, num_tokens=9),
                               R(request_id="A", is_prefill_chunk=True, num_computed_tokens=C, num_tokens=C + 12000,
                                 arrival_time=FRESH),
                               R(request_id="S", is_prefill_chunk=True, num_computed_tokens=C, num_tokens=C + 3000,
                                 arrival_time=FRESH)]
                obj.requests = {r.request_id: r for r in obj.running}
                out = obj.schedule()
                # two prefills pending while a request decodes: a throttled step whose hook picks the shorter one
                assert calls == [(True, "budget", False, 0)], calls
                assert out.num_scheduled_tokens == {"d": 8, "S": 864}, out.num_scheduled_tokens
                assert pol.stats["budget_steps"] == 1 and pol.debt == -(-864 * 12 // 1024)
                assert obj._sparse_yield_bypass is None
                obj.requests["A"].arrival_time = OVERDUE   # waited longer than MAX_WAIT: A goes first now
                while pol.debt:
                    obj.schedule()
                del calls[:]
                out = obj.schedule()
                assert calls[0][:2] == (True, "budget") and out.num_scheduled_tokens == {"d": 8, "A": 864}, (calls, out)
                assert pol.stats["budget_overdue"] == 1
            finally:
                Scheduler.schedule = real_sched
    finally:
        Scheduler.__init__ = real_init
        Scheduler.schedule = real_sched
        tc._SCHED_CLS_ACTIVE = False
    print("s17 real vLLM classes: SparseScheduler / SparseAsyncScheduler carry SPARSE_PREFILL_BUDGET_SHARE / _AGE / "
          "_MAX_WAIT, Scheduler.schedule left unpatched, a shared step runs through the real MRO: OK")


# ------------------------------------------------------------------------------------------ D. all knobs on
def test_all_knobs_on_together():
    import torch
    import test_s14_hot_cpu as HOT

    for mode in (1, 2):
        _env(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600, YIELD_CTX_UNIT=100000, YIELD_LOG=1, DEEP_CHUNK_DEPTH=8000,
             BUDGET_SHARE=mode, BUDGET_AGE=256, BUDGET_MAX_WAIT=3600)
        # the tier knobs are read when tier_connector is imported: reload it with them on (PREFILL knobs stay set)
        tc = HOT._tc(SPARSE_TIER_SPILL=1, SPARSE_TIER_HOT_GLOBAL=1, SPARSE_TIER_SHARED_PLAN=1)
        from sparsekv import sched as sp
        assert sp._tc is tc and tc.hot_global_enabled() and tc.shared_plan_enabled()
        calls = []

        def rec(fn):
            def f(kv, gbt, host_dev, hbt, idx, n_valid, st_kv, st_bt, hot, layer, slots, scratch, **kw):
                calls.append((fn, gbt.clone(), hbt.clone(), kw))
            return f
        tc.gather_rows_tiered = rec("slot")
        tc.gather_rows_tiered_global = rec("global")
        rt = HOT._rt(tc, True, True)          # shared plan + global pool, as the two knobs make it
        names = list(rt.layers)
        fwd = [0]

        def forward():
            """One decode slice of the tier runtime, all sparse layers, checked against the unshared tables."""
            bt, seq = HOT._batch(rt)
            rt.freed_upto[2] = fwd[0] % 9
            fwd[0] += 1
            ref = HOT._rt_unshared(tc, rt)
            del calls[:]
            for name, (l, g) in rt.layers.items():
                sk = rt.sketch_block_table(name, bt, seq)
                rt.gather(l, g, "kv", bt, seq, torch.zeros((4, HOT.C), dtype=torch.int32), "nv", "stkv", "stbt")
                assert torch.equal(sk, ref.sketch_block_table(name, bt, seq)), name
            assert {c[0] for c in calls} == {"global"} and calls[-1][3] == {"cached_slots": rt.max_reqs}
            assert all(torch.equal(c[1], calls[0][1]) and torch.equal(c[2], calls[0][2]) for c in calls)

        nruns = nsteps = shared = overdue = 0
        for fake in (S9.FakeSched, S15.AlignSched):
            for seed in SEEDS:
                s, pol = _build("class", fake, _start_s15(seed))
                assert (pol.budget_share, pol.budget_max_wait, pol.short_bypass, pol.deep_chunk_depth) == \
                    (mode, 3600.0, 2600, 8000)
                ndec, long_n, arrivals = S15._script(seed)
                want = {"L": long_n} if long_n else {}
                want.update({rid: n - cached for evs in arrivals.values() for rid, n, cached in evs})
                got = dict.fromkeys(want, 0)
                orig = s.schedule
                dec_steps = [0]

                def counted(*a, _orig=orig, _got=got, _s=s, _dec=dec_steps, **kw):
                    out = _orig(*a, **kw)
                    nst = out.num_scheduled_tokens
                    # (no bound on sum(nst): the fakes hand every decoder its 8 tokens without a budget check)
                    assert all(n > 0 for n in nst.values()), nst
                    pf_done = {rid for rid in _got if rid in _s.requests and not _s.requests[rid].is_prefill_chunk}
                    for rid, n in nst.items():
                        if rid in _got and _got[rid] < want[rid]:
                            _got[rid] += n
                    _dec[0] += all(r.request_id in nst for r in _s.running if not r.is_prefill_chunk)
                    if _dec[0] % 397 == 0:  # a tier decode slice between scheduler steps, a few per run
                        forward()
                    return out
                s.schedule = counted
                tr = mix_s15(s, pol, seed, True, steps=2400)  # the slowest of these mixes needs 1953 steps
                # decoders exist from step 0 only when ndec > 0; then every step must carry every decoder
                assert dec_steps[0] == len(tr), (seed, dec_steps[0], len(tr))
                assert not s.waiting and not any(r.is_prefill_chunk for r in s.running), (fake.__name__, seed)
                # every uncached prompt token prefilled exactly once (the fakes sample the first token with the last
                # chunk, so a finished request has had exactly its uncached tokens)
                assert got == want, (fake.__name__, seed, got, want)
                assert s._sparse_yield_bypass is None and pol.debt >= 0
                shared += pol.stats.get("budget_steps", 0)
                overdue += pol.stats.get("budget_overdue", 0)
                nruns, nsteps = nruns + 1, nsteps + len(tr)
            for explicit in ((), (5, 6, 50, 130)):
                s, pol = _build("class", fake, S14._start())
                tr = mix_s14(s, pol, explicit, True)
                assert s._sparse_yield_bypass is None and pol.stats["yield_steps"] > 0
                shared += pol.stats.get("budget_steps", 0)
                nruns, nsteps = nruns + 1, nsteps + len(tr)
        s, pol = _build("class", S15.AlignSched, [Req(f"d{i}", 30000) for i in range(4)])
        r = mix_followups(s, pol, True)
        assert r["got"] == {"L": 12000, "F1": 3000, "F2": 3000, "F3": 3000} and set(r["dec"].values()) == {r["steps"]}
        nruns, nsteps = nruns + 1, nsteps + r["steps"]
        forward()
        assert shared > 0 and overdue > 0 and fwd[0] > 1 and rt._plan_counts[0] >= fwd[0], (shared, overdue, fwd[0])
        assert rt._plan_counts[1] == 31 * rt._plan_counts[0], rt._plan_counts   # one leader, 31 followers per slice
        print(f"s17 all knobs on (scheduler class, budget mode {mode} + deadline, short bypass, deep chunk, ctx unit, "
              f"global hot pool, shared plan): {nruns} scripted runs, {nsteps} steps, {shared} shared steps, "
              f"{fwd[0]} tier slices with unshared-equal tables; all prefills complete, tokens exact: OK")
    # the hot-pool connector tests of test_s14_hot_cpu with the scheduler / budget environment still set
    HOT.test_shared_plan_same_tables_one_computation()
    HOT.test_connector_pool_lifecycle()
    for k in TIER_KNOBS:
        os.environ.pop(k, None)
    _env()
    tc = HOT._tc()
    assert not tc.hot_global_enabled() and not tc.shared_plan_enabled()
    assert tc.install_prefill_yield(type("Off", (S9.FakeSched,), {"schedule": S9.FakeSched.schedule}),
                                    monkeypatch=False).budget_share == 0
    print("s17 after the run: environment cleared, every knob reads off again: OK")


if __name__ == "__main__":
    import sparsekv.tier_connector as _tcm
    assert os.path.realpath(_tcm.__file__).startswith(os.path.realpath(ROOT) + os.sep), _tcm.__file__
    print("testing", _tcm.__file__)
    test_class_equals_monkeypatch_with_budget(deadline=False)
    test_class_equals_monkeypatch_with_budget(deadline=True)
    test_knobs_change_decisions_through_the_class()
    test_deadline_ranks_overdue_first_on_both_paths()
    test_real_vllm_classes_with_budget()
    test_all_knobs_on_together()
    print("ALL OK")
