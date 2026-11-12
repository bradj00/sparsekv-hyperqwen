# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""s14 CPU tests (no GPU): the prefill-yield policy as a vLLM scheduler class, and async scheduling.

PYTHONPATH=<tree>/pthfa/python:<tree> CUDA_VISIBLE_DEVICES= venv/bin/python sparsekv/test_s14_sched_cpu.py

A. sparsekv.sched.SparseScheduler (--scheduler-cls) takes the same schedule decisions, debt and stats as the
   Scheduler.schedule monkeypatch on scripted sequences; the policy is never applied twice; the class follows the
   server's async-scheduling setting.
B. Async scheduling (production runs --async-scheduling: step N+1 is scheduled before step N's output is processed).
   AsyncModelSched models vLLM 0.31's bookkeeping (scheduler.py:680-683 token count, :837-851 spec tokens,
   :1603-1630 _update_after_schedule, :2082-2089 rejection rollback, async_scheduler.py:25-50 and :59-78 placeholders).
   It is a model of those lines, not the real scheduler.
"""
import collections
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_fixes_cpu as T  # noqa: E402,F401  (sets sys.path / SPARSEKV_ROOT)
import test_s9_cpu as S9  # noqa: E402

KNOBS = ("SPARSE_PREFILL_YIELD_STEPS_PER_1K", "SPARSE_PREFILL_SHARE_CHUNK", "SPARSE_PREFILL_YIELD_CTX_UNIT",
         "SPARSE_PREFILL_YIELD_SHORT_BYPASS", "SPARSE_PREFILL_YIELD_SHORT_SCAN", "SPARSE_PREFILL_YIELD_SHORT_MAX_DEBT",
         "SPARSE_PREFILL_YIELD_LOG", "SPARSE_PREFILL_DEEP_CHUNK_DEPTH")


def _env(**kw):
    for k in KNOBS:
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ["SPARSE_PREFILL_" + k] = str(v)


def _tc():
    from sparsekv import tier_connector as tc
    tc._SCHED_CLS_ACTIVE = False
    return tc


class RecSched(S9.FakeSched):
    """FakeSched that records what the policy handed to the base schedule (throttle flag, chunk cap, bypass hook)."""

    def __init__(self, running, waiting=(), thr=0):
        super().__init__(running, waiting, thr)
        self.seen = []

    def schedule(self, throttle_prefills=False):
        self.seen.append((bool(throttle_prefills), self.scheduler_config.long_prefill_token_threshold,
                          self._sparse_yield_bypass is not None, self.prefill_capacity_bound))
        return super().schedule(throttle_prefills)


def _monkeypatch_sched():
    """The production path today: install_prefill_yield wraps the scheduler class's schedule."""
    tc = _tc()

    class M(RecSched):
        schedule = RecSched.schedule  # own attribute, so the patch does not leak into other tests

    pol = tc.install_prefill_yield(M)
    assert pol is not None and M.schedule._sparse_prefill_yield is pol
    return M, pol


def _subclass_sched():
    """The s14 path: SparseSchedulerMixin over an unpatched base; the connector builds the policy during __init__."""
    tc = _tc()
    from sparsekv import sched as sp

    class B(RecSched):
        def __init__(self, running, waiting=(), thr=0):
            super().__init__(running, waiting, thr)
            # HostTierConnector.__init__ (scheduler role, spill on) does exactly this
            self.connector = types.SimpleNamespace(prefill_yield=tc.install_prefill_yield(B))

    class Sub(sp.SparseSchedulerMixin, B):
        pass
    return Sub, B


def _script(s, pol, steps=420, explicit_throttle=()):
    """A scripted workload; returns the per-step trace."""
    trace = []
    for step in range(steps):
        if step == 3:
            s.add(S9.Req("S1", 31000, cached=30000))       # short follow-up after a prefix hit
        if step == 10:
            s.add(S9.Req("L2", 20000, cached=0))           # long request at the head of the queue
            s.add(S9.Req("S2", 31000, cached=30200))       # short one behind it
        if step == 14:
            s.add(S9.Req("S3", 31600, cached=30000))       # short, more than one capped chunk
        if step == 120:                                    # every decoder finishes
            for r in [r for r in s.running if not r.is_prefill_chunk]:
                s.running.remove(r)
                s.requests.pop(r.request_id, None)
        if step == 140:
            s.add(S9.Req("D9", 30008, cached=30000))       # a new decoder after 8 prompt tokens
        if step == 200:
            s.add(S9.Req("S4", 2500, cached=0))
        out = s.schedule(True) if step in explicit_throttle else s.schedule()
        trace.append((dict(out.num_scheduled_tokens), [n.req_id for n in out.scheduled_new_reqs], pol.debt,
                      [r.request_id for r in s.waiting], s.prefill_capacity_bound, s._sparse_yield_bypass is None))
    return trace


def _stats(pol):
    return {k: (list(v) if isinstance(v, list) else v) for k, v in pol.stats.items() if not k.endswith("_sum_s")}


CONFIGS = (
    dict(YIELD_STEPS_PER_1K=12),                                                   # s7 defaults
    dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600),                          # 12 steps per 1K  
    dict(YIELD_STEPS_PER_1K=3, YIELD_SHORT_BYPASS=2600),                           # 3 steps per 1K   
    dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=1728, YIELD_SHORT_SCAN=0),
    dict(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=1728, YIELD_SHORT_MAX_DEBT=10),
    dict(YIELD_STEPS_PER_1K=8, SHARE_CHUNK=0),
    dict(YIELD_STEPS_PER_1K=12, DEEP_CHUNK_DEPTH=8000),
    dict(YIELD_STEPS_PER_1K=12, YIELD_CTX_UNIT=100000, YIELD_SHORT_BYPASS=2600, YIELD_LOG=1),
)


def _start():
    return [S9.Req(f"d{i}", 30000) for i in range(3)] + [S9.Req("L", 64000, computed=0)]


def test_subclass_equals_monkeypatch():
    nsteps = 0
    for cfg in CONFIGS:
        for explicit in ((), (5, 6, 50, 130)):
            _env(**cfg)
            M, pol_m = _monkeypatch_sched()
            sm = M(_start())
            tr_m = _script(sm, pol_m, explicit_throttle=explicit)
            _env(**cfg)
            Sub, B = _subclass_sched()
            ss = Sub(_start())
            pol_s = ss._sparse_prefill_yield
            assert pol_s is not None and pol_s is ss.connector.prefill_yield and pol_s is not pol_m
            assert getattr(B.schedule, "_sparse_prefill_yield", None) is None, "class mode must not monkeypatch"
            tr_s = _script(ss, pol_s, explicit_throttle=explicit)
            for i, (a, b) in enumerate(zip(tr_m, tr_s)):
                assert a == b, (cfg, i, a, b)
            assert sm.seen == ss.seen, cfg                       # same throttle / chunk cap / bypass hook per step
            assert pol_m.debt == pol_s.debt and _stats(pol_m) == _stats(pol_s), (cfg, _stats(pol_m), _stats(pol_s))
            assert pol_m.stats["yield_steps"] > 0 and pol_m.stats["prefill_steps"] > 0
            if cfg.get("YIELD_SHORT_BYPASS") and cfg.get("YIELD_SHORT_MAX_DEBT") is None:
                assert pol_s.short_bypass == cfg["YIELD_SHORT_BYPASS"] and pol_s.stats["bypass_reqs"] > 0, pol_s.stats
            nsteps += len(tr_m)
    print(f"s14 scheduler class == monkeypatch: {len(CONFIGS)} knob sets x 2 scripts, {nsteps} steps, identical "
          "schedule outputs, base-call arguments, debt and stats: OK")


def test_never_applied_twice():
    # 1. class mode: install_prefill_yield builds the policy and leaves the class alone
    _env(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600)
    tc = _tc()

    class C(RecSched):
        schedule = RecSched.schedule
    orig = C.schedule
    tc._SCHED_CLS_ACTIVE = True
    pol = tc.install_prefill_yield(C)
    assert pol is not None and C.schedule is orig and pol.short_bypass == 2600
    tc._SCHED_CLS_ACTIVE = False
    assert tc.install_prefill_yield(C, monkeypatch=False) is not None and C.schedule is orig
    # 2. a base that already carries the monkeypatch: the scheduler class adds nothing (one application)
    from sparsekv import sched as sp
    _env(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600)
    M, pol_m = _monkeypatch_sched()

    class MB(M):
        def __init__(self, running, waiting=(), thr=0):
            super().__init__(running, waiting, thr)
            self.connector = types.SimpleNamespace(prefill_yield=pol_m)

    class Sub(sp.SparseSchedulerMixin, MB):
        pass
    s2 = Sub(_start())
    assert s2._sparse_prefill_yield is None
    tr2 = _script(s2, pol_m)
    _env(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600)
    M1, pol_1 = _monkeypatch_sched()
    s1 = M1(_start())
    tr1 = _script(s1, pol_1)
    assert tr1 == tr2 and _stats(pol_1) == _stats(pol_m) and s1.seen == s2.seen
    # 3. no policy (knob 0 / no spilling tier): plain schedule
    _env(YIELD_STEPS_PER_1K=0)
    Sub0, _ = _subclass_sched()
    s0 = Sub0(_start())
    assert s0._sparse_prefill_yield is None
    for _ in range(20):
        s0.schedule()
    assert all(t[0] is False and t[1] == 0 for t in s0.seen)
    tc._SCHED_CLS_ACTIVE = False
    print("s14 single application: class mode leaves Scheduler.schedule unpatched; over a patched base the class "
          "adds nothing; knob 0 = plain schedule: OK")


def test_real_vllm_classes():
    """The real vLLM 0.31 classes (no engine: __init__ is stubbed)."""
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    tc = _tc()
    from sparsekv import sched as sp
    assert resolve_obj_by_qualname("sparsekv.sched.SparseScheduler") is sp.SparseScheduler
    assert issubclass(sp.SparseScheduler, Scheduler) and not issubclass(sp.SparseScheduler, AsyncScheduler)
    assert issubclass(sp.SparseAsyncScheduler, AsyncScheduler)
    assert not issubclass(sp.SparseAsyncScheduler, sp.SparseScheduler)
    # AsyncScheduler does not override schedule(): the monkeypatch on Scheduler.schedule is what production's
    # AsyncScheduler runs today, and one mixin covers both classes
    assert "schedule" not in AsyncScheduler.__dict__
    assert sp.SparseAsyncScheduler._update_after_schedule is AsyncScheduler._update_after_schedule
    assert sp.SparseAsyncScheduler.schedule is sp.SparseSchedulerMixin.schedule
    assert tc._sched_has_bypass_hook(Scheduler), "installed vLLM lacks the sparse-prefill-yield-bypass hook"

    inits = []
    real_init, real_sched = Scheduler.__init__, Scheduler.schedule
    _env(YIELD_STEPS_PER_1K=12, YIELD_SHORT_BYPASS=2600)

    def fake_init(self, *a, **kw):
        inits.append(type(self).__name__)
        # vLLM builds the connector inside Scheduler.__init__; HostTierConnector then calls install_prefill_yield()
        self.connector = types.SimpleNamespace(prefill_yield=tc.install_prefill_yield())
        self.running, self.waiting, self.kv_holding_waiting, self.requests = [], [], [], {}
        self.prefill_capacity_bound = True
        self.scheduler_config = types.SimpleNamespace(long_prefill_token_threshold=0)
        # read by the real AsyncScheduler.__init__ (async_scheduler.py:18-19), which is NOT stubbed
        self.num_spec_tokens = 7
        self.parallel_config = types.SimpleNamespace(pipeline_parallel_size=1)

    calls = []

    def fake_schedule(self, throttle_prefills=False):
        calls.append((bool(throttle_prefills), self._sparse_yield_bypass is not None
                      if hasattr(self, "_sparse_yield_bypass") else False))
        nst = {r.request_id: 8 for r in self.running if not r.is_prefill_chunk}
        if not throttle_prefills:
            nst.update({r.request_id: 1728 for r in self.running if r.is_prefill_chunk})
        return types.SimpleNamespace(num_scheduled_tokens=nst, scheduled_new_reqs=[])

    def cfg(a):
        return types.SimpleNamespace(scheduler_config=types.SimpleNamespace(async_scheduling=a))
    Scheduler.__init__ = fake_init
    try:
        a = sp.SparseScheduler(vllm_config=cfg(True))
        assert type(a) is sp.SparseAsyncScheduler and inits == ["SparseAsyncScheduler"], (type(a), inits)
        s = sp.SparseScheduler(cfg(False))
        assert type(s) is sp.SparseScheduler and inits[1:] == ["SparseScheduler"], inits
        assert type(sp.SparseScheduler(vllm_config=cfg(None))) is sp.SparseScheduler
        assert type(sp.SparseAsyncScheduler(vllm_config=cfg(True))) is sp.SparseAsyncScheduler
        # the connector built the policy, with the bypass hook found on the real class, and nothing was patched
        assert Scheduler.schedule is real_sched and getattr(real_sched, "_sparse_prefill_yield", None) is None
        for obj in (a, s):
            pol = obj._sparse_prefill_yield
            assert pol is not None and pol.short_bypass == 2600 and pol.k == 12
            Scheduler.schedule = fake_schedule
            try:
                del calls[:]
                R = types.SimpleNamespace
                obj.running = [R(request_id="d", is_prefill_chunk=False, num_computed_tokens=9, num_tokens=9),
                               R(request_id="p", is_prefill_chunk=True, num_computed_tokens=0, num_tokens=90000)]
                obj.requests = {r.request_id: r for r in obj.running}
                for _ in range(1 + 21 + 1):
                    obj.schedule()
                # 1728 prefill tokens with a decode running -> ceil(1728 * 12 / 1024) = 21 decode-only steps
                assert [c[0] for c in calls] == [False] + [True] * 21 + [False], calls
                assert all(c[1] for c in calls[1:22]) and not calls[0][1] and not calls[22][1]
                assert pol.stats["yield_steps"] == 21 and pol.stats["prefill_steps"] == 2
                assert obj._sparse_yield_bypass is None and obj.prefill_capacity_bound is False
            finally:
                Scheduler.schedule = real_sched
    finally:
        Scheduler.__init__ = real_init
        Scheduler.schedule = real_sched
        tc._SCHED_CLS_ACTIVE = False
    print("s14 real vLLM classes: dotted path resolves; async setting picks SparseAsyncScheduler (one __init__); "
          "Scheduler.schedule left unpatched; policy drives the real MRO (1 prefill step : 21 decode-only): OK")


# ------------------------------------------------------------------------------------------- B. async scheduling
SPEC = 7


class AReq:
    def __init__(self, rid, prompt, cached=0, stop_after=None):
        self.request_id, self.num_prompt_tokens, self.num_tokens, self.cached = rid, prompt, prompt, cached
        self.num_computed_tokens = 0
        self.num_output_placeholders = 0
        self.nspec = 0                 # len(spec_token_ids)
        self.is_prefill_chunk = False  # as vLLM: an attribute, written only by _update_after_schedule
        self.stop_after, self.outputs, self.arrival_time = stop_after, 0, None


class AsyncModelSched:
    """vLLM 0.31 token bookkeeping with MTP-7. async_=True: placeholders (AsyncScheduler), and drive() processes
    step N's output only after step N+1 is scheduled (max_concurrent_batches = 2, config/vllm.py:591-600)."""
    BUDGET = 2048

    def __init__(self, async_, running=()):
        self.async_ = async_
        self.running, self.waiting, self.kv_holding_waiting = [], [], []
        self.requests = {}
        self.prefill_capacity_bound = False
        self.scheduler_config = types.SimpleNamespace(long_prefill_token_threshold=0)
        self._sparse_yield_bypass = None
        self.inflight = collections.deque()
        self.nsched = 0
        self.log = []      # per step: {rid: (prompt tokens, decode tokens)} by position, the ground truth
        self.pre = []      # per step: {rid: (is_prefill_chunk, prompt fully scheduled)} before scheduling
        self.left = []     # per step: token budget left over
        for r in running:  # already decoding: prompt computed, its first token sampled (sync) / in flight (async)
            r.num_computed_tokens = r.num_prompt_tokens
            r.nspec = SPEC
            if async_:
                r.num_output_placeholders = 1
            else:
                r.num_tokens += 1
                r.outputs = 1
            self.running.append(r)
            self.requests[r.request_id] = r

    def add(self, r):
        self.waiting.append(r)
        self.requests[r.request_id] = r

    def schedule(self, throttle_prefills=False):
        self.nsched += 1
        self.pre.append({r.request_id: (r.is_prefill_chunk, r.num_computed_tokens >= r.num_prompt_tokens)
                         for r in self.running})
        defer = (throttle_prefills and not self.prefill_capacity_bound
                 and any(not r.is_prefill_chunk for r in self.running))            # scheduler.py:607-609
        byp = self._sparse_yield_bypass
        thr = self.scheduler_config.long_prefill_token_threshold
        budget, nst, spec, new, truth = self.BUDGET, {}, {}, [], {}

        def place(r, n):
            c = r.num_computed_tokens
            p = max(0, min(c + n, r.num_prompt_tokens) - c)
            truth[r.request_id] = (p, n - p)
            nst[r.request_id] = n
        for r in self.running:
            if budget <= 0:
                break
            if defer and r.is_prefill_chunk and not (byp is not None and byp(r, r.num_computed_tokens, True)):
                continue                                                           # :657-667
            n = r.num_tokens + r.nspec + r.num_output_placeholders - r.num_computed_tokens   # :680-683
            if 0 < thr < n:
                n = thr
            n = min(n, budget)
            if n <= 0:
                continue
            ns = max(0, min(r.nspec, n + r.num_computed_tokens - r.num_tokens - r.num_output_placeholders))  # :837
            spec[r.request_id] = ns
            r.nspec = 0                                                            # :851
            place(r, n)
            budget -= n
        skipped = []
        while self.waiting and budget > 0:
            r = self.waiting[0]
            nc = r.cached
            if defer and nc < r.num_tokens - 1 and not (byp is not None and byp(r, nc, False)):   # :1075-1091
                if byp is not None and getattr(byp, "scan", False):
                    skipped.append(self.waiting.pop(0))
                    continue
                break
            self.waiting.pop(0)
            r.num_computed_tokens = nc
            n = r.num_tokens - nc
            if 0 < thr < n:
                n = thr
            n = min(n, budget)
            new.append(types.SimpleNamespace(req_id=r.request_id, num_computed_tokens=nc))
            self.running.append(r)
            spec[r.request_id] = 0
            place(r, n)
            budget -= n
        self.waiting = skipped + self.waiting
        gen = {}
        for rid, n in nst.items():                                                 # :1603-1630
            r = self.requests[rid]
            r.num_computed_tokens += n
            r.is_prefill_chunk = r.num_computed_tokens < r.num_tokens + r.num_output_placeholders
            gen[rid] = not r.is_prefill_chunk
            if self.async_ and not r.is_prefill_chunk:                             # async_scheduler.py:32-50
                r.num_output_placeholders += 1 + spec[rid]
                r.nspec = SPEC
        self.inflight.append((self.nsched, nst, spec, gen))
        self.log.append(truth)
        self.left.append(budget)
        return types.SimpleNamespace(num_scheduled_tokens=nst, scheduled_new_reqs=new)

    def process(self):
        """update_from_output for the oldest in-flight step."""
        step, nst, spec, gen = self.inflight.popleft()
        for rid in nst:
            r = self.requests.get(rid)
            if r is None or not gen[rid]:
                continue
            acc = min(spec[rid], (step * 5 + len(rid) * 3) % (SPEC + 1))           # deterministic acceptance
            rej = spec[rid] - acc
            r.num_computed_tokens -= rej                                           # scheduler.py:2085-2089
            if r.num_output_placeholders > 0:
                r.num_output_placeholders -= rej
            r.num_tokens += 1 + acc
            r.outputs += 1 + acc
            if self.async_:
                r.num_output_placeholders -= 1 + acc                               # async_scheduler.py:69-71
                assert r.num_output_placeholders >= 0
            else:
                r.nspec = SPEC                                                     # post_step update_draft_token_ids
            if r.stop_after is not None and r.outputs >= r.stop_after:
                self.running.remove(r)
                del self.requests[rid]

    def drive(self):
        out = self.schedule()
        while len(self.inflight) > (1 if self.async_ else 0):
            self.process()
        return out


def _amodel(async_, k=12, short=0, ndec=10, stop=None):
    from sparsekv.tier_connector import PrefillYield

    class S(AsyncModelSched):
        pass
    pol = PrefillYield(k, 864, 0.0, short_bypass=short)
    S.schedule = pol.wrap(AsyncModelSched.schedule)
    decs = [AReq(f"d{i}", 30000 + 7 * i, stop_after=stop) for i in range(ndec)]
    return pol, S(async_, decs)


def _aworkload(s, pol, steps=1200, llen=64123):
    debts = []
    for step in range(steps):
        if step == 2:
            s.add(AReq("L", llen))               # long, odd length: the last chunk leaves a few tokens of budget
        if step == 5:
            s.add(AReq("dz", 30000, cached=29990))   # decodes BEHIND L in running order: gets what L leaves
        if step == 9:
            s.add(AReq("S", 31007, cached=30000))
        if step == 300:
            s.add(AReq("L3", 20001))
        s.drive()
        debts.append(pol.debt)
    return debts


def test_async_decode_prefill_split_is_exact():
    """No stale classification: before every schedule() the policy's test (not is_prefill_chunk) equals 'prompt fully
    scheduled', and no token it counts as prefill is a placeholder or a decode token -- sync and async alike, with
    draft rejections and decode steps cut short by the token budget."""
    for async_ in (False, True):
        for short in (0, 2600):
            cut = dec_steps = 0
            # probe: what the step of L's last chunk gives to the requests scheduled behind L; then lengthen L so
            # that only 1, 4 or 7 tokens remain for the first decoder behind it (a decode step cut short by the
            # token budget: 8 tokens wanted)
            pol, s = _amodel(async_, short=short)
            _aworkload(s, pol)
            last = max(i for i, t in enumerate(s.log) if t.get("L", (0, 0))[0])
            order = list(s.log[last])
            behind = sum(sum(s.log[last][rid]) for rid in order[order.index("L") + 1:]) + s.left[last]
            assert behind > 8, behind
            for llen in [64123] + [64123 + behind - j for j in (1, 4, 7)]:
                pol, s = _amodel(async_, short=short)
                _aworkload(s, pol, llen=llen)
                for pre, truth in zip(s.pre, s.log):
                    for rid, (is_pf, prompt_done) in pre.items():
                        assert is_pf == (not prompt_done), (async_, rid, is_pf, prompt_done)
                    for rid, (p, d) in truth.items():
                        if rid in pre and not pre[rid][0]:
                            assert p == 0, (async_, rid, p, d)        # a decode: nothing of it is prompt
                            dec_steps += 1
                            cut += d < 1 + SPEC
                        else:
                            # counted as prefill by the policy: only prompt tokens (the first sampled position of
                            # a request is its last prompt token, so d == 0 always)
                            assert d == 0 and p > 0, (async_, rid, p, d)
                assert pol.stats["prefill_steps"] > 20 and pol.stats["yield_steps"] > 200
                if short:
                    assert pol.stats["bypass_reqs"] >= 1
                for r in s.running:
                    assert r.num_computed_tokens >= r.num_prompt_tokens, r.request_id   # every prefill finished
            assert cut > 0, "workload must include budget-truncated decode steps"
    print("s14 async: decode/prefill split equals ground truth at every step; placeholders and draft tokens are "
          "never counted as prefill (sync and async, bypass off/on): OK")


def test_async_same_duty_cycle_as_sync():
    """Debt is charged once per prefill step and paid one per decode-only step, the same under async scheduling."""
    for short in (0, 2600):
        res = []
        for async_ in (False, True):
            pol, s = _amodel(async_, short=short)
            debts = _aworkload(s, pol)
            prefill = [{rid: p for rid, (p, d) in t.items() if p} for t in s.log]
            res.append((debts, prefill, {k: v for k, v in pol.stats.items() if not k.endswith("_sum_s")}))
        assert res[0][0] == res[1][0], "debt trajectory differs between sync and async"
        assert res[0][1] == res[1][1], "prefill chunks differ between sync and async"
        assert res[0][2] == res[1][2], (res[0][2], res[1][2])
    print("s14 async: debt trajectory, prefill chunk sequence and stats identical to synchronous scheduling: OK")


def test_async_last_decoder_finish_is_bounded():
    """The one async effect: a decoder whose final token is in the in-flight step still counts as a decode for one
    schedule() call. With debt left that is one extra decode-only step, then the debt is dropped; never a double
    charge, never a stuck debt."""
    res = {}
    for async_ in (False, True):
        pol, s = _amodel(async_, ndec=1, stop=400)
        s.add(AReq("L", 64123))
        first_free = gone = None
        for step in range(400):
            s.drive()
            if gone is None and "d0" not in s.requests:
                gone = step
            if gone is not None and step > gone:
                assert pol.debt == 0, (async_, step, pol.debt)     # no decoders left: nothing owed
                assert s.log[step].get("L", (0, 0))[0] > 0 or s.requests["L"].num_computed_tokens >= 64123
        total = sum(t.get("L", (0, 0))[0] for t in s.log)
        assert total == 64123, total                                # every prompt token scheduled exactly once
        # steps after the decoder's last scheduled step in which L was still deferred
        last_dec = max(i for i, t in enumerate(s.log) if "d0" in t)
        res[async_] = (pol.stats["yield_steps"], last_dec, gone)
    ys, ya = res[False][0], res[True][0]
    assert 0 <= ya - ys <= 1, (ys, ya)
    print(f"s14 async: last decoder finishing costs at most one extra decode-only step (yield steps sync {ys}, "
          f"async {ya}); debt cleared, no double charge: OK")


def test_spill_never_reaches_in_flight_tokens():
    """Tier spill under async scheduling: vLLM calls remove_skipped_blocks on the PROCESSED basis
    (num_computed_tokens - num_in_flight_tokens: kv_cache_manager.py:549-555, scheduler.py:3017-3021), and the policy
    frees at most min(processed - tail, blocks whose host copy went out in an EARLIER step's meta)."""
    from sparsekv.tier_connector import HostTierSpillPolicy
    BS, TAIL, SINK = 16, 64, 1

    class Pool:
        def __init__(self):
            self.free, self.freed = 0, []

        def get_num_free_blocks(self):
            return self.free

        def free_blocks(self, blocks):
            self.freed.extend(blocks)
            self.free += len(blocks)

    class Mgr:
        block_size = BS

        def __init__(self):
            self.block_pool, self._null_block = Pool(), "null"
            self.req_to_blocks = {"r": list(range(100, 164))}
            self.base_calls = []

        def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
            self.base_calls.append(processed_computed_tokens)

        def pop_blocks_for_free(self, request_id):
            return self.req_to_blocks.pop(request_id, [])

    for policy in ("eager", "pressure"):
        conn = types.SimpleNamespace(safe={}, freed={}, freed_dirty=set())
        mgr = Mgr()
        pol = HostTierSpillPolicy(conn, TAIL, SINK, policy, watermark=8)
        pol.install(mgr)
        computed, in_flight = 0, 0
        for step in range(60):
            # async: this step's schedule sees the previous step still in flight (8 optimistic tokens)
            processed = max(0, computed - in_flight)
            before = conn.freed.get("r", 0)
            mgr.remove_skipped_blocks("r", processed, None)
            now = conn.freed.get("r", 0)
            assert now * BS <= max(SINK * BS, processed - TAIL), (step, now, processed)
            assert now <= max(conn.safe.get("r", 0), before), (step, now, conn.safe)
            # build_connector_meta at the end of this schedule: copies planned now are safe from the NEXT step on
            mirrored = max(0, (computed - 64) // BS)
            assert now <= mirrored
            conn.safe["r"] = mirrored
            in_flight = 8
            computed += 8 + 8 * (step % 3)
        assert conn.freed["r"] > SINK and mgr.block_pool.freed, policy
        assert all(b == "null" for b in mgr.req_to_blocks["r"][SINK:conn.freed["r"]])
        assert all(b != "null" for b in mgr.req_to_blocks["r"][conn.freed["r"]:])
        # finish / preemption drops the guard
        mgr.pop_blocks_for_free("r")
        assert "r" not in conn.safe
    print("s14 spill: frees stay below min(processed - tail, copies of earlier steps) with a step in flight "
          "(eager and pressure): OK")


if __name__ == "__main__":
    test_subclass_equals_monkeypatch()
    test_never_applied_twice()
    test_real_vllm_classes()
    test_async_decode_prefill_split_is_exact()
    test_async_same_duty_cycle_as_sync()
    test_async_last_decoder_finish_is_bounded()
    test_spill_never_reaches_in_flight_tokens()
    for k in KNOBS:
        os.environ.pop(k, None)
    print("ALL OK")
