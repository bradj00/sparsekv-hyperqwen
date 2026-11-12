# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""s14: the sparsekv scheduler policy as a vLLM scheduler class instead of a monkeypatch.

    vllm serve ... --scheduler-cls sparsekv.sched.SparseScheduler        (serve-sparse.sh: SPARSE_SCHEDULER_CLS=1)

schedule() applies the same PrefillYield policy object the monkeypatch path uses (tier_connector.PrefillYield.step:
one implementation, two ways to reach it). The policy is still built by the scheduler-side HostTierConnector from the
SPARSE_PREFILL_* environment and only when the tier spills, exactly as before; with this class in use
install_prefill_yield leaves Scheduler.schedule alone, so the policy is never applied twice.

SparseScheduler follows the server's async-scheduling setting: vLLM takes a custom class as given, and a plain
Scheduler subclass under --async-scheduling would run without AsyncScheduler's output placeholders. So constructing
SparseScheduler with async scheduling on returns a SparseAsyncScheduler (an AsyncScheduler subclass).

Still a vLLM patch, not this class: the short-request bypass hook inside Scheduler.schedule's two deferral points
(patches/sparse-prefill-yield-bypass.patch, Scheduler._sparse_yield_bypass).
"""
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from sparsekv import tier_connector as _tc


class SparseSchedulerMixin:
    """schedule() under the PrefillYield policy of this scheduler's HostTierConnector (none: plain vLLM schedule)."""

    _sparse_prefill_yield = None

    def __init__(self, *args, **kwargs):
        # before super().__init__: vLLM builds the KV connector there, and the connector builds the policy
        _tc._SCHED_CLS_ACTIVE = True
        super().__init__(*args, **kwargs)
        pol = getattr(getattr(self, "connector", None), "prefill_yield", None)
        patched = getattr(super().schedule, "_sparse_prefill_yield", None)
        if patched is not None:
            # the base schedule already carries the monkeypatch (installed earlier in this process): leave it to it
            _tc._log("sched: %s: Scheduler.schedule is already wrapped by install_prefill_yield; the scheduler class "
                     "adds nothing", type(self).__name__)
            pol = None
        self._sparse_prefill_yield = pol
        _tc._log("sched: %s in use (--scheduler-cls); prefill yield %s", type(self).__name__,
                 "applied by the scheduler class" if pol is not None
                 else ("applied by the monkeypatch" if patched is not None else "off (no spilling tier / knob 0)"))

    def schedule(self, throttle_prefills: bool = False):
        pol = self._sparse_prefill_yield
        if pol is None:
            return super().schedule(throttle_prefills)
        return pol.step(self, _base_schedule, throttle_prefills)


def _base_schedule(sched, throttle_prefills):
    return super(SparseSchedulerMixin, sched).schedule(throttle_prefills)


class SparseAsyncScheduler(SparseSchedulerMixin, AsyncScheduler):
    """--async-scheduling (production)."""


class SparseScheduler(SparseSchedulerMixin, Scheduler):
    """The --scheduler-cls entry point. Synchronous scheduling: this class; async scheduling: SparseAsyncScheduler."""

    def __new__(cls, *args, **kwargs):
        if cls is SparseScheduler:
            cfg = kwargs.get("vllm_config", args[0] if args else None)
            if getattr(getattr(cfg, "scheduler_config", None), "async_scheduling", False):
                # not a SparseScheduler instance, so Python does not run __init__ a second time
                return SparseAsyncScheduler(*args, **kwargs)
        return super().__new__(cls)
