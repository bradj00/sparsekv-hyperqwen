# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""CPU simulation of the s15 prefill-budget sharing knob on the measured 8-agent workload shape. No GPU.

PYTHONPATH=<tree>/pthfa/python:<tree> CUDA_VISIBLE_DEVICES= venv/bin/python sparsekv/sim_s15_budget.py

It drives the real PrefillYield policy (sparsekv/tier_connector.py) on the scripted scheduler of
test_s15_budget_cpu.py (a model of vLLM's token hand-out in mamba align mode) and attaches a clock to every step.

Assumptions (all of them are flags; the result is only meant to say whether a GPU run is worth it):
  - 8 agents in a closed loop: follow-up -> prefill -> decode -> think (tool call) -> next follow-up.
  - uncached tokens per follow-up: log-linear through p0/p10/p50/p90/p100 = 1000/2000/4000/11000/14000
    (about a quarter are <= 2600 and take the short bypass).
  - cached prefix 15-80K tokens, block aligned, always a hit (growing by each turn, restarting low past 80K).
  - decode length: log-normal, median 300 tokens, sigma 0.6, at --tok-per-step accepted tokens per decode step.
  - step time: 110 ms for a step with decodes (flat in the number of streams), plus for every prefill chunk
    tokens x (0.53 ms + 8.9e-6 ms x depth): the fitted cold cost of the class comment (1728 tokens: ~0.9 s at depth
    0, ~2.3 s at 90K).
  - think time between turns: exponential, mean --think seconds.
  - budget 2048 tokens per step, 864-token blocks, K = 12, short bypass 2600, max debt 240, SHARE_CHUNK 864.
  - not modelled: MTP draft slots, KV/GDN slot pressure, preemption, cache misses, async scheduling's one-step lag,
    image inputs, a second server.
TTFT = arrival to the end of the step that runs the request's last prefill chunk (that step samples its first token).
"""
import argparse
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_s15_budget_cpu as T15  # noqa: E402

BS = T15.BS
QUANT = ((0.0, 1000), (0.1, 2000), (0.5, 4000), (0.9, 11000), (1.0, 14000))


def uncached(rnd):
    q = rnd.random()
    for (q0, v0), (q1, v1) in zip(QUANT, QUANT[1:]):
        if q <= q1:
            return int(math.exp(math.log(v0) + (math.log(v1) - math.log(v0)) * (q - q0) / (q1 - q0)))
    return QUANT[-1][1]


def simulate(mode, seed, a):
    from sparsekv.tier_connector import PrefillYield
    rnd = random.Random(seed)
    pol = PrefillYield(a.k, BS, 0.0, short_bypass=a.short, short_scan=True, short_max_debt=240, budget_share=mode,
                       budget_age=a.age)
    S = type("S", (T15.AlignSched,), {})
    S.schedule = pol.wrap(T15.AlignSched.schedule)
    s = S([], tail_split=not a.run_through)
    ctx = [rnd.uniform(15000, 60000) for _ in range(a.agents)]
    arrivals = sorted((rnd.uniform(0, 5.0), i) for i in range(a.agents))
    live = {}  # request id -> dict(agent, arrival, unc, steps_left, t_first)
    done, now, nreq = [], 0.0, 0
    tsum = {"prefill": 0.0, "decode_only": 0.0}
    while now < a.minutes * 60:
        while arrivals and arrivals[0][0] <= now:
            t, ag = arrivals.pop(0)
            unc, cached = uncached(rnd), int(ctx[ag]) // BS * BS
            rid = f"r{nreq}"
            nreq += 1
            out_tok = max(20, int(rnd.lognormvariate(math.log(a.out_median), 0.6)))
            live[rid] = dict(agent=ag, arrival=t, unc=unc, out=out_tok, steps_left=math.ceil(out_tok / a.tok_per_step),
                             t_first=None, dec_steps=0)
            s.add(T15.Req(rid, cached + unc, cached=cached))
        if not s.running and not s.waiting:
            now = arrivals[0][0]
            continue
        depth = {r.request_id: r.num_computed_tokens for r in s.running}
        depth.update({r.request_id: r.cached for r in s.waiting})
        was_decode = {r.request_id for r in s.running if not r.is_prefill_chunk}
        out = s.schedule()
        nst = out.num_scheduled_tokens
        dt_pf = sum(n * (a.ms_tok + a.ms_attn * depth[rid]) for rid, n in nst.items() if rid not in was_decode) / 1e3
        dt = (a.dec_ms / 1e3 if any(rid in was_decode for rid in nst) else 0.005) + dt_pf
        now += dt
        tsum["prefill" if dt_pf else "decode_only"] += dt
        for rid in list(nst):
            L, r = live[rid], s.requests[rid]
            if rid in was_decode:
                L["steps_left"] -= 1
                L["dec_steps"] += 1
                if L["steps_left"] <= 0:
                    s.running.remove(r)
                    del s.requests[rid]
                    L["t_end"] = now
                    done.append(L)
                    del live[rid]
                    ctx[L["agent"]] += L["unc"] + L["out"]
                    if ctx[L["agent"]] > 80000:
                        ctx[L["agent"]] = rnd.uniform(15000, 30000)
                    arrivals.append((now + rnd.expovariate(1.0 / a.think), L["agent"]))
                    arrivals.sort()
            elif not r.is_prefill_chunk and L["t_first"] is None:
                L["t_first"] = now
    started = done + [L for L in live.values() if L["t_first"] is not None]
    # requests still waiting at the end count with the time they have waited so far (censored, a lower bound)
    pending = [L for L in live.values() if L["t_first"] is None]
    return dict(started=started, pending=pending, done=done, pol=pol, tsum=tsum, now=now)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def report(name, runs, a):
    ttft_l, ttft_s, rate, turns, ys, ps, pend, starve = [], [], [], 0, 0, 0, 0, []
    for r in runs:
        for L in r["started"]:
            (ttft_s if L["unc"] <= a.short else ttft_l).append(L["t_first"] - L["arrival"])
        for L in r["pending"]:
            pend += 1
            if L["unc"] > a.short:
                ttft_l.append(r["now"] - L["arrival"])
        for L in r["done"]:
            if L["dec_steps"] > 1:
                rate.append(L["dec_steps"] * a.tok_per_step / (L["t_end"] - L["t_first"]))
        turns += len(r["done"])
        ys += r["pol"].stats["yield_steps"]
        ps += r["pol"].stats["prefill_steps"]
        starve.append(r["tsum"]["prefill"] / (r["tsum"]["prefill"] + r["tsum"]["decode_only"]))
    n = len(runs)
    print("%-26s long TTFT p50 %5.1f s  p90 %5.1f s  max %5.1f s (n %4d) | short TTFT p50 %4.1f s  p90 %4.1f s | "
          "decode %4.1f tok/s per stream (p10 %4.1f) | turns %5.1f, prefill/yield steps %4d/%5d per run, "
          "prefill time share %.2f, unfinished %d" % (
              name, pct(ttft_l, 50), pct(ttft_l, 90), max(ttft_l), len(ttft_l), pct(ttft_s, 50), pct(ttft_s, 90),
              statistics.mean(rate), pct(rate, 10), turns / n, ps / n, ys / n, statistics.mean(starve), pend))
    return dict(p50=pct(ttft_l, 50), p90=pct(ttft_l, 90), rate=statistics.mean(rate), turns=turns / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--k", type=float, default=12.0)
    ap.add_argument("--short", type=int, default=2600)
    ap.add_argument("--age", type=int, default=256)
    ap.add_argument("--think", type=float, default=3.0)
    ap.add_argument("--out-median", type=float, default=300.0)
    ap.add_argument("--tok-per-step", type=float, default=3.0)
    ap.add_argument("--dec-ms", type=float, default=110.0)
    ap.add_argument("--ms-tok", type=float, default=0.53)
    ap.add_argument("--ms-attn", type=float, default=8.9e-6)
    ap.add_argument("--run-through", action="store_true", help="the last chunk may pass the last block boundary")
    ap.add_argument("--modes", default="0,1,2")
    a = ap.parse_args()
    print("args:", vars(a))
    res = {}
    for mode in (int(m) for m in a.modes.split(",")):
        runs = [simulate(mode, seed, a) for seed in range(a.seeds)]
        res[mode] = report({0: "off (s13)", 1: "on, mode 1 (shortest)", 2: "on, mode 2 (oldest+short)"}[mode], runs, a)
    return res


if __name__ == "__main__":
    main()
