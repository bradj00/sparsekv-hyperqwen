# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Per-layer decode selection cost (select + gather) vs context and batch, int8 vs int4 sketch, CUDA-graph replayed.
venv/bin/python run_s4/selbench.py   (one GPU)"""
import os, sys, time
import torch
sys.path.insert(0, os.environ.get("REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sparsekv import triton_ops as T

dev = "cuda"
if os.environ.get("HALF") == "1": T.CFG["half_exp"] = True
Hq, Hkv, D, BS, R, QL = 24, 4, 256, 864, 64, 8
torch.manual_seed(0)
basis = torch.stack([torch.linalg.qr(torch.randn(D, R, device=dev))[0].T for _ in range(Hkv)]).contiguous()


def bench(nreq, L, bits, parts=("select", "gather")):
    maxblk = L // BS + 2
    NB = nreq * maxblk + 1
    rows = NB * BS + 1
    q8 = torch.randint(-127, 127, (rows, Hkv, R if bits == 8 else R // 2), dtype=torch.int8, device=dev)
    sc = (torch.rand(rows, Hkv, device=dev) * 0.01 + 0.001).half()
    bt = (1 + torch.arange(nreq * maxblk, device=dev, dtype=torch.int32)).view(nreq, maxblk).contiguous()
    sl = torch.full((nreq,), L, dtype=torch.int32, device=dev)
    qsl = torch.arange(0, (nreq + 1) * QL, QL, dtype=torch.int32, device=dev)
    q = torch.randn(nreq * QL, Hq, D, device=dev, dtype=torch.bfloat16)
    C = 2048 + 64 + 1024
    nbs = -(-C // BS)
    kv = torch.empty((NB, Hkv, BS, 520), dtype=torch.int8, device=dev) if "gather" in parts else None
    st = torch.zeros(nreq * nbs, Hkv, BS, 520, dtype=torch.int8, device=dev)
    sbt = torch.arange(nreq * nbs, dtype=torch.int32, device=dev).view(nreq, nbs)

    def step():
        idx, nv = T.select(q, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=QL)
        if kv is not None:
            T.gather_rows(kv, bt, idx, nv, st, sbt)
    for _ in range(3):
        step()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    n = 50
    t = time.time()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    ms = (time.time() - t) / n * 1e3
    del q8, sc, kv, st
    torch.cuda.empty_cache()
    return ms


print(torch.cuda.get_device_name())
import itertools
for L in [int(x) for x in os.environ.get("LS", "122000,246000").split(",")]:
    for nreq in [int(x) for x in os.environ.get("NREQS", "1,2,4").split(",")]:
        r = {b: bench(nreq, L, b) for b in (8, 4)}
        print(f"L={L} nreq={nreq}: select+gather per layer-step int8 {r[8]:.3f} ms, int4 {r[4]:.3f} ms  "
              f"-> x16 layers: {16 * r[8]:.1f} / {16 * r[4]:.1f} ms per decode step ({16 * r[8] / nreq:.1f} / "
              f"{16 * r[4] / nreq:.1f} ms per request)", flush=True)
