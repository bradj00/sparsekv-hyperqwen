# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Kernel breakdown of one decode select+gather (nreq=4, L=246K) with torch.profiler."""
import os, sys
import torch
sys.path.insert(0, os.environ.get("REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sparsekv import triton_ops as T
from torch.profiler import profile, ProfilerActivity
dev = "cuda"; Hq, Hkv, D, BS, R, QL = 24, 4, 256, 864, 64, 8
nreq, L = int(os.environ.get("NREQ", 4)), int(os.environ.get("L", 246000))
basis = torch.stack([torch.linalg.qr(torch.randn(D, R, device=dev))[0].T for _ in range(Hkv)]).contiguous()
maxblk = L // BS + 2; NB = nreq * maxblk + 1; rows = NB * BS + 1
q8 = torch.randint(-127, 127, (rows, Hkv, R), dtype=torch.int8, device=dev)
sc = (torch.rand(rows, Hkv, device=dev) * 0.01 + 0.001).half()
bt = (1 + torch.arange(nreq * maxblk, device=dev, dtype=torch.int32)).view(nreq, maxblk).contiguous()
sl = torch.full((nreq,), L, dtype=torch.int32, device=dev)
qsl = torch.arange(0, (nreq + 1) * QL, QL, dtype=torch.int32, device=dev)
q = torch.randn(nreq * QL, Hq, D, device=dev, dtype=torch.bfloat16)
kv = torch.empty((NB, Hkv, BS, 520), dtype=torch.int8, device=dev)
C = 3136; nbs = 4
st = torch.zeros(nreq * nbs, Hkv, BS, 520, dtype=torch.int8, device=dev)
sbt = torch.arange(nreq * nbs, dtype=torch.int32, device=dev).view(nreq, nbs)
def step():
    idx, nv = T.select(q, qsl, sl, bt, q8, sc, basis, 2048, 64, 1024, 1.0 / 16, max_q_len=QL)
    T.gather_rows(kv, bt, idx, nv, st, sbt)
for _ in range(5): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(20): step()
    torch.cuda.synchronize()
tot = 0
rows_ = []
for e in p.key_averages():
    if e.device_time_total > 0:
        rows_.append((e.device_time_total / 20, e.key)); tot += e.device_time_total / 20
for t, k in sorted(rows_, reverse=True)[:12]:
    print(f"{t/1000:8.3f} ms  {100*t/tot:5.1f}%  {k[:70]}")
print(f"total {tot/1000:.3f} ms per layer-step (nreq={nreq}, L={L})")
