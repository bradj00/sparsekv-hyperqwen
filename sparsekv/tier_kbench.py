# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""Kernel split of gather_rows_tiered (stream E): resolve vs copy time per layer, and copy-kernel TPP variants.

python sparsekv/tier_kbench.py [--ctx 131072] [--hot 0.10]
Uses the lab trace of layer 0 for all 16 layer slots (timing only).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsekv import tier as T  # noqa: E402
from sparsekv.tier_bench import BS, C, HKV, KSEL, L, SINK, W, WIN, trace_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--hot", type=float, default=0.10)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    dev = "cuda"
    N = a.ctx
    nblk = (N + BS - 1) // BS
    tr = np.load(trace_path(N, "lab", 0, 384))
    mirror = T.HostKVMirror(L, nblk, HKV, BS, W, device=dev)
    kv_full = [torch.zeros((nblk + 1, HKV, BS, W), dtype=torch.int8, device=dev) for _ in range(L)]
    gpu_bt = torch.full((1, nblk), -1, dtype=torch.int32, device=dev)
    gpu_bt[0, 0] = 1
    gpu_bt[0, -3:] = torch.arange(nblk - 2, nblk + 1, dtype=torch.int32)
    full_bt = torch.arange(1, nblk + 1, dtype=torch.int32, device=dev)[None]
    host_bt = torch.arange(nblk, dtype=torch.int32, device=dev)[None]
    NBS = (C + BS - 1) // BS
    sbt = torch.arange(NBS, dtype=torch.int32, device=dev)[None]
    stg = torch.zeros((NBS, HKV, BS, W), dtype=torch.int8, device=dev)
    sc = T.TierScratch(1, C, dev)
    slot = torch.zeros(1, dtype=torch.int32, device=dev)
    nv = torch.full((1,), C, dtype=torch.int32, device=dev)
    allidx = torch.from_numpy(np.stack([np.concatenate([tr[t], np.arange(SINK), np.arange(N - WIN, N)])
                                        for t in range(len(tr))]).astype(np.int32)).to(dev)
    H = max(2 * C, int(a.hot * N))
    out = {"gpu": torch.cuda.get_device_name(0), "ctx": N, "H": H}
    for tag, bt in (("tiered", gpu_bt), ("resident", full_bt)):
        for tpp in (1, 2, 4, 8):
            hot = T.HotBuffer(L, 1, H, HKV, W, max_ctx=nblk * BS, device=dev)
            idx = torch.empty((1, C), dtype=torch.int32, device=dev)
            hl = [hot.layer(l) for l in range(L)]

            def resolve(l):
                T._tier_resolve_kernel[(1,)](
                    idx, nv, bt, host_bt, slot, hot.step, hl[l].pos2row, hl[l].last, hl[l].hand,
                    sc.code, sc.src, sc.rank, sc.victim, hot.stats, C, bt.stride(0), H, hot.max_ctx,
                    min(2 * C, H), hot.age, -1, BS=BS, BLOCK=min(4096, triton.next_power_of_2(C)), BLOCK2=1024,
                    num_warps=8)

            def copy(l):
                T._tier_copy_kernel[(1, triton.cdiv(C, tpp))](
                    idx, nv, slot, hot.step, sc.code, sc.src, sc.rank, sc.victim,
                    kv_full[l].view(torch.int32), mirror.dev[l].view(torch.int32), hl[l].kv.view(torch.int32),
                    stg.view(torch.int32), sbt, hl[l].tag, hl[l].last, hl[l].pos2row, C, sbt.stride(0), H,
                    hot.max_ctx, HKV=HKV, BS=BS, W4=W // 4, W4P=256, TPP=tpp, num_warps=4)

            e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            tr_ms = tc_ms = 0.0
            for t in range(a.steps):
                idx.copy_(allidx[t])
                hot.next_step()
                # resolve for all layers, then copy for all layers (same work as interleaved; separable timing)
                e[0].record()
                for l in range(L):
                    resolve(l)
                e[1].record()
                for l in range(L):
                    # NOTE: scratch is shared, so copy uses the resolve output of the LAST layer for every layer;
                    # identical byte volume, fine for timing.
                    copy(l)
                e[2].record()
                torch.cuda.synchronize()
                if t >= 64:
                    tr_ms += e[0].elapsed_time(e[1]); tc_ms += e[1].elapsed_time(e[2])
            n = a.steps - 64
            st = hot.stats.tolist()
            out[f"{tag}_tpp{tpp}"] = {"resolve_ms_16L": round(tr_ms / n, 3), "copy_ms_16L": round(tc_ms / n, 3),
                                      "miss_pct": round(100 * st[1] / max(1, st[0] + st[1]), 2)}
            print(tag, tpp, out[f"{tag}_tpp{tpp}"], flush=True)
            del hot
    print("RESULT " + json.dumps(out), flush=True)
    mirror.close()


if __name__ == "__main__":
    main()
