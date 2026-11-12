# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""s13 GPU test: GDN recover mode (SPARSE_GDN_RECOVER=1) == baseline per-draft-token states.
  CUDA_VISIBLE_DEVICES=<g> SPARSE_GDN_RECOVER=1 venv/bin/python sparsekv/test_s13_gdn_recover_gpu.py
Baseline: the verify kernel stores the state after every window token in its own block (columns c..c+K); the next
step reads column n-1. Recover: only column c exists; the kernel stores the state after token 0 there and
recover.gdn_recover_commit_kernel replays tokens 0..n-1 (and writes the block-aligned checkpoint). Checks, for several
accepted counts n, BIT-identical: outputs, state[c] == baseline state[c + n - 1], checkpoints; other blocks untouched."""
import os
import sys

os.environ["SPARSE_GDN_RECOVER"] = "1"
import torch
import triton

from vllm.model_executor.layers.mamba.gdn import recover
from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating_delta_rule_update as upd


def run(dtype, state_dtype, S=5, KD=7, H=16, HV=16, K=128, V=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    W = KD + 1
    T = S * W
    nblk = 1 + S * (W + 2)
    r = lambda *sh, sc=1.0: (torch.randn(*sh, generator=g, device=dev) * sc)  # noqa: E731
    h0 = r(nblk, HV, V, K, sc=0.3).to(state_dtype)
    a, b = r(T, HV).to(dtype), r(T, HV).to(dtype)
    q, k, v = r(1, T, H, K).to(dtype), r(1, T, H, K).to(dtype), r(1, T, HV, V).to(dtype)
    A_log, dt_bias = r(HV, sc=0.5), r(HV, sc=0.5)
    cu = torch.arange(0, T + 1, W, dtype=torch.int32, device=dev)
    run_blk = torch.tensor([1 + i * (W + 2) for i in range(S)], dtype=torch.int32, device=dev)
    base_idx = run_blk[:, None] + torch.arange(W, dtype=torch.int32, device=dev)[None, :]
    ckpt_blk_all = run_blk + W  # a spare block per row
    kw = dict(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, inplace_final_state=True, cu_seqlens=cu,
              use_qk_l2norm_in_kernel=True)
    # baseline (num_accepted = 1: read column 0)
    hb = h0.clone()
    ob, _ = upd(initial_state=hb, ssm_state_indices=base_idx.contiguous(),
                num_accepted_tokens=torch.ones(S, dtype=torch.int32, device=dev), **kw)
    # recover forward: every write column is NULL, the read column (the running block) is appended
    hr = h0.clone()
    rec_idx = torch.cat([torch.zeros_like(base_idx), run_blk[:, None]], dim=1).contiguous()
    orr, _ = upd(initial_state=hr, ssm_state_indices=rec_idx,
                 num_accepted_tokens=torch.full((S,), W + 1, dtype=torch.int32, device=dev), **kw)
    assert torch.equal(ob, orr), "verify outputs differ"
    assert torch.equal(hr, h0), "the recover forward wrote state"
    n = torch.tensor([1, 2, W, 4, 3][:S], dtype=torch.int32, device=dev)
    ckpt_at = torch.tensor([-1, 1, 3, -1, 2][:S], dtype=torch.int32, device=dev)
    ckpt_blk = torch.where(ckpt_at > 0, ckpt_blk_all, torch.zeros_like(ckpt_blk_all))
    bk, bv = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    recover.gdn_recover_commit_kernel[(triton.cdiv(V, bv), S * HV)](
        A_log, a, b, dt_bias, k[0].contiguous(), v[0].contiguous(), hr, cu, n, run_blk, ckpt_at,
        ckpt_blk, 1.0, 20.0, N=S, H=H, HV=HV, K=K, V=V, BK=bk, BV=bv, stride_state=hr.stride(0), num_warps=4)
    torch.cuda.synchronize()
    worst = 0.0
    for i in range(S):
        c = int(run_blk[i])
        assert torch.equal(hr[c], hb[c + int(n[i]) - 1]), f"row {i}: running state not bit-identical"
        if int(ckpt_at[i]) > 0:  # checkpoint after ckpt_at applied tokens == baseline column ckpt_at - 1
            assert torch.equal(hr[int(ckpt_blk[i])], hb[c + int(ckpt_at[i]) - 1]), f"row {i}: checkpoint differs"
        else:
            assert torch.equal(hr[c + W], h0[c + W]), "spare block written without a checkpoint"
        for j in range(1, W):
            assert torch.equal(hr[c + j], h0[c + j]), "recover wrote a draft column"
    return worst


if __name__ == "__main__":
    for dtype, sd, tol in ((torch.bfloat16, torch.bfloat16, 2e-2), (torch.float16, torch.float16, 3e-3),
                           (torch.float32, torch.float32, 1e-5)):
        for s in range(3):
            run(dtype, sd, seed=s)
        print(f"gdn recover vs baseline, dtype {dtype} state {sd}: outputs, running state and checkpoints bit-identical")
    print("ALL OK")
