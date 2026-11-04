#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""CPU self-test of pth_flash_attn.fa2_prefill (no GPU): the Triton dequant + LSE-merge kernels run under
TRITON_INTERPRET=1 and FA2 is replaced by an fp64 torch varlen attention with FA2's conventions (bottom-right
causal, softmax_lse [Hq, total_q], +inf LSE for a row with no keys). Checks the piece decomposition (end-aligned
pieces, empty pieces in multi-request slices, per-request piece size), cu_seqlens, strided cell views and the
merge against an fp64 reference over the dequantised cache.

  TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= venv/bin/python sparsekv/test_fa2_prefill_cpu.py
"""
import math
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pthfa", "python"))
import torch  # noqa: E402

import pth_flash_attn.fa2_prefill as fp  # noqa: E402

HQ, HKV, D = 24, 4, 256
G = HQ // HKV


def cell_views(kv):
    nb, nkv, bs, content = kv.shape
    padded = content // 2
    hs = padded - 4
    base = torch.tensor([], dtype=torch.float32).set_(kv.untyped_storage())
    st = kv.stride()
    off = kv.storage_offset()
    ks = torch.as_strided(base, (nb, bs, nkv), (st[0] // 4, st[2] // 4, st[1] // 4), (off + hs) // 4)
    vs = torch.as_strided(base, (nb, bs, nkv), (st[0] // 4, st[2] // 4, st[1] // 4), (off + padded + hs) // 4)
    key, val = kv.transpose(1, 2).split(padded, dim=-1)
    return key[..., :hs], val[..., :hs], ks, vs


def mock_fa2(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k=None, softmax_scale=None,
             causal=False, return_softmax_lse=False, out=None, fa_version=2, **kw):
    T = q.shape[0]
    lse = torch.full((HQ, T), float("inf"), dtype=torch.float32)
    o = torch.zeros(T, HQ, D, dtype=torch.float64)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    assert max(b - a for a, b in zip(cq, cq[1:])) <= max_seqlen_q
    assert max(b - a for a, b in zip(ck, ck[1:])) <= max_seqlen_k
    for r in range(len(cq) - 1):
        qs, qe, ks_, ke = cq[r], cq[r + 1], ck[r], ck[r + 1]
        nq, nk = qe - qs, ke - ks_
        if nq == 0 or nk == 0:
            continue
        qq = q[qs:qe].double()
        kk = k[ks_:ke].double().repeat_interleave(G, dim=1)
        vv = v[ks_:ke].double().repeat_interleave(G, dim=1)
        s = torch.einsum("qhd,khd->hqk", qq, kk) * softmax_scale
        if causal:
            i = torch.arange(nq)[:, None]
            j = torch.arange(nk)[None, :]
            s = s.masked_fill(j > (nk - nq) + i, float("-inf"))
        l_ = torch.logsumexp(s, -1)
        p = torch.exp(s - l_[..., None])
        o[qs:qe] = torch.einsum("hqk,khd->qhd", p, vv)
        lse[:, qs:qe] = l_.float()
    o = o.to(q.dtype)
    if out is not None:
        out.copy_(o)
        o = out
    return (o, lse) if return_softmax_lse else o


def run_case(name, seq, ql, bs, chunk, dtype, layout="HND", seed=0):
    g = torch.Generator().manual_seed(seed)
    R = len(seq)
    nblk = [math.ceil(s / bs) for s in seq]
    nb = sum(nblk) + 5
    shape = (nb, bs, HKV, 2 * (D + 4)) if layout == "NHD" else (nb, HKV, bs, 2 * (D + 4))
    kv = torch.zeros(shape, dtype=torch.int8)
    if layout == "NHD":
        kv = kv.transpose(1, 2)
    k8, v8, ks, vs = cell_views(kv)
    perm = torch.randperm(nb, generator=g)
    bt = torch.zeros(R, max(nblk) + 3, dtype=torch.int32)
    o = 0
    for r in range(R):
        bt[r, :nblk[r]] = perm[o:o + nblk[r]].to(torch.int32)
        o += nblk[r]
    # write random int8 KV + scales for every request
    Kf, Vf = [], []
    for r in range(R):
        pos = torch.arange(seq[r])
        blk, slot = bt[r, pos // bs].long(), pos % bs
        kq = torch.randint(-127, 128, (seq[r], HKV, D), generator=g, dtype=torch.int8)
        vq = torch.randint(-127, 128, (seq[r], HKV, D), generator=g, dtype=torch.int8)
        ksc = torch.rand(seq[r], HKV, generator=g) * 0.02 + 0.001
        vsc = torch.rand(seq[r], HKV, generator=g) * 0.05 + 0.001
        k8[blk, slot] = kq
        v8[blk, slot] = vq
        ks[blk, slot] = ksc
        vs[blk, slot] = vsc
        Kf.append(kq.double() * ksc.double()[..., None])
        Vf.append(vq.double() * vsc.double()[..., None])
    qsl = torch.tensor([0] + list(torch.tensor(ql).cumsum(0).tolist()), dtype=torch.int32)
    T = int(qsl[-1])
    q = (torch.randn(T, HQ, D, generator=g) * 2.0).to(torch.bfloat16)
    out = torch.full((T, HQ, D), float("nan"), dtype=torch.bfloat16)
    scale = D ** -0.5
    fp.prefill(q, out, k8, v8, ks, vs, bt, torch.tensor(seq, dtype=torch.int32), qsl, max(ql), max(seq) + 3,
               scale, chunk=chunk, dtype=dtype)
    # fp64 reference
    err = 0.0
    for r in range(R):
        qq = q[qsl[r]:qsl[r + 1]].double()
        kk = Kf[r].repeat_interleave(G, 1)
        vv = Vf[r].repeat_interleave(G, 1)
        s = torch.einsum("qhd,khd->hqk", qq, kk) * scale
        i = torch.arange(ql[r])[:, None]
        j = torch.arange(seq[r])[None, :]
        s = s.masked_fill(j > (seq[r] - ql[r]) + i, float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vv)
        got = out[qsl[r]:qsl[r + 1]].double()
        assert not torch.isnan(got).any(), f"{name}: NaN / unwritten rows in request {r}"
        err = max(err, float(((got - ref).abs().max() / ref.abs().max())))
    ch = max(chunk // R, max(ql))
    ch = -(-ch // 64) * 64
    npieces = max(1, -(-(max(seq) + 3 + fp.MARGIN) // ch))
    # bf16 output: 1 ulp at the row max is ~4e-3 relative (the interpreter may truncate); structural bugs give O(1)
    tol = 2.5e-2 if dtype == torch.bfloat16 else 1e-2
    ok = err < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: R={R} seq={seq} ql={ql} bs={bs} piece={ch} x{npieces} "
          f"{str(dtype).split('.')[-1]} {layout}: max rel err {err:.2e} (tol {tol:.0e})")
    return ok


def hook_gating():
    """try_prefill (the _run_slice hook): engages only on causal, window-0 prefill plans at or above MIN_CTX
    (host bound from the forward context, else the block-table capacity); writes only rows [t0, t1)."""
    from types import SimpleNamespace as NS
    calls = []
    real = fp.prefill
    fp.prefill = lambda *a, **k: calls.append((a[0].shape[0], a[10]))
    impl = NS(window=0, scale=0.0625)
    bt = torch.zeros(1, 20, dtype=torch.int32)
    k8 = torch.zeros(4, 864, HKV, D, dtype=torch.int8)
    caches = (k8, k8, None, None)
    q = torch.zeros(300, HQ, D)

    def part(pf=True, d=256):
        return NS(plan=NS(is_prefill=pf, D=d), block_table=bt, token_start=100, token_end=300, seq_lens=None,
                  query_start_loc=None, max_query_len=200)
    res = []
    fp.MIN_CTX = 8192
    res.append(fp.try_prefill(impl, part(), q, q, caches, True) is True and calls[-1] == (200, 20 * 864))
    res.append(fp.try_prefill(impl, part(pf=False), q, q, caches, True) is False)
    res.append(fp.try_prefill(impl, part(), q, q, caches, False) is False)
    res.append(fp.try_prefill(NS(window=2048, scale=1.0), part(), q, q, caches, True) is False)
    fp.MIN_CTX = 10 ** 9
    res.append(fp.try_prefill(impl, part(), q, q, caches, True) is False)
    fp.prefill = real
    ok = all(res)
    print(f"  {'PASS' if ok else 'FAIL'}  try_prefill gating {res}")
    return ok


def main():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    fp._flash = mock_fa2
    cases = [
        ("one piece", [300], [100], 48, 4096, torch.float16),
        ("many pieces", [1000], [70], 48, 128, torch.float16),
        ("piece == query", [500], [128], 32, 128, torch.float16),
        ("multi request, empty pieces", [1100, 90, 400], [64, 90, 30], 48, 384, torch.float16),
        ("bf16 workspace", [700], [50], 864, 192, torch.bfloat16),
        ("block 864 NHD", [1900], [100], 864, 512, torch.float16, "NHD"),
    ]
    ok = all(run_case(*c) for c in cases)
    ok &= hook_gating()
    print("fa2_prefill cpu selftest:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
