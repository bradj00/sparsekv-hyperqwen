# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Part of sparsekv-hyperqwen (v1.1); not part of upstream pth-flash-attn. Calls FlashAttention-2 through vLLM's
# vllm.vllm_flash_attn package; no FlashAttention code is included here.
"""Prefill attention through FlashAttention-2 over the int8 KV cache.

PTHFA_PREFILL_FA2=1 routes the PREFILL launches of the D=256 causal full-attention layers to FA2. The int8
per-token-head cache stays the only source of truth: per prefill chunk, the KV the chunk attends to is
dequantised in KV pieces of CH tokens into a 16-bit workspace and FA2 varlen runs on each piece; the pieces are
merged by their log-sum-exp. Decode, MTP verify and every non-prefill launch stay on pthfa (+ sparse-KV).

Why this design (kernel microbenchmark, 244K context, chunk 1728, fixed power limit): FA2 bf16 is 24% (3090 Ti) /
15% (3090) faster than pthfa per attention call. The dequant + merge cost here is memory traffic only:
~1.5 GB per layer-call at 244K (int8 read + fp16 write) plus one fp32 merge pass per extra piece, i.e. ~2-4 ms
next to ~160 ms of attention, so most of the measured gain survives. An FA2 kernel that reads int8 in its load
path (option b) would save only that ~2%, at weeks of kernel work; PREFILL_ATTN=int8 (option c) is a Triton
kernel over a bf16 cache (Triton prototypes ran ~1.85x slower than pthfa here) and quantises Q: not taken.

Pieces are aligned to the END of each request: piece 0 holds the last CH keys (all of the chunk's own query
positions, so FA2's bottom-right causal mask is exact there), pieces c >= 1 lie wholly before every query and
run non-causal. Everything is computed on the device from seq_lens / query_start_loc: no host sync. The host
only needs an upper bound of the longest sequence (the step's max_seq_len) to size the piece loop.

Numerics: K and V are dequantised as fp16(int8 * scale) (PTHFA_PREFILL_FA2_DTYPE=fp16, default; 2^-12
relative, ~1/16 of the int8 quantisation step) and Q is cast bf16 -> fp16 (exact in the normal range). pthfa
itself rounds Q and P to fp16, so this is the same rounding class, but not bit-identical to pthfa: it changes
upper-layer KV at fp16-rounding level, so it was gated on accuracy (call-site F1 within noise of pthfa prefill;
see docs/CHANGELOG.md and eval/prefill/README.md). dtype=bf16 is the
benchmark's variant (2^-9 relative).

Knobs (read at import; off unless set; serve-sparse.sh sets PTHFA_PREFILL_FA2=1):
  PTHFA_PREFILL_FA2=1                 enable
  PTHFA_PREFILL_FA2_CHUNK=32768       KV piece size in tokens (workspace = 2 * CH * Hkv * D * 2 B = 128 MiB at 32K)
  PTHFA_PREFILL_FA2_MIN_CTX=8192      below this max_seq_len, prefill stays on pthfa
  PTHFA_PREFILL_FA2_DTYPE=fp16|bf16
  PTHFA_PREFILL_FA2_CHECK=1           debug: sync and assert the piece loop covers every key (test window only)
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PTHFA_PREFILL_FA2", "0") == "1"
CHUNK = int(os.environ.get("PTHFA_PREFILL_FA2_CHUNK", "32768"))
MIN_CTX = int(os.environ.get("PTHFA_PREFILL_FA2_MIN_CTX", "8192"))
DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("PTHFA_PREFILL_FA2_DTYPE", "fp16")]
CHECK = os.environ.get("PTHFA_PREFILL_FA2_CHECK", "0") == "1"
# Keys past the host bound that the loop still covers (an off-by-a-few bound costs nothing).
MARGIN = 64
_TILE = 64

_flash = None
_stats = {"calls": 0, "pieces": 0}


def _fa2():
    global _flash
    if _flash is None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        _flash = flash_attn_varlen_func
    return _flash


@triton.jit
def _dequant_piece_kernel(
    k_ptr, v_ptr, ks_ptr, vs_ptr, bt_ptr, seq_ptr, cuk_ptr, ok_ptr, ov_ptr,
    piece, CH, BS,
    s_kb, s_ks, s_kh, s_vb, s_vs, s_vh,
    s_ksb, s_kss, s_ksh, s_vsb, s_vss, s_vsh, s_bt,
    HKV: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Piece `piece` of request r, KV head h: keys [max(end - CH, 0), end), end = seq_len - piece*CH, written
    densely at rows cuk[r] .. of the [rows, HKV, D] workspace as (int8 * fp32 scale) in the workspace dtype."""
    pid_t = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    seq = tl.load(seq_ptr + r)
    end = seq - piece * CH
    start = tl.maximum(end - CH, 0)
    n = tl.maximum(end - start, 0)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    ok = t < n
    pos = start + t
    blk = tl.load(bt_ptr + r * s_bt + pos // BS, mask=ok, other=0).to(tl.int64)
    slot = (pos % BS).to(tl.int64)
    d = tl.arange(0, D)
    row = (tl.load(cuk_ptr + r) + t).to(tl.int64)
    dst = (row * HKV + h)[:, None] * D + d[None, :]
    m2 = ok[:, None]
    k8 = tl.load(k_ptr + (blk * s_kb + slot * s_ks + h * s_kh)[:, None] + d[None, :], mask=m2, other=0)
    ks = tl.load(ks_ptr + blk * s_ksb + slot * s_kss + h * s_ksh, mask=ok, other=0.0)
    tl.store(ok_ptr + dst, (k8.to(tl.float32) * ks[:, None]).to(ok_ptr.dtype.element_ty), mask=m2)
    v8 = tl.load(v_ptr + (blk * s_vb + slot * s_vs + h * s_vh)[:, None] + d[None, :], mask=m2, other=0)
    vs = tl.load(vs_ptr + blk * s_vsb + slot * s_vss + h * s_vsh, mask=ok, other=0.0)
    tl.store(ov_ptr + dst, (v8.to(tl.float32) * vs[:, None]).to(ov_ptr.dtype.element_ty), mask=m2)


@triton.jit
def _merge_kernel(
    a_ptr, a_lse_ptr, s_alse_t, s_alse_h, b_ptr, b_lse_ptr, s_blse_t, s_blse_h,
    acc_ptr, L_ptr, out_ptr, s_out_t, s_out_h,
    HQ: tl.constexpr, D: tl.constexpr, FINAL: tl.constexpr,
):
    """acc/L <- logsumexp-merge of (a, lse_a) and (b, lse_b) for one (token, head) row; FINAL writes the
    normalised row to `out` instead. a is piece 0's FA2 output (first merge) or the fp32 accumulator; b is the
    new piece. FA2 reports +inf as the LSE of a row with no keys: treated as -inf (weight 0). lse_a is always
    finite (piece 0 holds the row's own key)."""
    pid = tl.program_id(0)
    t = pid // HQ
    h = pid % HQ
    d = tl.arange(0, D)
    la = tl.load(a_lse_ptr + t * s_alse_t + h * s_alse_h)
    lb = tl.load(b_lse_ptr + t * s_blse_t + h * s_blse_h)
    lb = tl.where((lb == float("inf")) | (lb != lb), float("-inf"), lb)
    m = tl.maximum(la, lb)
    wa = tl.exp(la - m)
    wb = tl.exp(lb - m)
    tot = wa + wb
    a = tl.load(a_ptr + (t * HQ + h) * D + d).to(tl.float32)
    b = tl.load(b_ptr + (t * HQ + h) * D + d).to(tl.float32)
    o = (a * wa + b * wb) / tot
    if FINAL:
        tl.store(out_ptr + t * s_out_t + h * s_out_h + d, o.to(out_ptr.dtype.element_ty))
    else:
        tl.store(acc_ptr + (t * HQ + h) * D + d, o)
        tl.store(L_ptr + t * HQ + h, m + tl.log(tot))


def _ws(name: str, shape, dtype, device):
    """Grow-only module-level workspaces (reused across layers and steps; one stream)."""
    n = math.prod(shape)
    buf = _WS.get(name)
    if buf is None or buf.numel() < n or buf.dtype != dtype or buf.device != device:
        _WS[name] = buf = torch.empty(n, dtype=dtype, device=device)
    return buf[:n].view(shape)


_WS: dict = {}


def prefill(q, out, k, v, ks, vs, block_table, seq_lens, query_start_loc, max_query_len: int, max_seq_len: int,
            softmax_scale: float, chunk: int | None = None, dtype: torch.dtype | None = None) -> None:
    """Causal prefill attention of one slice, same contract as pth_flash_attn_fwd's prefill launch.

    q/out [T, Hq, D] (query_start_loc rebased to this slice); k/v int8 [NB, BS, Hkv, D] and ks/vs fp32
    [NB, BS, Hkv] views (any outer strides); block_table [R, *] int32; seq_lens [R] int32 (KV length incl. the
    slice's tokens). max_query_len / max_seq_len are host upper bounds.
    """
    chunk = CHUNK if chunk is None else chunk
    dtype = DTYPE if dtype is None else dtype
    T, HQ, D = q.shape
    HKV = k.shape[2]
    R = seq_lens.shape[0]
    BS = k.shape[1]
    dev = q.device
    # Per-request piece length: the workspace holds R * CH rows; piece 0 must hold every query of the slice.
    ch = max(chunk // max(R, 1), max_query_len)
    ch = -(-ch // _TILE) * _TILE
    npieces = max(1, -(-(max_seq_len + MARGIN) // ch))
    if CHECK:
        smax = int(seq_lens.max().item())
        assert smax <= npieces * ch and int((query_start_loc[1:] - query_start_loc[:-1]).max().item()) <= ch, (
            f"fa2_prefill: pieces {npieces}x{ch} do not cover max seq_len {smax} (host bound {max_seq_len})")
    # cu_seqlens_k of every piece: klen[c, r] = clamp(seq_r - c*ch, 0, ch). Device only.
    c = torch.arange(npieces, device=dev, dtype=torch.int32).unsqueeze(1)
    klen = (seq_lens.unsqueeze(0) - c * ch).clamp_(0, ch)
    cuk = torch.zeros(npieces, R + 1, device=dev, dtype=torch.int32)
    cuk[:, 1:] = klen.cumsum(1, dtype=torch.int32)
    wk = _ws("k", (R * ch, HKV, D), dtype, dev)
    wv = _ws("v", (R * ch, HKV, D), dtype, dev)
    qh = q if q.dtype == dtype else q.to(dtype)
    flash = _fa2()
    grid = (ch // _TILE, R, HKV)
    bt = block_table

    def dequant(piece):
        _dequant_piece_kernel[grid](
            k, v, ks, vs, bt, seq_lens, cuk[piece], wk, wv, piece, ch, BS,
            k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
            ks.stride(0), ks.stride(1), ks.stride(2), vs.stride(0), vs.stride(1), vs.stride(2), bt.stride(0),
            HKV=HKV, D=D, BLOCK_T=_TILE, num_warps=4)

    def attn(piece, causal, o):
        return flash(qh, wk, wv, max_query_len, query_start_loc, ch, cu_seqlens_k=cuk[piece],
                     softmax_scale=softmax_scale, causal=causal, return_softmax_lse=True, out=o, fa_version=2)

    _stats["calls"] += 1
    _stats["pieces"] += npieces
    direct = npieces == 1 and out.dtype == dtype and out.is_contiguous()
    o0 = out if direct else _ws("o0", (T, HQ, D), dtype, dev)
    dequant(0)
    o0, lse0 = attn(0, True, o0)
    if npieces == 1:
        if not direct:
            out.copy_(o0)
        return
    ob = _ws("ob", (T, HQ, D), dtype, dev)
    acc = _ws("acc", (T, HQ, D), torch.float32, dev) if npieces > 2 else o0
    L = _ws("L", (T, HQ), torch.float32, dev)
    for piece in range(1, npieces):
        dequant(piece)
        ob, lseb = attn(piece, False, ob)
        first, final = piece == 1, piece == npieces - 1
        a, la = (o0, lse0) if first else (acc, L)
        s_alse = (la.stride(1), la.stride(0)) if first else (L.stride(0), L.stride(1))  # FA2 lse is [Hq, T]
        _merge_kernel[(T * HQ,)](
            a, la, s_alse[0], s_alse[1], ob, lseb, lseb.stride(1), lseb.stride(0),
            acc, L, out, out.stride(0), out.stride(1), HQ=HQ, D=D, FINAL=final, num_warps=2)


_md_cache: list = [None, 0]


def host_max_seq_len(fallback: int) -> int:
    """The step's host max_seq_len (an upper bound of every request's KV length in this forward), read from the
    forward context's attention metadata (any PthAttnMetadata of the step: one per KV-cache group, the same
    batch). Cached per metadata object; `fallback` (block-table capacity) if none is found."""
    try:
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
    except Exception:  # noqa: BLE001
        return fallback
    if isinstance(md, list):
        md = md[0] if md else None
    if md is _md_cache[0] and md is not None:
        return _md_cache[1]
    best = 0
    vals = md.values() if isinstance(md, dict) else ([md] if md is not None else [])
    for m in vals:
        if type(m).__name__ == "PthAttnMetadata":
            best = max(best, int(m.max_seq_len))
    if best <= 0:
        return fallback
    _md_cache[0], _md_cache[1] = md, best  # keeps md alive, so identity cannot be reused
    return best


def try_prefill(impl, part, query, output, caches, causal) -> bool:
    """vllm_backend._run_slice hook. True if the slice ran on FA2; False -> the caller runs pthfa."""
    plan = part.plan
    if not (plan.is_prefill and causal and not getattr(impl, "window", 0) and plan.D in (64, 128, 256)):
        return False
    k, v, ks, vs = caches
    fallback = int(part.block_table.shape[1]) * int(k.shape[1])
    msl = host_max_seq_len(fallback)
    if msl < MIN_CTX:
        return False
    t0, t1 = part.token_start, part.token_end
    if _stats["calls"] == 0 or _stats["calls"] % 20000 == 0:
        import logging
        logging.getLogger("vllm.pth_flash_attn.fa2_prefill").warning(
            "fa2_prefill: prefill on FA2 (calls %d, pieces %d; piece %d, dtype %s, min ctx %d; this slice T=%d, "
            "max_seq_len %d)", _stats["calls"], _stats["pieces"], CHUNK, str(DTYPE).split(".")[-1], MIN_CTX,
            t1 - t0, msl)
    prefill(query[t0:t1], output[t0:t1], k, v, ks, vs, part.block_table, part.seq_lens, part.query_start_loc,
            int(part.max_query_len), msl, float(impl.scale))
    return True


def stats() -> dict:
    return dict(_stats)
