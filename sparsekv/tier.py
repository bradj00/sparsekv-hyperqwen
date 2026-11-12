# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
"""sparsekv/tier.py -- host-RAM KV tier for the sparse-decode COMPACTION path (stream E, Track A).

Picture
-------
B's compaction path gathers the ~C = KSEL+SINK+WIN picked rows of every full-attention layer into a small staging
cache and runs stock pthfa over it. Those rows may come from three places, resolved per picked token:

  1. RESIDENT  the token's block is still in the GPU KV pool        (gpu_bt[r, pos // BS] >= 0)
  2. HOT       the token is in this request's GPU hot buffer         (pos2row[slot, pos] >= 0)
  3. HOST      otherwise: read from the pinned host mirror through a UVA device view (zero-copy, no CPU work),
               and also written into the hot buffer (replacing a victim row).

Convention: a spilled block has gpu_bt entry -1, or `null_block` (pass null_block=0 to feed a Model Runner V2 block
table directly: vLLM's null block 0 never holds data, spilled entries are rewritten to it). Every block of a request that may be spilled
must have a host copy: host_bt[r, blk] >= 0 (HostKVMirror.mirror_blocks writes it when a block fills).

Layouts (int8; W = 2*(D+4) = 520 bytes per token per KV head, inline fp32 scales included):
  GPU kv        [NB,  Hkv, BS, W]          (pthfa HND cache, unchanged)
  host mirror   [L, HB, Hkv, BS, W]         pinned, same block layout as the GPU -> mirror = plain DMA of whole blocks
  hot buffer    [L, S, H, Hkv, W]           token-granular: one hot row = all 4 heads of one token (2080 B)
  staging       [NBs, Hkv, BS, W]          B's staging cache; picked column j of request r -> staging_bt[r, j//BS], j%BS

Hot-buffer replacement ("windowed CLOCK", entirely on device, graph-safe):
  last[l, s, row] = global step at which the row was last used (hit or filled).
  Each step and layer, hits refresh last[] first. Then the n_miss victims are taken from a window of
  Wv = min(2*C, H) rows starting at the clock hand: first rows that are OLD (last < step - age), in ring order, then
  rows that are merely not used in this step (last != step). Rows used in this step are never victims, and because
  n_hit + n_miss <= C, a window of 2C rows always holds enough candidates. The hand moves past the last row consumed.
  age = 0 is FIFO-with-skip; age > 0 gives recently used rows a second lap (CLOCK). See tier_sim.py for miss rates
  versus exact LRU.

Requirements: H >= 2*C is recommended (H >= C is required); picks within one request row contain no duplicates
(contract: sinks/window excluded from top-k). Duplicates do not corrupt data (pos2row is cleared with a CAS), they
only waste a hot row.

Graph safety: gather_rows_tiered launches two Triton kernels with shapes depending only on (nreq, C); the global step
counter is a device tensor bumped by HotBuffer.next_step() (an in-place add, capturable). No host syncs.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice
    except ImportError:  # pragma: no cover
        from triton.language.extra.cuda import libdevice
except Exception:  # pragma: no cover - CPU-only simulation can still use the reference
    triton = None
    tl = None

NEG_AGE = -(1 << 30)


# ----------------------------------------------------------------------------------------------------------------
# pinned host memory without the caching host allocator's power-of-two rounding
# ----------------------------------------------------------------------------------------------------------------
def alloc_pinned(shape, dtype=torch.int8) -> torch.Tensor:
    """Page-locked CPU tensor of exactly the requested size (cudaHostRegister on a plain allocation).

    torch's pin_memory=True goes through the CUDA caching host allocator, which rounds large requests up to a power of
    two (a 17 GB pool would pin 32 GB). Registration pins exactly what we allocate.
    """
    t = torch.empty(shape, dtype=dtype)
    nbytes = t.numel() * t.element_size()
    if nbytes:
        rt = torch.cuda.cudart()
        err = rt.cudaHostRegister(t.data_ptr(), nbytes, 0)
        if int(err) != 0 and getattr(err, "value", err) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {err}")
    t._sk_registered = True  # noqa: SLF001  (marker for free_pinned)
    return t


def free_pinned(t: torch.Tensor) -> None:
    if getattr(t, "_sk_registered", False) and t.numel():
        torch.cuda.cudart().cudaHostUnregister(t.data_ptr())
        t._sk_registered = False


def uva_view(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """CUDA tensor aliasing pinned host memory (vLLM's helper; proven in the lab)."""
    import vllm  # noqa: F401  (registers torch.ops._C)
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    return get_accelerator_view_from_cpu_tensor(cpu_tensor)


# ----------------------------------------------------------------------------------------------------------------
# host mirror
# ----------------------------------------------------------------------------------------------------------------
class HostKVMirror:
    """Pinned host copy of the full-attention KV blocks of all sparse layers, same block layout as the GPU cache.

    Host blocks are handed out by a trivial free list (scheduler-side bookkeeping in the real integration).
    """

    def __init__(self, num_layers: int, num_host_blocks: int, hkv: int, bs: int, w: int, device="cuda",
                 use_uva: bool = True):
        self.L, self.HB, self.hkv, self.bs, self.w = num_layers, num_host_blocks, hkv, bs, w
        self.host = alloc_pinned((num_layers, num_host_blocks, hkv, bs, w), torch.int8)
        self.dev = uva_view(self.host) if use_uva else None  # CUDA view, same shape
        self.free = list(range(num_host_blocks - 1, -1, -1))
        self.copy_stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None

    @property
    def nbytes(self) -> int:
        return self.host.numel()

    def alloc(self, n: int) -> list[int]:
        if n > len(self.free):
            raise MemoryError(f"host KV pool exhausted ({n} > {len(self.free)} free blocks)")
        return [self.free.pop() for _ in range(n)]

    def release(self, ids) -> None:
        self.free.extend(int(i) for i in ids)

    def mirror_blocks(self, layer: int, kv: torch.Tensor, gpu_blocks, host_blocks, stream=None) -> None:
        """DMA whole GPU blocks of one layer to their host blocks (call when a block fills, or before a spill).

        Runs on the mirror's copy stream after the current stream's pending writes (the caller must keep the GPU
        block alive -- not reuse it -- until the copy stream has passed this point, e.g. by recording an event).
        """
        s = stream or self.copy_stream
        cur = torch.cuda.current_stream(kv.device)
        s.wait_stream(cur)
        with torch.cuda.stream(s):
            for gb, hb in zip(gpu_blocks, host_blocks):
                self.host[layer, int(hb)].copy_(kv[int(gb)], non_blocking=True)

    def mirror_all_layers(self, kvs, gpu_blocks, host_blocks, stream=None) -> torch.cuda.Event:
        for l, kv in enumerate(kvs):
            self.mirror_blocks(l, kv, gpu_blocks, host_blocks, stream)
        ev = torch.cuda.Event()
        ev.record(stream or self.copy_stream)
        return ev

    def close(self):
        self.dev = None
        free_pinned(self.host)


# ----------------------------------------------------------------------------------------------------------------
# hot buffer
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class HotLayer:
    kv: torch.Tensor  # [S, H, Hkv, W] int8
    tag: torch.Tensor  # [S, H] int32, logical position held, -1 empty
    last: torch.Tensor  # [S, H] int32, step last used
    pos2row: torch.Tensor  # [S, MAXCTX] int32, -1 = not hot
    hand: torch.Tensor  # [S] int32


class HotBuffer:
    """Per-request (slot) token-granular GPU cache of host-tier rows, one per sparse layer."""

    def __init__(self, num_layers: int, num_slots: int, rows: int, hkv: int, w: int, max_ctx: int,
                 device="cuda", age: int = 8, kv_slots: int | None = None):
        """kv_slots: slots that get hot KV rows (default all). S4: the CUDA-graph padding slot (the last one) never
        hits or misses (its rows have n_valid = 0), so it needs tag/last/pos2row/hand but no KV rows."""
        self.L, self.S, self.H, self.hkv, self.w, self.max_ctx = num_layers, num_slots, rows, hkv, w, max_ctx
        self.age = int(age)
        d = device
        self.kv_slots = num_slots if kv_slots is None else int(kv_slots)
        assert 1 <= self.kv_slots <= num_slots
        self.kv = torch.zeros((num_layers, self.kv_slots, rows, hkv, w), dtype=torch.int8, device=d)
        self.tag = torch.full((num_layers, num_slots, rows), -1, dtype=torch.int32, device=d)
        self.last = torch.full((num_layers, num_slots, rows), NEG_AGE, dtype=torch.int32, device=d)
        self.pos2row = torch.full((num_layers, num_slots, max_ctx), -1, dtype=torch.int32, device=d)
        self.hand = torch.zeros((num_layers, num_slots), dtype=torch.int32, device=d)
        self.step = torch.zeros(1, dtype=torch.int32, device=d)
        # statistics: [hit, miss, resident] summed over layers/requests since reset (device counters)
        self.stats = torch.zeros(3, dtype=torch.int64, device=d)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.kv, self.tag, self.last, self.pos2row, self.hand))

    def layer(self, l: int) -> HotLayer:
        return HotLayer(self.kv[l], self.tag[l], self.last[l], self.pos2row[l], self.hand[l])

    def next_step(self) -> None:
        """Advance the global step counter (once per decode/verify step; capturable)."""
        self.step.add_(1)

    def reset_slot(self, slot: int) -> None:
        """New request / preemption: forget everything cached for this slot."""
        self.tag[:, slot].fill_(-1)
        self.last[:, slot].fill_(NEG_AGE)
        self.pos2row[:, slot].fill_(-1)
        self.hand[:, slot].zero_()

    def invalidate_host_blocks(self, host_blocks: torch.Tensor, bs: int) -> None:
        """S14 global pool (one slot, keys = host tokens): the host blocks in `host_blocks` (int64 device tensor) are
        about to be rewritten, so drop their hot rows (caller guarantees 0 <= block < host pool blocks). No host sync,
        deterministic with duplicate / absent keys."""
        if host_blocks.numel() == 0:
            return
        assert self.S == 1, "invalidate_host_blocks is for the global pool"
        ar = torch.arange(bs, device=host_blocks.device)
        keys = (host_blocks.long()[:, None] * bs + ar).reshape(-1)
        p2r = self.pos2row[:, 0]  # [L, NK]
        rows = p2r.index_select(1, keys)  # [L, n]
        valid = rows >= 0
        r = rows.clamp_min(0).long()
        big = torch.iinfo(torch.int32).max
        # amin keeps untouched rows (min(x, INT_MAX) = x); a row named by a valid key gets -1 / NEG_AGE
        self.tag[:, 0].scatter_reduce_(1, r, torch.where(valid, -1, big).to(torch.int32), reduce="amin")
        self.last[:, 0].scatter_reduce_(1, r, torch.where(valid, NEG_AGE, big).to(torch.int32), reduce="amin")
        p2r.index_fill_(1, keys, -1)


# ----------------------------------------------------------------------------------------------------------------
# reference (plain torch; runs on CPU too, used by tier_sim.py and the tests)
# ----------------------------------------------------------------------------------------------------------------
def pick_victims_ref(last_s: torch.Tensor, hand: int, step: int, age: int, n_miss: int, C: int):
    """Windowed CLOCK victim choice for one (layer, slot). Returns (victims int64[n_miss], new_hand)."""
    H = last_s.numel()
    if n_miss == 0:
        return last_s.new_zeros(0, dtype=torch.int64), hand
    Wv = min(2 * C, H)
    t = torch.arange(Wv, device=last_s.device)
    rows = (hand + t) % H
    lst = last_s[rows]
    cls_a = lst < step - age
    cls_b = (lst != step) & ~cls_a
    pa = t[cls_a]
    pb = t[cls_b]
    na = min(n_miss, pa.numel())
    nb = n_miss - na
    assert nb <= pb.numel(), "hot buffer window too small (need H >= C and n_hit + n_miss <= C)"
    used = torch.cat([pa[:na], pb[:nb]])
    new_hand = (hand + int(used.max()) + 1) % H
    return rows[used], new_hand


def gather_rows_tiered_ref(kv, gpu_bt, host_kv, host_bt, idx, n_valid, staging_kv, staging_bt, hot: HotLayer,
                           req_slot, step: int, age: int, copy: bool = True, null_block: int = -1):
    """Reference of gather_rows_tiered for one layer. Mutates hot (and staging if copy). Returns counts [hit, miss, res].

    kv [NB,Hkv,BS,W] (GPU cache, may be None if copy=False), gpu_bt [nreq,maxblk] (-1 = spilled),
    host_kv [HB,Hkv,BS,W] (this layer's host mirror), host_bt [nreq,maxblk], idx [nreq,C], n_valid [nreq].
    """
    nreq, C = idx.shape
    BS = staging_kv.shape[2] if staging_kv is not None else host_kv.shape[2]
    tot = [0, 0, 0]
    for r in range(nreq):
        s = int(req_slot[r])
        n = int(n_valid[r])
        pos = idx[r, :n].long()
        blk, off = pos // BS, pos % BS
        gb = gpu_bt[r].long()[blk]
        res = (gb >= 0) & (gb != null_block)
        hrow = hot.pos2row[s].long()[pos]
        hit = (~res) & (hrow >= 0)
        miss = (~res) & (hrow < 0)
        hot.last[s][hrow[hit]] = step
        m = int(miss.sum())
        vic, nh = pick_victims_ref(hot.last[s], int(hot.hand[s]), step, age, m, C)
        hot.hand[s] = nh
        mpos = pos[miss]
        if m:
            old = hot.tag[s][vic].long()
            ok = old >= 0
            # CAS semantics: clear only if pos2row still points at the victim row
            oo = old[ok]
            still = hot.pos2row[s][oo].long() == vic[ok]
            hot.pos2row[s][oo[still]] = -1
            hot.tag[s][vic] = mpos.to(torch.int32)
            hot.pos2row[s][mpos] = vic.to(torch.int32)
            hot.last[s][vic] = step
            if copy:
                hb = host_bt[r].long()[blk[miss]]
                hot.kv[s][vic] = host_kv[hb, :, off[miss]].to(hot.kv.device)
        if copy:
            j = torch.arange(n, device=staging_kv.device)
            db = staging_bt[r].long()[j // BS]
            do = j % BS
            src = torch.empty((n, kv.shape[1], kv.shape[3]), dtype=kv.dtype, device=kv.device)
            src[res] = kv[gb[res], :, off[res]]
            src[hit] = hot.kv[s][hrow[hit]]
            # misses were just written to hot rows
            src[miss] = hot.kv[s][hot.pos2row[s].long()[mpos]]
            staging_kv[db, :, do] = src
        tot[0] += int(hit.sum()); tot[1] += m; tot[2] += int(res.sum())
    return tot


# ----------------------------------------------------------------------------------------------------------------
# S14: GLOBAL hot pool (SPARSE_TIER_HOT_GLOBAL=1, off by default). One pool of H rows per sparse layer shared by all
# requests, keyed by the GLOBAL HOST TOKEN (host_block * BS + offset) instead of (request slot, logical position).
#   * no per-request partition: idle slots hold no rows, a busy request can use the whole pool;
#   * rows survive the end of a request (a follow-up turn that maps the same host blocks starts warm) and are shared
#     by requests that map the same host blocks (common prefix);
#   * the price: a host block that is REWRITTEN must drop its hot rows (HotBuffer.invalidate_host_blocks, called by
#     the connector before every GPU->host block copy), and victims are chosen once for the whole batch (requests no
#     longer own disjoint rows), so the resolve step is split: classify + refresh hits for every request first, then
#     one victim scan. If the scan finds fewer victims than misses, the remaining misses are staged straight from
#     host and not cached (any H >= 1 is correct; H only sets the miss rate).
# Layout reuse: a HotBuffer with ONE slot (num_slots = kv_slots = 1) and max_ctx = host-pool tokens. Requests whose
# req_slot is < cached_slots use the pool; the others are uncached (CUDA-graph padding rows, prefill suffixes),
# exactly like the padding slot of the per-slot layout.
# ----------------------------------------------------------------------------------------------------------------
def global_window(H: int, nreq: int, C: int) -> int:
    """Rows after the hand that one victim scan may look at (bounds the serial scan like the per-slot 2C window)."""
    return min(H, 2 * nreq * C)


def pick_victims_global_ref(last_s: torch.Tensor, hand: int, step: int, age: int, n_miss: int, Wv: int):
    """Windowed CLOCK for the global pool: UP TO n_miss victims among the Wv rows after the hand (OLD rows first, then
    rows not used in this step). Returns (victims int64[<= n_miss], new_hand)."""
    H = last_s.numel()
    if n_miss <= 0:
        return last_s.new_zeros(0, dtype=torch.int64), hand
    t = torch.arange(min(Wv, H), device=last_s.device)
    rows = (hand + t) % H
    lst = last_s[rows]
    cls_a = lst < step - age
    cls_b = (lst != step) & ~cls_a
    pa = t[cls_a][:n_miss]
    pb = t[cls_b][: n_miss - pa.numel()]
    used = torch.cat([pa, pb])
    if used.numel() == 0:
        return used, hand
    return rows[used], (hand + int(used.max()) + 1) % H


def gather_rows_global_ref(kv, gpu_bt, host_kv, host_bt, idx, n_valid, staging_kv, staging_bt, hot: HotLayer,
                           req_slot, step: int, age: int, copy: bool = True, null_block: int = -1,
                           window: int | None = None, cached_slots: int = 1):
    """Reference of gather_rows_tiered_global, one layer (hot = HotBuffer(num_slots=1).layer(l)).
    Same arguments as gather_rows_tiered_ref; host_kv must hold int8 pthfa rows. Requests with req_slot <
    cached_slots use the pool, the others are uncached (padding rows, suffixes). Returns [hit, miss, resident,
    uncached] (uncached = cacheable misses that found no victim row, a subset of miss)."""
    nreq, C = idx.shape
    BS = staging_kv.shape[2] if staging_kv is not None else host_kv.shape[2]
    last, tag, p2r = hot.last[0], hot.tag[0], hot.pos2row[0]
    H, NK = last.numel(), p2r.numel()
    per = []
    M = 0
    tot = [0, 0, 0, 0]
    # ---- phase 1 (all requests): classify, refresh hits, count cacheable misses
    for r in range(nreq):
        n = int(n_valid[r])
        pos = idx[r, :n].long()
        blk, off = pos // BS, pos % BS
        gb = gpu_bt[r].long()[blk]
        res = (gb >= 0) & (gb != null_block)
        hb = host_bt[r].long()[blk]
        key = hb * BS + off
        keyok = (~res) & (hb >= 0) & (key < NK) & (int(req_slot[r]) < cached_slots)
        hrow = torch.full_like(pos, -1)
        hrow[keyok] = p2r[key[keyok]].long()
        hit = hrow >= 0
        miss = (~res) & ~hit
        cm = miss & keyok  # cacheable misses, in column order
        last[hrow[hit]] = step
        per.append((n, blk, off, gb, res, hb, key, hrow, hit, miss, cm, M))
        M += int(cm.sum())
        tot[0] += int(hit.sum()); tot[1] += int(miss.sum()); tot[2] += int(res.sum())
    # ---- phase 2 (once): victims for the whole batch
    Wv = global_window(H, nreq, C) if window is None else window
    vic, nh = pick_victims_global_ref(last, int(hot.hand[0]), step, age, M, Wv)
    hot.hand[0] = nh
    nv = vic.numel()
    tot[3] = M - nv
    # ---- phase 3 (per request): fill victim rows, stage
    for r in range(nreq):
        n, blk, off, gb, res, hb, key, hrow, hit, miss, cm, base = per[r]
        g = base + torch.arange(int(cm.sum()), device=idx.device)
        okc = g < nv
        ck = key[cm][okc]
        v = vic[g[okc]]
        if v.numel():
            old = tag[v].long()
            ok = old >= 0
            oo = old[ok]
            still = p2r[oo].long() == v[ok]  # CAS: clear only if the key still points at the victim row
            p2r[oo[still]] = -1
            tag[v] = ck.to(torch.int32)
            p2r[ck] = v.to(torch.int32)
            last[v] = step
            if copy:
                hot.kv[0][v] = host_kv[hb[cm][okc], :, off[cm][okc]].to(hot.kv.device)
        if copy:
            j = torch.arange(n, device=staging_kv.device)
            db = staging_bt[r].long()[j // BS]
            src = torch.empty((n, kv.shape[1], kv.shape[3]), dtype=kv.dtype, device=kv.device)
            src[res] = kv[gb[res], :, off[res]]
            src[hit] = hot.kv[0][hrow[hit]]
            src[miss] = host_kv[hb[miss], :, off[miss]].to(kv.device)
            staging_kv[db, :, j % BS] = src
    return tot


# ----------------------------------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _tier_resolve_kernel(
        idx_ptr, nvalid_ptr, gbt_ptr, hbt_ptr, slot_ptr, step_ptr,
        pos2row_ptr, last_ptr, hand_ptr,
        code_ptr, src_ptr, rank_ptr, victim_ptr, stats_ptr,
        C, MAXBLK, HBT_S0, HBW, H, MAXCTX, WV, AGE, NULLB,
        BS: tl.constexpr, BLOCK: tl.constexpr, BLOCK2: tl.constexpr,
    ):
        # MAXBLK = gpu_bt row stride (== its width), HBT_S0 / HBW = host_bt row stride / width (G-10)
        r = tl.program_id(0)
        s = tl.load(slot_ptr + r).to(tl.int64)
        n = tl.load(nvalid_ptr + r)
        step = tl.load(step_ptr)
        nmiss = 0
        nhit = 0
        nres = 0
        # ---- pass 1: classify every picked token, refresh hits
        for c0 in range(0, C, BLOCK):
            j = c0 + tl.arange(0, BLOCK)
            m = j < n
            pos = tl.load(idx_ptr + r * C + j, mask=m, other=0)
            m = m & (pos >= 0) & (pos < MAXCTX)  # garbage rows (e.g. CUDA-graph padding) never index out of range
            pos = tl.where(m, pos, 0)
            blk = pos // BS
            m = m & (blk < MAXBLK)
            off = pos - blk * BS
            gb = tl.load(gbt_ptr + r * MAXBLK + blk, mask=m, other=-1)
            res = m & (gb >= 0) & (gb != NULLB)
            nr = m & ~res
            hrow = tl.load(pos2row_ptr + s * MAXCTX + pos, mask=nr, other=-1)
            hit = nr & (hrow >= 0)
            miss = nr & (hrow < 0)
            tl.store(last_ptr + s * H + hrow, step, mask=hit)
            hb = tl.load(hbt_ptr + r * HBT_S0 + blk, mask=miss & (blk < HBW), other=0)
            code = tl.where(res, 1, tl.where(hit, 2, tl.where(miss, 3, 0)))
            src = tl.where(res, gb.to(tl.int64) * BS + off, tl.where(hit, hrow.to(tl.int64), hb.to(tl.int64) * BS + off))
            mi = miss.to(tl.int32)
            rank = tl.cumsum(mi, 0) - 1 + nmiss
            tl.store(code_ptr + r * C + j, code.to(tl.int8), mask=j < C)
            tl.store(src_ptr + r * C + j, src, mask=j < C)
            tl.store(rank_ptr + r * C + j, rank, mask=j < C)
            nmiss += tl.sum(mi, 0)
            nhit += tl.sum(hit.to(tl.int32), 0)
            nres += tl.sum(res.to(tl.int32), 0)
        tl.debug_barrier()
        # ---- pass 2: victims in ring order from the clock hand. OLD rows (idle > AGE steps) first; scan stops as
        # soon as enough are found (usually within the first BLOCK2 rows). Only if the whole window holds fewer
        # OLD rows than misses, a second scan takes rows that are merely not used in this step.
        hand = tl.load(hand_ptr + s)
        ca = nmiss * 0
        maxt = nmiss * 0 - 1
        t0 = nmiss * 0
        while (t0 < WV) & (ca < nmiss):
            t = t0 + tl.arange(0, BLOCK2)
            mt = t < WV
            row = (hand + t) % H
            lst = tl.load(last_ptr + s * H + row, mask=mt, other=step, volatile=True)
            isa = mt & (lst < step - AGE)
            ra = tl.cumsum(isa.to(tl.int32), 0) - 1 + ca
            use = isa & (ra < nmiss)
            tl.store(victim_ptr + r * C + ra, row, mask=use)
            maxt = tl.maximum(maxt, tl.max(tl.where(use, t, -1), 0))
            ca += tl.sum(isa.to(tl.int32), 0)
            t0 += BLOCK2
        na = tl.minimum(ca, nmiss)
        cb = nmiss * 0
        t0 = nmiss * 0
        while (t0 < WV) & (cb < nmiss - na):
            t = t0 + tl.arange(0, BLOCK2)
            mt = t < WV
            row = (hand + t) % H
            lst = tl.load(last_ptr + s * H + row, mask=mt, other=step, volatile=True)
            isb = mt & (lst != step) & (lst >= step - AGE)
            rb = tl.cumsum(isb.to(tl.int32), 0) - 1 + cb
            use = isb & (rb < nmiss - na)
            tl.store(victim_ptr + r * C + na + rb, row, mask=use)
            maxt = tl.maximum(maxt, tl.max(tl.where(use, t, -1), 0))
            cb += tl.sum(isb.to(tl.int32), 0)
            t0 += BLOCK2
        newhand = tl.where(maxt >= 0, (hand + maxt + 1) % H, hand)
        tl.store(hand_ptr + s, newhand)
        tl.atomic_add(stats_ptr + 0, nhit.to(tl.int64))
        tl.atomic_add(stats_ptr + 1, nmiss.to(tl.int64))
        tl.atomic_add(stats_ptr + 2, nres.to(tl.int64))

    @triton.jit
    def _tier_copy_kernel(
        idx_ptr, nvalid_ptr, slot_ptr, step_ptr, code_ptr, src_ptr, rank_ptr, victim_ptr,
        kv_ptr, host_ptr, hot_ptr, stg_ptr, sbt_ptr,
        tag_ptr, last_ptr, pos2row_ptr,
        C, NBS, H, MAXCTX, KVS,
        KV_SB, KV_SH, KV_ST, HO_SB, HO_SH, HO_ST, ST_SB, ST_SH, ST_ST,
        HKV: tl.constexpr, BS: tl.constexpr, W4: tl.constexpr, W4P: tl.constexpr, TPP: tl.constexpr,
        HOST4: tl.constexpr = False,
    ):
        # int32 element units: one head row = W4 int32 (520 B -> 130)
        # HOST4 (int4 host tier): misses were already dequantised by _tier_fill4_kernel into the victim hot row (slots
        # with hot rows) or straight into the staging row (others)
        r = tl.program_id(0)
        j0 = tl.program_id(1) * TPP
        s = tl.load(slot_ptr + r).to(tl.int64)
        n = tl.load(nvalid_ptr + r)
        step = tl.load(step_ptr)
        h = tl.arange(0, HKV)[:, None]
        e = tl.arange(0, W4P)[None, :]
        em = e < W4
        for jj in range(TPP):
            j = j0 + jj
            if j < n:
                code = tl.load(code_ptr + r * C + j).to(tl.int32)
                src = tl.load(src_ptr + r * C + j)  # int64
                sblk = src // BS
                soff = src - sblk * BS
                kv_off = sblk * KV_SB + h * KV_SH + soff * KV_ST + e  # resident rows (strided cache view)
                ho_off = sblk * HO_SB + h * HO_SH + soff * HO_ST + e  # host rows
                hot_off = (s * H + src) * (HKV * W4) + h * W4 + e
                x = tl.load(kv_ptr + kv_off, mask=em & (code == 1), other=0)
                x += tl.load(hot_ptr + hot_off, mask=em & (code == 2), other=0)
                if HOST4:
                    if (code == 3) & (s < KVS):
                        v3 = tl.load(victim_ptr + r * C + tl.load(rank_ptr + r * C + j)).to(tl.int64)
                        x += tl.load(hot_ptr + (s * H + v3) * (HKV * W4) + h * W4 + e, mask=em, other=0)
                else:
                    x += tl.load(host_ptr + ho_off, mask=em & (code == 3), other=0)
                dblk = tl.load(sbt_ptr + r * NBS + j // BS).to(tl.int64)
                doff = j % BS
                if HOST4:
                    keep = (code != 3) | (s < KVS)  # padding-slot misses were written by the fill kernel
                else:
                    keep = code >= 0
                tl.store(stg_ptr + dblk * ST_SB + h * ST_SH + doff * ST_ST + e, x, mask=em & keep)
                if (code == 3) & (s < KVS):  # slots without hot KV rows (graph padding) stage from host only
                    pos = tl.load(idx_ptr + r * C + j)
                    rank = tl.load(rank_ptr + r * C + j)
                    v = tl.load(victim_ptr + r * C + rank).to(tl.int64)
                    tl.store(hot_ptr + (s * H + v) * (HKV * W4) + h * W4 + e, x, mask=em)
                    old = tl.load(tag_ptr + s * H + v)
                    if old >= 0:
                        tl.atomic_cas(pos2row_ptr + s * MAXCTX + old, v.to(tl.int32), -1)
                    tl.store(tag_ptr + s * H + v, pos)
                    tl.store(pos2row_ptr + s * MAXCTX + pos, v.to(tl.int32))
                    tl.store(last_ptr + s * H + v, step)


class TierScratch:
    """Per-(nreq, C) scratch buffers for gather_rows_tiered (allocate once; reused across layers/steps)."""

    def __init__(self, max_reqs: int, C: int, device="cuda", glob: bool = False):
        self.code = torch.zeros((max_reqs, C), dtype=torch.int8, device=device)
        self.src = torch.zeros((max_reqs, C), dtype=torch.int64, device=device)
        self.rank = torch.zeros((max_reqs, C), dtype=torch.int32, device=device)
        self.victim = torch.zeros((max_reqs, C), dtype=torch.int32, device=device)
        if glob:  # S14 global pool: per-request cacheable-miss counts / victim-list offsets, victims found
            self.nmiss = torch.zeros(max_reqs, dtype=torch.int32, device=device)
            self.base = torch.zeros(max_reqs, dtype=torch.int32, device=device)
            self.nvict = torch.zeros(1, dtype=torch.int32, device=device)


def gather_rows_tiered(kv, gpu_bt, host_dev, host_bt, idx, n_valid, staging_kv, staging_bt, hot: HotBuffer, layer: int,
                       req_slot, scratch: TierScratch, tpp: int = 4, block: int = 0, block2: int = 1024,
                       null_block: int = -1, stats: torch.Tensor | None = None, cache_rows: bool = True):
    """Tiered version of the contract's gather_rows for one sparse layer (graph-capturable).

    kv         [NB, Hkv, BS, W] int8 GPU cache of this layer
    gpu_bt     [nreq, maxblk] int32, -1 where the block lives only on the host
    host_dev   [HB, Hkv, BS, W] int8 CUDA (UVA) view of this layer's host mirror (HostKVMirror.dev[layer])
    host_bt    [nreq, maxblk] int32 host block of every logical block that may be non-resident
    idx, n_valid  selection (contract layout; columns >= n_valid ignored)
    staging_kv [NBs, Hkv, BS, W], staging_bt [nreq, nbs]
    req_slot   [nreq] int32 hot-buffer slot of each request (a slot >= hot.kv_slots: rows come from host, nothing
               is cached -- S4 uses the padding slot this way for prefill suffixes)
    null_block block id that also means "not resident" (vLLM's null block 0 when fed a Model Runner V2 table)
    stats      int64[3] device counters [hit, miss, resident] to add to (default hot.stats)
    cache_rows S5: False = the caller guarantees every req_slot is >= hot.kv_slots (the padding slot: nothing is
               cached, misses are staged straight from host), so C may exceed hot.H. The S4 suffix path always uses the
               padding slot; requiring C <= hot.H there made SPARSE_TIER_SUFFIX_KSEL=16384 (C = 17472 > H = 6553 at
               HOT_FRAC 0.025 x 262144) die with an AssertionError inside the forward -> EngineCore fatal -> HTTP 500.
               The padding slot's pos2row is never filled (no hits), victims are computed but never written (copy
               kernel: code 3 & s < KVS is false), so no hot-buffer state is touched.
    """
    nreq, C = idx.shape
    NB, HKV, BS, W = kv.shape
    assert W % 4 == 0 and (C <= hot.H or not cache_rows), (C, hot.H)
    # G-10: host_bt may be wider/padded (own row stride); gpu_bt row stride is its width; idx rows are C apart
    assert gpu_bt.stride(1) == 1 and gpu_bt.stride(0) == gpu_bt.shape[1], "gpu_bt rows must be dense"
    assert host_bt.stride(1) == 1 and host_bt.shape[1] >= gpu_bt.shape[1], "host_bt narrower than gpu_bt"
    assert idx.is_contiguous()
    hl = hot.layer(layer)
    WV = min(2 * C, hot.H)
    if block <= 0:
        block = min(4096, triton.next_power_of_2(C))
    _tier_resolve_kernel[(nreq,)](
        idx, n_valid, gpu_bt, host_bt, req_slot, hot.step,
        hl.pos2row, hl.last, hl.hand,
        scratch.code, scratch.src, scratch.rank, scratch.victim, hot.stats if stats is None else stats,
        C, gpu_bt.stride(0), host_bt.stride(0), host_bt.shape[1], hot.H, hot.max_ctx, WV, hot.age, null_block,
        BS=BS, BLOCK=block, BLOCK2=block2, num_warps=8,
    )
    W4 = W // 4
    host4 = host_dev.shape[-1] == W4H
    if host4:
        hk = hl.kv
        _tier_fill4_kernel[(nreq, triton.cdiv(C, tpp))](
            n_valid, req_slot, scratch.code, scratch.src, scratch.rank, scratch.victim,
            host_dev, host_dev.stride(0), host_dev.stride(1), host_dev.stride(2), hk, staging_kv,
            staging_kv.stride(0), staging_kv.stride(1), staging_kv.stride(2), staging_bt,
            C, staging_bt.stride(0), hot.H, hot.kv_slots, HKV=HKV, BS=BS, W=W, TPP=tpp, num_warps=4)
    kv4, ho4, st4 = kv.view(torch.int32), host_dev.view(torch.int32), staging_kv.view(torch.int32)
    for t in (kv4, ho4, st4):
        assert t.stride(3) == 1, "token-head rows must be contiguous"
    _tier_copy_kernel[(nreq, triton.cdiv(C, tpp))](
        idx, n_valid, req_slot, hot.step, scratch.code, scratch.src, scratch.rank, scratch.victim,
        kv4, ho4, hl.kv.view(torch.int32), st4,
        staging_bt, hl.tag, hl.last, hl.pos2row,
        C, staging_bt.stride(0), hot.H, hot.max_ctx, hot.kv_slots,
        kv4.stride(0), kv4.stride(1), kv4.stride(2), ho4.stride(0), ho4.stride(1), ho4.stride(2),
        st4.stride(0), st4.stride(1), st4.stride(2),
        HKV=HKV, BS=BS, W4=W4, W4P=triton.next_power_of_2(W4), TPP=tpp, HOST4=host4, num_warps=4,
    )


# ----------------------------------------------------------------------------------------------------------------
# S14 global hot pool: Triton kernels (see the S14 block above gather_rows_global_ref for the design).
# GPU check: sparsekv/test_s14_hot_gpu.py; CPU reference check: sparsekv/test_s14_hot_cpu.py.
# Codes written by the resolve kernel: 1 resident, 2 hot hit, 3 host miss that may be cached (rank = its index among
# the request's cacheable misses), 4 host miss that is never cached (uncached request, unmapped / out-of-range key).
# ----------------------------------------------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _tier_resolve_g_kernel(
        idx_ptr, nvalid_ptr, gbt_ptr, hbt_ptr, slot_ptr, step_ptr,
        pos2row_ptr, last_ptr,
        code_ptr, src_ptr, rank_ptr, nmiss_ptr, stats_ptr,
        C, MAXBLK, HBT_S0, HBW, NK, CSLOTS, NULLB,
        BS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # one program per request: classify, refresh hits (pool = slot 0), count cacheable misses. No victims here:
        # every request's hits must be marked before the single victim scan (_tier_victims_g_kernel).
        r = tl.program_id(0)
        cached = tl.load(slot_ptr + r) < CSLOTS
        n = tl.load(nvalid_ptr + r)
        step = tl.load(step_ptr)
        ncm = 0
        nmiss = 0
        nhit = 0
        nres = 0
        for c0 in range(0, C, BLOCK):
            j = c0 + tl.arange(0, BLOCK)
            m = j < n
            pos = tl.load(idx_ptr + r * C + j, mask=m, other=0)
            m = m & (pos >= 0)
            pos = tl.where(m, pos, 0)
            blk = pos // BS
            m = m & (blk < MAXBLK)
            off = pos - blk * BS
            gb = tl.load(gbt_ptr + r * MAXBLK + blk, mask=m, other=-1)
            res = m & (gb >= 0) & (gb != NULLB)
            nr = m & ~res
            hb = tl.load(hbt_ptr + r * HBT_S0 + blk, mask=nr & (blk < HBW), other=-1)
            key = hb.to(tl.int64) * BS + off
            kok = nr & (hb >= 0) & (key < NK) & cached
            hrow = tl.load(pos2row_ptr + key, mask=kok, other=-1)
            hit = kok & (hrow >= 0)
            miss = nr & ~hit
            cm = miss & kok
            tl.store(last_ptr + hrow, step, mask=hit)
            code = tl.where(res, 1, tl.where(hit, 2, tl.where(cm, 3, tl.where(miss, 4, 0))))
            hsrc = tl.where(hb >= 0, key, off.to(tl.int64))  # unmapped block: read host block 0 (in range), as before
            src = tl.where(res, gb.to(tl.int64) * BS + off, tl.where(hit, hrow.to(tl.int64), hsrc))
            ci = cm.to(tl.int32)
            rank = tl.cumsum(ci, 0) - 1 + ncm
            tl.store(code_ptr + r * C + j, code.to(tl.int8), mask=j < C)
            tl.store(src_ptr + r * C + j, src, mask=j < C)
            tl.store(rank_ptr + r * C + j, rank, mask=j < C)
            ncm += tl.sum(ci, 0)
            nmiss += tl.sum(miss.to(tl.int32), 0)
            nhit += tl.sum(hit.to(tl.int32), 0)
            nres += tl.sum(res.to(tl.int32), 0)
        tl.store(nmiss_ptr + r, ncm)
        tl.atomic_add(stats_ptr + 0, nhit.to(tl.int64))
        tl.atomic_add(stats_ptr + 1, nmiss.to(tl.int64))
        tl.atomic_add(stats_ptr + 2, nres.to(tl.int64))

    @triton.jit
    def _tier_victims_g_kernel(
        nmiss_ptr, base_ptr, nvict_ptr, step_ptr, last_ptr, hand_ptr, victim_ptr,
        NREQ, H, WV, AGE,
        NRP: tl.constexpr, BLOCK2: tl.constexpr,
    ):
        # ONE program for the batch: prefix-sum the requests' cacheable misses (base[r] = first entry of request r in
        # the victim list), then the same two-pass windowed CLOCK scan as _tier_resolve_kernel, over the pool.
        rr = tl.arange(0, NRP)
        cnt = tl.load(nmiss_ptr + rr, mask=rr < NREQ, other=0)
        cs = tl.cumsum(cnt, 0)
        tl.store(base_ptr + rr, cs - cnt, mask=rr < NREQ)
        nmiss = tl.sum(cnt, 0)
        step = tl.load(step_ptr)
        hand = tl.load(hand_ptr)
        ca = nmiss * 0
        maxt = nmiss * 0 - 1
        t0 = nmiss * 0
        while (t0 < WV) & (ca < nmiss):
            t = t0 + tl.arange(0, BLOCK2)
            mt = t < WV
            row = (hand + t) % H
            lst = tl.load(last_ptr + row, mask=mt, other=step, volatile=True)
            isa = mt & (lst < step - AGE)
            ra = tl.cumsum(isa.to(tl.int32), 0) - 1 + ca
            use = isa & (ra < nmiss)
            tl.store(victim_ptr + ra, row, mask=use)
            maxt = tl.maximum(maxt, tl.max(tl.where(use, t, -1), 0))
            ca += tl.sum(isa.to(tl.int32), 0)
            t0 += BLOCK2
        na = tl.minimum(ca, nmiss)
        cb = nmiss * 0
        t0 = nmiss * 0
        while (t0 < WV) & (cb < nmiss - na):
            t = t0 + tl.arange(0, BLOCK2)
            mt = t < WV
            row = (hand + t) % H
            lst = tl.load(last_ptr + row, mask=mt, other=step, volatile=True)
            isb = mt & (lst != step) & (lst >= step - AGE)
            rb = tl.cumsum(isb.to(tl.int32), 0) - 1 + cb
            use = isb & (rb < nmiss - na)
            tl.store(victim_ptr + na + rb, row, mask=use)
            maxt = tl.maximum(maxt, tl.max(tl.where(use, t, -1), 0))
            cb += tl.sum(isb.to(tl.int32), 0)
            t0 += BLOCK2
        newhand = tl.where(maxt >= 0, (hand + maxt + 1) % H, hand)
        tl.store(hand_ptr, newhand)
        tl.store(nvict_ptr, na + tl.minimum(cb, nmiss - na))  # victims found (< nmiss: the rest stays uncached)

    @triton.jit
    def _tier_fill4_g_kernel(nvalid_ptr, code_ptr, src_ptr, rank_ptr, victim_ptr, base_ptr, nvict_ptr,
                             host_ptr, HO_SB, HO_SH, HO_ST, hot_ptr, stg_ptr, ST_SB, ST_SH, ST_ST, sbt_ptr,
                             C, NBS, HKV: tl.constexpr, BS: tl.constexpr, W: tl.constexpr, TPP: tl.constexpr):
        # int4 host tier: dequantise host misses into their victim pool row (code 3 with a victim) or straight into
        # the staging row (code 4, or code 3 beyond the victims found)
        r = tl.program_id(0)
        j0 = tl.program_id(1) * TPP
        n = tl.load(nvalid_ptr + r)
        base = tl.load(base_ptr + r)
        nv = tl.load(nvict_ptr)
        h = tl.arange(0, HKV)[:, None]
        m = h >= 0
        for jj in range(TPP):
            j = j0 + jj
            if j < n:
                code = tl.load(code_ptr + r * C + j).to(tl.int32)
                if code >= 3:
                    sv = tl.load(src_ptr + r * C + j)
                    sblk = sv // BS
                    soff = sv - sblk * BS
                    src = host_ptr + sblk * HO_SB + h * HO_SH + soff * HO_ST
                    g = base + tl.load(rank_ptr + r * C + j)
                    if (code == 3) & (g < nv):
                        v = tl.load(victim_ptr + g).to(tl.int64)
                        dst = hot_ptr + v * (HKV * W) + h * W
                    else:
                        dblk = tl.load(sbt_ptr + r * NBS + j // BS).to(tl.int64)
                        dst = stg_ptr + dblk * ST_SB + h * ST_SH + (j % BS) * ST_ST
                    _dq4_half(src, dst, m, HKV)
                    _dq4_half(src + 136, dst + 260, m, HKV)

    @triton.jit
    def _tier_copy_g_kernel(
        nvalid_ptr, step_ptr, code_ptr, src_ptr, rank_ptr, victim_ptr, base_ptr, nvict_ptr,
        kv_ptr, host_ptr, hot_ptr, stg_ptr, sbt_ptr,
        tag_ptr, last_ptr, pos2row_ptr,
        C, NBS,
        KV_SB, KV_SH, KV_ST, HO_SB, HO_SH, HO_ST, ST_SB, ST_SH, ST_ST,
        HKV: tl.constexpr, BS: tl.constexpr, W4: tl.constexpr, W4P: tl.constexpr, TPP: tl.constexpr,
        HOST4: tl.constexpr = False,
    ):
        # _tier_copy_kernel for the pool: rows are pool rows (slot 0), tags / pos2row are keyed by the host token
        # (= src of a miss), and a code-3 miss is cached only if the victim scan reached it (base + rank < nvict)
        r = tl.program_id(0)
        j0 = tl.program_id(1) * TPP
        n = tl.load(nvalid_ptr + r)
        step = tl.load(step_ptr)
        base = tl.load(base_ptr + r)
        nv = tl.load(nvict_ptr)
        h = tl.arange(0, HKV)[:, None]
        e = tl.arange(0, W4P)[None, :]
        em = e < W4
        for jj in range(TPP):
            j = j0 + jj
            if j < n:
                code = tl.load(code_ptr + r * C + j).to(tl.int32)
                src = tl.load(src_ptr + r * C + j)  # int64
                sblk = src // BS
                soff = src - sblk * BS
                kv_off = sblk * KV_SB + h * KV_SH + soff * KV_ST + e
                ho_off = sblk * HO_SB + h * HO_SH + soff * HO_ST + e
                g = base + tl.load(rank_ptr + r * C + j)
                put = (code == 3) & (g < nv)  # this miss gets a pool row
                v = tl.load(victim_ptr + g, mask=put, other=0).to(tl.int64)
                x = tl.load(kv_ptr + kv_off, mask=em & (code == 1), other=0)
                x += tl.load(hot_ptr + src * (HKV * W4) + h * W4 + e, mask=em & (code == 2), other=0)
                if HOST4:
                    # the fill kernel dequantised cached misses into pool row v, the others into the staging row
                    x += tl.load(hot_ptr + v * (HKV * W4) + h * W4 + e, mask=em & put, other=0)
                    keep = (code < 3) | put
                else:
                    x += tl.load(host_ptr + ho_off, mask=em & (code >= 3), other=0)
                    keep = code >= 0
                dblk = tl.load(sbt_ptr + r * NBS + j // BS).to(tl.int64)
                doff = j % BS
                tl.store(stg_ptr + dblk * ST_SB + h * ST_SH + doff * ST_ST + e, x, mask=em & keep)
                if put:
                    tl.store(hot_ptr + v * (HKV * W4) + h * W4 + e, x, mask=em)
                    old = tl.load(tag_ptr + v)
                    if old >= 0:
                        tl.atomic_cas(pos2row_ptr + old, v.to(tl.int32), -1)
                    tl.store(tag_ptr + v, src.to(tl.int32))
                    tl.store(pos2row_ptr + src, v.to(tl.int32))
                    tl.store(last_ptr + v, step)


def gather_rows_tiered_global(kv, gpu_bt, host_dev, host_bt, idx, n_valid, staging_kv, staging_bt, hot: HotBuffer,
                              layer: int, req_slot, scratch: TierScratch, cached_slots: int = 1, tpp: int = 4,
                              block: int = 0, block2: int = 1024, null_block: int = -1,
                              stats: torch.Tensor | None = None):
    """gather_rows_tiered for the S14 GLOBAL hot pool (graph-capturable: three or four Triton launches with shapes
    depending only on (nreq, C); no host sync).

    hot          HotBuffer(num_slots=1, kv_slots=1, max_ctx=<host-pool tokens>): pool rows keyed by host token
    host_bt      [nreq, >= maxblk] host block of every logical block (-1 = none); REQUIRED for every non-resident pick
    req_slot     [nreq] int32; rows with req_slot < cached_slots use the pool, the others never touch it
    scratch      TierScratch(..., glob=True) with at least nreq rows
    C may exceed hot.H (misses beyond the victims found are staged straight from host).
    """
    nreq, C = idx.shape
    NB, HKV, BS, W = kv.shape
    assert hot.S == 1 and hot.kv_slots == 1, "global pool: HotBuffer with one slot"
    assert W % 4 == 0 and nreq <= scratch.nmiss.numel() and nreq * C <= scratch.victim.numel()
    assert gpu_bt.stride(1) == 1 and gpu_bt.stride(0) == gpu_bt.shape[1], "gpu_bt rows must be dense"
    assert host_bt.stride(1) == 1 and host_bt.shape[1] >= gpu_bt.shape[1], "host_bt narrower than gpu_bt"
    assert idx.is_contiguous()
    hl = hot.layer(layer)
    if block <= 0:
        block = min(4096, triton.next_power_of_2(C))
    _tier_resolve_g_kernel[(nreq,)](
        idx, n_valid, gpu_bt, host_bt, req_slot, hot.step,
        hl.pos2row, hl.last,
        scratch.code, scratch.src, scratch.rank, scratch.nmiss, hot.stats if stats is None else stats,
        C, gpu_bt.stride(0), host_bt.stride(0), host_bt.shape[1], hot.max_ctx, cached_slots, null_block,
        BS=BS, BLOCK=block, num_warps=8,
    )
    _tier_victims_g_kernel[(1,)](
        scratch.nmiss, scratch.base, scratch.nvict, hot.step, hl.last, hl.hand, scratch.victim,
        nreq, hot.H, global_window(hot.H, nreq, C), hot.age,
        NRP=triton.next_power_of_2(max(nreq, 2)), BLOCK2=block2, num_warps=8,
    )
    W4 = W // 4
    host4 = host_dev.shape[-1] == W4H
    if host4:
        _tier_fill4_g_kernel[(nreq, triton.cdiv(C, tpp))](
            n_valid, scratch.code, scratch.src, scratch.rank, scratch.victim, scratch.base, scratch.nvict,
            host_dev, host_dev.stride(0), host_dev.stride(1), host_dev.stride(2), hl.kv, staging_kv,
            staging_kv.stride(0), staging_kv.stride(1), staging_kv.stride(2), staging_bt,
            C, staging_bt.stride(0), HKV=HKV, BS=BS, W=W, TPP=tpp, num_warps=4)
    kv4, ho4, st4 = kv.view(torch.int32), host_dev.view(torch.int32), staging_kv.view(torch.int32)
    for t in (kv4, ho4, st4):
        assert t.stride(3) == 1, "token-head rows must be contiguous"
    _tier_copy_g_kernel[(nreq, triton.cdiv(C, tpp))](
        n_valid, hot.step, scratch.code, scratch.src, scratch.rank, scratch.victim, scratch.base, scratch.nvict,
        kv4, ho4, hl.kv.view(torch.int32), st4,
        staging_bt, hl.tag, hl.last, hl.pos2row,
        C, staging_bt.stride(0),
        kv4.stride(0), kv4.stride(1), kv4.stride(2), ho4.stride(0), ho4.stride(1), ho4.stride(2),
        st4.stride(0), st4.stride(1), st4.stride(2),
        HKV=HKV, BS=BS, W4=W4, W4P=triton.next_power_of_2(W4), TPP=tpp, HOST4=host4, num_warps=4,
    )


# ----------------------------------------------------------------------------------------------------------------
# Track B adapter: HiSparse physical indices (token-major BLNHC rows) -> HND staging for stock pthfa
# ----------------------------------------------------------------------------------------------------------------
def gather_physical_rows_ref(rows, phys, n_valid, staging_kv, staging_bt):
    """rows [R, Hkv*W] int8 (HiSparse resident+hot cache viewed as token rows), phys [nreq, C] int32 physical row
    per picked column (-1 = invalid), staging as in the contract. Column j -> staging_bt[r, j//BS], j%BS."""
    nreq, C = phys.shape
    _, HKV, BS, W = staging_kv.shape
    for r in range(nreq):
        n = int(n_valid[r])
        p = phys[r, :n].long()
        ok = p >= 0
        j = torch.arange(n, device=phys.device)[ok]
        staging_kv[staging_bt[r].long()[j // BS], :, j % BS] = rows[p[ok]].view(-1, HKV, W)


if triton is not None:

    @triton.jit
    def _gather_physical_kernel(rows_ptr, phys_ptr, nvalid_ptr, stg_ptr, sbt_ptr, C, NBS,
                                HKV: tl.constexpr, BS: tl.constexpr, W4: tl.constexpr, W4P: tl.constexpr,
                                TPP: tl.constexpr):
        r = tl.program_id(0)
        j0 = tl.program_id(1) * TPP
        n = tl.load(nvalid_ptr + r)
        h = tl.arange(0, HKV)[:, None]
        e = tl.arange(0, W4P)[None, :]
        em = e < W4
        for jj in range(TPP):
            j = j0 + jj
            if j < n:
                p = tl.load(phys_ptr + r * C + j).to(tl.int64)
                if p >= 0:
                    x = tl.load(rows_ptr + p * (HKV * W4) + h * W4 + e, mask=em)
                    dblk = tl.load(sbt_ptr + r * NBS + j // BS).to(tl.int64)
                    tl.store(stg_ptr + dblk * (HKV * BS * W4) + h * (BS * W4) + (j % BS) * W4 + e, x, mask=em)


def gather_physical_rows(rows, phys, n_valid, staging_kv, staging_bt, tpp: int = 4):
    """Graph-safe Triton version of gather_physical_rows_ref (rows must be 4-byte aligned per token-head: W % 4 == 0)."""
    nreq, C = phys.shape
    _, HKV, BS, W = staging_kv.shape
    _gather_physical_kernel[(nreq, triton.cdiv(C, tpp))](
        rows.view(torch.int32), phys, n_valid, staging_kv.view(torch.int32), staging_bt, C, staging_bt.stride(0),
        HKV=HKV, BS=BS, W4=W // 4, W4P=triton.next_power_of_2(W // 4), TPP=tpp, num_warps=4)


# ----------------------------------------------------------------------------------------------------------------
# S4: batched whole-block copies (prefill staging) without host syncs or per-block launches
# ----------------------------------------------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _copy_blocks_kernel(dst_ptr, dst_ids_ptr, src_ptr, src_ids_ptr, D_SB, D_SH, D_ST, S_SB, S_SH, S_ST,
                            BS: tl.constexpr, W4: tl.constexpr, W4P: tl.constexpr, TB: tl.constexpr):
        i = tl.program_id(0)
        h = tl.program_id(1)
        t0 = tl.program_id(2) * TB
        db = tl.load(dst_ids_ptr + i).to(tl.int64)
        sb = tl.load(src_ids_ptr + i).to(tl.int64)
        t = t0 + tl.arange(0, TB)[:, None]
        e = tl.arange(0, W4P)[None, :]
        m = (t < BS) & (e < W4)
        x = tl.load(src_ptr + sb * S_SB + h * S_SH + t * S_ST + e, mask=m)
        tl.store(dst_ptr + db * D_SB + h * D_SH + t * D_ST + e, x, mask=m)


def copy_blocks(dst, dst_ids, src, src_ids, tb: int = 32):
    """dst[dst_ids[i]] = src[src_ids[i]] for whole [Hkv, BS, W] int8 blocks (strided block views allowed; src may be a
    UVA view of pinned host memory). One launch, no temporaries, no host sync."""
    n = dst_ids.numel()
    if n == 0:
        return
    _, HKV, BS, W = dst.shape
    d4, s4 = dst.view(torch.int32), src.view(torch.int32)
    assert d4.stride(3) == 1 and s4.stride(3) == 1
    W4 = W // 4
    _copy_blocks_kernel[(n, HKV, triton.cdiv(BS, tb))](
        d4, dst_ids, s4, src_ids, d4.stride(0), d4.stride(1), d4.stride(2), s4.stride(0), s4.stride(1), s4.stride(2),
        BS=BS, W4=W4, W4P=triton.next_power_of_2(W4), TB=tb, num_warps=4)


# ----------------------------------------------------------------------------------------------------------------
# S4: int4 host KV tier (SPARSE_TIER_HOST_BITS=4). Host row per token-head (W4H = 272 B) for each of K and V:
#   128 B packed nibbles (byte j = dim j low | dim j + 128 high) + 4 fp16 scales of 64-dim groups = 136 B.
# Made from the pthfa int8 row [256 int8 | fp32 scale] (x 2) by group requantisation (q4 = rint(q8 * 7 / amax_g),
# s_g = s8 * amax_g / 7) and turned back into a valid pthfa int8 row on fetch (s8' = max_g s_g * 7 / 127,
# q8' = rint(q4 * s_g / s8')), so the attention kernels and the GPU cache layout are unchanged.
# ----------------------------------------------------------------------------------------------------------------
W4H = 272


def host_bits() -> int:
    import os
    b = int(os.environ.get("SPARSE_TIER_HOST_BITS", "8"))
    if b not in (4, 8):
        raise ValueError(f"SPARSE_TIER_HOST_BITS={b} (expected 4 or 8)")
    return b


def host_row_width(w: int) -> int:
    """Host bytes per token-head row for a GPU row of w bytes (520 for D=256)."""
    return w if host_bits() == 8 else W4H


def quant_rows4_ref(rows: torch.Tensor) -> torch.Tensor:
    """[..., 520] int8 pthfa rows -> [..., 272] int8 host rows (reference)."""
    out = torch.empty((*rows.shape[:-1], W4H), dtype=torch.int8, device=rows.device)
    for hf in range(2):
        v = rows[..., hf * 260: hf * 260 + 256].to(torch.int32)
        s = rows[..., hf * 260 + 256: hf * 260 + 260].contiguous().view(torch.float32)[..., 0]
        g = v.view(*v.shape[:-1], 4, 64)
        a = g.abs().amax(-1).clamp_min(1)
        q = torch.round(g.float() * 7.0 / a[..., None]).clamp(-7, 7).to(torch.int32).view(*v.shape)
        packed = (q[..., 128:] * 16 + (q[..., :128] & 15)).to(torch.int8)
        sg = (s[..., None] * a.float() / 7.0).to(torch.float16)
        out[..., hf * 136: hf * 136 + 128] = packed
        out[..., hf * 136 + 128: hf * 136 + 136] = sg.contiguous().view(torch.int8)
    return out


def dequant_rows4_ref(h: torch.Tensor) -> torch.Tensor:
    """[..., 272] int8 host rows -> [..., 520] int8 pthfa rows (reference)."""
    out = torch.empty((*h.shape[:-1], 520), dtype=torch.int8, device=h.device)
    for hf in range(2):
        b = h[..., hf * 136: hf * 136 + 128].to(torch.int32)
        q = torch.cat([(b << 28) >> 28, b >> 4], -1)                                  # [..., 256]
        sg = h[..., hf * 136 + 128: hf * 136 + 136].contiguous().view(torch.float16).float()  # [..., 4]
        smax = sg.amax(-1)
        s8 = torch.where(smax > 0, smax * 7.0 / 127.0, torch.ones_like(smax))
        f = (sg / s8[..., None]).repeat_interleave(64, -1)
        q8 = torch.round(q.float() * f).clamp(-127, 127).to(torch.int8)
        out[..., hf * 260: hf * 260 + 256] = q8
        out[..., hf * 260 + 256: hf * 260 + 260] = s8.to(torch.float32)[..., None].contiguous().view(torch.int8)
    return out


if triton is not None:

    @triton.jit
    def _q4_half(src, dst, m, TB: tl.constexpr):
        # src, dst: [R, 1] int8 pointer bases of one half (src: 256 values + fp32 scale; dst: 128 B + 4 fp16)
        d = tl.arange(0, 128)[None, :]
        lo = tl.load(src + d, mask=m, other=0).to(tl.float32)
        hi = tl.load(src + 128 + d, mask=m, other=0).to(tl.float32)
        s = tl.load((src + 256).to(tl.pointer_type(tl.float32)), mask=m, other=0.0)        # [R, 1]
        lo3 = tl.reshape(lo, [TB, 2, 64])
        hi3 = tl.reshape(hi, [TB, 2, 64])
        alo = tl.maximum(tl.max(tl.abs(lo3), 2), 1.0)                                      # [R, 2]
        ahi = tl.maximum(tl.max(tl.abs(hi3), 2), 1.0)
        qlo = tl.minimum(tl.maximum(libdevice.rint(lo3 * 7.0 / alo[:, :, None]), -7.0), 7.0)
        qhi = tl.minimum(tl.maximum(libdevice.rint(hi3 * 7.0 / ahi[:, :, None]), -7.0), 7.0)
        qlo = tl.reshape(qlo, [TB, 128]).to(tl.int32)
        qhi = tl.reshape(qhi, [TB, 128]).to(tl.int32)
        tl.store(dst + d, (qhi * 16 + (qlo & 15)).to(tl.int8), mask=m)
        g = tl.arange(0, 2)[None, :]
        sp = (dst + 128).to(tl.pointer_type(tl.float16))
        tl.store(sp + g, (s * alo / 7.0).to(tl.float16), mask=m)
        tl.store(sp + 2 + g, (s * ahi / 7.0).to(tl.float16), mask=m)

    @triton.jit
    def _dq4_half(src, dst, m, R: tl.constexpr):
        # inverse of _q4_half: src [R,1] base of 136 B, dst [R,1] base of a 260 B pthfa half row
        d = tl.arange(0, 128)[None, :]
        x = tl.load(src + d, mask=m, other=0).to(tl.int32)
        lo = ((x << 28) >> 28).to(tl.float32)
        hi = (x >> 4).to(tl.float32)
        g = tl.arange(0, 2)[None, :]
        sp = (src + 128).to(tl.pointer_type(tl.float16))
        s01 = tl.load(sp + g, mask=m, other=0.0).to(tl.float32)                            # [R, 2]
        s23 = tl.load(sp + 2 + g, mask=m, other=0.0).to(tl.float32)
        smax = tl.maximum(tl.max(s01, 1), tl.max(s23, 1))                                  # [R]
        s8 = tl.where(smax > 0, smax * 7.0 / 127.0, 1.0)
        f01 = s01 / s8[:, None]
        f23 = s23 / s8[:, None]
        q0 = tl.reshape(tl.reshape(lo, [R, 2, 64]) * f01[:, :, None], [R, 128])
        q1 = tl.reshape(tl.reshape(hi, [R, 2, 64]) * f23[:, :, None], [R, 128])
        q0 = tl.minimum(tl.maximum(libdevice.rint(q0), -127.0), 127.0).to(tl.int8)
        q1 = tl.minimum(tl.maximum(libdevice.rint(q1), -127.0), 127.0).to(tl.int8)
        tl.store(dst + d, q0, mask=m)
        tl.store(dst + 128 + d, q1, mask=m)
        tl.store((dst + 256).to(tl.pointer_type(tl.float32)), s8[:, None], mask=m)

    @triton.jit
    def _quant4_blocks_kernel(src_ptr, S0, S1, S2, src_ids_ptr, dst_ptr, D0, D1, D2, dst_ids_ptr,
                              BS: tl.constexpr, TB: tl.constexpr):
        i = tl.program_id(0)
        h = tl.program_id(1)
        t = tl.program_id(2) * TB + tl.arange(0, TB)[:, None]
        m = t < BS
        sb = tl.load(src_ids_ptr + i).to(tl.int64)
        db = tl.load(dst_ids_ptr + i).to(tl.int64)
        src = src_ptr + sb * S0 + h * S1 + t * S2
        dst = dst_ptr + db * D0 + h * D1 + t * D2
        _q4_half(src, dst, m, TB)
        _q4_half(src + 260, dst + 136, m, TB)

    @triton.jit
    def _dequant4_blocks_kernel(src_ptr, S0, S1, S2, src_ids_ptr, dst_ptr, D0, D1, D2, dst_ids_ptr,
                                BS: tl.constexpr, TB: tl.constexpr):
        i = tl.program_id(0)
        h = tl.program_id(1)
        t = tl.program_id(2) * TB + tl.arange(0, TB)[:, None]
        m = t < BS
        sb = tl.load(src_ids_ptr + i).to(tl.int64)
        db = tl.load(dst_ids_ptr + i).to(tl.int64)
        src = src_ptr + sb * S0 + h * S1 + t * S2
        dst = dst_ptr + db * D0 + h * D1 + t * D2
        _dq4_half(src, dst, m, TB)
        _dq4_half(src + 136, dst + 260, m, TB)

    @triton.jit
    def _tier_fill4_kernel(nvalid_ptr, slot_ptr, code_ptr, src_ptr, rank_ptr, victim_ptr,
                           host_ptr, HO_SB, HO_SH, HO_ST, hot_ptr, stg_ptr, ST_SB, ST_SH, ST_ST, sbt_ptr,
                           C, NBS, H, KVS, HKV: tl.constexpr, BS: tl.constexpr, W: tl.constexpr, TPP: tl.constexpr):
        # host misses (code 3) of an int4 host tier: dequantise into the victim hot row (or, for slots without hot
        # rows, straight into the staging row); the copy kernel then reads them from there (HOST4)
        r = tl.program_id(0)
        j0 = tl.program_id(1) * TPP
        s = tl.load(slot_ptr + r).to(tl.int64)
        n = tl.load(nvalid_ptr + r)
        h = tl.arange(0, HKV)[:, None]
        m = h >= 0
        for jj in range(TPP):
            j = j0 + jj
            if j < n:
                code = tl.load(code_ptr + r * C + j).to(tl.int32)
                if code == 3:
                    sv = tl.load(src_ptr + r * C + j)
                    sblk = sv // BS
                    soff = sv - sblk * BS
                    src = host_ptr + sblk * HO_SB + h * HO_SH + soff * HO_ST
                    if s < KVS:
                        v = tl.load(victim_ptr + r * C + tl.load(rank_ptr + r * C + j)).to(tl.int64)
                        dst = hot_ptr + (s * H + v) * (HKV * W) + h * W
                    else:
                        dblk = tl.load(sbt_ptr + r * NBS + j // BS).to(tl.int64)
                        dst = stg_ptr + dblk * ST_SB + h * ST_SH + (j % BS) * ST_ST
                    _dq4_half(src, dst, m, HKV)
                    _dq4_half(src + 136, dst + 260, m, HKV)


def quant_blocks_to_host4(src_kv, src_ids, host4, dst_ids, tb: int = 16):
    """host4[dst_ids[i]] = int4(src_kv[src_ids[i]]) for whole blocks; host4 is a UVA view [HB, Hkv, BS, 272]."""
    n = src_ids.numel()
    if n == 0:
        return
    _, HKV, BS, W = src_kv.shape
    assert W == 520 and host4.shape[-1] == W4H and src_kv.stride(3) == 1 and host4.stride(3) == 1
    _quant4_blocks_kernel[(n, HKV, triton.cdiv(BS, tb))](
        src_kv, src_kv.stride(0), src_kv.stride(1), src_kv.stride(2), src_ids,
        host4, host4.stride(0), host4.stride(1), host4.stride(2), dst_ids, BS=BS, TB=tb, num_warps=4)


def copy_blocks_from_host4(dst, dst_ids, host4, src_ids, tb: int = 16):
    """dst[dst_ids[i]] = pthfa int8 rows of host4[src_ids[i]] (dense prefill staging from an int4 host tier)."""
    n = src_ids.numel()
    if n == 0:
        return
    _, HKV, BS, W = dst.shape
    assert W == 520 and host4.shape[-1] == W4H and dst.stride(3) == 1 and host4.stride(3) == 1
    _dequant4_blocks_kernel[(n, HKV, triton.cdiv(BS, tb))](
        host4, host4.stride(0), host4.stride(1), host4.stride(2), src_ids,
        dst, dst.stride(0), dst.stride(1), dst.stride(2), dst_ids, BS=BS, TB=tb, num_warps=4)
