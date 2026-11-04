#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Sparse-KV release launcher (sk-release). Same dense baseline config as the stock HyperQwen launcher (PTH_FA=1 int8
# KV + MTP-7), plus query-aware sparse decode (SPARSE_KV=1) and, by default, the system-RAM KV tier (SPARSE_TIER=1).
# Install first: bash sparsekv/install.sh <repo>/venv. Read sparsekv/RELEASE.md (knobs, limits); README.md has uninstall/rollback.
#
# Defaults: accuracy-first knobs (the fixed configuration, docs/CHANGELOG.md):
#   SPARSE_TIER_SUFFIX_SPARSE=0  follow-up prefill is exact (dense over the staged context), not sparse
#   SPARSE_STATS_SUB=1           exact softmax normaliser in selection
#   SPARSE_SELECT_EVERY=1        re-select every decode step
#   SPARSE_TIER_STAGE_SERIAL=1   crash-safe staging: each dense prefill request is staged alone (bounded memory)
# Experimental, default off: SPARSE_TIER_SUFFIX_EXACT=1 (streamed exact follow-up path), SPARSE_TIER_S5_DMA=1
# (copy-engine prefetch for staging; int4 host copy only; untested at scale).
# v1.1 defaults (validated on GPU, docs/CHANGELOG.md), each overridable from the environment:
#   SPARSE_TIER_HOST_MAMBA_SLOTS=32       host GDN checkpoint slots (was 8): no full re-prefill after long tool waits
#   SPARSE_TIER_MAMBA_SAVE_AT_FINISH=1    host-save the last decode-boundary GDN state of each request
#   SPEC_NO_BLOCK_DROP=1                  keep the trailing prefix-cache block on a hit (MTP eagle block drop off)
#   PTHFA_PREFILL_FA2=1                   prefill attention through FlashAttention-2 over the int8 cache
#   SPARSE_PREFILL_YIELD_SHORT_BYPASS=2600  short follow-ups (<= 2600 uncached tokens) skip the prefill yield
#   SPARSE_TIER_PIN_PREFIX_MAX=2          up to 2 pinned warm prefixes (requests with vllm_xargs sparse_pin_prefix)
# Off by default: PTHFA_FLAT_GQA=1 (flat GQA rows in pthfa prefill; small speedup, not bit-exact).
#
# Defaults are a DEV server: 127.0.0.1:8093. Production would set BIND=0.0.0.0 PORT=8080 HQ_GPU=0 HQ_TMP=... in its unit.
# The GPU is required (HQ_GPU or CUDA_VISIBLE_DEVICES): nothing is picked for you.
#
# Profiles (SPARSE_PROFILE):
#   tier   (default) sparse decode + host-RAM tier with pressure spill: MAX_LEN 200000, MAX_SEQS 2, GPU_UTIL 0.90.
#          2 slots validated on the release build (3090, 3 runs: 177K + 120K users, forced spill + preemption, GPU
#          prefix cache wiped between follow-ups -> host KV + host GDN checkpoint restore, all PASS). GPU pool 71K
#          tokens at 2 slots (111K at MAX_SEQS=1), host pool ~14 GB pinned. Spill forces always-sparse decode.
#   sparse sparse decode only, no tier: MAX_LEN 153600, MAX_SEQS 2, GPU_UTIL 0.92 (production's window and slots;
#          stream I validated 0.92 = 184K pool with the reservation). Per-step dense/sparse switch (sk-switch) on:
#          short contexts replay the dense graphs, long ones the sparse graphs.
# The sparse buffers (sketches, staging, tier hot buffer) are reserved out of the KV budget (SPARSE_MEM_RESERVE=1,
# review G-1/G-17), so GPU_UTIL can sit near production's 0.93; the pool is correspondingly smaller.
# Every knob can be overridden from the environment.
set -euo pipefail
REPO=${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$REPO"
VENV=$REPO/venv
SP=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

if [ -z "${HQ_GPU:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "serve-sparse: set HQ_GPU (or CUDA_VISIBLE_DEVICES) to the card to use" >&2; exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${HQ_GPU:-$CUDA_VISIBLE_DEVICES}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TMPDIR=${HQ_TMP:-$REPO/run_release/tmp}
mkdir -p "$TMPDIR"
export VLLM_API_KEY=${VLLM_API_KEY:-$(cat "${API_KEY_FILE:-api_key.txt}")} BIND=${BIND:-127.0.0.1} PORT=${PORT:-8093}
case "$PORT" in 8080|8082) [ "${SPARSE_PRODUCTION:-0}" = 1 ] || {
  echo "serve-sparse: port $PORT is production; set SPARSE_PRODUCTION=1 to confirm" >&2; exit 2; } ;; esac

# ---- dense baseline (HyperQwen PTH_FA=1 + MTP-7) ----
export SPEC=${SPEC:-mtp} CTX=${CTX:-long} PREFIX_CACHE=${PREFIX_CACHE:-1} EMBED_UVA=${EMBED_UVA:-1}
export PTH_FA=1 DRAFT_TOKENS=${DRAFT_TOKENS:-7}
export INT8_ACT=${INT8_ACT-int8} PREFILL_ATTN=${PREFILL_ATTN-int8}
# v1.1: keep the trailing 864-token prefix-cache block on a hit (start_qwen.sh: disable_eagle_block_drop); prefill
# attention through FlashAttention-2 over the int8 cache (pthfa fa2_prefill.py; needs pthfa rebuilt from this tree)
export SPEC_NO_BLOCK_DROP=${SPEC_NO_BLOCK_DROP:-1} PTHFA_PREFILL_FA2=${PTHFA_PREFILL_FA2:-1}

# ---- sparse decode ----
export SPARSE_KV=1 SPARSE_IMPL=${SPARSE_IMPL:-triton}
export SPARSE_BASIS=${SPARSE_BASIS:-$VENV/share/sparsekv/basis27.pt}
export SPARSE_KSEL=${SPARSE_KSEL:-2048} SPARSE_SINK=${SPARSE_SINK:-64} SPARSE_WIN=${SPARSE_WIN:-1024}
export SPARSEKV_ROOT=${SPARSEKV_ROOT:-$SP}     # the installed sparsekv package (sparsekv/install.sh)
export SPARSE_MEM_RESERVE=${SPARSE_MEM_RESERVE:-1}   # budget sparse buffers out of the KV pool (G-1)
export SPARSE_SWITCH=${SPARSE_SWITCH:-1}             # per-step dense/sparse graph families; SPARSE_CUTOFF overrides
export SPARSE_SKETCH_BITS=${SPARSE_SKETCH_BITS:-4}   # int4 rank-64 sketch (half the GPU bytes of int8)
# accuracy-first selection (fixed configuration); STATS_SUB=4 / SELECT_EVERY=2 are speed shortcuts with a
# measured accuracy cost (docs/CHANGELOG.md)
export SPARSE_STATS_SUB=${SPARSE_STATS_SUB:-1} SPARSE_SELECT_EVERY=${SPARSE_SELECT_EVERY:-1}

PROFILE=${SPARSE_PROFILE:-tier}
TIER_ARGS=""
# v1.1: short requests (<= N uncached tokens) are still admitted on the prefill yield's decode-only steps
export SPARSE_PREFILL_YIELD_SHORT_BYPASS=${SPARSE_PREFILL_YIELD_SHORT_BYPASS:-2600}
case "$PROFILE" in
  tier)
    export MAX_LEN=${MAX_LEN-200000} MAX_SEQS=${MAX_SEQS:-2} GPU_UTIL=${GPU_UTIL:-0.90}
    export SPARSE_TIER=1 SPARSE_TIER_SPILL=${SPARSE_TIER_SPILL:-1}
    export SPARSE_TIER_SPILL_POLICY=${SPARSE_TIER_SPILL_POLICY:-pressure}
    export SPARSE_TIER_HOT_FRAC=${SPARSE_TIER_HOT_FRAC:-0.10}  # 0.15-0.20 recommended on the x4 card (STREAM_tier.md)
    export SPARSE_TIER_HOST_BITS=${SPARSE_TIER_HOST_BITS:-4}   # int4 host copy: half the RAM, no measurable accuracy cost
    export SPARSE_TIER_SUFFIX_SPARSE=${SPARSE_TIER_SUFFIX_SPARSE:-0}  # exact follow-up prefill (fixed configuration)
    export SPARSE_TIER_STAGE_SERIAL=${SPARSE_TIER_STAGE_SERIAL:-1}    # crash-safe staging (bounded staging memory)
    export SPARSE_TIER_HOST_PREFIX=${SPARSE_TIER_HOST_PREFIX:-1}
    # Host pool (stream E sizing): default MAX_LEN x MAX_SEQS tokens of KV mirror (33.3 KB/token: 13.3 GB for 2 x 200K)
    # + SPARSE_TIER_HOST_MAMBA_SLOTS GDN checkpoints (85.3 MB each: 0.68 GB) = ~14 GB pinned per server, capped at
    # SPARSE_TIER_HOST_GB_CAP; SPARSE_TIER_HOST_GB=<n> sets it explicitly. The worker refuses to start if the pool would
    # leave < 8 GB MemAvailable (78 GB VM; keep <= ~24 GB per server when two servers run).
    export SPARSE_TIER_HOST_GB_CAP=${SPARSE_TIER_HOST_GB_CAP:-16}
    export SPARSE_TIER_HOST_MAMBA_SLOTS=${SPARSE_TIER_HOST_MAMBA_SLOTS:-32} SPARSE_TIER_HOST_MAMBA_PER_REQ=${SPARSE_TIER_HOST_MAMBA_PER_REQ:-2}
    # v1.1: 32 GDN slots (2.7 GB pinned at 85.3 MB each) so checkpoints survive agents' tool waits; the newest
    # decode-boundary GDN state of each request is host-saved at finish; up to 2 pinned warm prefixes.
    export SPARSE_TIER_MAMBA_SAVE_AT_FINISH=${SPARSE_TIER_MAMBA_SAVE_AT_FINISH:-1}
    export SPARSE_TIER_PIN_PREFIX_MAX=${SPARSE_TIER_PIN_PREFIX_MAX:-2}
    # SPARSE_TIER_HOST_TOKENS (default MAX_LEN*MAX_SEQS; 8.4 GB pinned per 250K tokens), SPARSE_TIER_WATERMARK_BLOCKS
    # (default num_gpu_blocks//10), SPARSE_TIER_TAIL / _SINK_BLOCKS / _MARGIN: passed through when set.
    TIER_ARGS='--kv-transfer-config {"kv_connector":"HostTierConnector","kv_connector_module_path":"sparsekv.tier_connector","kv_role":"kv_both"}'
    [ "$SPARSE_TIER_SPILL" = 1 ] && TIER_ARGS="$TIER_ARGS --no-scheduler-reserve-full-isl"
    ;;
  sparse)
    export MAX_LEN=${MAX_LEN-153600} MAX_SEQS=${MAX_SEQS:-2} GPU_UTIL=${GPU_UTIL:-0.92}
    export SPARSE_TIER=0 SPARSE_TIER_SPILL=0
    ;;
  *) echo "serve-sparse: SPARSE_PROFILE must be tier or sparse" >&2; exit 2 ;;
esac

# ---- preflight: refuse to boot a silently-dense or random-basis server (review G-2, G-7) ----
[ -f "$SPARSE_BASIS" ] || { echo "serve-sparse: basis $SPARSE_BASIS missing (sparse_kv would use a RANDOM basis)" >&2; exit 1; }
bash "$REPO/sparsekv/install.sh" "$VENV" --check >/dev/null || {
  echo "serve-sparse: release not (fully) installed in $VENV:" >&2; bash "$REPO/sparsekv/install.sh" "$VENV" --check >&2 || true; exit 1; }

echo "[serve-sparse] profile=$PROFILE gpu=$CUDA_VISIBLE_DEVICES port=$PORT bind=$BIND MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS" \
     "GPU_UTIL=$GPU_UTIL KSEL=$SPARSE_KSEL SINK=$SPARSE_SINK WIN=$SPARSE_WIN impl=$SPARSE_IMPL tier=$SPARSE_TIER" \
     "spill=$SPARSE_TIER_SPILL sketch_bits=$SPARSE_SKETCH_BITS stats_sub=$SPARSE_STATS_SUB select_every=$SPARSE_SELECT_EVERY" \
     "host_bits=${SPARSE_TIER_HOST_BITS:-} suffix_sparse=${SPARSE_TIER_SUFFIX_SPARSE:-}" \
     "gdn_slots=${SPARSE_TIER_HOST_MAMBA_SLOTS:-} save_at_finish=${SPARSE_TIER_MAMBA_SAVE_AT_FINISH:-} no_block_drop=$SPEC_NO_BLOCK_DROP" \
     "fa2_prefill=$PTHFA_PREFILL_FA2 yield_bypass=$SPARSE_PREFILL_YIELD_SHORT_BYPASS pin_max=${SPARSE_TIER_PIN_PREFIX_MAX:-} release=$(head -1 "$SP/sparsekv/RELEASE")"
export EXTRA_ARGS="--served-model-name ${HQ_NAMES:-qwen3.8-27b-q4 qwen3.8-27b} $TIER_ARGS ${EXTRA_ARGS:-}"
exec bash single-user/start_qwen.sh
