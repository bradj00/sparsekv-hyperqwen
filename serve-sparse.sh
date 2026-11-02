#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Sparse-KV release launcher (sk-release). Same dense baseline config as the stock HyperQwen launcher (PTH_FA=1 int8
# KV + MTP-7), plus query-aware sparse decode (SPARSE_KV=1) and, by default, the system-RAM KV tier (SPARSE_TIER=1).
# Install first: bash sparsekv/install.sh <repo>/venv. Read sparsekv/RELEASE.md (knobs, limits); README.md has uninstall/rollback.
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

# ---- sparse decode ----
export SPARSE_KV=1 SPARSE_IMPL=${SPARSE_IMPL:-triton}
export SPARSE_BASIS=${SPARSE_BASIS:-$VENV/share/sparsekv/basis27.pt}
export SPARSE_KSEL=${SPARSE_KSEL:-2048} SPARSE_SINK=${SPARSE_SINK:-64} SPARSE_WIN=${SPARSE_WIN:-1024}
export SPARSEKV_ROOT=${SPARSEKV_ROOT:-$SP}     # the installed sparsekv package (sparsekv/install.sh)
export SPARSE_MEM_RESERVE=${SPARSE_MEM_RESERVE:-1}   # budget sparse buffers out of the KV pool (G-1)
export SPARSE_SWITCH=${SPARSE_SWITCH:-1}             # per-step dense/sparse graph families; SPARSE_CUTOFF overrides

PROFILE=${SPARSE_PROFILE:-tier}
TIER_ARGS=""
case "$PROFILE" in
  tier)
    export MAX_LEN=${MAX_LEN-200000} MAX_SEQS=${MAX_SEQS:-2} GPU_UTIL=${GPU_UTIL:-0.90}
    export SPARSE_TIER=1 SPARSE_TIER_SPILL=${SPARSE_TIER_SPILL:-1}
    export SPARSE_TIER_SPILL_POLICY=${SPARSE_TIER_SPILL_POLICY:-pressure}
    export SPARSE_TIER_HOT_FRAC=${SPARSE_TIER_HOT_FRAC:-0.10}  # 0.15-0.20 recommended on the x4 card (STREAM_tier.md)
    export SPARSE_TIER_HOST_PREFIX=${SPARSE_TIER_HOST_PREFIX:-1}
    # Host pool (stream E sizing): default MAX_LEN x MAX_SEQS tokens of KV mirror (33.3 KB/token: 13.3 GB for 2 x 200K)
    # + SPARSE_TIER_HOST_MAMBA_SLOTS GDN checkpoints (85.3 MB each: 0.68 GB) = ~14 GB pinned per server, capped at
    # SPARSE_TIER_HOST_GB_CAP; SPARSE_TIER_HOST_GB=<n> sets it explicitly. The worker refuses to start if the pool would
    # leave < 8 GB MemAvailable (78 GB VM; keep <= ~24 GB per server when two servers run).
    export SPARSE_TIER_HOST_GB_CAP=${SPARSE_TIER_HOST_GB_CAP:-16}
    export SPARSE_TIER_HOST_MAMBA_SLOTS=${SPARSE_TIER_HOST_MAMBA_SLOTS:-8} SPARSE_TIER_HOST_MAMBA_PER_REQ=${SPARSE_TIER_HOST_MAMBA_PER_REQ:-2}
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
     "spill=$SPARSE_TIER_SPILL release=$(head -1 "$SP/sparsekv/RELEASE")"
export EXTRA_ARGS="--served-model-name ${HQ_NAMES:-qwen3.8-27b-q4 qwen3.8-27b} $TIER_ARGS ${EXTRA_ARGS:-}"
exec bash single-user/start_qwen.sh
