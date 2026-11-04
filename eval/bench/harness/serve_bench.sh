#!/bin/bash
# serve_bench.sh <dense|sparse|tier8|tier4>   (HQ_GPU, PORT required; MTP-7 speculative decoding ON = the served config)
# Every config uses the same venv, weights and tokenizer. Overrides (MAX_LEN, MAX_SEQS, GPU_UTIL) come from the env.
set -euo pipefail
CFG=$1
REL=${REL:?release checkout with the sparse release installed in REL/venv}
BENCH_TMP=${BENCH_TMP:?}
export HOST=127.0.0.1 BIND=127.0.0.1
export HQ_TMP=$BENCH_TMP/tmp-$CFG-g$HQ_GPU; mkdir -p "$HQ_TMP"
export SPARSE_SWITCH_LOG=1
case "$CFG" in
  dense)  # the base config (PTH_FA int8 KV, int8 prefill/activations, prefix cache, MTP-7), no SPARSE_* at all
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$HQ_GPU
    export HF_HOME=${HF_HOME:?} TMPDIR=$HQ_TMP
    [ -n "${BENCH_TOKEN:-}" ] && export VLLM_API_KEY=$BENCH_TOKEN  # optional server auth
    export SPEC=mtp CTX=long PREFIX_CACHE=1 EMBED_UVA=1 PTH_FA=1 DRAFT_TOKENS=7 INT8_ACT=int8 PREFILL_ATTN=int8
    export MAX_LEN=${MAX_LEN:-199000} MAX_SEQS=${MAX_SEQS:-1} GPU_UTIL=${GPU_UTIL:-0.93}
    export EXTRA_ARGS="--served-model-name qwen3.8-27b-q4 qwen3.8-27b --block-size 864"
    cd "$REL"; exec bash single-user/start_qwen.sh ;;
  sparse) # GPU-only sparse decode (int8 sketch, exact statistics, selection every step, no tier), forced always-sparse
    export SPARSE_PROFILE=sparse MAX_LEN=${MAX_LEN:-199000} MAX_SEQS=${MAX_SEQS:-1} GPU_UTIL=${GPU_UTIL:-0.93}
    export SPARSE_CUTOFF=0 ;;
  tier8|tier4) # the production 4 x 250K command (top-level README); tier8 differs only in the host copy's bits
    export SPARSE_PROFILE=tier MAX_LEN=${MAX_LEN:-262144} MAX_SEQS=${MAX_SEQS:-4} GPU_UTIL=${GPU_UTIL:-0.94}
    export SPARSE_TIER_HOT_FRAC=0.025 SPARSE_SKETCH_BITS=${SPARSE_SKETCH_BITS:-4} SPARSE_TIER_HOST_GB=${SPARSE_TIER_HOST_GB:-17.8}
    export SPARSE_STATS_SUB=${SPARSE_STATS_SUB:-4} SPARSE_SELECT_EVERY=${SPARSE_SELECT_EVERY:-2}
    if [ "$CFG" = tier4 ]; then export SPARSE_TIER_HOST_BITS=4; else export SPARSE_TIER_HOST_BITS=8; fi ;;
  *) echo "unknown config $CFG" >&2; exit 2 ;;
esac
export EXTRA_ARGS="--block-size 864"
cd "$REL"; exec bash "$REL/serve-sparse.sh"
