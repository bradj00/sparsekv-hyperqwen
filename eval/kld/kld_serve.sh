#!/bin/bash
# kld_serve.sh <ref|toggle|tier8|tier4>   (HQ_GPU, PORT required). Speculative decoding off, logits recorder on.
set -euo pipefail
CFG=$1
KLD_DIR=${KLD_DIR:-$(cd "$(dirname "$0")" && pwd)}
export REPO=$KLD_DIR/launch
export PYTHONPATH=$KLD_DIR${PYTHONPATH:+:$PYTHONPATH}
export KLD_OUT=$KLD_DIR/out/$CFG
export ASYNC_SCHED=0 HOST=127.0.0.1 BIND=127.0.0.1
export HQ_TMP=$KLD_DIR/tmp-$CFG; mkdir -p "$HQ_TMP" "$KLD_OUT"
export SPARSE_SWITCH_LOG=1
case "$CFG" in
  ref)    # dense: the same base config as the sparse launcher (PTH_FA int8 KV, int8 prefill/activations, prefix cache), no SPARSE_*; MTP removed by kld_exec
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$HQ_GPU
    export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TMPDIR=$HQ_TMP
    export VLLM_API_KEY=${VLLM_API_KEY:-$(cat "${API_KEY_FILE:?}")}
    export SPEC=mtp CTX=long PREFIX_CACHE=1 EMBED_UVA=1 PTH_FA=1 DRAFT_TOKENS=7 INT8_ACT=int8 PREFILL_ATTN=int8
    export MAX_LEN=${MAX_LEN:-199000} MAX_SEQS=2 GPU_UTIL=${GPU_UTIL:-0.93}
    export EXTRA_ARGS="--served-model-name qwen3.8-27b-q4 qwen3.8-27b --block-size 864"
    cd "$REPO"; exec bash single-user/start_qwen.sh ;;
  toggle) # SPARSE_PROFILE=sparse (GPU-only sparse decode); per-request dense/sparse via the switch table
    export SPARSE_PROFILE=sparse MAX_LEN=${MAX_LEN:-199000} GPU_UTIL=${GPU_UTIL:-0.94}
    export SPARSE_CUTOFF=1:1000000000 KLD_TOGGLE=1 ;;
  tier8|tier4) # the production 4 x 250K configuration (top-level README); tier8 differs only in the host copy bits
    export SPARSE_PROFILE=tier MAX_LEN=262144 MAX_SEQS=4 GPU_UTIL=0.94 SPARSE_TIER_HOT_FRAC=0.025 SPARSE_SKETCH_BITS=4
    export SPARSE_TIER_HOST_GB=17.8 SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2
    if [ "$CFG" = tier4 ]; then export SPARSE_TIER_HOST_BITS=4; else export SPARSE_TIER_HOST_BITS=8; fi ;;
  *) echo "unknown config $CFG" >&2; exit 2 ;;
esac
export EXTRA_ARGS="--block-size 864"   # production page size; without MTP vLLM would pick 800, and the sparse ops are built for 864
exec bash "$REPO/serve-sparse.sh"
