#!/bin/bash
# kld_serve_q4.sh <toggle_q4|toggle_ctl|dense_q4>  (HQ_GPU, PORT required) -- Q4TEST copy of kld_serve.sh.
# Same server configurations as kld_serve.sh's toggle / ref, run from the q4test launcher tree ($KLD_DIR/launch-q4,
# venv = $Q4REL/venv, a checkout carrying the SPARSE_DEBUG_KV_FAKEQ4 debug hook). Speculative decoding off, logits recorder on.
#   toggle_q4   = toggle + SPARSE_DEBUG_KV_FAKEQ4=1 (one server: dense-switch and sparse-switch run per sample)
#   toggle_ctl  = toggle with the knob OFF from the q4test venv (control: must reproduce out/toggle exactly)
#   dense_q4    = ref (no SPARSE_*) + SPARSE_DEBUG_KV_FAKEQ4=1 (optional cross-check of the dense int4 cell)
set -euo pipefail
CFG=$1
KLD_DIR=${KLD_DIR:?}
export REPO=$KLD_DIR/launch-q4
export PYTHONPATH=$KLD_DIR${PYTHONPATH:+:$PYTHONPATH}
export KLD_OUT=$KLD_DIR/out/$CFG
export ASYNC_SCHED=0 HOST=127.0.0.1 BIND=127.0.0.1
export HQ_TMP=$KLD_DIR/tmp-$CFG; mkdir -p "$HQ_TMP" "$KLD_OUT"
export SPARSE_SWITCH_LOG=1
case "$CFG" in
  dense_q4)  # = kld_serve.sh ref + the knob
    export SPARSE_DEBUG_KV_FAKEQ4=1
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$HQ_GPU
    export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TMPDIR=$HQ_TMP
    [ -n "${BENCH_TOKEN:-}" ] && export VLLM_API_KEY=$BENCH_TOKEN  # optional server auth
    export SPEC=mtp CTX=long PREFIX_CACHE=1 EMBED_UVA=1 PTH_FA=1 DRAFT_TOKENS=7 INT8_ACT=int8 PREFILL_ATTN=int8
    export MAX_LEN=${MAX_LEN:-199000} MAX_SEQS=2 GPU_UTIL=${GPU_UTIL:-0.93}
    export EXTRA_ARGS="--served-model-name qwen3.8-27b-q4 qwen3.8-27b --block-size 864"
    cd "$REPO"; exec bash single-user/start_qwen.sh ;;
  toggle_q4|toggle_ctl)  # = kld_serve.sh toggle (+ the knob for toggle_q4)
    [ "$CFG" = toggle_q4 ] && export SPARSE_DEBUG_KV_FAKEQ4=1 || unset SPARSE_DEBUG_KV_FAKEQ4
    export SPARSE_PROFILE=sparse MAX_LEN=${MAX_LEN:-199000} GPU_UTIL=${GPU_UTIL:-0.94}
    export SPARSE_CUTOFF=1:1000000000 KLD_TOGGLE=1 ;;
  *) echo "unknown config $CFG" >&2; exit 2 ;;
esac
export EXTRA_ARGS="--block-size 864"
exec bash "$REPO/serve-sparse.sh"
