#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Boot a dev server from this worktree (own venv). Env: PORT (8090-8099), SPARSE_* knobs, GPU_UTIL, TAG.
# Must be called inside a gpu lock with CUDA_VISIBLE_DEVICES/HQ_GPU set. Writes run/server.<TAG>.pid
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd $WT
TAG=${TAG:-x}
mkdir -p $WT/run_tier/tmp $WT/run_tier/hf $WT/run_tier/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME=$WT/run_tier/hf TMPDIR=$WT/run_tier/tmp
export VLLM_API_KEY=${VLLM_API_KEY:-$(cat "${API_KEY_FILE:-api_key.txt}")} HOST=127.0.0.1 BIND=127.0.0.1 PORT=${PORT:-8091}
export SPEC=mtp CTX=long PREFIX_CACHE=1 EMBED_UVA=1 PTH_FA=1
export MAX_LEN=${MAX_LEN:-153600} MAX_SEQS=${MAX_SEQS:-2} DRAFT_TOKENS=7
export INT8_ACT=int8 PREFILL_ATTN=int8
export GPU_UTIL=${GPU_UTIL:-0.88}
export SPARSEKV_ROOT=$WT PYTHONPATH=$WT${PYTHONPATH:+:$PYTHONPATH}
if [ "${SPARSE_TIER:-0}" = 1 ]; then TIER_ARGS="--kv-transfer-config {\"kv_connector\":\"HostTierConnector\",\"kv_connector_module_path\":\"sparsekv.tier_connector\",\"kv_role\":\"kv_both\"}"; fi
export EXTRA_ARGS="--served-model-name qwen3.8-27b-q4 qwen3.8-27b ${TIER_ARGS:-} ${TIER_EXTRA:-}"
LOG=$WT/run_tier/logs/server.$TAG.log
setsid bash single-user/start_qwen.sh > $LOG 2>&1 < /dev/null &
echo $! > $WT/run_tier/server.$TAG.pid
echo "[serve] pid $(cat $WT/run_tier/server.$TAG.pid) log $LOG"
