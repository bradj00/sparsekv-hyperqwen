#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# inside a GPU lock: real-spill capacity session. Env: NTOK (prompt tokens), MAX_LEN, GPU_UTIL, TAG, PORT
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPARSE_KV=1 SPARSE_IMPL=triton SPARSE_TIER=1 SPARSE_TIER_SPILL=1 HQ_GPU=$CUDA_VISIBLE_DEVICES
export GPU_UTIL=${GPU_UTIL:-0.78} MAX_LEN=${MAX_LEN:-200000} MAX_SEQS=${MAX_SEQS:-1}
export SPARSE_TIER_HOT_FRAC=${SPARSE_TIER_HOT_FRAC:-0.10} TIER_EXTRA="--no-scheduler-reserve-full-isl"
TAG=${TAG:-spill1}; PORT=${PORT:-8095}
export TAG PORT SPARSE_WT=$WT
trap "$WT/run_tier/stop.sh" EXIT TERM INT
ss -ltn | grep -q ":$PORT " && { echo "[spill] port $PORT busy"; exit 1; }
$WT/run_tier/serve.sh
LOG=$WT/run_tier/logs/server.$TAG.log
for i in $(seq 1 120); do
  sleep 5
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "[spill] healthy after $((i*5))s"; break; }
  P=$(cat $WT/run_tier/server.$TAG.pid); kill -0 $P 2>/dev/null || { echo "[spill] server died"; grep -E "Error|error" $LOG | grep -v frames | tail -20; exit 1; }
done
grep -h "tier:\|GPU KV cache size\|Available KV cache\|SPARSE_TIER" $LOG | tail -8 | cut -c1-300
QS=needle,code,speed timeout 300 $WT/venv/bin/python $WT/run_tier/bigctx.py $PORT $WT/run_tier/results.$TAG.fit.json 100000 2>&1 | grep -v Warn
grep -h "tier: stats" $LOG | tail -1 | cut -c1-400
QS=needle,code,needle2,speed timeout 560 $WT/venv/bin/python $WT/run_tier/bigctx.py $PORT $WT/run_tier/results.$TAG.big.json 180000 2>&1 | grep -v Warn
grep -h "tier: stats\|Error\|Traceback" $LOG | tail -5 | cut -c1-300
