#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# session.sh TAG PORT SUITES  (inside a gpu lock). Boots, waits for health, runs client, always stops the server.
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TAG=$1 PORT=$2 SPARSE_WT=$WT; SUITES=$3
trap "$WT/run_tier/stop.sh" EXIT TERM INT
ss -ltn | grep -q ":$PORT " && { echo "[session] port $PORT busy"; exit 1; }
$WT/run_tier/serve.sh
LOG=$WT/run_tier/logs/server.$TAG.log
for i in $(seq 1 120); do
  sleep 5
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "[session] healthy after $((i*5))s"; break; }
  P=$(cat $WT/run_tier/server.$TAG.pid); kill -0 $P 2>/dev/null || { echo "[session] server died"; tail -40 $LOG; exit 1; }
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "[session] boot timeout"; tail -40 $LOG; exit 1; }
grep -h "sparse_kv\|GPU KV cache size\|Available KV cache\|Capturing CUDA graphs.*100%" $LOG | sed "s/\r/\n/g" | grep -v "^$" | tail -8
timeout ${CLIENT_TIMEOUT:-900} $WT/venv/bin/python $WT/run/client.py $PORT $WT/run_tier/results.$TAG.json $SUITES 2>&1 | grep -v "Warning\|warn("
