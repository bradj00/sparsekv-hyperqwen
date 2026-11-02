#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# 2-user tier job (inside a GPU lock, HQ_GPU set): boots serve-sparse.sh tier profile with 2 slots, forced spill
# (watermark 1, E's setting) and the dev endpoint for /reset_prefix_cache; runs conc.py; stops the server by PID.
set -u
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TAG=${TAG:-conc}; export PORT=${PORT:-$(( HQ_GPU == 1 ? 8090 : 8093 ))}
export SPARSE_PROFILE=tier MAX_SEQS=${MAX_SEQS:-2} SPARSE_TIER_WATERMARK_BLOCKS=${SPARSE_TIER_WATERMARK_BLOCKS:-1}
export VLLM_SERVER_DEV_MODE=1
mkdir -p $REPO/run_release/logs
LOG=$REPO/run_release/logs/server.$TAG.log; PIDF=$REPO/run_release/server.$TAG.pid
stop() {
  P=$(cat $PIDF 2>/dev/null) || return 0; [ -n "$P" ] || return 0
  PIDS=$(ps -o pid= -g $P 2>/dev/null | tr -d " ")
  [ -n "$PIDS" ] && { echo "[conc] stopping group $P: $(echo $PIDS)"; kill -TERM $PIDS 2>/dev/null; }
  for i in $(seq 1 60); do sleep 1; [ -z "$(ps -o pid= -g $P 2>/dev/null)" ] && break; done
  ALIVE=$(ps -o pid= -g $P 2>/dev/null | tr -d " "); [ -n "$ALIVE" ] && { echo "[conc] SIGKILL $ALIVE"; kill -KILL $ALIVE; sleep 3; }
  rm -f $PIDF; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}
trap stop EXIT TERM INT
ss -ltn | grep -q ":$PORT " && { echo "[conc] port $PORT busy"; exit 1; }
T0=$(date +%s)
setsid bash $REPO/serve-sparse.sh > $LOG 2>&1 < /dev/null &
echo $! > $PIDF
for i in $(seq 1 120); do
  sleep 5
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "[conc] healthy after $(( $(date +%s) - T0 ))s"; break; }
  kill -0 $(cat $PIDF) 2>/dev/null || { echo "[conc] server died"; grep -a -E "Error|error" $LOG | grep -v frames | tail -20; exit 1; }
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "[conc] not healthy"; tail -20 $LOG; exit 1; }
grep -a -h -E "serve-sparse\]|reserving|GPU KV cache size|tier: runtime|mamba checkpoint host store|SPILL scheduler hook" $LOG | cut -c1-300
timeout ${CLIENT_TIMEOUT:-1080} $REPO/venv/bin/python $REPO/run_release/conc.py $PORT $REPO/run_release/results.$TAG.json 177000 120000 2>&1 | grep -v -i warn
echo "[conc] preempt/resume lines: $(grep -a -c -E "tier: request .* (preempted|re-admitted|resumed)" $LOG)"
grep -a -h -E "tier: request .* (preempted|re-admitted|resumed)" $LOG | cut -c60-260 | head -6
grep -a -h "tier: stats" $LOG | tail -1 | cut -c1-500
grep -a -h -E "Traceback|CUDA out of memory|> reserved" $LOG | head -5
echo "[conc] job time $(( $(date +%s) - T0 ))s"
