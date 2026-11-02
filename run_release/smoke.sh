#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# sk-release smoke job: run INSIDE a GPU lock with HQ_GPU set. Boots serve-sparse.sh (env passes through: SPARSE_PROFILE,
# MAX_LEN, GPU_UTIL, ...), runs smoke.py, stops the server by its process group (explicit PIDs), prints GPU memory.
# usage: TAG=x PORT=8093 HQ_GPU=0 run_release/smoke.sh [NTOK]
set -u
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TAG=${TAG:-smoke}; export PORT=${PORT:-$(( HQ_GPU == 1 ? 8090 : 8093 ))}
mkdir -p $REPO/run_release/logs
LOG=$REPO/run_release/logs/server.$TAG.log; PIDF=$REPO/run_release/server.$TAG.pid
stop() {
  P=$(cat $PIDF 2>/dev/null) || return 0; [ -n "$P" ] || return 0
  PIDS=$(ps -o pid= -g $P 2>/dev/null | tr -d " ")
  [ -n "$PIDS" ] && { echo "[smoke] stopping group $P: $(echo $PIDS)"; kill -TERM $PIDS 2>/dev/null; }
  for i in $(seq 1 60); do sleep 1; [ -z "$(ps -o pid= -g $P 2>/dev/null)" ] && break; done
  ALIVE=$(ps -o pid= -g $P 2>/dev/null | tr -d " "); [ -n "$ALIVE" ] && { echo "[smoke] SIGKILL $ALIVE"; kill -KILL $ALIVE; sleep 3; }
  rm -f $PIDF
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}
trap stop EXIT TERM INT
ss -ltn | grep -q ":$PORT " && { echo "[smoke] port $PORT busy"; exit 1; }
T0=$(date +%s)
setsid bash $REPO/${SERVE:-serve-sparse.sh} > $LOG 2>&1 < /dev/null &
echo $! > $PIDF
echo "[smoke] server pid $(cat $PIDF), log $LOG"
for i in $(seq 1 120); do
  sleep 5
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "[smoke] healthy after $(( $(date +%s) - T0 ))s"; break; }
  kill -0 $(cat $PIDF) 2>/dev/null || { echo "[smoke] server died"; grep -E "Error|error|refus" $LOG | grep -v frames | tail -20; exit 1; }
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "[smoke] not healthy after 600 s"; tail -20 $LOG; exit 1; }
grep -h -E "serve-sparse\]|sparse_kv:|tier:|GPU KV cache size|Available KV cache|SPARSE_TIER|Maximum concurrency" $LOG | cut -c1-260 | head -20
timeout ${CLIENT_TIMEOUT:-900} $REPO/venv/bin/python $REPO/run_release/smoke.py $PORT $REPO/run_release/results.$TAG.json ${1:-128000} 2>&1 | grep -v -i "warn"
grep -h -E "tier: stats|Traceback|Error" $LOG | tail -5 | cut -c1-300
echo "[smoke] job time $(( $(date +%s) - T0 ))s"
