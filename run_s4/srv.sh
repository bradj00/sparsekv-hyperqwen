#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# run_s4 dev server: srv.sh start|stop GPU PORT TAG  (env passes through: MAX_LEN, MAX_SEQS, GPU_UTIL, SPARSE_TIER_*, ...)
# REPO defaults to the s4 worktree (own venv). start: tier profile, setsid + pidfile, waits for health. stop: kill the
# process group by explicit PID.
set -u
REPO=${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}; D=$REPO/run_s4; mkdir -p $D/logs
CMD=$1 G=$2 PORT=$3 TAG=$4; PIDF=$D/server.g$G.pid; LOG=$D/logs/server.$TAG.log
CPUS=$([ "$G" = 1 ] && echo 8-15 || echo 0-7)
stop() {
  P=$(cat $PIDF 2>/dev/null) || return 0; [ -n "$P" ] || return 0
  PIDS=$(ps -o pid= -g $P 2>/dev/null | tr -d " ")
  [ -n "$PIDS" ] && kill -TERM $PIDS 2>/dev/null
  for i in $(seq 1 60); do sleep 1; [ -z "$(ps -o pid= -g $P 2>/dev/null)" ] && break; done
  ALIVE=$(ps -o pid= -g $P 2>/dev/null | tr -d " "); [ -n "$ALIVE" ] && { kill -KILL $ALIVE; sleep 3; }
  rm -f $PIDF; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}
if [ "$CMD" = stop ]; then stop; exit 0; fi
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 1; }
export HQ_GPU=$G PORT SPARSE_PROFILE=${SPARSE_PROFILE:-tier} MAX_LEN=${MAX_LEN:-262144} MAX_SEQS=${MAX_SEQS:-2} GPU_UTIL=${GPU_UTIL:-0.93}
export SPARSE_TIER_HOST_GB_CAP=${SPARSE_TIER_HOST_GB_CAP:-20} HQ_TMP=$D/tmp$G HQ_NAMES="qwen3.8-27b"
export SPARSE_CG_STATS=${SPARSE_CG_STATS:-1}
T0=$(date +%s)
setsid taskset -c $CPUS bash $REPO/serve-sparse.sh > $LOG 2>&1 < /dev/null &
echo $! > $PIDF
for i in $(seq 1 120); do
  sleep 5
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "[srv] $TAG healthy after $(( $(date +%s) - T0 ))s"; break; }
  kill -0 $(cat $PIDF) 2>/dev/null || { echo "[srv] $TAG died"; grep -a -E "Error|error" $LOG | grep -v frames | tail -15; rm -f $PIDF; exit 1; }
done
grep -a -h -E "serve-sparse\]|reserving|GPU KV cache size|tier: runtime|mamba checkpoint host store|allocated >|out of memory|captured per family|Capturing CUDA graphs \(FULL\): 100" $LOG | sed 's/.*Capturing CUDA graphs (FULL): 100%[^|]*|[^|]*| \([0-9/]*\).*/FULL capture \1/' | cut -c1-330
