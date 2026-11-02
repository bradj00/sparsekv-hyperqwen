#!/bin/bash
# run_gpu.sh GPU PORT "cfg:role[:mode[:switch]]" ...
# For each stage: boot `kld_serve.sh cfg` on GPU (or attach if its server is already up), drive kld_client.py, stop
# the server by its process group. Env: HQ_REPO (HyperQwen checkout + venv with the release installed),
# KLD_DIR (work dir: data/, out/, logs/; default this directory), KLD_API_KEY_FILE (+ what kld_serve.sh needs).
# What we ran (2 GPUs in parallel):
#   ./run_gpu.sh 1 8099 ref:ref tier4:forced:tier4
#   ./run_gpu.sh 0 8097 toggle:toggle            # then, on the same server, a dense repeat of 6 samples:
#   kld_client.py --role forced --mode dense2 --switch dense --only prose_mc.128k.*,prose_mc.191k.* (sids listed)
#   ./run_gpu.sh 0 8097 toggle:stop tier8:forced:tier8
set -u
HQ_REPO=${HQ_REPO:?}
KLD_DIR=${KLD_DIR:-$(cd "$(dirname "$0")" && pwd)}; export KLD_DIR
PY=$HQ_REPO/venv/bin/python
GPU=$1 PORT=$2; shift 2
cd "$KLD_DIR"; mkdir -p logs
for stage in "$@"; do
  IFS=: read cfg role mode switch <<<"$stage"
  echo "[$(date -u +%T)] stage $cfg ($role) gpu $GPU"
  if [ -f logs/server.$cfg.pid ] && kill -0 $(cat logs/server.$cfg.pid) 2>/dev/null; then
    SPID=$(cat logs/server.$cfg.pid); echo "attaching to running $cfg server $SPID"
  else
    HQ_GPU=$GPU PORT=$PORT setsid bash kld_serve.sh $cfg > logs/server.$cfg.log 2>&1 &
    SPID=$!; echo $SPID > logs/server.$cfg.pid
  fi
  ok=0
  for i in $(seq 1 180); do
    kill -0 $SPID 2>/dev/null || break
    curl -sf -m 5 http://127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }
    sleep 5
  done
  if [ $ok = 1 ]; then
    echo "[$(date -u +%T)] $cfg healthy"; grep -m1 "GPU KV cache size" logs/server.$cfg.log
    case $role in
      ref) $PY kld_client.py --role ref --port $PORT --data data --out out/ref ;;
      toggle) $PY kld_client.py --role toggle --port $PORT --data data --out out/toggle --ref-dir out/ref ;;
      forced) $PY kld_client.py --role forced --mode $mode ${switch:+--switch $switch} --port $PORT --data data \
                --out out/$cfg --ref-dir out/ref ;;
      stop) ;;
    esac
  else
    echo "[$(date -u +%T)] $cfg FAILED to become healthy"; tail -5 logs/server.$cfg.log
  fi
  kill -TERM -- -$SPID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  kill -KILL -- -$SPID 2>/dev/null; sleep 5
  echo "[$(date -u +%T)] $cfg stopped"
done
