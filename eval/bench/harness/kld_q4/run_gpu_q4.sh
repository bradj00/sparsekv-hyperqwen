#!/bin/bash
# run_gpu_q4.sh GPU PORT "cfg:role[:only]" ...  -- Q4TEST copy of run_gpu.sh driving kld_serve_q4.sh.
#   roles: toggle  -> out/<cfg>/<sid>.{dense,sparse}.npz     (toggle_q4, toggle_ctl)
#          forced  -> out/<cfg>/<sid>.<cfg>.npz              (dense_q4)
#   only:  optional --only list for kld_client.py (comma-separated sids/docs/lengths; use ';' between items here)
# Run on GPU0 (the card the existing toggle/ cells came from) for same-card pairs.
set -u
# Env: KLD_DIR (KLD work dir with data/ and out/ref from eval/kld), REL (release checkout + venv), Q4REL (q4test checkout)
KLD_DIR=${KLD_DIR:?}; REL=${REL:?}; export KLD_DIR REL
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KLD_CODE=${KLD_CODE:-$HERE/../../../kld}   # eval/kld (kld_client.py, kld_proc.py)
GPU=$1 PORT=$2; shift 2
cd $KLD_DIR; mkdir -p logs
[ -L launch-q4/venv ] || bash "$HERE/make_launch_q4.sh" || exit 1
for stage in "$@"; do
  IFS=: read cfg role only <<<"$stage"
  ONLY=(); [ -n "${only:-}" ] && ONLY=(--only "${only//;/,}")
  echo "[$(date -u +%T)] stage $cfg ($role) gpu $GPU ${only:+only=$only}"
  HQ_GPU=$GPU PORT=$PORT setsid bash "$HERE/kld_serve_q4.sh" $cfg > logs/server.$cfg.log 2>&1 &
  SPID=$!; echo $SPID > logs/server.$cfg.pid
  ok=0
  for i in $(seq 1 180); do
    kill -0 $SPID 2>/dev/null || break
    curl -sf -m 5 http://127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }
    sleep 5
  done
  if [ $ok = 1 ]; then
    echo "[$(date -u +%T)] $cfg healthy"; grep -m1 "GPU KV cache size" logs/server.$cfg.log
    echo "fakeq4 layers ON: $(grep -c 'fakeq4: SPARSE_DEBUG_KV_FAKEQ4=1 ON' logs/server.$cfg.log) (expect 16 for *_q4 once traffic starts, 0 for toggle_ctl)"
    case $role in
      toggle) $REL/venv/bin/python "$KLD_CODE"/kld_client.py --role toggle --port $PORT --data data --out out/$cfg --ref-dir out/ref "${ONLY[@]}" ;;
      forced) $REL/venv/bin/python "$KLD_CODE"/kld_client.py --role forced --mode $cfg --port $PORT --data data --out out/$cfg --ref-dir out/ref "${ONLY[@]}" ;;
    esac
    echo "fakeq4 layers ON after the stage: $(grep -c 'fakeq4: SPARSE_DEBUG_KV_FAKEQ4=1 ON' logs/server.$cfg.log)"
  else
    echo "[$(date -u +%T)] $cfg FAILED to become healthy"; tail -5 logs/server.$cfg.log
  fi
  kill -TERM -- -$SPID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  kill -KILL -- -$SPID 2>/dev/null; sleep 5
  echo "[$(date -u +%T)] $cfg stopped; gpu mem $(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader)"
done
echo "[$(date -u +%T)] ALL DONE gpu $GPU"
