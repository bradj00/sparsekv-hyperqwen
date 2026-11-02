#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Run a command under whichever GPU lock frees first (gpu0 = 3090 Ti x16, gpu1 = 3090 x4).
# One BLOCKING flock waiter per card (fair against other streams' blocking waiters); the first to get its lock claims
# the job (mkdir claim dir), the other one gives up. The command sees HQ_GPU=<card>; capped at JOB_MAX s (default 1500).
# usage: TAG=x run_release/gpujob.sh [--only 0|1] CMD...    env: WAIT_MAX (s, default 7200)
ONLY=""; [ "${1:-}" = --only ] && { ONLY=$2; shift 2; }
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CLAIM=$REPO/run_release/claim.${TAG:-job}.$$
WAIT_MAX=${WAIT_MAX:-7200}; JOB_MAX=${JOB_MAX:-1500}; T0=$(date +%s)
waiter() {
  local g=$1; shift
  exec 9>${GPU_LOCK_DIR:-/tmp}/gpu$g.lock
  flock -w $WAIT_MAX 9 || { echo "[gpujob] gpu$g: gave up waiting"; return 3; }
  if ! mkdir "$CLAIM" 2>/dev/null; then flock -u 9; return 0; fi
  echo "[gpujob] gpu$g lock taken after $(( $(date +%s) - T0 ))s at $(date -u +%T)"
  HQ_GPU=$g timeout -s TERM $JOB_MAX "$@"; local rc=$?
  flock -u 9; exec 9>&-
  echo "[gpujob] released gpu$g rc=$rc at $(date -u +%T)"
  echo $rc > "$CLAIM/rc"
}
PIDS=()
for g in ${ONLY:-0 1}; do waiter $g "$@" & PIDS+=($!); done
# when one waiter has run the job, stop the other (explicit PIDs; it holds no lock unless it already lost the claim)
while :; do
  if [ -f "$CLAIM/rc" ]; then
    for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && { pkids=$(ps -o pid= --ppid $p | tr -d ' '); kill $p $pkids 2>/dev/null; }; done
    rc=$(cat "$CLAIM/rc"); rm -rf "$CLAIM"; exit $rc
  fi
  alive=0; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
  [ $alive = 0 ] && { rm -rf "$CLAIM"; echo "[gpujob] no job ran"; exit 3; }
  sleep 5
done
