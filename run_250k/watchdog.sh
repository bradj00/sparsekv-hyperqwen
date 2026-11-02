#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# soak watchdog: watchdog.sh PORT LOG RESULTS  -> prints a line on any failure signal, a summary every 10 min
P=$1 LOG=$2 R=$3; n=0; fails=0
base=$(grep -a -c -E "Traceback|out of memory|allocated >|EngineDeadError|MemoryError" $LOG)
while true; do
  sleep 30; n=$((n+1))
  if ! curl -sf -m 10 localhost:$P/health >/dev/null; then fails=$((fails+1)); echo "WD $P health FAIL ($fails)"; else fails=0; fi
  e=$(grep -a -c -E "Traceback|out of memory|allocated >|EngineDeadError|MemoryError" $LOG)
  [ "$e" != "$base" ] && { echo "WD $P new error lines: $(grep -a -E "Traceback|out of memory|allocated >|EngineDeadError|MemoryError" $LOG | tail -2 | cut -c1-200)"; base=$e; }
  grep -q soak_end $R 2>/dev/null && { echo "WD $P soak_end: $(grep soak_end $R)"; exit 0; }
  if [ $((n % 20)) = 0 ]; then
    echo "WD $P $(date -u +%T) turns=$(grep -c "\"q\"" $R) wrong=$(grep -c "\"ok\": false" $R) err=$(grep -c "\"error\"" $R) uncached_long=$(python3 -c "import json;print(sum(1 for l in open(\"$R\") if (j:=json.loads(l)).get(\"prompt\",0)>100000 and (j.get(\"cached\") or 0)<100000))") mem=$(free -g | awk "/Mem/{print \$7}")G"
  fi
done
