#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# GPU1: after the 250K k=1,2 job -> 128K x2 ablations. GPU0: after the 128K x4 sweep -> 4x250K (big250.sh).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
P_=${PROGRESS:-run_s4/progress.log}
( until grep -q sweep_end run_s4/d.g1_250a.jsonl; do sleep 5; done; run_s4/srv.sh stop 1 8097 x > /dev/null 2>&1
  echo "- $(date -u +%FT%TZ) GPU1: 128K x2 ablations (run_s4/abl.sh: STATS_SUB, SELECT_EVERY, suffix sparse, sync slots)" >> $P_
  run_s4/abl.sh 1 8097 > run_s4/abl.g1.out 2>&1 ) &
( until grep -q sweep_end run_s4/d.s4c128n4.jsonl; do sleep 5; done; run_s4/srv.sh stop 0 8099 x > /dev/null 2>&1
  until [ -f run_s4/server.g1.pid ] && grep -q abl run_s4/abl.g1.out 2>/dev/null || [ -n "$(ls run_s4/logs/server.abl_*g1.log 2>/dev/null)" ]; do sleep 5; done
  sleep 30
  run_s4/big250.sh 0 8099 big250_g0 > run_s4/big250_g0.out 2>&1 ) &
wait
