#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Max concurrent 250K agents on the box: GPU0 N0 x 250K + GPU1 N1 x 250K at once (int4 host tier by default).
# Job A (both GPUs in parallel): 2 cold prefills each (rounds 0); 120 s pause; job B: remaining cold prefills + 2 rounds
# with all agents of each server concurrent. duo.sh TAG [N0] [N1]   env: HOSTBITS (4), GB_PER_AGENT (4.4 at int4)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
T=$1 N0=${2:-4} N1=${3:-4}
HB=${HOSTBITS:-4}; GPA=${GB_PER_AGENT:-$([ "$HB" = 4 ] && echo 4.4 || echo 8.4)}
P_=${PROGRESS:-run_s4/progress.log}
boot() {  # GPU PORT N TAG
  SPARSE_TIER_HOST_BITS=$HB SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2 SPARSE_SKETCH_BITS=4 MAX_LEN=262144 MAX_SEQS=$3 \
  GPU_UTIL=0.94 SPARSE_TIER_HOT_FRAC=0.025 SPARSE_TIER_HOST_GB=$(python3 -c "print(round($3*$GPA+0.2,1))") \
    run_s4/srv.sh start $1 $2 $4 2>&1 | grep -a "healthy\|died\|tier: runtime\|MemoryError" | cut -c1-300
}
boot 0 8099 $N0 ${T}_g0; boot 1 8097 $N1 ${T}_g1
curl -sf localhost:8099/health >/dev/null && curl -sf localhost:8097/health >/dev/null || { echo "duo boot failed"; exit 1; }
echo "- $(date -u +%FT%TZ) duo $T: GPU0 ${N0}x250K + GPU1 ${N1}x250K booted (host bits $HB); job A" >> $P_
a0=$([ $N0 -ge 2 ] && echo 2 || echo 1); a1=$([ $N1 -ge 2 ] && echo 2 || echo 1)
venv/bin/python run_s4/dec.py 8099 run_s4/d.${T}_g0.jsonl 246000 $a0 --log run_s4/logs/server.${T}_g0.log --ans 1024 --rounds 0 > /dev/null 2>&1 &
venv/bin/python run_s4/dec.py 8097 run_s4/d.${T}_g1.jsonl 246000 $a1 --uoff 4 --log run_s4/logs/server.${T}_g1.log --ans 1024 --rounds 0 > /dev/null 2>&1 &
wait; sleep 120
echo "- $(date -u +%FT%TZ) duo $T: job B (all agents)" >> $P_
venv/bin/python run_s4/dec.py 8099 run_s4/d.${T}_g0.jsonl 246000 $N0 --log run_s4/logs/server.${T}_g0.log --ans 1024 --rounds 2 > /dev/null 2>&1 &
venv/bin/python run_s4/dec.py 8097 run_s4/d.${T}_g1.jsonl 246000 $N1 --uoff 4 --log run_s4/logs/server.${T}_g1.log --ans 1024 --rounds 2 > /dev/null 2>&1 &
wait
echo DUO_END
