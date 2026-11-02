#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# 4 x 250K on one GPU (S4 config). Job A: k=1,2 (2 cold prefills + 2 rounds); 120 s pause; job B: k=3,4 on the same
# warm server (users 0,1 hit the host cache, 2 more cold prefills). Each job < 20 min. big250.sh GPU PORT TAG
cd "$(dirname "${BASH_SOURCE[0]}")/.."
G=$1 P=$2 T=$3
P_=${PROGRESS:-run_s4/progress.log}
SPARSE_STATS_SUB=${SUB:-4} SPARSE_SELECT_EVERY=${EVERY:-2} SPARSE_SKETCH_BITS=4 MAX_LEN=262144 MAX_SEQS=4 GPU_UTIL=${UTIL:-0.94} \
  SPARSE_TIER_HOT_FRAC=${HOT:-0.025} SPARSE_TIER_HOST_GB=${HOSTGB:-33.6} run_s4/srv.sh start $G $P $T 2>&1 | grep -a "healthy\|died\|tier: runtime\|reserving\|MemoryError" | cut -c1-300
curl -sf localhost:$P/health > /dev/null || { echo "$T boot failed"; exit 1; }
echo "- $(date -u +%FT%TZ) GPU$G: 4x250K job A (k=1,2) on $T" >> $P_
venv/bin/python run_s4/dec.py $P run_s4/d.$T.jsonl 246000 ${KS_A:-1,2} --log run_s4/logs/server.$T.log --ans 1024 --rounds ${ROUNDS_A:-2} > run_s4/d.$T.outA 2>&1
sleep 120
echo "- $(date -u +%FT%TZ) GPU$G: 4x250K job B (k=3,4) on $T" >> $P_
venv/bin/python run_s4/dec.py $P run_s4/d.$T.jsonl 246000 ${KS_B:-3,4} --log run_s4/logs/server.$T.log --ans 1024 --rounds 2 > run_s4/d.$T.outB 2>&1
echo BIG_END
