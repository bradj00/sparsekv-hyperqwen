#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# GPU1 part of the 8-agent run: wait for GPU1 to be free, boot N x 250K (int4 host), cold-prefill all docs in two jobs
# (docs uoff 4), then the JOINT job: both servers run 2 rounds with all agents at once. g1part.sh N
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N=${1:-4}; T=int4_g1; P_=${PROGRESS:-run_s4/progress.log}
until [ ! -f run_s4/server.g1.pid ] && [ -f run_s4/GPU1_FREE ]; do sleep 10; done
SPARSE_TIER_HOST_BITS=4 SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2 SPARSE_SKETCH_BITS=4 MAX_LEN=262144 MAX_SEQS=$N GPU_UTIL=0.94 \
  SPARSE_TIER_HOT_FRAC=0.025 SPARSE_TIER_HOST_GB=$(python3 -c "print(round($N*4.4,1))") run_s4/srv.sh start 1 8097 $T 2>&1 | grep -a "healthy\|died\|tier: runtime\|MemoryError" | cut -c1-300
echo "- $(date -u +%FT%TZ) GPU1: ${N}x250K int4 host booted ($T); cold-prefilling docs (uoff 4)" >> $P_
venv/bin/python run_s4/dec.py 8097 run_s4/d.$T.jsonl 246000 2 --uoff 4 --log run_s4/logs/server.$T.log --ans 1024 --rounds 0 > /dev/null 2>&1
sleep 60
venv/bin/python run_s4/dec.py 8097 run_s4/d.$T.jsonl 246000 $N --uoff 4 --log run_s4/logs/server.$T.log --ans 1024 --rounds 0 > /dev/null 2>&1
until grep -q BIG_END run_s4/int4_g0.out; do sleep 10; done
sleep 30
echo "- $(date -u +%FT%TZ) JOINT: GPU0 4x250K + GPU1 ${N}x250K, all agents concurrently, 2 rounds" >> $P_
venv/bin/python run_s4/dec.py 8099 run_s4/d.joint_g0.jsonl 246000 4 --log run_s4/logs/server.int4_g0.log --ans 1024 --rounds 2 > /dev/null 2>&1 &
venv/bin/python run_s4/dec.py 8097 run_s4/d.joint_g1.jsonl 246000 $N --uoff 4 --log run_s4/logs/server.$T.log --ans 1024 --rounds 2 > /dev/null 2>&1 &
wait
echo JOINT_END
