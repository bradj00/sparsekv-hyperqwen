#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# paired int8 / int4 sketch recall + speed at 128K x2 (3 rounds each), after the ablations: bits.sh GPU PORT
cd "$(dirname "${BASH_SOURCE[0]}")/.."
G=$1 P=$2

echo "- $(date -u +%FT%TZ) GPU$G: int8 vs int4 sketch recall pair (128K x2, 3 rounds)" >> ${PROGRESS:-run_s4/progress.log}
for b in 4 8; do
  tag=bits${b}_g$G
  SPARSE_SKETCH_BITS=$b SPARSE_STATS_SUB=4 SPARSE_SELECT_EVERY=2 MAX_LEN=131072 MAX_SEQS=2 GPU_UTIL=0.93 SPARSE_TIER_HOT_FRAC=0.05 SPARSE_TIER_HOST_GB=4.5 \
    run_s4/srv.sh start $G $P $tag > /dev/null 2>&1 || { echo "$tag boot failed"; continue; }
  venv/bin/python run_s4/dec.py $P run_s4/d.$tag.jsonl 122000 1 --log run_s4/logs/server.$tag.log --ans 1024 --rounds 4 > /dev/null 2>&1
  run_s4/srv.sh stop $G $P $tag > /dev/null 2>&1
  echo "== bits $b"; venv/bin/python run_s4/sum.py run_s4/d.$tag.jsonl | tail -1
done
echo BITS_END
