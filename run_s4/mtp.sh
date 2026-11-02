#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# MTP draft-length sweep on the sparse profile (no host tier): 2 x 60K docs, 1024-token answers. mtp.sh GPU PORT
cd "$(dirname "${BASH_SOURCE[0]}")/.."
G=$1 P=$2
for d in 3 5 7; do
  tag=mtp${d}_g$G
  DRAFT_TOKENS=$d SPARSE_PROFILE=sparse SPARSE_STATS_SUB=4 MAX_LEN=131072 MAX_SEQS=2 GPU_UTIL=0.92 run_s4/srv.sh start $G $P $tag > /dev/null 2>&1 || { echo "$tag boot failed"; continue; }
  grep -a -o "GPU KV cache size: [0-9,]* tokens" run_s4/logs/server.$tag.log | head -1
  venv/bin/python run_s4/dec.py $P run_s4/d.$tag.jsonl 60000 2 --ans 1024 --rounds 2 > /dev/null 2>&1
  run_s4/srv.sh stop $G $P $tag > /dev/null 2>&1
  echo "== MTP $d"; venv/bin/python run_s4/sum.py run_s4/d.$tag.jsonl | tail -1
done
echo MTP_END
