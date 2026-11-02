#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# 128K x2 ablations on one GPU (each <= ~8 min): abl.sh GPU PORT ; variants of SPARSE_STATS_SUB / SPARSE_SELECT_EVERY /
# SPARSE_TIER_SUFFIX_SPARSE / SPARSE_TIER_SYNC_SLOTS on the current install, MAX_SEQS 2, host pool 8.8 GB.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
G=$1 P=$2
for v in "SUB=1 EVERY=1 SFX=1 SYNC=0" "SUB=4 EVERY=1 SFX=1 SYNC=0" "SUB=4 EVERY=2 SFX=1 SYNC=0" "SUB=4 EVERY=2 SFX=1 SYNC=1" "SUB=4 EVERY=2 SFX=0 SYNC=0"; do
  eval $v
  tag=abl_s${SUB}_e${EVERY}_x${SFX}_y${SYNC}_g$G
  SPARSE_STATS_SUB=$SUB SPARSE_SELECT_EVERY=$EVERY SPARSE_TIER_SUFFIX_SPARSE=$SFX SPARSE_TIER_SYNC_SLOTS=$SYNC \
    MAX_LEN=131072 MAX_SEQS=2 GPU_UTIL=0.93 SPARSE_TIER_HOT_FRAC=0.05 run_s4/srv.sh start $G $P $tag > /dev/null 2>&1 || { echo "$tag boot failed"; continue; }
  venv/bin/python run_s4/dec.py $P run_s4/d.$tag.jsonl 122000 2 --log run_s4/logs/server.$tag.log --ans 1024 --rounds 2 > /dev/null 2>&1
  run_s4/srv.sh stop $G $P $tag > /dev/null 2>&1
  echo "== $v"; venv/bin/python run_s4/sum.py run_s4/d.$tag.jsonl | tail -1
done
echo ABL_END
