#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# One GPU job (run via gpurun.sh): correctness test, kernel split, full tier benchmark. Output tagged by card.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
G=$(nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-gpu=name --format=csv,noheader | tr ' ' '_')
OUT=results/$G; mkdir -p $OUT
python sparsekv/test_tier.py 2>&1 | tee $OUT/test.log
python sparsekv/tier_kbench.py --ctx 131072 2>&1 | tee $OUT/kbench.log
rm -f $OUT/bench.jsonl
python sparsekv/tier_bench.py run --ctx 131072 262144 --preset lab heavy brief --hot 0.05 0.10 0.20 --baseline --out $OUT/bench.jsonl 2>&1 | grep -v Warn | tail -3
python sparsekv/tier_bench.py run --ctx 131072 --nreq 2 --preset lab --hot 0.10 --baseline --out $OUT/bench.jsonl 2>&1 | tail -1
REAL=$(ls ${TRACE_RUNS:-traces/runs}/*_full_picks.npy 2>/dev/null | head -8)
[ -n "$REAL" ] && python sparsekv/tier_bench.py run --real $REAL --hot 0.10 0.20 --out $OUT/bench.jsonl 2>&1 | tail -1 || true
echo ALLDONE
