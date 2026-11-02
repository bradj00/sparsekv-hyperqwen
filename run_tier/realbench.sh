#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
cd "$(dirname "${BASH_SOURCE[0]}")/.."
G=$(nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-gpu=name --format=csv,noheader | tr " " "_")
python sparsekv/tier_bench.py run --real $(ls ${TRACE_RUNS:-traces/runs}/*_picks.npy) --hot 0.10 0.15 0.20 --out results/$G/real.jsonl 2>&1 | grep -c RESULT
