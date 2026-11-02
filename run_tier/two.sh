#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# inside a GPU lock: tier session (shadow spill) on exact+speed suites
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPARSE_KV=1 SPARSE_IMPL=triton GPU_UTIL=${GPU_UTIL:-0.83} MAX_LEN=${MAX_LEN:-131072} HQ_GPU=$CUDA_VISIBLE_DEVICES
SPARSE_TIER=${SPARSE_TIER:-1} $WT/run_tier/session.sh ${TAG:-tier1} ${PORT:-8094} ${SUITES:-exact,speed}
