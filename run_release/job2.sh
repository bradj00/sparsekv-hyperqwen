#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# job 2 (inside a GPU lock, HQ_GPU set): GPU unit tests on the installed modules, then the sparse-profile smoke test.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); cd $REPO
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$HQ_GPU
echo "[job2] test_fixes_gpu"; timeout 300 venv/bin/python sparsekv/test_fixes_gpu.py 2>&1 | grep -v -i warn | tail -8
echo "[job2] test_tier"; timeout 300 venv/bin/python sparsekv/test_tier.py 2>&1 | grep -v -i warn | tail -6
echo "[job2] smoke sparse profile"
unset CUDA_VISIBLE_DEVICES
SPARSE_PROFILE=sparse TAG=smoke2 run_release/smoke.sh 128000
