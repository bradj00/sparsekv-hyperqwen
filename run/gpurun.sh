#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# usage: gpurun.sh <cmd...>  -- takes whichever GPU lock frees first (waits up to ~40 min)
cmd="$*"
for i in $(seq 1 240); do
  for g in ${GPUS:-0 1}; do
    if flock -n -E 75 ${GPU_LOCK_DIR:-/tmp}/gpu$g.lock -c "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g HQ_GPU=$g; echo [gpurun] GPU$g; $cmd"; then exit 0; else rc=$?; [ $rc -ne 75 ] && exit $rc; fi
  done
  sleep 10
done
echo "[gpurun] timeout"; exit 99
