#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# lockrun.sh GPU cmd...  -- blocking wait (kernel queue) on one GPU lock, up to 3 x 25 min
G=$1; shift; cmd="$*"
for i in 1 2 3; do
  flock -w 1500 -E 75 ${GPU_LOCK_DIR:-/tmp}/gpu$G.lock -c "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G HQ_GPU=$G; echo [lock] GPU$G \$(date +%T); $cmd" && exit 0
  rc=$?; [ $rc -ne 75 ] && exit $rc
done
echo "[lockrun] gave up"; exit 99
