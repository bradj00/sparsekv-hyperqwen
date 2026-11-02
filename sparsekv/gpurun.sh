#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# usage: [GPU=0|1] [WAIT=secs] gpurun.sh <cmd...>
# Polls both dev-GPU locks (gpu0 = 3090 Ti x16, gpu1 = 3090 x4) non-blockingly until one is free, then runs the
# command (max 18 min) on that card. GPU=0|1 restricts to one card.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=${DEV_VENV:+$DEV_VENV/bin:}$PATH
CMD="$*"; WAIT=${WAIT:-3600}; t0=$(date +%s)
cards=${GPU:-"0 1"}
while :; do
  for g in $cards; do
    CUDA_VISIBLE_DEVICES=$g flock -n -E 99 ${GPU_LOCK_DIR:-/tmp}/gpu$g.lock -c "echo [gpu$g \$(nvidia-smi -i $g --query-gpu=name --format=csv,noheader)]; timeout 1080 $CMD"
    rc=$?; [ $rc -ne 99 ] && exit $rc
  done
  [ $(( $(date +%s) - t0 )) -gt $WAIT ] && { echo "gpurun: no GPU lock within ${WAIT}s"; exit 98; }
  sleep 2
done
