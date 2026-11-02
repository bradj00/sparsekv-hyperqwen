#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# MemAvailable + GPU util every 20 s; flags < 8 GB
while true; do
  m=$(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo)
  g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr "\n" " ")
  echo "$(date -u +%T) avail=${m}G gpu=[$g]$(awk -v m=$m "BEGIN{if (m<8) print \" LOW\"}")"
  sleep 20
done
