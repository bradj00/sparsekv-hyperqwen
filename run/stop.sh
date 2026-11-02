#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Stop the dev server by PID (its process group, created by setsid), wait for GPU memory release.
WT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); TAG=${TAG:-x}
P=$(cat $WT/run/server.$TAG.pid 2>/dev/null) || exit 0
[ -n "$P" ] || exit 0
PIDS=$(ps -o pid= -g $P 2>/dev/null | tr -d " ")
echo "[stop] group $P: $(echo $PIDS)"
kill -TERM $PIDS 2>/dev/null
for i in $(seq 1 60); do
  sleep 1; ALIVE=$(ps -o pid= -g $P 2>/dev/null | tr -d " "); [ -z "$ALIVE" ] && break
done
ALIVE=$(ps -o pid= -g $P 2>/dev/null | tr -d " ")
[ -n "$ALIVE" ] && { echo "[stop] SIGKILL $ALIVE"; kill -KILL $ALIVE; sleep 3; }
rm -f $WT/run/server.$TAG.pid
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
