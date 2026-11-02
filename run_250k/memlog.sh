#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# memlog.sh OUT: MemAvailable (MB) every 5 s
while true; do echo "$(date +%s) $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)" >> $1; sleep 5; done
