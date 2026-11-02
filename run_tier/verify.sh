#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SPARSE_TIER_VERIFY=1 TAG=tierv SPARSE_TIER=1 SUITES=exact PORT=8094 run_tier/two.sh
grep "tier: stats" run_tier/logs/server.tierv.log | tail -3 | cut -c1-400
