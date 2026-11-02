#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python sparsekv/test_tier.py || exit 1
TAG=tier1b SPARSE_TIER=1 SUITES=exact PORT=8098 run_tier/two.sh
