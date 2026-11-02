#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# GPU0 (production card): sparse profile then tier profile, merged release
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SPARSE_PROFILE=sparse TAG=sparse${HQ_GPU}b run_release/smoke.sh 128000
SPARSE_PROFILE=tier TAG=tier${HQ_GPU}b run_release/smoke.sh 128000
