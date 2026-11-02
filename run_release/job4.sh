#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# tier profile (spill) smoke on the merged release
cd "$(dirname "${BASH_SOURCE[0]}")/.."; SPARSE_PROFILE=tier TAG=tier${HQ_GPU}b run_release/smoke.sh 128000
