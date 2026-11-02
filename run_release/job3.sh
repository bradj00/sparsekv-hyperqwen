#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# job 3 (inside a GPU lock): dense control on the installed release venv, same smoke client
cd "$(dirname "${BASH_SOURCE[0]}")/.."; SERVE=run_release/dense-control.sh TAG=dense${HQ_GPU} run_release/smoke.sh 128000
