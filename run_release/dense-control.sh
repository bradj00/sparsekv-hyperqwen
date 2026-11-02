#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Dense control on the INSTALLED release venv: the dense baseline config with no SPARSE_* knobs set, so every sparse patch is
# inert. Shows the dense path is unchanged by the install and gives same-GPU baselines. Dev port 8093, HQ_GPU required.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
[ -n "${HQ_GPU:-}" ] || { echo "dense-control: set HQ_GPU" >&2; exit 2; }
unset SPARSE_KV SPARSE_TIER SPARSE_TIER_SPILL
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$HQ_GPU
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TMPDIR=$REPO/run_release/tmp
export VLLM_API_KEY=${VLLM_API_KEY:-$(cat "${API_KEY_FILE:-api_key.txt}")} HOST=127.0.0.1 BIND=127.0.0.1 PORT=${PORT:-8093}
export SPEC=mtp CTX=long PREFIX_CACHE=1 EMBED_UVA=1 PTH_FA=1 MAX_LEN=${MAX_LEN:-153600} MAX_SEQS=${MAX_SEQS:-2}
export DRAFT_TOKENS=7 INT8_ACT=int8 PREFILL_ATTN=int8
export EXTRA_ARGS="--served-model-name qwen3.8-27b-q4 qwen3.8-27b"
exec bash single-user/start_qwen.sh
