#!/bin/bash
# run_gpu.sh GPU PORT : the per-GPU queue runner, serialised on the GPU with a lock file.
# Env: BENCH (work dir: src/ = this harness, reqs/, data/, out/, queue/), REL (checkout of this repo with the
# release installed in REL/venv), HF_HOME, MODEL_DIR; optional BENCH_TOKEN (server auth token).
B=${BENCH:?}
export BENCH=$B REL=${REL:?} BENCH_TMP=$B/tmp HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
mkdir -p $B/tmp
exec flock $B/gpu$1.lock $REL/venv/bin/python -u $B/src/runner.py $1 $2
