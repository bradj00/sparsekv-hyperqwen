#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Regenerate sparsekv/patches/pthfa-sparse-kv.patch from this repo's pthfa sources (S4): stock vllm_backend.py is
# blob 0b29ca4 (the base of the release patch); sparse_kv.py is a new file.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d); mkdir -p $T/a $T/b
git cat-file -p 0b29ca4 > $T/a/vllm_backend.py
cp pthfa/python/pth_flash_attn/vllm_backend.py pthfa/python/pth_flash_attn/sparse_kv.py $T/b/
(cd $T && diff -u --label a/sparse_kv.py --label b/sparse_kv.py /dev/null b/sparse_kv.py | sed "1i diff --git a/sparse_kv.py b/sparse_kv.py\nnew file mode 100644" ; \
          echo "diff --git a/vllm_backend.py b/vllm_backend.py"; diff -u --label a/vllm_backend.py --label b/vllm_backend.py a/vllm_backend.py b/vllm_backend.py) > sparsekv/patches/pthfa-sparse-kv.patch || true
rm -rf $T
echo "regenerated sparsekv/patches/pthfa-sparse-kv.patch ($(wc -l < sparsekv/patches/pthfa-sparse-kv.patch) lines)"
