#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# mk_draft_vocab_variant.sh SRC_MODEL_DIR DST_MODEL_DIR IDS.json
#
# Build a model-directory variant with a different MTP draft vocabulary WITHOUT touching the source model directory.
# Hard-links the weight shards (read-only, ~0 disk; SRC and DST must be on one filesystem), makes real copies of the
# three files HyperQwen's prepare/build_draft_vocab.py rewrites, then slices the draft head for IDS.json from the
# quantized lm_head. IDS.json: a JSON list of token ids (sparsekv/tools/mk_draft_vocab_ids.py writes one).
#
# Run from the HyperQwen tree (it needs prepare/build_draft_vocab.py). CPU only. Environment:
#   PY=<python>      interpreter with torch + safetensors (default: venv/bin/python)
#   PREPARE=<dir>    HyperQwen's prepare/ directory (default: ./prepare)
# Then serve with MODEL=DST_MODEL_DIR. A different draft list changes the output text, so run the accuracy gate
# (eval/v12/README.md) before relying on it.
set -euo pipefail
[ $# -eq 3 ] || { sed -n '4,15p' "$0" >&2; exit 2; }
SRC=$(cd -- "$1" && pwd); DST=$2; IDS=$(cd -- "$(dirname -- "$3")" && pwd)/$(basename -- "$3")
PY=${PY:-venv/bin/python}; case "$PY" in */*) PY=$(cd -- "$(dirname -- "$PY")" && pwd)/$(basename -- "$PY") ;; esac
PREPARE=$(cd -- "${PREPARE:-prepare}" && pwd)
[ -f "$PREPARE/build_draft_vocab.py" ] || { echo "no build_draft_vocab.py in $PREPARE (set PREPARE)" >&2; exit 2; }
[ -f "$IDS" ] || { echo "no id list $IDS" >&2; exit 2; }
[ -e "$DST" ] && { echo "$DST exists; nothing done"; exit 0; }
mkdir -p "$(dirname -- "$DST")"
DST=$(cd -- "$(dirname -- "$DST")" && pwd)/$(basename -- "$DST")
[ "$(stat -c %d "$SRC")" = "$(stat -c %d "$(dirname -- "$DST")")" ] || { echo "SRC and DST are on different filesystems (hard links needed)" >&2; exit 1; }
FILES=(model_extra_tensors.safetensors mtp_draft_vocab_ids.pt model.safetensors.index.json)
sums() { (cd -- "$SRC" && sha256sum "${FILES[@]}") | sha256sum; }
before=$(sums)
cp -al "$SRC" "$DST"
for f in "${FILES[@]}"; do
  rm -f "$DST/$f"; cp -a "$SRC/$f" "$DST/$f"
  [ "$(stat -c %h "$DST/$f")" = 1 ] || { echo "still linked: $f" >&2; exit 1; }
done
(cd -- "$PREPARE" && CUDA_VISIBLE_DEVICES= "$PY" build_draft_vocab.py "$DST" --ids "$IDS") 2>&1 | tail -4
[ "$before" = "$(sums)" ] || { echo "THE SOURCE MODEL DIRECTORY CHANGED" >&2; exit 1; }
"$PY" -c "import torch,sys; t=torch.load(sys.argv[1]+'/mtp_draft_vocab_ids.pt'); print('variant', sys.argv[1], 'draft ids', t.numel())" "$DST"
