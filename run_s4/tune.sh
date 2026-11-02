#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# suffix-knob tuning on a warm server (run_s4/sfx.json is hot-reloaded by the tier): tune.sh PORT LOG
cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=$1 LOG=$2
echo "{}" > run_s4/sfx.json
venv/bin/python run_s4/dec.py $P run_s4/t.warm.jsonl 122000 4 --log $LOG --rounds 1 --qs code > run_s4/t.warm.out 2>&1
for cfg in "{\"sparse\": false}" "{}" "{\"seg_q\": 32}" "{\"ksel\": 4096}" "{\"seg_q\": 16, \"ksel\": 4096}" "{\"seg_last\": 256, \"seg_q\": 32}" "{\"seg\": 128, \"seg_q\": 32}"; do
  echo "$cfg" > run_s4/sfx.json; sleep 2
  tag=$(echo "$cfg" | tr -dc "a-z0-9_")
  venv/bin/python run_s4/dec.py $P run_s4/t.$tag.jsonl 122000 4 --log $LOG --rounds 2 --qs code > /dev/null 2>&1
  echo "CFG $cfg: $(venv/bin/python -c "
import json,sys
rs=[json.loads(l) for l in open(\"run_s4/t.$tag.jsonl\")]
c=[r for r in rs if r.get(\"q\")==\"code\"]
print(\"code ok\", sorted((r[\"user\"],r[\"ok\"]) for r in c), \"ttft\", [r[\"ttft\"] for r in c])")"
done
echo TUNE_END
