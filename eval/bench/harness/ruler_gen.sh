#!/bin/bash
# ruler_gen.sh N : RULER (official scripts, pinned commit) data for 13 tasks x 4 lengths, Qwen chat template, thinking off.
B=${BENCH:?}; N=${1:-10}
TOK=${MODEL_DIR:?}
export PATH=$B/venv/bin:$PATH
cd $B/ext/RULER/scripts/data
DATA_DIR=$B/data
mkdir -p $DATA_DIR/ruler $B/logs/ruler_gen
run() { L=$1 T=$2
  $B/venv/bin/python prepare.py --save_dir $DATA_DIR/ruler/$L --benchmark synthetic --task $T --tokenizer_path $TOK \
    --tokenizer_type hf --max_seq_length $L --model_template_type qwen3-nothink --num_samples $N --random_seed 42 \
    > $B/logs/ruler_gen/$T.$L.log 2>&1; echo "$T $L rc=$?"; }
export -f run; export B N TOK DATA_DIR
for L in 32768 65536 131072 196608; do for T in niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2; do echo "$L $T"; done; done | xargs -P 12 -n 2 bash -c 'run $0 $1'
