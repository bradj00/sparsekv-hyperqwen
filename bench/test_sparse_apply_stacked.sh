#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# CPU test for `patches/apply.sh --sparse` / `--sparse-state` with STACKED sparse patches (s17): two sparse-*.patch
# entries that touch the same file (as sparse-parser-qwen3-backports.patch and sparse-frontend-backports.patch do in
# parser/qwen3.py). Before s17 a re-run tested each patch with an isolated `patch -R --dry-run`; the earlier patch of
# a stack does not reverse in a fully installed tree, so the re-run tried to apply it again and left .rej files.
#
# Builds a scratch copy of patches/apply.sh with a synthetic series (a stacked pair on one file, an independent patch
# and one that creates a file) and a scratch package dir; applies, re-applies, and asserts no .rej/.orig files and
# identical content. No GPU, no venv, no network.
#
#   bash bench/test_sparse_apply_stacked.sh
#   VLLM_DIR=<site-packages>/vllm bash bench/test_sparse_apply_stacked.sh
#       also: copy the files the REAL sparse patches touch out of an installed (sparse-patched) vllm, and check that
#       the real apply.sh --sparse finds every patch applied there and changes nothing. VLLM_DIR is only read.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/repo/patches" "$T/repo/kvarn" "$T/v0/sub" "$T/v1/sub" "$T/v2/sub" "$T/v3/sub" "$T/v4/sub"
cp "$REPO/patches/apply.sh" "$T/repo/patches/apply.sh"
AP="$T/repo/patches/apply.sh"

# tree states: v0 base, v1 = +lower, v2 = +upper (same lines again), v3 = +other, v4 = +newfile
printf 'def f():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n' > "$T/v0/sub/stack.py"
printf 'x = 1\ny = 2\nz = 3\n' > "$T/v0/other.py"
for v in 1 2 3 4; do cp -r "$T/v0/." "$T/v$v/"; done
printf 'def f():\n    a = 1\n    b = 20\n    c = 3\n    return a + b + c\n' > "$T/v1/sub/stack.py"
for v in 2 3 4; do printf 'def f():\n    a = 1\n    b = 200\n    c = 3\n    d = 4\n    return a + b + c + d\n' > "$T/v$v/sub/stack.py"; done
for v in 3 4; do printf 'x = 1\ny = 22\nz = 3\n' > "$T/v$v/other.py"; done
printf 'new = True\n' > "$T/v4/sub/new.py"
mk() { (cd "$T" && diff -urN "$1" "$2" | sed -e "s|^--- $1/|--- a/|" -e "s|^+++ $2/|+++ b/|" -e '/^diff /d') > "$T/repo/patches/$3"; }
mk v0 v1 sparse-lower.patch
mk v1 v2 sparse-upper.patch
mk v2 v3 sparse-other.patch
mk v3 v4 sparse-newfile.patch
printf 'sparse-lower.patch\nsparse-other.patch\nsparse-upper.patch\nsparse-newfile.patch\n' > "$T/repo/patches/series"
# series order differs from the order the patches were cut in only for the independent one (other.py), which is fine

sum_tree() { (cd "$1" && find . -type f | sort | xargs sha256sum | sha256sum | cut -c1-16); }
litter() { find "$1" \( -name '*.rej' -o -name '*.orig' \) | wc -l | tr -d ' '; }
WANT=$(sum_tree "$T/v4")

echo "== fresh apply"
cp -r "$T/v0" "$T/pkg"
out=$(bash "$AP" --sparse "$T/pkg" 2>&1); check 0 $? "apply.sh --sparse on the base tree exits 0"
check 4 "$(printf '%s\n' "$out" | grep -c '^== sparse-[a-z]*\.patch$')" "four patches applied"
check "$WANT" "$(sum_tree "$T/pkg")" "content is the fully patched tree"
check 0 "$(litter "$T/pkg")" "no .rej/.orig"

echo "== the old isolated check is wrong for the lower patch of a stack (negative control)"
patch -p1 -R --dry-run -s --fuzz 0 -d "$T/pkg" -i "$T/repo/patches/sparse-lower.patch" >/dev/null 2>&1
check 1 $? "isolated patch -R --dry-run of sparse-lower.patch fails in the fully patched tree"

echo "== re-apply (idempotence)"
out=$(bash "$AP" --sparse "$T/pkg" 2>&1); check 0 $? "second apply.sh --sparse exits 0"
check 4 "$(printf '%s\n' "$out" | grep -c '(already applied)$')" "all four reported already applied"
check "$WANT" "$(sum_tree "$T/pkg")" "content identical after the re-apply"
check 0 "$(litter "$T/pkg")" "no .rej/.orig after the re-apply"
check "sparse-lower.patch applied,sparse-other.patch applied,sparse-upper.patch applied,sparse-newfile.patch applied," \
  "$(bash "$AP" --sparse-state "$T/pkg" | tr '\n' ',')" "--sparse-state: all applied, series order"

echo "== only the upper patch of the stack missing"
patch -p1 -R -s --fuzz 0 --no-backup-if-mismatch -d "$T/pkg" -i "$T/repo/patches/sparse-upper.patch"
check "sparse-lower.patch applied,sparse-other.patch applied,sparse-upper.patch absent,sparse-newfile.patch applied," \
  "$(bash "$AP" --sparse-state "$T/pkg" | tr '\n' ',')" "--sparse-state names the missing one"
out=$(bash "$AP" --sparse "$T/pkg" 2>&1); check 0 $? "apply.sh --sparse exits 0"
check "== sparse-upper.patch" "$(printf '%s\n' "$out" | grep -v 'already applied')" "only sparse-upper.patch applied"
check "$WANT" "$(sum_tree "$T/pkg")" "content is the fully patched tree again"
check 0 "$(litter "$T/pkg")" "no .rej/.orig"

echo "== independent patch missing under later ones (a tree installed before the patch joined the series)"
patch -p1 -R -s --fuzz 0 --no-backup-if-mismatch -d "$T/pkg" -i "$T/repo/patches/sparse-other.patch"
out=$(bash "$AP" --sparse "$T/pkg" 2>&1); check 0 $? "apply.sh --sparse exits 0"
check "== sparse-other.patch" "$(printf '%s\n' "$out" | grep -v 'already applied')" "only sparse-other.patch applied"
check "$WANT" "$(sum_tree "$T/pkg")" "content is the fully patched tree again"

echo "== --sparse-state changes nothing and leaves no scratch dir"
mkdir "$T/tmp"
TMPDIR="$T/tmp" bash "$AP" --sparse-state "$T/pkg" >/dev/null; TMPDIR="$T/tmp" bash "$AP" --sparse "$T/pkg" >/dev/null
check 0 "$(find "$T/tmp" -mindepth 1 | wc -l | tr -d ' ')" "scratch copies removed"
check "$WANT" "$(sum_tree "$T/pkg")" "content unchanged"

if [ -n "${VLLM_DIR:-}" ]; then
  echo "== real series on a copy of the files the sparse patches touch in $VLLM_DIR"
  mkdir "$T/real"
  for name in $(bash "$REPO/patches/apply.sh" --list | grep '^sparse-.*\.patch$'); do
    for f in $(sed -n -e 's|^+++ b/||p' "$REPO/patches/$name" | cut -f1); do
      [ -f "$VLLM_DIR/$f" ] && { mkdir -p "$T/real/$(dirname "$f")"; cp "$VLLM_DIR/$f" "$T/real/$f"; }
    done
  done
  before=$(sum_tree "$T/real")
  n=$(bash "$REPO/patches/apply.sh" --list | grep -c '^sparse-.*\.patch$')
  out=$(bash "$REPO/patches/apply.sh" --sparse "$T/real" 2>&1); check 0 $? "real apply.sh --sparse exits 0"
  check "$n" "$(printf '%s\n' "$out" | grep -c '(already applied)$')" "all $n real sparse patches already applied"
  check "$before" "$(sum_tree "$T/real")" "content identical"
  check 0 "$(litter "$T/real")" "no .rej/.orig"
fi

if [ $FAILS = 0 ]; then echo "ALL OK"; else echo "$FAILS FAILED"; exit 1; fi
