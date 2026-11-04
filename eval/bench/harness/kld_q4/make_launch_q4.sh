#!/bin/bash
# Q4TEST copy of eval/kld/make_launch.sh: the same symlink launcher tree (start_qwen.sh's final exec
# drops --speculative-config and adds the KLD recorder), but mirroring the q4test tree (its venv carries the
# SPARSE_DEBUG_KV_FAKEQ4 hook) into $KLD_DIR/launch-q4. The existing $KLD_DIR/launch is not touched.
set -euo pipefail
REL=${Q4REL:?q4test checkout}
KLD_DIR=${KLD_DIR:?}
L=$KLD_DIR/launch-q4
rm -rf "$L"; mkdir -p "$L/single-user"
for e in "$REL"/* "$REL"/.[!.]*; do [ -e "$e" ] || continue; b=$(basename "$e"); [ "$b" = single-user ] && continue; ln -s "$e" "$L/$b"; done
for e in "$REL"/single-user/*; do b=$(basename "$e"); [ "$b" = start_qwen.sh ] && continue; ln -s "$e" "$L/single-user/$b"; done
{
  echo '#!/bin/bash'
  cat <<'FN'
kld_exec() {
  local a=() skip=0 x
  for x in "$@"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    if [ "$x" = --speculative-config ]; then skip=1; continue; fi
    a+=("$x")
  done
  a+=(--logits-processors kld_proc:KLDRecorder)
  if [ "${PRINT_ARGV:-0}" = 1 ]; then printf '%s\n' "${a[@]}"; exit 0; fi
  exec "${a[@]}"
}
FN
  tail -n +2 "$REL/single-user/start_qwen.sh" | sed 's/^qwen_exec venv\/bin\/vllm serve/kld_exec venv\/bin\/vllm serve/'
} > "$L/single-user/start_qwen.sh"
n=$(grep -c '^kld_exec venv/bin/vllm serve' "$L/single-user/start_qwen.sh")
[ "$n" = 1 ] || { echo "make_launch_q4: expected 1 kld_exec line, got $n" >&2; exit 1; }
[ "$(readlink -f "$L/venv")" = "$(readlink -f "$REL/venv")" ] || { echo "make_launch_q4: venv link wrong" >&2; exit 1; }
echo "make_launch_q4: $L -> $REL (venv $(readlink -f "$L/venv"))"
