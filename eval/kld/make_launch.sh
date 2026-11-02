#!/bin/bash
# Builds a launcher tree that mirrors the release repo (symlinks) with one change: single-user/start_qwen.sh's
# final exec drops --speculative-config and adds --logits-processors kld_proc:KLDRecorder.
set -euo pipefail
HQ_REPO=${HQ_REPO:?HyperQwen checkout with the release installed}
L=${KLD_DIR:?}/launch
rm -rf "$L"; mkdir -p "$L/single-user"
for e in "$HQ_REPO"/* "$HQ_REPO"/.[!.]*; do [ -e "$e" ] || continue; b=$(basename "$e"); [ "$b" = single-user ] && continue; ln -s "$e" "$L/$b"; done
for e in "$HQ_REPO"/single-user/*; do b=$(basename "$e"); [ "$b" = start_qwen.sh ] && continue; ln -s "$e" "$L/single-user/$b"; done
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
  tail -n +2 "$HQ_REPO/single-user/start_qwen.sh" | sed 's/^qwen_exec venv\/bin\/vllm serve/kld_exec venv\/bin\/vllm serve/'
} > "$L/single-user/start_qwen.sh"
grep -c '^kld_exec venv/bin/vllm serve' "$L/single-user/start_qwen.sh"
