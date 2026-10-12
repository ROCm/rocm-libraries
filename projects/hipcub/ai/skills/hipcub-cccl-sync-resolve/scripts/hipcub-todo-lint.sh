#!/usr/bin/env bash
# hipcub-todo-lint.sh
#
# Checks every ticked (`- [X]`) todo.md item against the upstream commit's
# own diff to see whether it *should* have recorded a backend-parity,
# rocPRIM follow-up, test counterpart or benchmark counterpart disposition
# (per hipcub-show-upstream-commit.sh's counterpart checks), and flags any
# ticked item whose tick-note is silent on one.
#
# Why this exists: in the rocThrust family, an AI-driven sync used its
# counterpart check correctly for the first handful of items, then silently
# stopped recording dispositions for the rest, including the commits that
# later turned out to have left AMD-only files behind. This script makes the
# "was a disposition recorded" check mechanical instead of a prose reminder.
#
# This is a presence check, not a correctness check: it doesn't judge whether
# a recorded disposition was the right call, only that one was written down.
#
# Usage: hipcub-todo-lint.sh --repo <path-to-rocm-libraries> --todo <path-to-todo.md>
#
#   --repo <path>   Absolute path to the rocm-libraries working tree that
#                   holds the fetched upstream CCCL commit history (required).
#   --todo <path>   Path to the todo.md being linted (required).

set -euo pipefail

HIPCUB_REPO=""
TODO_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) HIPCUB_REPO="$2"; shift 2 ;;
    --repo=*) HIPCUB_REPO="${1#--repo=}"; shift ;;
    --todo) TODO_FILE="$2"; shift 2 ;;
    --todo=*) TODO_FILE="${1#--todo=}"; shift ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 64 ;;
  esac
done

if [[ -z "$HIPCUB_REPO" || -z "$TODO_FILE" ]]; then
  echo "ERROR: --repo and --todo are both required" >&2
  echo "Usage: $0 --repo <path> --todo <path>" >&2
  exit 64
fi

if [[ ! -f "$TODO_FILE" ]]; then
  echo "ERROR: todo file not found: $TODO_FILE" >&2
  exit 64
fi

cd "$HIPCUB_REPO"

violations=0
checked=0

sha=""
note=""
in_item=0

check_item() {
  local item_sha="$1" item_note="$2"
  [[ -z "$item_sha" ]] && return

  mapfile -t touched < <(git diff-tree --no-commit-id --name-only -r "$item_sha" \
    -- cub/ 2>/dev/null || true)
  [[ ${#touched[@]} -eq 0 ]] && return

  checked=$((checked + 1))

  local needs_parity=0 needs_rocprim=0 needs_test=0 needs_bench=0 path
  for path in "${touched[@]}"; do
    case "$path" in
      cub/cub/device/dispatch/* | cub/cub/agent/* | cub/cub/detail/*) needs_rocprim=1 ;;
      cub/cub/*.cuh) needs_parity=1 ;;
      cub/test/*catch2_test_*.cu) needs_test=1 ;;
      cub/benchmarks/bench/*.cu) needs_bench=1 ;;
    esac
  done

  if [[ "$needs_parity" -eq 1 ]] && ! grep -qiE 'backend parity|backend/rocprim|backend/cub|rocprim backend|cub backend' <<<"$item_note"; then
    echo "VIOLATION: $item_sha touches a public cub/cub/ header but tick-note has no backend parity disposition"
    violations=$((violations + 1))
  fi

  if [[ "$needs_rocprim" -eq 1 ]] && ! grep -qiE 'rocprim follow-up|projects/hipccl/hipccl3/rocprim' <<<"$item_note"; then
    echo "VIOLATION: $item_sha touches cub dispatch/agent/detail but tick-note has no rocPRIM follow-up disposition"
    violations=$((violations + 1))
  fi

  if [[ "$needs_test" -eq 1 ]] && ! grep -qiE 'test counterpart|test_hipcub_' <<<"$item_note"; then
    echo "VIOLATION: $item_sha touches a cub/test/**/catch2_test_*.cu file but tick-note has no test counterpart disposition"
    violations=$((violations + 1))
  fi

  if [[ "$needs_bench" -eq 1 ]] && ! grep -qiE 'benchmark counterpart|benchmark_(device|block|warp)_' <<<"$item_note"; then
    echo "VIOLATION: $item_sha touches a cub/benchmarks/bench/**/*.cu file but tick-note has no benchmark counterpart disposition"
    violations=$((violations + 1))
  fi
}

while IFS= read -r line; do
  if [[ "$line" =~ ^-\ \[X\]\ ([0-9a-f]{7,40})\  ]]; then
    if [[ "$in_item" -eq 1 ]]; then
      check_item "$sha" "$note"
    fi
    sha="${BASH_REMATCH[1]}"
    note=""
    in_item=1
  elif [[ "$line" =~ ^-\ \[\ \] ]]; then
    if [[ "$in_item" -eq 1 ]]; then
      check_item "$sha" "$note"
    fi
    sha=""
    note=""
    in_item=0
  elif [[ "$in_item" -eq 1 ]]; then
    note+="$line"$'\n'
  fi
done < "$TODO_FILE"

if [[ "$in_item" -eq 1 ]]; then
  check_item "$sha" "$note"
fi

echo
echo "Checked $checked ticked items with cub-scoped diffs; found $violations violation(s)."

if [[ "$violations" -gt 0 ]]; then
  exit 1
fi
exit 0
