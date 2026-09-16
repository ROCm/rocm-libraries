#!/usr/bin/env bash
# rocthrust-todo-lint.sh
#
# Checks every ticked (`- [X]`) todo.md item against the upstream commit's
# own diff to see whether it *should* have recorded a CUDA -> HIP or
# testing/ -> test/ counterpart disposition (per
# rocthrust-show-upstream-commit.sh's two counterpart checks), and flags any
# ticked item whose tick-note is silent on a disposition it should have
# recorded.
#
# Why this exists: PR 12112 (an AI-driven CCCL -> rocThrust sync) used the
# testing/ -> test/ counterpart check correctly for its first ~5-6 ported
# commits, then silently stopped recording dispositions for the remaining
# ~80 -- including the exact three commits later shown (by diffing against
# the human-authored PR 11296) to have left `test/test_*.cpp` files behind.
# The counterpart-check mechanism already existed and was demonstrably used;
# the gap was a mid-session discipline lapse, not a missing tool. This
# script makes the "was a disposition recorded" check mechanical instead of
# a prose reminder a human (or an agent) can quietly stop following.
#
# This is a presence check, not a correctness check: it does not judge
# whether a recorded disposition was the *right* call (e.g. the historical
# `reduce_into` item recorded "HIP backend deliberately untouched," which
# later turned out to be wrong on the merits) -- only that some disposition
# was written down at all, so the trail exists for later review.
#
# Usage: rocthrust-todo-lint.sh --repo <path-to-rocm-libraries> --todo <path-to-todo.md>
#
#   --repo <path>   Absolute path to the rocm-libraries working tree that
#                    holds the fetched upstream CCCL commit history
#                    (required) -- same repo rocthrust-show-upstream-commit.sh
#                    operates in.
#   --todo <path>   Path to the todo.md being linted (required).

set -euo pipefail

ROCTHRUST_REPO=""
TODO_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) ROCTHRUST_REPO="$2"; shift 2 ;;
    --repo=*) ROCTHRUST_REPO="${1#--repo=}"; shift ;;
    --todo) TODO_FILE="$2"; shift 2 ;;
    --todo=*) TODO_FILE="${1#--todo=}"; shift ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 64 ;;
  esac
done

if [[ -z "$ROCTHRUST_REPO" || -z "$TODO_FILE" ]]; then
  echo "ERROR: --repo and --todo are both required" >&2
  echo "Usage: $0 --repo <path> --todo <path>" >&2
  exit 64
fi

if [[ ! -f "$TODO_FILE" ]]; then
  echo "ERROR: todo file not found: $TODO_FILE" >&2
  exit 64
fi

cd "$ROCTHRUST_REPO"

violations=0
checked=0

# Parse todo.md into (status, sha, subject) triples for ticked items, plus
# the indented note block that follows each one (everything up to the next
# top-level "- [" line).
sha=""
note=""
in_item=0

check_item() {
  local item_sha="$1" item_note="$2"
  [[ -z "$item_sha" ]] && return

  mapfile -t touched < <(git diff-tree --no-commit-id --name-only -r "$item_sha" \
    -- thrust/thrust/ thrust/testing/ thrust/examples/ 2>/dev/null || true)
  [[ ${#touched[@]} -eq 0 ]] && return

  checked=$((checked + 1))

  local needs_hip=0 needs_test=0
  for path in "${touched[@]}"; do
    case "$path" in
      thrust/thrust/system/cuda/*) needs_hip=1 ;;
    esac
    case "$path" in
      thrust/testing/*/*) ;; # subdirectory, no test/ counterpart pattern
      thrust/testing/*.cu) needs_test=1 ;;
    esac
  done

  if [[ "$needs_hip" -eq 1 ]] && ! grep -qiE 'hip counterpart|hip backend|system/hip' <<<"$item_note"; then
    echo "VIOLATION: $item_sha touches thrust/system/cuda/ but tick-note has no HIP counterpart disposition"
    violations=$((violations + 1))
  fi

  if [[ "$needs_test" -eq 1 ]] && ! grep -qiE 'test/ counterpart|test/test_|test_[a-z_]+\.cpp' <<<"$item_note"; then
    echo "VIOLATION: $item_sha touches a top-level thrust/testing/*.cu file but tick-note has no test/ counterpart disposition"
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
echo "Checked $checked ticked items with thrust-scoped diffs; found $violations violation(s)."

if [[ "$violations" -gt 0 ]]; then
  exit 1
fi
exit 0
