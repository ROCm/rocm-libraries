#!/usr/bin/env bash
# rocthrust-commit-list.sh
#
# Lists, in strict chronological (oldest-first) order, every upstream CCCL
# commit that touches the Thrust subtree between two confirmed tags, and
# flags (never reorders) commits that touch a path matching
# sensitive-files.md.
#
# There is no MERGED/NOT_MERGED classification here: rocThrust has no
# subtree-merge history to check commits against, and the tag range is
# already human-confirmed by the time this script runs (see
# rocthrust-cccl-sync-investigate's Signal A-D version check). Every commit
# this script prints needs to be ported.
#
# Scope: the whole upstream `thrust/` subtree, so no commit that touches
# Thrust can be missed. Earlier versions scanned only thrust/thrust/, then
# only thrust/{thrust,testing,examples}/; each time, commits confined to the
# unscanned paths (tests, then benchmarks, CMake, README/.gitignore) were
# silently never enumerated, even though rocThrust's parity list includes
# them. Listing a commit doesn't mean it changes rocThrust: the resolve
# stage decides that, and records "N/A" when it doesn't.
#
# Path translation (see translate_path below):
#   thrust/benchmarks/<x>  -> projects/rocthrust/benchmark/<x>  (plural -> singular)
#   thrust/<anything else> -> projects/rocthrust/<anything else>
#
# Usage: rocthrust-commit-list.sh --repo <path-to-rocm-libraries> \
#          --from <tag> --to <tag> [--remote cccl] [--sensitive-file <path>]
#
#   --repo <path>            Absolute path to the rocm-libraries working tree (required).
#   --from <tag>             Confirmed current tag, exclusive (required).
#   --to <tag>               Confirmed target tag, inclusive (required).
#   --remote <name>          Upstream CCCL remote name (default: cccl).
#   --sensitive-file <path>  Path to sensitive-files.md (default: sibling
#                            rocthrust-cccl-sync-investigate/sensitive-files.md).
#
# Output columns (tab-separated), one row per commit, oldest-first:
#   SHA   SUBJECT   PR   FLAG   SCOPE
#
# PR is the trailing "(#NNNN)" pulled out of the subject line, or '-' if none.
# FLAG is '⚠' if the commit touches a path matching a sensitive-files.md
# pattern (after translating thrust/... to projects/rocthrust/thrust/...),
# else '-'.
# SCOPE is a comma-joined subset of the tags below, naming which parts of
# thrust/ the commit touches. Read it before assuming a commit needs
# CUDA/HIP source-level porting; most non-HEADER commits don't.
#   HEADER   thrust/thrust/
#   TEST     thrust/testing/
#   EXAMPLE  thrust/examples/
#   BENCH    thrust/benchmarks/  (rocThrust uses Google Benchmark, not nvbench)
#   CMAKE    thrust/cmake/, thrust/**/CMakeLists.txt outside the dirs above
#            (rocThrust's CMake is its own; usually N/A)
#   OTHER    anything else, e.g. thrust/README.md, thrust/.gitignore, docs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SENSITIVE_FILE="$SCRIPT_DIR/../../rocthrust-cccl-sync-investigate/sensitive-files.md"

ROCTHRUST_REPO=""
FROM_TAG=""
TO_TAG=""
REMOTE="cccl"
SENSITIVE_FILE="$DEFAULT_SENSITIVE_FILE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) ROCTHRUST_REPO="$2"; shift 2 ;;
    --repo=*) ROCTHRUST_REPO="${1#--repo=}"; shift ;;
    --from) FROM_TAG="$2"; shift 2 ;;
    --from=*) FROM_TAG="${1#--from=}"; shift ;;
    --to) TO_TAG="$2"; shift 2 ;;
    --to=*) TO_TAG="${1#--to=}"; shift ;;
    --remote) REMOTE="$2"; shift 2 ;;
    --remote=*) REMOTE="${1#--remote=}"; shift ;;
    --sensitive-file) SENSITIVE_FILE="$2"; shift 2 ;;
    --sensitive-file=*) SENSITIVE_FILE="${1#--sensitive-file=}"; shift ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 64 ;;
  esac
done

if [[ -z "$ROCTHRUST_REPO" || -z "$FROM_TAG" || -z "$TO_TAG" ]]; then
  echo "ERROR: --repo, --from, and --to are all required" >&2
  echo "Usage: $0 --repo <path> --from <tag> --to <tag> [--remote cccl] [--sensitive-file <path>]" >&2
  exit 64
fi

# See the "Scope" note at the top of this file.
SCOPE_PATHS=("thrust/")

cd "$ROCTHRUST_REPO"

git remote get-url "$REMOTE" >/dev/null 2>&1 || {
  echo "ERROR: remote '$REMOTE' not found in $ROCTHRUST_REPO. Add it with:" >&2
  echo "  git remote add $REMOTE https://github.com/NVIDIA/cccl.git" >&2
  exit 1
}

# Extract sensitive path patterns (relative to projects/rocthrust/) from the
# bullet list under '## Patterns' in sensitive-files.md. Skip if the file is
# missing rather than fail — the caller is warned but the list still prints.
declare -a PATTERNS=()
if [[ -f "$SENSITIVE_FILE" ]]; then
  while IFS= read -r pattern; do
    PATTERNS+=("$pattern")
  done < <(sed -n '/^## Patterns/,/^## /p' "$SENSITIVE_FILE" \
             | grep -oE '`[^`]+`' | tr -d '`' | grep -v '^$')
else
  echo "WARNING: sensitive-files.md not found at $SENSITIVE_FILE — flags will all be '-'" >&2
fi

translate_path() {
  # arg: upstream path (thrust/... form) -> rocThrust path
  case "$1" in
    thrust/benchmarks/*) echo "projects/rocthrust/benchmark/${1#thrust/benchmarks/}" ;;
    *) echo "projects/rocthrust/${1#thrust/}" ;;
  esac
}

compute_scope() {
  # args: list of touched upstream paths (thrust/... form)
  local path
  local -A seen=()
  for path in "$@"; do
    case "$path" in
      thrust/thrust/*) seen[HEADER]=1 ;;
      thrust/testing/*) seen[TEST]=1 ;;
      thrust/examples/*) seen[EXAMPLE]=1 ;;
      thrust/benchmarks/*) seen[BENCH]=1 ;;
      thrust/cmake/* | */CMakeLists.txt) seen[CMAKE]=1 ;;
      *) seen[OTHER]=1 ;;
    esac
  done
  local -a tags=()
  local tag
  for tag in HEADER TEST EXAMPLE BENCH CMAKE OTHER; do
    [[ -n "${seen[$tag]:-}" ]] && tags+=("$tag")
  done
  local IFS=,
  echo "${tags[*]}"
}

is_sensitive() {
  # args: list of touched upstream paths (thrust/... form)
  local path translated pattern
  for path in "$@"; do
    translated="$(translate_path "$path")"
    for pattern in "${PATTERNS[@]:-}"; do
      [[ -z "$pattern" ]] && continue
      # Translate a '**' glob suffix to a simple prefix match.
      local prefix="${pattern%\*\*}"
      if [[ "$prefix" != "$pattern" ]]; then
        [[ "$translated" == "projects/rocthrust/$prefix"* ]] && return 0
      else
        [[ "$translated" == "projects/rocthrust/$pattern" ]] && return 0
      fi
    done
  done
  return 1
}

git log --no-merges --reverse --format='%H%x09%s' "${FROM_TAG}..${TO_TAG}" -- "${SCOPE_PATHS[@]}" \
  | while IFS=$'\t' read -r sha subject; do
      pr="-"
      if [[ "$subject" =~ \(#([0-9]+)\)[[:space:]]*$ ]]; then
        pr="${BASH_REMATCH[1]}"
      fi

      mapfile -t touched < <(git diff-tree --no-commit-id --name-only -r "$sha" -- "${SCOPE_PATHS[@]}")

      flag="-"
      if is_sensitive "${touched[@]}"; then
        flag="⚠"
      fi

      scope="$(compute_scope "${touched[@]}")"

      printf '%s\t%s\t%s\t%s\t%s\n' "$sha" "$subject" "$pr" "$flag" "$scope"
    done
