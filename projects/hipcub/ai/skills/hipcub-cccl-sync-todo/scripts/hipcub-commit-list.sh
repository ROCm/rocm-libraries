#!/usr/bin/env bash
# hipcub-commit-list.sh
#
# Lists, in strict chronological (oldest-first) order, every upstream CCCL
# commit that touches the CUB subtree between two confirmed tags, and flags
# (never reorders) commits that touch a path matching sensitive-files.md or
# that are already in the baseline tag via a backport.
#
# There is no MERGED/NOT_MERGED classification here: hipCUB has no
# subtree-merge history to check commits against, and the tag range is
# already human-confirmed by the time this script runs (see
# hipcub-cccl-sync-investigate's version check).
#
# Scope: the whole upstream `cub/` subtree, so no commit that touches CUB can
# be missed (the rocThrust family learned this twice: narrower scans silently
# dropped test-only, benchmark-only and CMake-only commits). Listing a commit
# doesn't mean it changes hipCUB: the resolve stage decides that, and
# records "N/A" when it doesn't.
#
# Path translation (see header_candidates below; string-only, no existence
# check): cub/cub/<dir>/<name>.cuh -> hipcub/include/hipcub/<dir>/<name>.hpp,
# plus the same under backend/cub/ and backend/rocprim/. Only header
# translations are matched against sensitive-files.md.
#
# Usage: hipcub-commit-list.sh --repo <path-to-rocm-libraries> \
#          --from <tag> --to <tag> [--remote cccl] [--sensitive-file <path>]
#
#   --repo <path>            Absolute path to the rocm-libraries working tree (required).
#   --from <tag>             Confirmed current tag, exclusive (required).
#   --to <tag>               Confirmed target tag, inclusive (required).
#   --remote <name>          Upstream CCCL remote name (default: cccl).
#   --sensitive-file <path>  Path to sensitive-files.md (default: sibling
#                            hipcub-cccl-sync-investigate/sensitive-files.md).
#
# Output columns (tab-separated), one row per commit, oldest-first:
#   SHA   SUBJECT   PR   FLAG   SCOPE   BASELINE
#
# PR is the trailing "(#NNNN)" pulled out of the subject line, or '-' if none.
# FLAG is '⚠' if the commit touches a path whose hipCUB translation matches a
# sensitive-files.md pattern, else '-'.
# SCOPE is a comma-joined subset of the tags below:
#   HEADER    cub/cub/ public headers (everything not listed under INTERNAL)
#   INTERNAL  cub/cub/device/dispatch/, cub/cub/agent/, cub/cub/detail/
#             (no hipCUB copy; N/A or a rocPRIM follow-up)
#   TEST      cub/test/
#   EXAMPLE   cub/examples/
#   BENCH     cub/benchmarks/  (hipCUB doesn't use nvbench)
#   CMAKE     cub/cmake/, cub/**/CMakeLists.txt outside the dirs above
#   OTHER     anything else, e.g. cub/README.md, cub/.clang-tidy, docs
# BASELINE says whether the change is already in --from. CCCL tags its
# releases on branch/X.Y.x, so --from is usually not an ancestor of --to and
# FROM..TO also lists main-line commits whose backports are already in FROM:
#   =   a patch-identical commit is already in FROM (git --cherry-mark)
#   ~   a commit in FROM cites the same PR number (non-identical backport)
#   -   neither

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SENSITIVE_FILE="$SCRIPT_DIR/../../hipcub-cccl-sync-investigate/sensitive-files.md"

HIPCUB_REPO=""
FROM_TAG=""
TO_TAG=""
REMOTE="cccl"
SENSITIVE_FILE="$DEFAULT_SENSITIVE_FILE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) HIPCUB_REPO="$2"; shift 2 ;;
    --repo=*) HIPCUB_REPO="${1#--repo=}"; shift ;;
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

if [[ -z "$HIPCUB_REPO" || -z "$FROM_TAG" || -z "$TO_TAG" ]]; then
  echo "ERROR: --repo, --from, and --to are all required" >&2
  echo "Usage: $0 --repo <path> --from <tag> --to <tag> [--remote cccl] [--sensitive-file <path>]" >&2
  exit 64
fi

# See the "Scope" note at the top of this file.
SCOPE_PATHS=("cub/")
HIPCUB_INC="projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub"

cd "$HIPCUB_REPO"

git remote get-url "$REMOTE" >/dev/null 2>&1 || {
  echo "ERROR: remote '$REMOTE' not found in $HIPCUB_REPO. Add it with:" >&2
  echo "  git remote add $REMOTE https://github.com/NVIDIA/cccl.git" >&2
  exit 1
}

# Extract sensitive path patterns (relative to projects/hipccl/hipccl3/hipcub/) from the
# bullet list under '## Patterns' in sensitive-files.md.
declare -a PATTERNS=()
if [[ -f "$SENSITIVE_FILE" ]]; then
  while IFS= read -r pattern; do
    PATTERNS+=("$pattern")
  done < <(sed -n '/^## Patterns/,/^## /p' "$SENSITIVE_FILE" \
             | grep -oE '`[^`]+`' | tr -d '`' | grep -v '^$')
else
  echo "WARNING: sensitive-files.md not found at $SENSITIVE_FILE — flags will all be '-'" >&2
fi

# Keep in sync with header_candidates in hipcub-show-upstream-commit.sh.
header_candidates() {
  # arg: upstream path; prints hipCUB header paths it may correspond to
  case "$1" in
    cub/cub/*.cuh)
      local rel="${1#cub/cub/}"
      rel="${rel%.cuh}.hpp"
      echo "$HIPCUB_INC/$rel"
      echo "$HIPCUB_INC/backend/cub/$rel"
      echo "$HIPCUB_INC/backend/rocprim/$rel"
      ;;
  esac
}

compute_scope() {
  # args: list of touched upstream paths (cub/... form)
  local path
  local -A seen=()
  for path in "$@"; do
    case "$path" in
      cub/cub/device/dispatch/* | cub/cub/agent/* | cub/cub/detail/*) seen[INTERNAL]=1 ;;
      cub/cub/*) seen[HEADER]=1 ;;
      cub/test/*) seen[TEST]=1 ;;
      cub/examples/*) seen[EXAMPLE]=1 ;;
      cub/benchmarks/*) seen[BENCH]=1 ;;
      cub/cmake/* | */CMakeLists.txt) seen[CMAKE]=1 ;;
      *) seen[OTHER]=1 ;;
    esac
  done
  local -a tags=()
  local tag
  for tag in HEADER INTERNAL TEST EXAMPLE BENCH CMAKE OTHER; do
    [[ -n "${seen[$tag]:-}" ]] && tags+=("$tag")
  done
  local IFS=,
  echo "${tags[*]}"
}

is_sensitive() {
  # args: list of touched upstream paths (cub/... form)
  local path translated pattern prefix
  for path in "$@"; do
    while IFS= read -r translated; do
      [[ -z "$translated" ]] && continue
      for pattern in "${PATTERNS[@]:-}"; do
        [[ -z "$pattern" ]] && continue
        # Translate a '**' glob suffix to a simple prefix match.
        prefix="${pattern%\*\*}"
        if [[ "$prefix" != "$pattern" ]]; then
          [[ "$translated" == "projects/hipccl/hipccl3/hipcub/$prefix"* ]] && return 0
        else
          [[ "$translated" == "projects/hipccl/hipccl3/hipcub/$pattern" ]] && return 0
        fi
      done
    done < <(header_candidates "$path")
  done
  return 1
}

# Commits already in FROM_TAG, for the BASELINE column. Empty when FROM_TAG
# is an ancestor of TO_TAG (the symmetric difference has no left side).
declare -A PATCH_EQUIV=()
declare -A BASELINE_PRS=()
while read -r mark sha; do
  [[ "$mark" == "=" ]] && PATCH_EQUIV[$sha]=1
done < <(git log --no-merges --cherry-mark --right-only --format='%m %H' \
           "${FROM_TAG}...${TO_TAG}" -- "${SCOPE_PATHS[@]}")
while IFS= read -r left_subject; do
  while [[ "$left_subject" =~ \(#([0-9]+)\) ]]; do
    BASELINE_PRS[${BASH_REMATCH[1]}]=1
    left_subject="${left_subject/"${BASH_REMATCH[0]}"/}"
  done
done < <(git log --no-merges --left-only --format='%s' \
           "${FROM_TAG}...${TO_TAG}" -- "${SCOPE_PATHS[@]}")

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

      baseline="-"
      if [[ -n "${PATCH_EQUIV[$sha]:-}" ]]; then
        baseline="="
      elif [[ "$pr" != "-" && -n "${BASELINE_PRS[$pr]:-}" ]]; then
        baseline="~"
      fi

      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$sha" "$subject" "$pr" "$flag" "$scope" "$baseline"
    done
