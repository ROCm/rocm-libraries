#!/usr/bin/env bash
# cccl-branch-drift-check.sh
#
# Phase B helper: checks whether CCCL's upstream maintenance branch for
# $TO_TAG's minor line (branch/X.Y.x, e.g. branch/3.1.x) has moved ahead of
# $TO_TAG itself in ways relevant to Thrust.
#
# Why this exists: a real manual process used elsewhere in this org (for
# hipCUB/rocPRIM's CUB sync) walks `branch/$PREV..branch/$NEW` tips, not
# release tags — because CCCL keeps backporting fixes onto branch/X.Y.x
# *after* a release is tagged, sometimes rolled into a later patch tag
# (v3.1.1, v3.1.2, ...), sometimes landed but not yet tagged at all.
# Confirmed directly against a real clone: branch/3.1.x's tip sat 31 commits
# (~7 months) ahead of v3.1.0, including a "Update branch/3.1.x to v3.1.5"
# bump commit with no v3.1.5 tag yet. Of those 31, 8 touched
# thrust/thrust, thrust/testing, or thrust/examples, and at least 3 were
# substantive fixes, not version bumps (e.g. "Always include <new> when we
# need operator new for clang-cuda", "Fix offset_iterator tests", "Ensure
# detect_wrong_difference is a valid output iterator").
#
# This check is corroboration only, like Signals B/C in cccl-version-delta.sh
# — never a hard gate. A missing branch/X.Y.x (e.g. because $TO_TAG predates
# the per-minor maintenance-branch convention, or there's no network) is
# reported plainly and is NOT treated as an error.
#
# Usage: cccl-branch-drift-check.sh --repo <path-to-rocm-libraries> \
#                                   --to <tag> [--remote cccl] [--paths "p1 p2 ..."]
#
#   --repo <path>   Absolute path to the rocm-libraries working tree (required).
#   --to <tag>      The confirmed/candidate TO_TAG, e.g. v3.1.0 (required).
#   --remote <name> Upstream CCCL remote name (default: cccl).
#   --paths <list>  Space-separated pathspecs to check, relative to the
#                   upstream repo root (default: "thrust/thrust/ thrust/testing/
#                   thrust/examples/" — the same three paths
#                   rocthrust-commit-list.sh scans).
#
# Output: a human-readable report on stdout, ending in an eval-able
# BRANCH_DRIFT_STATUS='...' line for downstream steps. Always exits 0.

set -euo pipefail

ROCTHRUST_REPO=""
TO_TAG=""
REMOTE="cccl"
PATHS="thrust/thrust/ thrust/testing/ thrust/examples/"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)    ROCTHRUST_REPO="$2"; shift 2 ;;
    --repo=*)  ROCTHRUST_REPO="${1#--repo=}"; shift ;;
    --to)      TO_TAG="$2"; shift 2 ;;
    --to=*)    TO_TAG="${1#--to=}"; shift ;;
    --remote)  REMOTE="$2"; shift 2 ;;
    --remote=*) REMOTE="${1#--remote=}"; shift ;;
    --paths)   PATHS="$2"; shift 2 ;;
    --paths=*) PATHS="${1#--paths=}"; shift ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [[ -z "$ROCTHRUST_REPO" ]]; then
  echo "ERROR: --repo <path-to-rocm-libraries> is required" >&2
  exit 64
fi
if [[ -z "$TO_TAG" ]]; then
  echo "ERROR: --to <tag> is required" >&2
  exit 64
fi

cd "$ROCTHRUST_REPO"

echo "==================== CCCL maintenance-branch drift check ===================="
echo "Target tag (--to)  : $TO_TAG"
echo "Remote              : $REMOTE"
echo "Paths checked       : $PATHS"
echo

if [[ ! "$TO_TAG" =~ ^v([0-9]+)\.([0-9]+)\.[0-9]+$ ]]; then
  echo "RESULT: could not parse major.minor out of '$TO_TAG' (expected vX.Y.Z) —"
  echo "skipping drift check."
  echo
  echo "# ---- eval-able summary ----"
  echo "BRANCH_DRIFT_STATUS='unknown (could not parse --to)'"
  exit 0
fi
TAG_MAJOR="${BASH_REMATCH[1]}"
TAG_MINOR="${BASH_REMATCH[2]}"
BRANCH_NAME="branch/${TAG_MAJOR}.${TAG_MINOR}.x"
REMOTE_BRANCH_REF="refs/remotes/${REMOTE}/${BRANCH_NAME}"

if ! git fetch "$REMOTE" "refs/heads/${BRANCH_NAME}:${REMOTE_BRANCH_REF}" -q 2>/dev/null; then
  echo "RESULT: NO MAINTENANCE BRANCH FOUND for ${BRANCH_NAME} on remote '$REMOTE'"
  echo "(not necessarily an error — older CCCL releases may predate the"
  echo "per-minor maintenance-branch convention, or the remote/network may be"
  echo "unavailable). Skipping drift check."
  echo
  echo "# ---- eval-able summary ----"
  echo "BRANCH_DRIFT_STATUS='no maintenance branch found for ${BRANCH_NAME}'"
  exit 0
fi

# shellcheck disable=SC2086 # PATHS is an intentionally word-split pathspec list
mapfile -t drift_commits < <(git log --no-merges --oneline "${TO_TAG}..${REMOTE_BRANCH_REF}" -- $PATHS)

if [[ ${#drift_commits[@]} -eq 0 ]]; then
  echo "RESULT: clean — no thrust-relevant commits beyond $TO_TAG on ${BRANCH_NAME}."
  echo
  echo "# ---- eval-able summary ----"
  echo "BRANCH_DRIFT_STATUS='clean (no thrust-relevant commits beyond ${TO_TAG} on ${BRANCH_NAME})'"
  exit 0
fi

echo "RESULT: DRIFT — ${#drift_commits[@]} commit(s) touch thrust-relevant paths on"
echo "${BRANCH_NAME} since $TO_TAG:"
for line in "${drift_commits[@]}"; do
  echo "  $line"
done
echo
echo "These are NOT included in a \$CURRENT_TAG..$TO_TAG range. Decide with the"
echo "human: bump --to to a later patch tag if one already exists and covers"
echo "them, or record them as an explicit, deliberate exclusion in the report."
echo
echo "# ---- eval-able summary ----"
echo "BRANCH_DRIFT_STATUS='DRIFT (${#drift_commits[@]} commits touch thrust-relevant paths on ${BRANCH_NAME} since ${TO_TAG})'"
