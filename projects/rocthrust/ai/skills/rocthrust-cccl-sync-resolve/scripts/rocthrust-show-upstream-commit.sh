#!/usr/bin/env bash
# rocthrust-show-upstream-commit.sh
#
# Shows a single upstream CCCL commit's message and diff, scoped to the
# Thrust subtree, plus a table mapping each touched upstream path to its
# translated rocThrust destination path, and two AMD-only counterpart
# checks (CUDA -> HIP, and testing/ -> test/) for files that have no
# upstream commit trail of their own. This is the two-way alternative to
# RCCL's meld-based 3-way diff tool — rocThrust has no merge in progress, so
# there is no "ours"/"theirs"/"working" tree to snapshot (see
# rocthrust-cccl-sync-resolve/SKILL.md's "No 3-way diff tool" section).
#
# Scope matches rocthrust-commit-list.sh: thrust/thrust/, thrust/testing/,
# and thrust/examples/ (NOT thrust/benchmarks/, thrust/cmake/,
# thrust/internal/, or thrust/scripts/ — see that script's header comment
# for why). A commit enumerated by rocthrust-commit-list.sh may show nothing
# here only if it's a false edge case; if that happens, the two scripts have
# drifted out of sync and need reconciling.
#
# Usage: rocthrust-show-upstream-commit.sh --repo <path-to-rocm-libraries> --sha <sha> [--sync-base <ref>]
#
#   --repo <path>       Absolute path to the rocm-libraries working tree (required).
#   --sha <sha>         Upstream CCCL commit to show (required).
#   --sync-base <ref>   Local commit the sync branch started from (optional).
#                       When given, the CUDA->HIP and testing/->test/
#                       counterpart checks report whether each counterpart
#                       file has already changed since this point in the
#                       current sync; without it, both checks still run but
#                       can't report that status.

set -euo pipefail

ROCTHRUST_REPO=""
SHA=""
SYNC_BASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) ROCTHRUST_REPO="$2"; shift 2 ;;
    --repo=*) ROCTHRUST_REPO="${1#--repo=}"; shift ;;
    --sha) SHA="$2"; shift 2 ;;
    --sha=*) SHA="${1#--sha=}"; shift ;;
    --sync-base) SYNC_BASE="$2"; shift 2 ;;
    --sync-base=*) SYNC_BASE="${1#--sync-base=}"; shift ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 64 ;;
  esac
done

if [[ -z "$ROCTHRUST_REPO" || -z "$SHA" ]]; then
  echo "ERROR: --repo and --sha are both required" >&2
  echo "Usage: $0 --repo <path> --sha <sha>" >&2
  exit 64
fi

cd "$ROCTHRUST_REPO"

SCOPE_PATHS=("thrust/thrust/" "thrust/testing/" "thrust/examples/")

echo "=== Commit message ==="
git show --no-patch --format='%H%n%an <%ae>%n%ad%n%n%B' "$SHA"

echo
echo "=== Touched paths (upstream -> rocThrust) ==="
mapfile -t touched < <(git diff-tree --no-commit-id --name-only -r "$SHA" -- "${SCOPE_PATHS[@]}")
if [[ ${#touched[@]} -eq 0 ]]; then
  echo "(no files under thrust/thrust/, thrust/testing/, or thrust/examples/ touched by this commit)"
else
  for path in "${touched[@]}"; do
    printf '%s\t->\tprojects/rocthrust/%s\n' "$path" "${path#thrust/}"
  done
fi

echo
echo "=== CUDA -> HIP counterpart check ==="
# Unconditional, per-commit, no de-duplication against earlier commits: if
# this commit touches thrust/system/cuda/**, report the corresponding
# thrust/system/hip/** file's existence and (with --sync-base) whether it
# has already been touched by this sync. Re-runs the same way even if an
# earlier todo.md item already flagged the same HIP file.
found_cuda_touch=0
for path in "${touched[@]}"; do
  case "$path" in
    thrust/thrust/system/cuda/*)
      found_cuda_touch=1
      local_path="projects/rocthrust/${path#thrust/}"
      hip_path="${local_path/\/system\/cuda\//\/system\/hip\/}"
      if [[ -f "$hip_path" ]]; then
        counterpart="$hip_path"
      else
        base="$(basename "$path")"
        counterpart="$(git ls-files "projects/rocthrust/thrust/system/hip/*$base" 2>/dev/null | head -1)"
      fi
      if [[ -z "${counterpart:-}" ]]; then
        printf '%s\n\t-> no HIP counterpart found (CUDA-only concept)\n' "$path"
      else
        status="unknown (no --sync-base given)"
        if [[ -n "$SYNC_BASE" ]]; then
          if git diff --quiet "$SYNC_BASE" -- "$counterpart" 2>/dev/null; then
            status="UNCHANGED since \$SYNC_BASE -- consider whether this commit's CUDA change needs a HIP-side counterpart too"
          else
            status="already changed since \$SYNC_BASE"
          fi
        fi
        printf '%s\n\t-> %s\n\t[%s]\n' "$path" "$counterpart" "$status"
      fi
      ;;
  esac
done
if [[ "$found_cuda_touch" -eq 0 ]]; then
  echo "(no thrust/system/cuda/ paths touched by this commit)"
fi

echo
echo "=== testing/ -> test/ counterpart check (AMD-only mirror) ==="
# projects/rocthrust/test/ (singular) is a separate, hand-maintained GTest
# suite with NO upstream counterpart at all -- unlike testing/, which is a
# translated 1:1 port of upstream's Catch2 suite. There is a reliable naming
# convention (testing/<name>.cu -> test/test_<name>.cpp) but no git history
# ever links the two, so a change to testing/<name>.cu (or a header it
# exercises) can silently leave test/test_<name>.cpp behind. Same
# unconditional, no-de-duplication model as the CUDA -> HIP check above.
#
# Only triggers on top-level thrust/testing/*.cu touches -- thrust/testing/
# has subdirectories (cuda/, cpp/, omp/, unittest/, cmake/, docs/) with no
# test/ counterpart pattern at all; thrust/testing/cuda/ in particular is
# its own direct-port case (projects/rocthrust/testing/cuda/), see
# rocthrust-cccl-sync-resolve/SKILL.md step 3. A commit that only touches a
# header (no thrust/testing/*.cu touch) is not caught by this check either,
# same limitation the CUDA -> HIP check has for header-only commits.
found_testing_touch=0
for path in "${touched[@]}"; do
  case "$path" in
    thrust/testing/*/*) ;; # subdirectory -- no test/ counterpart pattern, skip
    thrust/testing/*.cu)
      found_testing_touch=1
      base="$(basename "$path" .cu)"
      direct="projects/rocthrust/test/test_${base}.cpp"
      if [[ -f "$direct" ]]; then
        counterpart="$direct"
      else
        counterpart="$(git ls-files "projects/rocthrust/test/test_${base}*.cpp" 2>/dev/null | head -1)"
      fi
      if [[ -z "${counterpart:-}" ]]; then
        printf '%s\n\t-> no test/ counterpart found (testing/-only, or split across differently-named test_* files)\n' "$path"
      else
        status="unknown (no --sync-base given)"
        if [[ -n "$SYNC_BASE" ]]; then
          if git diff --quiet "$SYNC_BASE" -- "$counterpart" 2>/dev/null; then
            status="UNCHANGED since \$SYNC_BASE -- consider whether this commit's testing/ change needs a test/-side counterpart too"
          else
            status="already changed since \$SYNC_BASE"
          fi
        fi
        printf '%s\n\t-> %s\n\t[%s]\n' "$path" "$counterpart" "$status"
      fi
      ;;
  esac
done
if [[ "$found_testing_touch" -eq 0 ]]; then
  echo "(no top-level thrust/testing/*.cu paths touched by this commit)"
fi

echo
echo "=== Diff (scoped to thrust/thrust/, thrust/testing/, thrust/examples/) ==="
git show "$SHA" -- "${SCOPE_PATHS[@]}"
