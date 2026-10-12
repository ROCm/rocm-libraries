#!/usr/bin/env bash
# hipcub-show-upstream-commit.sh
#
# Shows a single upstream CCCL commit's message and diff, scoped to the CUB
# subtree, plus a table mapping each touched upstream path to the hipCUB
# file(s) it may correspond to, and four counterpart checks:
#
#   - backend parity: for each touched public header, which of hipCUB's
#     top-level forwarder, backend/rocprim/ and backend/cub/ headers exist,
#     and (with --sync-base) whether each has changed during this sync;
#   - rocPRIM follow-up: for touched dispatch/agent/detail files, the
#     rocPRIM header that implements the same algorithm;
#   - tests: cub/test/catch2_test_<x>.cu -> test/hipcub/test_hipcub_<x>.cpp;
#   - benchmarks: cub/benchmarks/bench/<algo>/ -> benchmark/benchmark_device_<algo>.cpp.
#
# hipCUB is a wrapper, not a copy of CUB, so none of these mappings is
# exact: they're name heuristics. A miss means "look for it by hand", not
# "hipCUB doesn't have it".
#
# Scope matches hipcub-commit-list.sh: the whole upstream cub/ subtree.
# Every commit that script enumerates shows a non-empty diff here; if one
# doesn't, the two scripts have drifted apart.
#
# Usage: hipcub-show-upstream-commit.sh --repo <path-to-rocm-libraries> --sha <sha> [--sync-base <ref>]
#
#   --repo <path>       Absolute path to the rocm-libraries working tree (required).
#   --sha <sha>         Upstream CCCL commit to show (required).
#   --sync-base <ref>   Local commit the sync branch started from (optional).
#                       When given, the counterpart checks report whether each
#                       hipCUB file has already changed since this point in
#                       the current sync.

set -euo pipefail

HIPCUB_REPO=""
SHA=""
SYNC_BASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) HIPCUB_REPO="$2"; shift 2 ;;
    --repo=*) HIPCUB_REPO="${1#--repo=}"; shift ;;
    --sha) SHA="$2"; shift 2 ;;
    --sha=*) SHA="${1#--sha=}"; shift ;;
    --sync-base) SYNC_BASE="$2"; shift 2 ;;
    --sync-base=*) SYNC_BASE="${1#--sync-base=}"; shift ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 64 ;;
  esac
done

if [[ -z "$HIPCUB_REPO" || -z "$SHA" ]]; then
  echo "ERROR: --repo and --sha are both required" >&2
  echo "Usage: $0 --repo <path> --sha <sha> [--sync-base <ref>]" >&2
  exit 64
fi

cd "$HIPCUB_REPO"

SCOPE_PATHS=("cub/")
HIPCUB="projects/hipccl/hipccl3/hipcub"
HIPCUB_INC="$HIPCUB/hipcub/include/hipcub"
ROCPRIM_DEVICE="projects/hipccl/hipccl3/rocprim/rocprim/include/rocprim/device"

# Keep in sync with header_candidates in hipcub-commit-list.sh.
header_candidates() {
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

is_internal() {
  case "$1" in
    cub/cub/device/dispatch/* | cub/cub/agent/* | cub/cub/detail/*) return 0 ;;
  esac
  return 1
}

# Glob for a file named <prefix><stem>*<suffix-glob>, dropping trailing
# _word segments from <stem> until something matches. Prints the matches.
progressive_glob() {
  local dir="$1" prefix="$2" stem="$3" matches
  while [[ -n "$stem" ]]; do
    matches="$(git ls-files "$dir/${prefix}${stem}*" 2>/dev/null)"
    if [[ -n "$matches" ]]; then
      echo "$matches"
      return 0
    fi
    [[ "$stem" == *_* ]] || break
    stem="${stem%_*}"
  done
  return 1
}

test_counterpart() {
  # cub/test/**/catch2_test_<x>.cu -> test/hipcub/test_hipcub_<x>*.cpp(.in)
  local base
  base="$(basename "$1" .cu)"
  base="${base#catch2_test_}"
  progressive_glob "$HIPCUB/test/hipcub" "test_hipcub_" "$base" \
    || progressive_glob "$HIPCUB/test/hipcub" "test_hipcub_" "${base}s"
}

bench_counterpart() {
  # cub/benchmarks/bench/<algo>/... -> benchmark/benchmark_device_<algo>*.cpp
  local algo="$1"
  progressive_glob "$HIPCUB/benchmark" "benchmark_device_" "$algo" \
    || git ls-files "$HIPCUB/benchmark/benchmark_*${algo}*.cpp" 2>/dev/null | grep .
}

example_counterpart() {
  # cub/examples/<sub>/<name>.cu -> examples/<sub>/<name>.cu or .cpp
  local rel="${1#cub/examples/}" stem
  stem="${rel%.cu}"
  git ls-files "$HIPCUB/examples/$stem.cu" "$HIPCUB/examples/$stem.cpp" 2>/dev/null | grep .
}

sync_status() {
  # arg: hipCUB path; prints its change status since --sync-base
  if [[ -z "$SYNC_BASE" ]]; then
    echo "unknown (no --sync-base given)"
  elif git diff --quiet "$SYNC_BASE" -- "$1" 2>/dev/null; then
    echo "UNCHANGED since \$SYNC_BASE"
  else
    echo "already changed since \$SYNC_BASE"
  fi
}

echo "=== Commit message ==="
git show --no-patch --format='%H%n%an <%ae>%n%ad%n%n%B' "$SHA"

mapfile -t touched < <(git diff-tree --no-commit-id --name-only -r "$SHA" -- "${SCOPE_PATHS[@]}")

echo
echo "=== Touched paths (upstream -> hipCUB) ==="
# A = added, M = modified, D = deleted upstream. Existing hipCUB files are
# listed; "[no local file]" means none of the heuristics found one: upstream
# tooling hipCUB doesn't use, an internal file, or a hipCUB file under a
# different name (look for it before calling it N/A).
if [[ ${#touched[@]} -eq 0 ]]; then
  echo "(no files under cub/ touched by this commit)"
else
  while IFS=$'\t' read -r status path; do
    printf '%s %s\n' "$status" "$path"
    if is_internal "$path"; then
      printf '\t-> [internal: no hipCUB copy; see rocPRIM follow-up check]\n'
      continue
    fi
    local_files=""
    case "$path" in
      cub/cub/*.cuh)
        local_files="$(header_candidates "$path" | while read -r f; do
                         if git cat-file -e "HEAD:$f" 2>/dev/null; then echo "$f"; fi
                       done)" ;;
      cub/test/*catch2_test_*.cu) local_files="$(test_counterpart "$path" || true)" ;;
      cub/benchmarks/bench/*/*)
        algo="${path#cub/benchmarks/bench/}"; algo="${algo%%/*}"
        local_files="$(bench_counterpart "$algo" || true)" ;;
      cub/examples/*.cu) local_files="$(example_counterpart "$path" || true)" ;;
    esac
    if [[ -z "$local_files" ]]; then
      printf '\t-> [no local file]\n'
    else
      while read -r f; do printf '\t-> %s\n' "$f"; done <<< "$local_files"
    fi
  done < <(git diff-tree --no-commit-id --name-status -r "$SHA" -- "${SCOPE_PATHS[@]}" | cut -f1,2)
fi

echo
echo "=== Backend parity check (public headers) ==="
# Unconditional, per-commit, no de-duplication against earlier items. For
# each touched cub/cub/ public header, report all three hipCUB locations.
# A change usually needs both backends: the rocPRIM backend reimplements it,
# the CUB backend forwards to it (and can only use it once
# CCCL_MINIMUM_VERSION allows). backend/cub/ has no file for most block/,
# warp/ and some thread/ and iterator/ headers: backend/cub/hipcub.hpp
# re-exports those namespaces wholesale.
found_header=0
for path in "${touched[@]}"; do
  case "$path" in
    cub/cub/*.cuh) ;;
    *) continue ;;
  esac
  is_internal "$path" && continue
  found_header=1
  echo "$path"
  while read -r f; do
    if git cat-file -e "HEAD:$f" 2>/dev/null; then
      printf '\t%s\n\t\t[%s]\n' "$f" "$(sync_status "$f")"
    else
      note="missing"
      [[ "$f" == "$HIPCUB_INC/backend/cub/"* ]] && note="missing (may be re-exported by backend/cub/hipcub.hpp)"
      printf '\t%s\n\t\t[%s]\n' "$f" "$note"
    fi
  done < <(header_candidates "$path")
done
if [[ "$found_header" -eq 0 ]]; then
  echo "(no public cub/cub/ headers touched by this commit)"
fi

echo
echo "=== rocPRIM follow-up check (dispatch/agent/detail) ==="
# hipCUB has no copy of CUB's dispatch/agent/detail layers; the rocPRIM
# backend calls rocPRIM's device algorithms instead. When upstream changes
# an algorithm's behavior here (not just tuning or CUDA plumbing), rocPRIM
# may need the same change. This only names the likely rocPRIM header.
found_internal=0
for path in "${touched[@]}"; do
  is_internal "$path" || continue
  found_internal=1
  name="$(basename "$path" .cuh)"
  name="${name#dispatch_}"; name="${name#agent_}"; name="${name#tuning_}"
  name="${name#device_}"
  # CUB names whose rocPRIM algorithm is spelled differently.
  case "$name" in
    batch_memcpy) name="memcpy" ;;
    rle) name="run_length_encode" ;;
    single_pass_scan_operators) name="scan" ;;
  esac
  match="$(progressive_glob "$ROCPRIM_DEVICE" "device_" "$name" | grep -v '_config\.hpp$' | head -3 || true)"
  if [[ -z "$match" ]]; then
    printf '%s\n\t-> no rocPRIM device header matched\n' "$path"
  else
    printf '%s\n' "$path"
    while read -r f; do printf '\t-> %s\n' "$f"; done <<< "$match"
  fi
done
if [[ "$found_internal" -eq 0 ]]; then
  echo "(no dispatch/agent/detail paths touched by this commit)"
fi

echo
echo "=== Test counterpart check ==="
# cub/test/**/catch2_test_<x>.cu -> test/hipcub/test_hipcub_<x>.cpp. hipCUB
# implements upstream Catch2 tests with Google Test, so the counterpart is a
# rewrite, not a copy. Legacy non-Catch2 tests (test_*_fail.cu, nvtx, link
# tests) and c2h/ test plumbing have no hipCUB counterpart.
found_test=0
for path in "${touched[@]}"; do
  case "$path" in
    cub/test/*catch2_test_*.cu) ;;
    *) continue ;;
  esac
  found_test=1
  counterpart="$(test_counterpart "$path" || true)"
  if [[ -z "$counterpart" ]]; then
    printf '%s\n\t-> no test_hipcub_* counterpart found (new test, or under a different name)\n' "$path"
  else
    printf '%s\n' "$path"
    while read -r f; do printf '\t-> %s\n\t\t[%s]\n' "$f" "$(sync_status "$f")"; done <<< "$counterpart"
  fi
done
if [[ "$found_test" -eq 0 ]]; then
  echo "(no cub/test/**/catch2_test_*.cu paths touched by this commit)"
fi

echo
echo "=== Benchmark counterpart check ==="
# cub/benchmarks/bench/<algo>/*.cu -> benchmark/benchmark_device_<algo>.cpp.
# Upstream benchmarks are nvbench with tuning axes; hipCUB's are its own
# harness. Tuning-only (%RANGE%) and nvbench_helper changes are N/A.
found_bench=0
declare -A seen_algo=()
for path in "${touched[@]}"; do
  case "$path" in
    cub/benchmarks/bench/*/*) ;;
    *) continue ;;
  esac
  found_bench=1
  algo="${path#cub/benchmarks/bench/}"; algo="${algo%%/*}"
  [[ -n "${seen_algo[$algo]:-}" ]] && continue
  seen_algo[$algo]=1
  counterpart="$(bench_counterpart "$algo" || true)"
  if [[ -z "$counterpart" ]]; then
    printf 'bench/%s\n\t-> no hipCUB benchmark found\n' "$algo"
  else
    printf 'bench/%s\n' "$algo"
    while read -r f; do printf '\t-> %s\n\t\t[%s]\n' "$f" "$(sync_status "$f")"; done <<< "$counterpart"
  fi
done
if [[ "$found_bench" -eq 0 ]]; then
  echo "(no cub/benchmarks/bench/ paths touched by this commit)"
fi

echo
echo "=== Diff (scoped to cub/) ==="
git show "$SHA" -- "${SCOPE_PATHS[@]}"
