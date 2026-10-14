#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build the three epilogue variants with TensileLite, then collect timing, hardware counters and
# one ATT trace per variant with rocprofv3. Results land in --out-dir; see rocprof_analyze.py.

set -euo pipefail

GPU=0
OUT_DIR="${HOME}/epilogue-profile"
SKIP_BUILD=0
TIMING_REPEATS=3
FORCE=0
YAML=epilogues/YAMLs/benchmark_k1_bf16_top_config.yaml
VARIANTS=(rms beta none)

usage() {
  echo "usage: $0 [--gpu N] [--out-dir DIR] [--skip-build] [--timing-repeats R] [--force]" >&2
  echo "  defaults: --gpu 0 --out-dir \$HOME/epilogue-profile --timing-repeats 3" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --timing-repeats) TIMING_REPEATS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYZE="$SCRIPT_DIR/rocprof_analyze.py"
mkdir -p "$OUT_DIR"
OUT="$(cd "$OUT_DIR" && pwd)"
START_S=$SECONDS

# shellcheck disable=SC1091
source "${TENSILE_VENV:-$HOME/.tensile}/bin/activate"
mkdir -p "$OUT/logs"

echo "== S1 preflight (gpu $GPU)"
PREFLIGHT_ARGS=(preflight --gpu "$GPU" --out-dir "$OUT" --tensile-dir "$TENSILE_DIR")
[[ $FORCE -eq 1 ]] && PREFLIGHT_ARGS+=(--force)
SELECTOR="$(python3 "$ANALYZE" "${PREFLIGHT_ARGS[@]}")"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES="$SELECTOR"
echo "   ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"

build_outputs_exist() {
  local count
  count="$(find "$OUT/tensile/2_BenchmarkData" -name '*_CSVWinner.csv' 2>/dev/null | wc -l)"
  [[ "$count" -eq ${#VARIANTS[@]} ]]
}

# The stamp ties a build to the YAML, the git HEAD and the tensilelite working tree (excluding these
# scripts, which do not influence the built kernels), so --skip-build cannot reuse a stale build.
tree_status_hash() {
  (
    cd "$TENSILE_DIR"
    git status --porcelain --untracked-files=all -- . ':(exclude)epilogues/scripts'
    git diff HEAD -- . ':(exclude)epilogues/scripts'
    git ls-files --others --exclude-standard -z -- . ':(exclude)epilogues/scripts' | xargs -0 -r sha256sum
  ) | sha256sum | cut -d' ' -f1
}

# The toolchain that generates the kernels is part of the stamp: a ROCm or compiler update changes the
# code objects even when the sources are unchanged.
rocm_core_version() {
  local header=/opt/rocm/include/rocm-core/rocm_version.h
  [[ -f "$header" ]] || { echo unavailable; return; }
  awk '/#define ROCM_VERSION_(MAJOR|MINOR|PATCH)[ \t]/ { v = v (v ? "." : "") $3 } END { print v }' "$header"
}

clang_version() {
  /opt/rocm/llvm/bin/clang --version 2>/dev/null | head -n 1 || echo unavailable
}

current_stamp() {
  printf '{\n  "clang_version": "%s",\n  "git_head": "%s",\n  "rocm_core_version": "%s",\n  "rocm_root": "%s",\n  "tree_status_sha256": "%s",\n  "yaml_sha256": "%s"\n}\n' \
    "$(clang_version)" "$(git -C "$TENSILE_DIR" rev-parse HEAD)" "$(rocm_core_version)" \
    "$(readlink -f /opt/rocm/bin)" "$(tree_status_hash)" \
    "$(sha256sum "$TENSILE_DIR/$YAML" | cut -d' ' -f1)"
}

STAMP="$OUT/tensile/build_stamp.json"

require_fresh_build() {
  if [[ ! -f "$STAMP" ]]; then
    echo "error: --skip-build refused: $STAMP is missing; rerun without --skip-build" >&2
    exit 3
  fi
  if ! diff <(current_stamp) "$STAMP" > "$OUT/logs/stamp_diff.txt"; then
    echo "error: --skip-build refused: the build is stale (< current, > stamped):" >&2
    cat "$OUT/logs/stamp_diff.txt" >&2
    exit 3
  fi
}

echo "== S2 build"
if [[ $SKIP_BUILD -eq 1 ]] && build_outputs_exist; then
  require_fresh_build
  echo "   --skip-build: reusing $OUT/tensile (stamp matches)"
else
  rm -rf "$OUT/tensile"
  STAMP_BEFORE="$(current_stamp)"
  (cd "$TENSILE_DIR" && TENSILE_DISABLE_HELPER_CACHE=1 ./tensilelite/bin/Tensile "$YAML" "$OUT/tensile") \
    > "$OUT/logs/tensile_build.log" 2>&1
  printf '%s\n' "$STAMP_BEFORE" > "$STAMP"
fi

echo "== S3 discover variants"
python3 "$ANALYZE" discover --tensile-out "$OUT/tensile" --out "$OUT/variants.json"

echo "== P0 counter availability"
python3 "$ANALYZE" list-avail --out "$OUT/rocprofv3_list_avail.txt"

echo "== S4 per-variant passes"
rm -rf "$OUT/variants"
for v in "${VARIANTS[@]}"; do
  mkdir -p "$OUT/variants/$v"
  python3 "$ANALYZE" run --variant "$v" --variants-json "$OUT/variants.json" \
    --metadata "$OUT/run_metadata.json" --avail "$OUT/rocprofv3_list_avail.txt" \
    --out "$OUT/variants/$v" --gpu-selector "$SELECTOR" --timing-repeats "$TIMING_REPEATS"
done

echo "== S5 merge"
STATUS=0
python3 "$ANALYZE" merge --root "$OUT" || STATUS=$?
echo "total runtime: $((SECONDS - START_S)) s (exit $STATUS)"
exit "$STATUS"
