#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail

# Builds rocjitsu and runs a hipBLASLt GEMM workload under its race detector,
# emulating the GPU target in AMDGPU_TARGETS with hipBLASLt artifacts for that
# target already unpacked at ROCM_PATH. The workload runners live in
# projects/hipblaslt/clients/scripts/rocjitsu and write their reports, including
# summary.md for the job summary, to RACE_REPORT_DIR/<workload>.
#
# GEMM_WORKLOAD selects the workload:
#   random-gemms  RANDOM_GEMM_COUNT fixed-seed sampled GEMMs, chosen by the
#                 library's normal kernel selection.

ROCM_PATH="${ROCM_PATH:-${PWD}/build}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-}"
ROCJITSU_SOURCE_DIR="${ROCJITSU_SOURCE_DIR:-${PWD}/rocm-systems/emulation/rocjitsu}"
ROCJITSU_BUILD_DIR="${ROCJITSU_BUILD_DIR:-${PWD}/rocjitsu-build}"
RACE_REPORT_DIR="${RACE_REPORT_DIR:-${PWD}/race-reports}"
HIPBLASLT_BENCH="${HIPBLASLT_BENCH:-${ROCM_PATH}/bin/hipblaslt-bench}"
RUNNER_DIR="${RUNNER_DIR:-${PWD}/rocm-libraries/projects/hipblaslt/clients/scripts/rocjitsu}"
GEMM_WORKLOAD="${GEMM_WORKLOAD:-random-gemms}"
GEMM_WORKERS="${GEMM_WORKERS:-8}"
GEMM_TIMEOUT_SECONDS="${GEMM_TIMEOUT_SECONDS:-900}"
RANDOM_GEMM_COUNT="${RANDOM_GEMM_COUNT:-100}"
RANDOM_GEMM_SEED="${RANDOM_GEMM_SEED:-0}"
TIMING_FILE="${RACE_REPORT_DIR}/timing.tsv"

# Use the fetched compiler and runtime libraries for the build and benchmark.
LLVM_RUNTIME_ROOT="${ROCM_PATH}/lib/llvm/lib"
export ROCM_PATH
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:"\
"${LLVM_RUNTIME_ROOT}:${LLVM_RUNTIME_ROOT}/$(uname -m)-unknown-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# rocjitsu emulates one GPU at a time, so the job runs one target and selects
# that target's emulator profile from the pinned rocjitsu source.
case "${AMDGPU_TARGETS}" in
  gfx942) default_config="gfx942_cdna3_kmd.json" ;;
  gfx950) default_config="gfx950_mi355x_kmd.json" ;;
  *)
    echo "AMDGPU_TARGETS must name one supported target (gfx942 or gfx950), not '${AMDGPU_TARGETS}'" >&2
    exit 1
    ;;
esac
ROCJITSU_CONFIG="${ROCJITSU_CONFIG:-${ROCJITSU_SOURCE_DIR}/configs/${default_config}}"

mkdir -p "${RACE_REPORT_DIR}"
: >"${TIMING_FILE}"

# Print each stage's duration and exit status at exit, and add them to the
# GitHub job summary ahead of the workload's own summary.
print_timing_summary() {
  echo ""
  echo "=== rocjitsu GEMM race check stages ==="
  printf "%-28s %10s %6s\n" "stage" "seconds" "status"
  while IFS=$'\t' read -r label seconds status; do
    printf "%-28s %10s %6s\n" "${label}" "${seconds}" "${status}"
  done <"${TIMING_FILE}"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo "### rocjitsu GEMM race check stages (${GEMM_WORKLOAD}, ${AMDGPU_TARGETS})"
      echo ""
      echo "| Stage | Seconds | Exit status |"
      echo "| --- | ---: | ---: |"
      while IFS=$'\t' read -r label seconds status; do
        echo "| ${label} | ${seconds} | ${status} |"
      done <"${TIMING_FILE}"
      echo ""
    } >>"${GITHUB_STEP_SUMMARY}"
  fi
}
trap print_timing_summary EXIT

run_timed() {
  local label="$1"
  shift
  echo "::group::${label}"
  local start status
  start="$(date +%s)"
  set +e
  "$@"
  status=$?
  set -e
  echo "::endgroup::"
  printf "%s\t%s\t%s\n" "${label}" "$(($(date +%s) - start))" "${status}" \
    | tee -a "${TIMING_FILE}"
  return "${status}"
}

BENCH_DIR="$(dirname "${HIPBLASLT_BENCH}")"
for path in "${HIPBLASLT_BENCH}" "${BENCH_DIR}/hipblaslt_gentest.py" \
  "${BENCH_DIR}/hipblaslt_template.yaml" "${BENCH_DIR}/hipblaslt_common.yaml" \
  "${ROCJITSU_CONFIG}" "${RUNNER_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required input not found: ${path}" >&2
    exit 1
  fi
done
if ! command -v amdclang >/dev/null 2>&1 || ! command -v amdclang++ >/dev/null 2>&1; then
  echo "amdclang and amdclang++ must be available from the fetched ROCm payload" >&2
  exit 1
fi

# Build only the CLI, its LD_PRELOAD runtime and the race and logging plugins.
# The warning flags keep compiler/header warning mismatches visible without
# failing the build.
run_timed "configure rocjitsu" cmake \
  -S "${ROCJITSU_SOURCE_DIR}" \
  -B "${ROCJITSU_BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DROCM_PATH="${ROCM_PATH}" \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
  -DFETCHCONTENT_BASE_DIR="${ROCJITSU_BUILD_DIR}/third_party" \
  -DCMAKE_CXX_FLAGS="-Wno-error=unknown-warning-option -Wno-error=nested-anon-types" \
  -DCMAKE_C_COMPILER="$(command -v amdclang)" \
  -DCMAKE_CXX_COMPILER="$(command -v amdclang++)"
run_timed "build rocjitsu" cmake --build "${ROCJITSU_BUILD_DIR}" --target \
  rocjitsu_bin rocjitsu_shared rocjitsu_plugin_race_so rocjitsu_plugin_logging_so

ROCJITSU_BIN="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"
ROCJITSU_REVISION="$(git -C "${ROCJITSU_SOURCE_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
"${ROCJITSU_BIN}" --version

case "${GEMM_WORKLOAD}" in
  random-gemms)
    run_timed "random GEMMs" python3 "${RUNNER_DIR}/rocjitsu_random_gemms.py" \
      --rocjitsu "${ROCJITSU_BIN}" \
      --client "${HIPBLASLT_BENCH}" \
      --config "${ROCJITSU_CONFIG}" \
      --reports "${RACE_REPORT_DIR}/${GEMM_WORKLOAD}" \
      --target "${AMDGPU_TARGETS}" \
      --workers "${GEMM_WORKERS}" \
      --timeout "${GEMM_TIMEOUT_SECONDS}" \
      --rocjitsu-revision "${ROCJITSU_REVISION}" \
      --continue-on-failure \
      --count "${RANDOM_GEMM_COUNT}" \
      --seed "${RANDOM_GEMM_SEED}"
    ;;
  *)
    echo "Unsupported GEMM_WORKLOAD: ${GEMM_WORKLOAD}" >&2
    exit 1
    ;;
esac
