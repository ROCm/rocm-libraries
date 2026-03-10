#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for convolution tests during migration.
# - No ctest usage.
# - Supports both gtest and non-gtest test binaries.
# - Uses branch profiles for split PR branches to run only relevant targets.
#
# Usage:
#   ./script/launch_tests.sh [build_dir] [mode] [jobs]
#
# Args:
#   build_dir : default <repo_root>/build
#   mode      : quick|full (default quick)
#   jobs      : parallel jobs for build step (default 0: backend default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"

BUILD_DIR="${1:-${REPO_ROOT}/build}"
MODE="${2:-quick}"
JOBS="${3:-0}"

if [[ "$MODE" != "quick" && "$MODE" != "full" ]]; then
  echo "[ERROR] mode must be quick or full"
  exit 2
fi

echo "[INFO] Source dir: ${SRC_DIR}"
echo "[INFO] Build dir : ${BUILD_DIR}"
echo "[INFO] Mode      : ${MODE}"
echo "[INFO] Jobs      : ${JOBS}"

echo "[INFO] Build step..."
"${SCRIPT_DIR}/build_all_test_targets.sh" "${BUILD_DIR}" "${JOBS}"

BIN_DIR="${BUILD_DIR}/bin"
if [[ ! -d "${BIN_DIR}" ]]; then
  echo "[ERROR] Binary directory not found: ${BIN_DIR}"
  exit 3
fi

branch="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "[INFO] Branch    : ${branch}"

declare -a CANDIDATES
case "${branch}" in
  users/kokolchin/pr4573-part1-infra)
    CANDIDATES=(
      test_conv2d
      test_conv2d_find2
      test_conv2d_bias
      test_conv3d
      test_conv3d_find2
      test_conv3d_bias
      test_immed_conv2d
      test_immed_conv3d
      test_conv_group
    )
    ;;
  users/kokolchin/pr4573-part2-conv2d)
    CANDIDATES=(
      test_conv2d
      test_conv2d_find2
      test_conv2d_bias
      test_immed_conv2d
    )
    ;;
  users/kokolchin/pr4573-part3-conv3d)
    CANDIDATES=(
      test_conv3d
      test_conv3d_find2
      test_conv3d_bias
      test_immed_conv3d
      test_conv_group
    )
    ;;
  users/kokolchin/pr4573-part4-solver-gtests)
    CANDIDATES=(
      test_conv_ck_igemm_fwd_v6r1_dlops_nchw
      test_conv_embed_db
      test_conv_extra
      test_conv_for_implicit_gemm
      test_conv_hip_igemm_xdlops
      test_conv_igemm_dynamic_dlops
      test_conv_igemm_dynamic_xdlops_nhwc_bf16
      test_conv_igemm_dynamic_xdlops_nhwc_nchw
      test_conv_igemm_mlir_bwd_wrw
      test_conv_igemm_mlir_fwd
      test_conv_igemm_mlir_xdlops_bwd_wrw
      test_conv_igemm_mlir_xdlops_fwd
      test_conv_trans
      test_deepbench_conv
      test_miopen_conv
      test_regression_float_mi100
      test_regression_issue_2012
      test_smoke_solver_ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC
      test_smoke_solver_ConvAsmImplicitGemmGTCDynamicXdlopsNHWC_bf16
      test_smoke_solver_ConvAsmImplicitGemmGTCDynamicXdlopsNHWC_fp32_fp16
      test_smoke_solver_ConvCkIgemmFwdV6r1DlopsNchw
      test_smoke_solver_convasmbwdwrw
    )
    ;;
  *)
    CANDIDATES=(
      test_conv2d
      test_conv2d_find2
      test_conv2d_bias
      test_conv3d
      test_conv3d_find2
      test_conv3d_bias
      test_immed_conv2d
      test_immed_conv3d
      test_conv_group
    )
    echo "[WARN] Unknown branch profile. Using default candidate list."
    ;;
esac

declare -a TARGETS=()
for t in "${CANDIDATES[@]}"; do
  if [[ -x "${BIN_DIR}/${t}" ]]; then
    TARGETS+=("${t}")
  fi
done

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "[ERROR] No convolution test binaries found in ${BIN_DIR}"
  exit 4
fi

echo "[INFO] Launch targets: ${TARGETS[*]}"

pass=0
fail=0
skip_env=0
declare -a FAILED=()
declare -a SKIPPED_ENV=()

for t in "${TARGETS[@]}"; do
  bin="${BIN_DIR}/${t}"
  echo
  echo "[RUN] ${t}"

  # Probe whether this binary is a gtest executable.
  set +e
  "${bin}" --gtest_list_tests >/dev/null 2>&1
  is_gtest=$?
  set -e

  set +e
  if [[ $is_gtest -eq 0 ]]; then
    if [[ "$MODE" == "quick" ]]; then
      out="$("${bin}" --gtest_filter="Smoke*" 2>&1)"
      rc=$?
    else
      out="$("${bin}" 2>&1)"
      rc=$?
    fi
  else
    if [[ "$MODE" == "quick" ]]; then
      out="$("${bin}" --float 2>&1)"
      rc=$?
    else
      out="$("${bin}" --float --all 2>&1)"
      rc=$?
    fi
  fi
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[PASS] ${t}"
    pass=$((pass + 1))
    continue
  fi

  # Environment-specific skip: missing rocBLAS Tensile DB for this GPU arch.
  if [[ "$out" == *"rocBLAS error:"* && "$out" == *"TensileLibrary.dat"* ]]; then
    echo "[SKIP_ENV] ${t} (rocBLAS Tensile DB missing for current GPU arch)"
    skip_env=$((skip_env + 1))
    SKIPPED_ENV+=("${t}")
    continue
  fi

  echo "[FAIL] ${t}"
  echo "$out"
  fail=$((fail + 1))
  FAILED+=("${t}")
done

echo
echo "[SUMMARY] pass=${pass} fail=${fail} skip_env=${skip_env}"
if [[ "${#SKIPPED_ENV[@]}" -gt 0 ]]; then
  echo "[SUMMARY] skipped (env): ${SKIPPED_ENV[*]}"
fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "[SUMMARY] failed: ${FAILED[*]}"
  exit 1
fi

echo "[PASS] launch_tests completed."
