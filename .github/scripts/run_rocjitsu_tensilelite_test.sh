#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail

# Run TensileLite common GEMM tests under rocjitsu gfx1250/gfx942 emulation.
#
# Follows the same pattern as run_rocjitsu_hipblaslt_race_check.sh:
#   1. Use TheRock artifact tree at ROCM_PATH.
#   2. Build rocjitsu from rocm-systems checkout.
#   3. Run pytest under rocjitsu emulation.
#
# Advisory job (continue-on-error in workflow). Validates that TensileLite
# kernels build and execute correctly under CPU emulation.

ROCM_PATH="${ROCM_PATH:-${PWD}/build}"
AMDGPU_FAMILIES="${AMDGPU_FAMILIES:-}"
ROCJITSU_GPU_TARGET="${ROCJITSU_GPU_TARGET:-}"
ROCJITSU_SOURCE_DIR="${ROCJITSU_SOURCE_DIR:-${PWD}/rocm-systems/emulation/rocjitsu}"
ROCJITSU_BUILD_DIR="${ROCJITSU_BUILD_DIR:-${PWD}/rocjitsu-build}"
ROCJITSU_CONFIG="${ROCJITSU_CONFIG:-}"
REPORT_DIR="${REPORT_DIR:-${PWD}/rocjitsu-tensilelite-reports}"
TENSILELITE_ROOT="${TENSILELITE_ROOT:-${ROCM_PATH}/share/hipblaslt/tensilelite}"
TENSILELITE_CLIENT="${TENSILELITE_CLIENT:-${ROCM_PATH}/libexec/hipblaslt/tensilelite/tensilelite-client}"
PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-2700}"
TIMING_FILE="${REPORT_DIR}/timing.tsv"

select_rocjitsu_target() {
  local target_selector="${ROCJITSU_GPU_TARGET:-${AMDGPU_FAMILIES}}"
  local default_config

  case "${target_selector}" in
    gfx94*)
      ROCJITSU_GPU_TARGET="gfx942"
      default_config="${ROCJITSU_SOURCE_DIR}/configs/gfx942_cdna3_kmd.json"
      ;;
    gfx950*)
      ROCJITSU_GPU_TARGET="gfx950"
      default_config="${ROCJITSU_SOURCE_DIR}/configs/gfx950_cdna4_kmd.json"
      ;;
    gfx125*)
      ROCJITSU_GPU_TARGET="gfx1250"
      default_config="${ROCJITSU_SOURCE_DIR}/configs/gfx1250.json"
      ;;
    *)
      echo "Unsupported rocjitsu target: ${target_selector}" >&2
      echo "Supported: gfx94*, gfx950*, gfx125*." >&2
      exit 1
      ;;
  esac

  ROCJITSU_CONFIG="${ROCJITSU_CONFIG:-${default_config}}"
}

print_timing_summary() {
  if [[ -f "${TIMING_FILE}" ]]; then
    echo ""
    echo "=== rocjitsu tensilelite test timing summary ==="
    printf "%-36s %10s %6s\n" "stage" "seconds" "status"
    while IFS=$'\t' read -r label seconds status; do
      printf "%-36s %10s %6s\n" "${label}" "${seconds}" "${status}"
    done <"${TIMING_FILE}"
  fi
}
trap print_timing_summary EXIT

run_timed() {
  local label="$1"
  shift
  echo "::group::${label}"
  local start
  start="$(date +%s)"
  set +e
  "$@"
  local status=$?
  set -e
  local end
  end="$(date +%s)"
  local elapsed=$((end - start))
  echo "::endgroup::"
  printf "%s\t%s\t%s\n" "${label}" "${elapsed}" "${status}" | tee -a "${TIMING_FILE}"
  return "${status}"
}

# ── Validate artifact layout ──────────────────────────────────────────────────

if [[ ! -d "${ROCM_PATH}" ]]; then
  echo "ROCM_PATH does not exist: ${ROCM_PATH}" >&2
  exit 1
fi

if [[ ! -d "${TENSILELITE_ROOT}" ]]; then
  echo "TensileLite artifacts not found: ${TENSILELITE_ROOT}" >&2
  exit 1
fi

if [[ ! -x "${TENSILELITE_CLIENT}" ]]; then
  echo "tensilelite-client not found: ${TENSILELITE_CLIENT}" >&2
  exit 1
fi

# The compiler artifact layout depends on whether TheRock enables LLVM's host
# per-target runtime directories. Prefer the flat lib/llvm/lib layout, then the
# matching host-triple directory when per-target runtime directories are used.
LLVM_RUNTIME_ROOT="${ROCM_PATH}/lib/llvm/lib"
LIBOMP_CANDIDATES=(
  "${LLVM_RUNTIME_ROOT}/libomp.so"
  "${LLVM_RUNTIME_ROOT}/$(uname -m)-unknown-linux-gnu/libomp.so"
)
LIBOMP_PATH=""
for candidate in "${LIBOMP_CANDIDATES[@]}"; do
  if [[ -e "${candidate}" ]]; then
    LIBOMP_PATH="${candidate}"
    break
  fi
done

if [[ -z "${LIBOMP_PATH}" ]]; then
  echo "OpenMP runtime not found at either expected path:" >&2
  printf "  %s\n" "${LIBOMP_CANDIDATES[@]}" >&2
  exit 1
fi

LLVM_HOST_RUNTIME_DIR="$(dirname "${LIBOMP_PATH}")"
LLVM_RUNTIME_LIBRARY_PATH="${LLVM_RUNTIME_ROOT}"
if [[ "${LLVM_HOST_RUNTIME_DIR}" != "${LLVM_RUNTIME_ROOT}" ]]; then
  LLVM_RUNTIME_LIBRARY_PATH="${LLVM_RUNTIME_LIBRARY_PATH}:${LLVM_HOST_RUNTIME_DIR}"
fi

select_rocjitsu_target

if [[ -z "${ROCJITSU_CONFIG}" || ! -f "${ROCJITSU_CONFIG}" ]]; then
  echo "rocjitsu config not found: ${ROCJITSU_CONFIG}" >&2
  exit 1
fi

mkdir -p "${REPORT_DIR}"
: >"${TIMING_FILE}"

# ── Environment ───────────────────────────────────────────────────────────────

export ROCM_PATH
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:${LLVM_RUNTIME_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TENSILELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "ROCM_PATH=${ROCM_PATH}"
echo "AMDGPU_FAMILIES=${AMDGPU_FAMILIES}"
echo "ROCJITSU_GPU_TARGET=${ROCJITSU_GPU_TARGET}"
echo "ROCJITSU_CONFIG=${ROCJITSU_CONFIG}"
echo "TENSILELITE_ROOT=${TENSILELITE_ROOT}"
echo "TENSILELITE_CLIENT=${TENSILELITE_CLIENT}"
echo "PER_TEST_TIMEOUT=${PER_TEST_TIMEOUT}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ── Build rocjitsu ────────────────────────────────────────────────────────────

configure_rocjitsu() {
  cmake \
    -S "${ROCJITSU_SOURCE_DIR}" \
    -B "${ROCJITSU_BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DROCM_PATH="${ROCM_PATH}" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
    -DCMAKE_CXX_FLAGS="-Wno-error=unknown-warning-option -Wno-error=nested-anon-types" \
    -DCMAKE_C_COMPILER="$(command -v amdclang)" \
    -DCMAKE_CXX_COMPILER="$(command -v amdclang++)"
}

build_rocjitsu() {
  cmake --build "${ROCJITSU_BUILD_DIR}" --target rocjitsu_bin rocjitsu_shared
}

ROCJITSU_BIN="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"

show_rocjitsu_version() {
  "${ROCJITSU_BIN}" --version
}

run_timed "configure rocjitsu" configure_rocjitsu
run_timed "build rocjitsu" build_rocjitsu
run_timed "rocjitsu version" show_rocjitsu_version

# ── Install pytest dependencies ───────────────────────────────────────────────

install_pytest_deps() {
  # Python 3.12+ required for rocisa stable-ABI extension
  local python_bin
  python_bin="$(command -v python3.12 || command -v python3)"
  PYTHON="${python_bin}"

  if command -v uv >/dev/null 2>&1; then
    uv pip install \
      pytest pyyaml msgpack \
      pytest-xdist pytest-timeout \
      syrupy tqdm joblib numpy filelock
  else
    "${PYTHON}" -m pip install --quiet \
      pytest pyyaml msgpack \
      pytest-xdist pytest-timeout \
      syrupy tqdm joblib numpy filelock
  fi
}

run_timed "install pytest deps" install_pytest_deps

# ── Run TensileLite tests ─────────────────────────────────────────────────────

run_tensilelite_tests() {
  local junit_dir="${REPORT_DIR}/junit"
  mkdir -p "${junit_dir}"

  # pytest-timeout handles per-test kills. No outer timeout here — the workflow
  # step timeout (160 min) is the process-tree kill backstop.
  "${ROCJITSU_BIN}" \
      --config "${ROCJITSU_CONFIG}" \
      -- "${PYTHON}" -m pytest \
        "${TENSILELITE_ROOT}/Tensile/Tests/common" \
        -m "${ROCJITSU_GPU_TARGET}" \
        -v -s \
        --timeout="${PER_TEST_TIMEOUT}" \
        --junit-xml="${junit_dir}/tensilelite.xml" \
        --prebuilt-client="${TENSILELITE_CLIENT}" \
        --global-parameters="LibraryFormat='msgpack'" \
        "--tensile-options=--cxx-compiler,${ROCM_PATH}/bin/amdclang++,--gpu-targets,${ROCJITSU_GPU_TARGET}" \
    2>&1 | tee "${REPORT_DIR}/tensilelite-test.log"

  local status=${PIPESTATUS[0]}

  # Parse JUnit XML for per-test timing summary
  if [[ -f "${junit_dir}/tensilelite.xml" ]]; then
    "${PYTHON}" << 'JUNIT_PARSE'
import xml.etree.ElementTree as ET, os
junit_dir = os.environ.get('REPORT_DIR', '.') + '/junit'
tree = ET.parse(junit_dir + '/tensilelite.xml')
tests = [(tc.get('time','0'), tc.get('name',''), 'PASSED' if tc.find('failure') is None and tc.find('error') is None else 'FAILED') for tc in tree.iter('testcase')]
tests.sort(key=lambda x: float(x[0]), reverse=True)
total = sum(float(t) for t,_,_ in tests)
passed = sum(1 for _,_,s in tests if s == 'PASSED')
failed = sum(1 for _,_,s in tests if s == 'FAILED')
print()
print('=' * 80)
print('Per-test timing from JUnit XML (rocjitsu emulation)')
print('=' * 80)
print('%-55s %10s %8s' % ('Test', 'Time', 'Status'))
print('-' * 80)
for t, name, status in tests:
    secs = float(t)
    time_str = '%dm%02ds' % (int(secs//60), int(secs%60)) if secs >= 60 else '%.1fs' % secs
    short = name.split('[')[-1].rstrip(']').split('/')[-1].replace('.yaml','') if '[' in name else name
    print('%-55s %10s %8s' % (short, time_str, status))
print('-' * 80)
print('Total: %d tests, %d passed, %d failed, %.0fs (%.0f min)' % (len(tests), passed, failed, total, total/60))
print('=' * 80)
JUNIT_PARSE
  fi

  return "${status}"
}

set +e
run_timed "tensilelite tests (${ROCJITSU_GPU_TARGET})" run_tensilelite_tests
test_status=$?
set -e

if [[ "${test_status}" -ne 0 ]]; then
  echo "tensilelite rocjitsu tests exited with status ${test_status}" >&2
  # Exit code 1 = test failures, 5 = no tests collected. Both are useful signal.
  # Don't treat as infra failure.
fi

exit "${test_status}"
