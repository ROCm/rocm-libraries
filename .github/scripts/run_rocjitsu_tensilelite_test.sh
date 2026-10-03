#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Functions are invoked through run_timed and the EXIT trap.
# shellcheck disable=SC2329
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
# Activate the test venv and install the artifact's requirements-test.txt first.
# Extra command-line arguments are forwarded to pytest for focused local runs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"

ROCM_PATH="${ROCM_PATH:-${PWD}/build}"
AMDGPU_FAMILIES="${AMDGPU_FAMILIES:-}"
ROCJITSU_GPU_TARGET="${ROCJITSU_GPU_TARGET:-}"
# Architecture to select and build tests for when it differs from the emulated
# device, e.g. gfx1250-strict configs on the gfx1250 emulator.
ROCJITSU_TEST_TARGET="${ROCJITSU_TEST_TARGET:-}"
ROCJITSU_SOURCE_DIR="${ROCJITSU_SOURCE_DIR:-${PWD}/rocm-systems/emulation/rocjitsu}"
ROCJITSU_BUILD_DIR="${ROCJITSU_BUILD_DIR:-${PWD}/rocjitsu-build}"
ROCJITSU_CONFIG="${ROCJITSU_CONFIG:-}"
REPORT_DIR="${REPORT_DIR:-${PWD}/rocjitsu-tensilelite-reports}"
TENSILELITE_ROOT="${TENSILELITE_ROOT:-${ROCM_PATH}/share/hipblaslt/tensilelite}"
TENSILELITE_CLIENT="${TENSILELITE_CLIENT:-${ROCM_PATH}/libexec/hipblaslt/tensilelite/tensilelite-client}"
PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-2700}"
# Leave time within the 345-minute step for cleanup and report generation.
SUITE_TIMEOUT_SECONDS="${SUITE_TIMEOUT_SECONDS:-19800}"
HOST_CORES="$(nproc)"
PYTEST_WORKERS="${PYTEST_WORKERS:-$(( HOST_CORES < 16 ? HOST_CORES : 16 ))}"
ROCJITSU_BUILD_JOBS="${ROCJITSU_BUILD_JOBS:-$(( HOST_CORES < 48 ? HOST_CORES : 48 ))}"
# AVX2 build; -march=native breaks (libstdc++ <experimental/simd> AVX-512 assert).
ROCJITSU_MARCH="${ROCJITSU_MARCH:-x86-64-v3}"
ROCJITSU_LTO="${ROCJITSU_LTO:-ON}"
# Per-CU step cap; empty = upstream default.
ROCJITSU_FUNCTIONAL_QUANTUM="${ROCJITSU_FUNCTIONAL_QUANTUM:-}"
# num_threads (XCD) and cpu_dispatch_threads (CU, #10074) sized so
# PYTEST_WORKERS x num_threads x cpu_dispatch ~= host_cores. "auto" derives them.
ROCJITSU_CPU_DISPATCH_THREADS="${ROCJITSU_CPU_DISPATCH_THREADS:-auto}"
ROCJITSU_NUM_THREADS="${ROCJITSU_NUM_THREADS:-auto}"
ROCJITSU_MAX_XCD_THREADS="${ROCJITSU_MAX_XCD_THREADS:-8}"  # gfx1250/94x/950 have 8 XCDs
# Compiler capability probes can write into the current directory. Resolve
# inputs before running pytest from the report directory to retain those files.
for setting in ROCM_PATH ROCJITSU_SOURCE_DIR ROCJITSU_BUILD_DIR REPORT_DIR TENSILELITE_ROOT TENSILELITE_CLIENT; do
  printf -v "${setting}" '%s' "$(realpath -m -- "${!setting}")"
done
if [[ -n "${ROCJITSU_CONFIG}" ]]; then
  ROCJITSU_CONFIG="$(realpath -m -- "${ROCJITSU_CONFIG}")"
fi
PYTHON="$(realpath -ms -- "$(command -v "${PYTHON}")")"
TIMING_FILE="${REPORT_DIR}/timing.tsv"

for setting in PYTEST_WORKERS ROCJITSU_BUILD_JOBS PER_TEST_TIMEOUT SUITE_TIMEOUT_SECONDS ROCJITSU_MAX_XCD_THREADS; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${setting} must be a positive integer, got '${!setting}'" >&2
    exit 1
  fi
done
for setting in ROCJITSU_NUM_THREADS ROCJITSU_CPU_DISPATCH_THREADS; do
  if [[ "${!setting}" != auto && ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${setting} must be 'auto' or a positive integer, got '${!setting}'" >&2
    exit 1
  fi
done

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
      default_config="${ROCJITSU_SOURCE_DIR}/configs/gfx950_mi355x_kmd.json"
      ;;
    gfx125*)
      ROCJITSU_GPU_TARGET="gfx1250"
      default_config="${ROCJITSU_SOURCE_DIR}/configs/gfx1250_mi455x.json"
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
  local had_errexit=0
  case "$-" in *e*) had_errexit=1 ;; esac
  set +e
  "$@"
  local status=$?
  local end
  end="$(date +%s)"
  local elapsed=$((end - start))
  echo "::endgroup::"
  printf "%s\t%s\t%s\n" "${label}" "${elapsed}" "${status}" | tee -a "${TIMING_FILE}"
  if [[ "${had_errexit}" -ne 0 ]]; then
    set -e
  fi
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
TEST_TARGET="${ROCJITSU_TEST_TARGET:-${ROCJITSU_GPU_TARGET}}"

if [[ -z "${ROCJITSU_CONFIG}"|| ! -f "${ROCJITSU_CONFIG}" ]]; then
  echo "rocjitsu config not found: ${ROCJITSU_CONFIG}" >&2
  exit 1
fi

mkdir -p "${REPORT_DIR}"
: >"${TIMING_FILE}"

# Inject a bounded per-process thread budget into a config copy; repoint ROCJITSU_CONFIG.
apply_dispatch_sizing() {
  local py host_cores budget num_threads cpu_dispatch injected
  py="${PYTHON}"
  host_cores="${HOST_CORES}"

  budget=$(( host_cores / PYTEST_WORKERS ))
  (( budget < 1 )) && budget=1

  if [[ "${ROCJITSU_NUM_THREADS}" == "auto" ]]; then
    num_threads="${budget}"
    (( num_threads > ROCJITSU_MAX_XCD_THREADS )) && num_threads="${ROCJITSU_MAX_XCD_THREADS}"
  else
    num_threads="${ROCJITSU_NUM_THREADS}"
  fi

  if [[ "${ROCJITSU_CPU_DISPATCH_THREADS}" == "auto" ]]; then
    cpu_dispatch=$(( budget / num_threads ))
    (( cpu_dispatch < 1 )) && cpu_dispatch=1
    (( cpu_dispatch > 32 )) && cpu_dispatch=32
  else
    cpu_dispatch="${ROCJITSU_CPU_DISPATCH_THREADS}"
  fi

  injected="${REPORT_DIR}/rocjitsu-config.json"
  SRC_CONFIG="${ROCJITSU_CONFIG}" INJECT_NUM_THREADS="${num_threads}" \
    INJECT_CPU_DISPATCH="${cpu_dispatch}" INJECT_BUDGET="${budget}" \
    INJECT_FQ="${ROCJITSU_FUNCTIONAL_QUANTUM}" \
    OUT_CONFIG="${injected}" \
    "${py}" - <<'PYEOF'
import json, os
cfg = json.load(open(os.environ["SRC_CONFIG"]))
em = cfg.get("exec_mode", "functional")
if em != "functional":
    print(f"::warning::exec_mode={em} — 'functional' is the fast path; clocked/other is far slower")
cfg["num_threads"] = int(os.environ["INJECT_NUM_THREADS"])
cfg["cpu_dispatch_threads"] = int(os.environ["INJECT_CPU_DISPATCH"])
cfg["cpu_thread_budget"] = int(os.environ["INJECT_BUDGET"])
cfg["async_helper_threads"] = 0

# functional_quantum is a per-CU config entry under topology.
fq = os.environ.get("INJECT_FQ", "")
if fq != "":
    def set_cu_quantum(node):
        if isinstance(node, dict):
            if node.get("type") == "compute_unit":
                conf = node.setdefault("config", [])
                for e in conf:
                    if e.get("key") == "functional_quantum":
                        e["value"] = str(fq); break
                else:
                    conf.append({"key": "functional_quantum", "value": str(fq)})
            for v in node.values():
                set_cu_quantum(v)
        elif isinstance(node, list):
            for v in node:
                set_cu_quantum(v)
    set_cu_quantum(cfg)

with open(os.environ["OUT_CONFIG"], "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF

  ROCJITSU_CONFIG="${injected}"
  echo "dispatch sizing: host_cores=${host_cores} xdist_workers=${PYTEST_WORKERS}" \
       "per-process budget=${budget} -> num_threads(XCD)=${num_threads}" \
       "cpu_dispatch_threads(CU)=${cpu_dispatch}" \
       "functional_quantum=${ROCJITSU_FUNCTIONAL_QUANTUM:-<upstream default>}" \
       "(total ~= $(( PYTEST_WORKERS * num_threads * cpu_dispatch )) host threads)"
}

apply_dispatch_sizing

# ── Environment ───────────────────────────────────────────────────────────────

export ROCM_PATH
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:${LLVM_RUNTIME_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SCRIPT_DIR}:${TENSILELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "ROCM_PATH=${ROCM_PATH}"
echo "AMDGPU_FAMILIES=${AMDGPU_FAMILIES}"
echo "ROCJITSU_GPU_TARGET=${ROCJITSU_GPU_TARGET}"
echo "TEST_TARGET=${TEST_TARGET}"
echo "ROCJITSU_CONFIG=${ROCJITSU_CONFIG}"
echo "ROCJITSU_MARCH=${ROCJITSU_MARCH}"
echo "TENSILELITE_ROOT=${TENSILELITE_ROOT}"
echo "TENSILELITE_CLIENT=${TENSILELITE_CLIENT}"
echo "PER_TEST_TIMEOUT=${PER_TEST_TIMEOUT}"
echo "SUITE_TIMEOUT_SECONDS=${SUITE_TIMEOUT_SECONDS}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

"${PYTHON}" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("TensileLite emulation requires Python 3.12 or newer")
import pytest, xdist, yaml, numpy, msgpack, filelock, rocisa
print(f"Test interpreter: {sys.executable}")
PY
"${PYTHON}" - <<'PY' > "${REPORT_DIR}/python-packages.txt"
from importlib.metadata import distributions
print('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())))
PY

# ── Build rocjitsu ────────────────────────────────────────────────────────────

configure_rocjitsu() {
  local cxx_flags="-Wno-error=unknown-warning-option -Wno-error=nested-anon-types"
  [[ -n "${ROCJITSU_MARCH}" ]] && cxx_flags="-march=${ROCJITSU_MARCH} ${cxx_flags}"
  echo "rocjitsu build flags: CXX_FLAGS='${cxx_flags}' LTO=${ROCJITSU_LTO}"
  cmake \
    -S "${ROCJITSU_SOURCE_DIR}" \
    -B "${ROCJITSU_BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DLTO="${ROCJITSU_LTO}" \
    -DROCM_PATH="${ROCM_PATH}" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
    -DCMAKE_CXX_FLAGS="${cxx_flags}" \
    -DCMAKE_C_COMPILER="$(command -v amdclang)" \
    -DCMAKE_CXX_COMPILER="$(command -v amdclang++)"
}

build_rocjitsu() {
  cmake --build "${ROCJITSU_BUILD_DIR}" --parallel "${ROCJITSU_BUILD_JOBS}" \
    --target rocjitsu_bin rocjitsu_shared hsa_hotswap_rocjitsu
}

ROCJITSU_BIN="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"

show_rocjitsu_version() {
  "${ROCJITSU_BIN}" --version
}

run_timed "configure rocjitsu" configure_rocjitsu
run_timed "build rocjitsu" build_rocjitsu
run_timed "rocjitsu version" show_rocjitsu_version

# The HSA hotswap hook translates gfx1250 code objects for emulation.
HOTSWAP_LIB=$(find "${ROCJITSU_BUILD_DIR}" -name "libhsa_hotswap_rocjitsu.so" -type f -print -quit)
if [[ -n "${HOTSWAP_LIB}" ]]; then
  cp "${HOTSWAP_LIB}" "${ROCM_PATH}/lib/"
  echo "Installed hotswap lib: ${ROCM_PATH}/lib/libhsa_hotswap_rocjitsu.so"
else
  echo "::error::libhsa_hotswap_rocjitsu.so was not built"
  exit 1
fi

# Log the rocjitsu commit and warn on env that slows a functional run.
log_provenance_and_hygiene() {
  local repo="${ROCJITSU_SOURCE_DIR%/emulation/rocjitsu}"
  echo "::group::rocjitsu provenance + perf hygiene"
  git -C "${repo}" log -1 --format='rocm-systems HEAD %h  %cd  %s' --date=iso 2>/dev/null \
    || echo "rocm-systems commit unavailable"
  echo "RJ_FORCE_SCALAR=${RJ_FORCE_SCALAR:-<unset: SIMD fast path ON>}"
  for v in RJ_VMEM_TRACE HSA_HOTSWAP_VERBOSE HSA_HOTSWAP_DUMP_SOURCE; do
    [[ -n "${!v:-}" ]] && echo "::warning::${v}=${!v} set — adds tracing/logging overhead; unset for perf"
  done
  echo "::endgroup::"
}

log_provenance_and_hygiene

# ── Run TensileLite tests ─────────────────────────────────────────────────────

run_tensilelite_tests() {
  local junit_dir="${REPORT_DIR}/junit"
  mkdir -p "${junit_dir}"

  # The supervisor bounds the entire suite and reaps orphaned descendants.
  # The plugin writes results incrementally, including before a suite timeout.
  (
    cd "${REPORT_DIR}" || exit 1
    exec "${PYTHON}" "${SCRIPT_DIR}/rocjitsu_pytest.py" \
      --report-dir "${REPORT_DIR}" --timeout "${SUITE_TIMEOUT_SECONDS}" -- \
      "${ROCJITSU_BIN}" \
      --config "${ROCJITSU_CONFIG}" \
      -- "${PYTHON}" -m pytest \
        "${TENSILELITE_ROOT}/Tensile/Tests/common/test_config.py" \
        -p rocjitsu_pytest --rocjitsu-report-dir="${REPORT_DIR}" \
        -m "${TEST_TARGET}" \
        --gpu-targets="${TEST_TARGET}" \
        -v -s \
        -n "${PYTEST_WORKERS}" \
        --client-lock-scope=worker \
        --timeout="${PER_TEST_TIMEOUT}" \
        --junit-xml="${junit_dir}/tensilelite.xml" \
        --prebuilt-client="${TENSILELITE_CLIENT}" \
        --global-parameters="LibraryFormat='msgpack'" \
        "--tensile-options=--cxx-compiler,${ROCM_PATH}/bin/amdclang++,--gpu-targets,${TEST_TARGET}" \
        "$@"
  ) 2>&1 | tee "${REPORT_DIR}/tensilelite-test.log"

  local statuses=("${PIPESTATUS[@]}")
  local status=${statuses[0]}
  if [[ "${status}" -eq 0 ]]; then
    status=${statuses[1]}
  fi

  # Parse JUnit XML for per-test timing summary
  if [[ -f "${junit_dir}/tensilelite.xml" ]]; then
    REPORT_DIR="${REPORT_DIR}" "${PYTHON}" << 'JUNIT_PARSE'
import xml.etree.ElementTree as ET, os
junit_dir = os.environ.get('REPORT_DIR', '.') + '/junit'
tree = ET.parse(junit_dir + '/tensilelite.xml')
def outcome(tc):
    if tc.find('failure') is not None or tc.find('error') is not None:
        return 'FAILED'
    if tc.find('skipped') is not None:
        return 'SKIPPED'
    return 'PASSED'
tests = [(tc.get('time','0'), tc.get('name',''), outcome(tc)) for tc in tree.iter('testcase')]
tests.sort(key=lambda x: float(x[0]), reverse=True)
total = sum(float(t) for t,_,_ in tests)
passed = sum(1 for _,_,s in tests if s == 'PASSED')
failed = sum(1 for _,_,s in tests if s == 'FAILED')
skipped = sum(1 for _,_,s in tests if s == 'SKIPPED')
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
print('Total: %d tests, %d passed, %d failed, %d skipped, %.0fs aggregate test time (%.0f min)' % (len(tests), passed, failed, skipped, total, total/60))
print('=' * 80)
JUNIT_PARSE
    local report_status=$?
    if [[ "${status}" -eq 0 ]]; then
      status=${report_status}
    fi
  fi

  return "${status}"
}

set +e
run_timed "tensilelite tests (${TEST_TARGET})" run_tensilelite_tests "$@"
test_status=$?
set -e

if [[ "${test_status}" -ne 0 ]]; then
  echo "tensilelite rocjitsu tests exited with status ${test_status}" >&2
  # Preserve pytest failures, missing-test errors, and the suite timeout (124).
fi

exit "${test_status}"
