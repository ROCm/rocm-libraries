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
# xdist worker count. Emulation is CPU-bound and ~15-20 min/test; serial
# execution of the full gfx1250 suite exceeds the workflow step timeout. Local
# validation used 12-16 workers. Override via PYTEST_WORKERS.
PYTEST_WORKERS="${PYTEST_WORKERS:-16}"
# Host SIMD target for the rocjitsu *emulator* build (x86, not the GPU kernels).
# rocm-systems #10702/#10710 give a native-SIMD-width fast path for f32 MFMA/WMMA.
# DEFAULT x86-64-v3 (AVX2, 8-lane): locally build-verified clean on develop
# 6366fe4f with amdclang++ + GCC14 libstdc++ (matches the Ubuntu-24.04 CI base).
# AVX-512 is NOT usable: -march=native reproduces the Aug-14 break (adc2acb2c43) —
# a libstdc++ <experimental/simd> static_assert (simd_x86.h:4232, is_same_v<long
# long, long>) at 512-bit width; a toolchain bug the rocjitsu-side fix can't touch.
# Override: ROCJITSU_MARCH= (empty = portable 4-lane) or =native/=x86-64-v4 (breaks).
ROCJITSU_MARCH="${ROCJITSU_MARCH:-x86-64-v3}"
# Link-time optimization (IPO) for the emulator build. rocm-systems cmake option
# `LTO`, OFF by default; free perf in a Release build (no sanitizers here). Tier-1.
ROCJITSU_LTO="${ROCJITSU_LTO:-ON}"
# Per-CU functional_quantum: max CU step() iters per dispatch quantum (default 1024
# upstream; 0 = unbounded). Higher/0 = fewer scheduler yields on long StreamK/MX
# kernels. Empty = leave upstream default (A/B lever). Injected into every CU when set.
ROCJITSU_FUNCTIONAL_QUANTUM="${ROCJITSU_FUNCTIONAL_QUANTUM:-}"
# rocjitsu exposes TWO host-thread axes; both multiply, and xdist multiplies again:
#   effective_host_threads ≈ PYTEST_WORKERS × num_threads × cpu_dispatch_threads
#   - num_threads (config, top-level): 1 engine thread per XCD, XCDs run
#     concurrently. Default 0 = min(host, #XCDs) → up to 8 on gfx1250. This is
#     the RELIABLE axis today.
#   - cpu_dispatch_threads (config, top-level, rocm-systems #10074): per-SoC CU
#     dispatch width. Default 1 = serial. NOTE rocm-systems #11333: same-SoC CU
#     work currently serializes, so this axis may not materialize as speedup yet.
# We pin BOTH so the product stays ≈ host_cores/worker (no oversubscription), and
# spend the per-process budget on the XCD axis first (reliable), remainder on CU.
# "auto" derives them; set ROCJITSU_CPU_DISPATCH_THREADS/ROCJITSU_NUM_THREADS to
# pin explicitly (e.g. 1/1 to force fully serial).
ROCJITSU_CPU_DISPATCH_THREADS="${ROCJITSU_CPU_DISPATCH_THREADS:-auto}"
ROCJITSU_NUM_THREADS="${ROCJITSU_NUM_THREADS:-auto}"
# Max XCD engine threads to request (gfx1250/gfx94x/gfx950 all have 8 XCDs).
ROCJITSU_MAX_XCD_THREADS="${ROCJITSU_MAX_XCD_THREADS:-8}"
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

# ── Size functional host-thread parallelism (#10074 + XCD engine threads) ─────
# Upstream configs leave both axes at defaults; auto num_threads (up to 8 XCDs)
# alone can oversubscribe once xdist runs PYTEST_WORKERS instances. We inject a
# bounded budget = host_cores/PYTEST_WORKERS per process, spent XCD-axis first
# (num_threads, reliable) then CU-axis (cpu_dispatch_threads, #10074 — may be
# inert per #11333). Writes a modified config copy and repoints ROCJITSU_CONFIG.
apply_dispatch_sizing() {
  local py host_cores budget num_threads cpu_dispatch injected
  py="$(command -v python3.12 || command -v python3)"
  host_cores="$(nproc)"

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
    INJECT_CPU_DISPATCH="${cpu_dispatch}" INJECT_FQ="${ROCJITSU_FUNCTIONAL_QUANTUM}" \
    OUT_CONFIG="${injected}" \
    "${py}" - <<'PYEOF'
import json, os
cfg = json.load(open(os.environ["SRC_CONFIG"]))
em = cfg.get("exec_mode", "functional")
if em != "functional":
    print(f"::warning::exec_mode={em} — 'functional' is the fast path; clocked/other is far slower")
cfg["num_threads"] = int(os.environ["INJECT_NUM_THREADS"])
cfg["cpu_dispatch_threads"] = int(os.environ["INJECT_CPU_DISPATCH"])

# functional_quantum is a per-CU field carried in each compute_unit node's
# "config" [{key,value}] list under topology. Set it on every CU when requested.
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
    set_cu_quantum(cfg.get("topology", {}))

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
export PYTHONPATH="${TENSILELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "ROCM_PATH=${ROCM_PATH}"
echo "AMDGPU_FAMILIES=${AMDGPU_FAMILIES}"
echo "ROCJITSU_GPU_TARGET=${ROCJITSU_GPU_TARGET}"
echo "ROCJITSU_CONFIG=${ROCJITSU_CONFIG}"
echo "ROCJITSU_MARCH=${ROCJITSU_MARCH}"
echo "TENSILELITE_ROOT=${TENSILELITE_ROOT}"
echo "TENSILELITE_CLIENT=${TENSILELITE_CLIENT}"
echo "PER_TEST_TIMEOUT=${PER_TEST_TIMEOUT}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ── Build rocjitsu ────────────────────────────────────────────────────────────

configure_rocjitsu() {
  local cxx_flags="-Wno-error=unknown-warning-option -Wno-error=nested-anon-types"
  # Only add -march when explicitly requested (default empty = portable/safe;
  # see the ROCJITSU_MARCH note re: the Aug-14 AVX-512 build break).
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
  cmake --build "${ROCJITSU_BUILD_DIR}" --target rocjitsu_bin rocjitsu_shared hsa_hotswap_rocjitsu
}

ROCJITSU_BIN="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"

show_rocjitsu_version() {
  "${ROCJITSU_BIN}" --version
}

run_timed "configure rocjitsu" configure_rocjitsu
run_timed "build rocjitsu" build_rocjitsu
run_timed "rocjitsu version" show_rocjitsu_version

# The HSA hotswap hook lets rocjitsu intercept the HIP runtime's device query.
# Without it, tensilelite-client fails with hipErrorNoDevice.
HOTSWAP_LIB=$(find "${ROCJITSU_BUILD_DIR}" -name "libhsa_hotswap_rocjitsu.so" -type f | head -1)
if [[ -n "${HOTSWAP_LIB}" ]]; then
  cp "${HOTSWAP_LIB}" "${ROCM_PATH}/lib/"
  echo "Installed hotswap lib: ${ROCM_PATH}/lib/libhsa_hotswap_rocjitsu.so"
else
  echo "::warning::libhsa_hotswap_rocjitsu.so not found — tests may fail with hipErrorNoDevice"
fi

# ── Log emulator provenance + perf hygiene ────────────────────────────────────
# The one fact logged nowhere else: the exact rocjitsu commit (compare its date
# to a fix's merge date to confirm inclusion). Plus warn on env that silently
# slows a functional run. Build flags, config knobs, and exec_mode are already
# logged at the points they're set (env dump, build flags line, dispatch sizing).
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

  # pytest-timeout handles per-test kills; xdist (-n) runs tests concurrently so
  # the suite fits inside the workflow step timeout. The step timeout is the
  # process-tree kill backstop.
  "${ROCJITSU_BIN}" \
      --config "${ROCJITSU_CONFIG}" \
      -- "${PYTHON}" -m pytest \
        "${TENSILELITE_ROOT}/Tensile/Tests/common" \
        -m "${ROCJITSU_GPU_TARGET}" \
        -v -s \
        -n "${PYTEST_WORKERS}" \
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
