#!/usr/bin/env bash
set -euo pipefail

# The workflow that calls this script fetches hipBLASLt/TensileLite artifacts
# from the current TheRock build, then checks out rocm-systems.
#
# TODO(newling) Until rocjitsu is packaged as a complete runnable TheRock
# artifact, we build the rocjitsu CLI locally. Monitor progress on packaging rocjitsu.
#
# The script runs small hipBLASLt and TensileLite GEMMs under the race detector.
# TODO(newling) expand the GEMM-space tested.
#
# Basic flow:
#   1. Use the TheRock artifact tree unpacked at ROCM_PATH.
#   2. Build rocjitsu from the rocm-systems checkout in ROCJITSU_SOURCE_DIR.
#   3. Select a rocjitsu config for gfx942 or gfx950.
#   4. Run hipblaslt-bench and a reduced TensileLite smoke under RJ_RACE=1.

# These defaults match the GitHub Actions workspace layout: ROCM_PATH is the
# unpacked TheRock artifact tree, ROCJITSU_SOURCE_DIR is the checked-out
# rocm-systems source tree, ROCJITSU_BUILD_DIR is a local build directory, and
# RACE_REPORT_DIR is uploaded at the end of the job. They can be overridden for
# local reproduction. The brittle part is not the path names themselves, but the
# artifact layout underneath ROCM_PATH; the checks below fail early if
# hipBLASLt/TensileLite files move in TheRock artifacts.
ROCM_PATH="${ROCM_PATH:-${PWD}/build}"
AMDGPU_FAMILIES="${AMDGPU_FAMILIES:-}"
ROCJITSU_GPU_TARGET="${ROCJITSU_GPU_TARGET:-}"
ROCJITSU_SOURCE_DIR="${ROCJITSU_SOURCE_DIR:-${PWD}/rocm-systems/emulation/rocjitsu}"
ROCJITSU_BUILD_DIR="${ROCJITSU_BUILD_DIR:-${PWD}/rocjitsu-build}"
ROCJITSU_CONFIG="${ROCJITSU_CONFIG:-}"
RACE_REPORT_DIR="${RACE_REPORT_DIR:-${PWD}/race-reports}"
HIPBLASLT_BENCH="${HIPBLASLT_BENCH:-${ROCM_PATH}/bin/hipblaslt-bench}"
TENSILELITE_ROOT="${TENSILELITE_ROOT:-${ROCM_PATH}/share/hipblaslt/tensilelite}"
TENSILE_DRIVER="${TENSILE_DRIVER:-${TENSILELITE_ROOT}/Tensile/bin/Tensile}"
TENSILELITE_CLIENT="${TENSILELITE_CLIENT:-${ROCM_PATH}/libexec/hipblaslt/tensilelite/tensilelite-client}"
RACE_TIMEOUT_SECONDS="${RACE_TIMEOUT_SECONDS:-180}"
TENSILELITE_TIMEOUT_SECONDS="${TENSILELITE_TIMEOUT_SECONDS:-420}"
TIMING_FILE="${RACE_REPORT_DIR}/timing.tsv"

# Map TheRock's artifact-group name to the concrete GPU target used by both
# rocjitsu and TensileLite. The workflow pins one rocm-systems revision, so the
# corresponding target-specific KMD config paths are part of this script's
# contract instead of a compatibility search across historical config names.
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
    *)
      echo "Unsupported rocjitsu race-check target: ${target_selector}" >&2
      echo "Supported groups are gfx94* and gfx950*." >&2
      exit 1
      ;;
  esac

  ROCJITSU_CONFIG="${ROCJITSU_CONFIG:-${default_config}}"
}

# Record the duration and exit status of each stage in a file that is uploaded
# with the other race reports. The same information is also printed at exit.
print_timing_summary() {
  if [[ -f "${TIMING_FILE}" ]]; then
    echo ""
    echo "=== rocjitsu race check timing summary ==="
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
  case "$-" in
    *e*) had_errexit=1 ;;
  esac

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
  else
    set +e
  fi
  return "${status}"
}


# These checks validate the artifact layout before any workload is launched
# under rocjitsu. A failure here means the expected TheRock BLAS test payload is
# not available to this job.
if [[ ! -d "${ROCM_PATH}" ]]; then
  echo "ROCM_PATH does not exist: ${ROCM_PATH}" >&2
  exit 1
fi

if [[ ! -x "${HIPBLASLT_BENCH}" ]]; then
  echo "hipblaslt-bench not found or not executable: ${HIPBLASLT_BENCH}" >&2
  exit 1
fi

if [[ ! -d "${TENSILELITE_ROOT}" ]]; then
  echo "TensileLite artifacts not found: ${TENSILELITE_ROOT}" >&2
  echo "Expected TheRock BLAS test artifacts to contain share/hipblaslt/tensilelite" >&2
  exit 1
fi

if [[ ! -f "${TENSILE_DRIVER}" ]]; then
  echo "Tensile driver not found: ${TENSILE_DRIVER}" >&2
  exit 1
fi

if [[ ! -x "${TENSILELITE_CLIENT}" ]]; then
  echo "tensilelite-client not found or not executable: ${TENSILELITE_CLIENT}" >&2
  exit 1
fi

# The compiler artifact layout depends on whether TheRock enables LLVM's host
# per-target runtime directories. Prefer the flat lib/llvm/lib layout, then the
# matching host-triple directory when per-target runtime directories are used.
LLVM_RUNTIME_ROOT="${ROCM_PATH}/lib/llvm/lib"
if [[ ! -d "${LLVM_RUNTIME_ROOT}" ]]; then
  echo "LLVM runtime directory not found: ${LLVM_RUNTIME_ROOT}" >&2
  exit 1
fi

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
  echo "rocjitsu ${ROCJITSU_GPU_TARGET} config not found under ${ROCJITSU_SOURCE_DIR}/configs" >&2
  exit 1
fi

mkdir -p "${RACE_REPORT_DIR}"
: >"${TIMING_FILE}"

# Everything below must resolve against the fetched ROCm payload, not whatever
# happens to be installed in the CI image. ROCM_PATH is exported for subprocesses
# that use it to find the ROCm install root.
export ROCM_PATH
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:${LLVM_RUNTIME_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TENSILELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "ROCM_PATH=${ROCM_PATH}"
echo "AMDGPU_FAMILIES=${AMDGPU_FAMILIES}"
echo "ROCJITSU_GPU_TARGET=${ROCJITSU_GPU_TARGET}"
echo "ROCJITSU_SOURCE_DIR=${ROCJITSU_SOURCE_DIR}"
echo "ROCJITSU_BUILD_DIR=${ROCJITSU_BUILD_DIR}"
echo "ROCJITSU_CONFIG=${ROCJITSU_CONFIG}"
echo "HIPBLASLT_BENCH=${HIPBLASLT_BENCH}"
echo "TENSILELITE_ROOT=${TENSILELITE_ROOT}"
echo "TENSILE_DRIVER=${TENSILE_DRIVER}"
echo "TENSILELITE_CLIENT=${TENSILELITE_CLIENT}"
echo "LIBOMP_PATH=${LIBOMP_PATH}"
echo "RACE_REPORT_DIR=${RACE_REPORT_DIR}"
echo "PATH=${PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "PYTHONPATH=${PYTHONPATH}"


# TODO(newling): Track migration to a packaged rocjitsu once TheRock provides a
# complete runnable artifact. Until then, build rocjitsu from the rocm-systems
# checkout selected by the workflow so this job controls the tool source.
# Build only the CLI and its LD_PRELOAD runtime; this keeps the job independent
# of full rocm-systems packaging.
#
# The warning suppressions keep the local rocjitsu build from failing on
# compiler/header warning mismatches. `nested-anon-types` is a Clang warning for
# anonymous structs/unions nested inside another type; use -Wno-error so those
# warnings remain visible but do not fail this bridge build.
cmake_args=(
  -S "${ROCJITSU_SOURCE_DIR}"
  -B "${ROCJITSU_BUILD_DIR}"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_TESTING=OFF
  -DROCM_PATH="${ROCM_PATH}"
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}"
  -DCMAKE_CXX_FLAGS="-Wno-error=unknown-warning-option -Wno-error=nested-anon-types"
)

if ! command -v amdclang >/dev/null 2>&1 || ! command -v amdclang++ >/dev/null 2>&1; then
  echo "amdclang and amdclang++ must be available from the fetched ROCm payload" >&2
  exit 1
fi
cmake_args+=(
  -DCMAKE_C_COMPILER="$(command -v amdclang)"
  -DCMAKE_CXX_COMPILER="$(command -v amdclang++)"
)

run_timed "configure rocjitsu" cmake "${cmake_args[@]}"

# The pinned source revision exposes stable targets for the CLI launcher and its
# LD_PRELOAD runtime. Both are required to execute the workloads below.
run_timed "build rocjitsu" \
  cmake --build "${ROCJITSU_BUILD_DIR}" \
  --target rocjitsu_bin rocjitsu_shared

ROCJITSU_BIN="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"
if [[ ! -x "${ROCJITSU_BIN}" ]]; then
  echo "rocjitsu binary not found after build: ${ROCJITSU_BIN}" >&2
  exit 1
fi

show_rocjitsu_version() {
  echo "rocjitsu version:"
  "${ROCJITSU_BIN}" --version
}

run_hipblaslt_bench_check() {
  mkdir -p "${RACE_REPORT_DIR}/hipblaslt-bench"
  echo "running hipblaslt-bench under rocjitsu race detection"
  # Use zero initialization because HPL initialization currently triggers a
  # separate device-fill kernel LDS race. Keep this check scoped to the
  # hipBLASLt GEMM dispatch path while that issue is investigated separately.
  timeout "${RACE_TIMEOUT_SECONDS}" \
    env \
      HSA_ENABLE_SDMA=1 \
      RJ_RACE=1 \
      RJ_LOG=1 \
      RJ_SINKS=stderr,file \
      RJ_SINK_DIR="${RACE_REPORT_DIR}/hipblaslt-bench" \
      "${ROCJITSU_BIN}" \
        --config "${ROCJITSU_CONFIG}" \
        -- "${HIPBLASLT_BENCH}" \
          --precision f32_r \
          --initialization zero \
          --verify \
          -m 128 \
          -n 128 \
          -k 128 \
          --iters 1 \
          --cold_iters 0 \
    2>&1 | tee "${RACE_REPORT_DIR}/hipblaslt-bench.log"
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "hipblaslt-bench race check command failed with status ${status}" >&2
    return "${status}"
  fi

  # rocjitsu writes race findings to its sink directory. The application may
  # still exit normally, so explicitly inspect race.log and make any report a CI
  # failure with the report echoed into the job log.
  if [[ -f "${RACE_REPORT_DIR}/hipblaslt-bench/race.log" ]] &&
    grep -q '^RACE ' "${RACE_REPORT_DIR}/hipblaslt-bench/race.log"; then
    echo "rocjitsu race detector reported a race in hipblaslt-bench:" >&2
    cat "${RACE_REPORT_DIR}/hipblaslt-bench/race.log" >&2
    return 1
  fi
}


write_tensilelite_check_yaml() {
  local yaml="$1"

  # One reduced config is embedded here to keep this bridge job self-contained.
  # If the race check grows to multiple configs, move them into a small checked-in
  # data directory or install them with the TensileLite test artifacts, then have
  # this script iterate over that list.
  cat >"${yaml}" <<YAML
GlobalParameters:
  NumElementsToValidate: -1
  DataInitTypeBeta: 0
  DataInitTypeAlpha: 1
  Device: 0
  CpuThreads: 1
  PrintSolutionRejectionReason: True

BenchmarkProblems:
  -
    - OperationType: GEMM
      DataType: s
      TransposeA: False
      TransposeB: True
      UseBeta: True
      Batched: True
    - InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - MatrixInstruction:
          - [16, 16, 4, 1, 1, 2, 3, 2, 2]
        - PrefetchGlobalRead: [2]
        - PrefetchLocalRead: [1]
        - DepthU: [32]
        - VectorWidthA: [-1]
        - VectorWidthB: [-1]
        - GlobalReadVectorWidthA: [-1]
        - GlobalReadVectorWidthB: [-1]
        - LocalReadVectorWidth: [-1]
        - TransposeLDS: [-1]
        - LdsBlockSizePerPadA: [-1]
        - LdsBlockSizePerPadB: [-1]
        - LdsPadA: [-1]
        - LdsPadB: [-1]
        - StaggerU: [16]
        - StaggerUStride: [-1]
        - WorkGroupMapping: [1]
        - 1LDSBuffer: [-1]
        - WorkGroupMappingXCC: [8]
        - WorkGroupMappingXCCGroup: [-1]
        - GlobalSplitU: [1]
        - GlobalSplitUAlgorithm: ["MultipleBuffer"]
        - GlobalReadPerMfma: [1.0]
        - LocalWritePerMfma: [-1]
        - StoreRemapVectorWidth: [0]
        - StoreVectorWidth: [-1]
        - SourceSwap: [True]
        - NumElementsPerBatchStore: [16]
        - ClusterLocalRead: [1]
        - DirectToVgprA: [True]
        - DirectToVgprB: [False]
        - WorkGroup: [[32, 4, 4]]
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [32, 32, 1, 32]
YAML
}

run_tensilelite_client_check() {
  local test_dir="${RACE_REPORT_DIR}/tensilelite"
  local output_dir="${test_dir}/output"
  local sink_dir="${test_dir}/sinks"
  local yaml="${test_dir}/f32_nt_32x32.yaml"
  mkdir -p "${output_dir}" "${sink_dir}"
  write_tensilelite_check_yaml "${yaml}"

  local tensile_args=(
    "${yaml}"
    "${output_dir}"
    --prebuilt-client "${TENSILELITE_CLIENT}"
    --gpu-targets "${ROCJITSU_GPU_TARGET}"
    --library-format msgpack
  )
  if [[ -x "${ROCM_PATH}/bin/amdclang++" ]]; then
    tensile_args+=(--cxx-compiler "${ROCM_PATH}/bin/amdclang++")
  fi

  echo "running tensilelite-client under rocjitsu race detection"
  # Drive the normal Tensile front end with the client from the TheRock artifact.
  # This covers a different surface from hipblaslt-bench: Python-side Tensile
  # setup, rocisa imports, generated client config, and the standalone
  # tensilelite-client runtime path.
  timeout "${TENSILELITE_TIMEOUT_SECONDS}" \
    env \
      HSA_ENABLE_SDMA=1 \
      RJ_RACE=1 \
      RJ_LOG=1 \
      RJ_SINKS=stderr,file \
      RJ_SINK_DIR="${sink_dir}" \
      "${ROCJITSU_BIN}" \
        --config "${ROCJITSU_CONFIG}" \
        -- python3 "${TENSILE_DRIVER}" "${tensile_args[@]}" \
    2>&1 | tee "${RACE_REPORT_DIR}/tensilelite-client.log"
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "tensilelite-client race check command failed with status ${status}" >&2
    return "${status}"
  fi

  if ! grep -q "PASSED" "${RACE_REPORT_DIR}/tensilelite-client.log"; then
    echo "tensilelite-client race check did not report validation success" >&2
    return 1
  fi

  # As above, client success and race-detector success are distinct signals.
  # Require both: the client must finish cleanly, and rocjitsu must leave the
  # race sink free of RACE records.
  if [[ -f "${sink_dir}/race.log" ]] && grep -q '^RACE ' "${sink_dir}/race.log"; then
    echo "rocjitsu race detector reported a race in tensilelite-client:" >&2
    cat "${sink_dir}/race.log" >&2
    return 1
  fi
}

run_timed "rocjitsu version" show_rocjitsu_version

check_status=0

# Run both workload checks even if the first one fails. That gives the uploaded
# race-reports artifact a complete picture for the failing CI attempt instead of
# forcing a second long run just to learn whether the other workload also broke.
set +e
run_timed "hipblaslt-bench race check" run_hipblaslt_bench_check
hipblaslt_status=$?
run_timed "tensilelite-client race check" run_tensilelite_client_check
tensilelite_status=$?
set -e

if [[ "${hipblaslt_status}" -ne 0 ]]; then
  echo "hipblaslt-bench race check failed with status ${hipblaslt_status}" >&2
  check_status=1
fi

if [[ "${tensilelite_status}" -ne 0 ]]; then
  echo "tensilelite-client race check failed with status ${tensilelite_status}" >&2
  check_status=1
fi

if [[ "${check_status}" -ne 0 ]]; then
  echo "one or more rocjitsu race checks failed" >&2
  exit "${check_status}"
fi

echo "rocjitsu race checks completed without race reports"
