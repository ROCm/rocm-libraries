#!/usr/bin/env bash
set -euo pipefail

# Temporary bridge for rocjitsu race detection in rocm-libraries CI.
#
# The calling workflow fetches hipBLASLt/TensileLite artifacts from the current
# TheRock build, then checks out rocm-systems. Until rocjitsu is packaged as a
# complete runnable TheRock artifact, build the rocjitsu CLI locally and run
# small gfx950 hipBLASLt and TensileLite smokes under the race detector.
#
# Keep this script deliberately narrow. It is not a replacement for the normal
# hipBLASLt or TensileLite test suites; it is a post-build instrumentation pass
# that proves the TheRock BLAS artifacts can be launched through rocjitsu and do
# not emit race reports for one small, deterministic gfx950 GEMM path. The
# normal package tests still own broad functional coverage.

ROCM_PATH="${ROCM_PATH:-${PWD}/build}"
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

# The rocjitsu smoke currently has several expensive pieces: artifact setup in
# the caller, local rocjitsu configure/build here, and the two instrumented GPU
# workloads. Keep a durable timing file as well as grouped console output so
# slow CI runs are diagnosable without re-running the workflow.
print_timing_summary() {
  if [[ -f "${TIMING_FILE}" ]]; then
    echo ""
    echo "=== rocjitsu race smoke timing summary ==="
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

# Fail before doing any rocjitsu work if the expected TheRock BLAS test
# artifacts are missing. Most failures here mean the artifact fetch/layout
# changed, not that the race detector found a race.
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

# Prefer the architecture-specific config name when present, but accept the
# older generic CDNA4 config while rocm-systems and TheRock packaging are still
# converging.
if [[ -z "${ROCJITSU_CONFIG}" ]]; then
  for candidate in \
    "${ROCJITSU_SOURCE_DIR}/configs/gfx950_cdna4_kmd.json" \
    "${ROCJITSU_SOURCE_DIR}/configs/amdgpu_cdna4_kmd.json"; do
    if [[ -f "${candidate}" ]]; then
      ROCJITSU_CONFIG="${candidate}"
      break
    fi
  done
fi

if [[ -z "${ROCJITSU_CONFIG}" || ! -f "${ROCJITSU_CONFIG}" ]]; then
  echo "rocjitsu gfx950 config not found under ${ROCJITSU_SOURCE_DIR}/configs" >&2
  exit 1
fi

mkdir -p "${RACE_REPORT_DIR}"
: >"${TIMING_FILE}"

# Everything below must resolve against the fetched ROCm payload, not whatever
# happens to be installed in the CI image. In particular, lib/rocm_sysdeps/lib
# is needed by some packaged dependencies, and the TensileLite Python tree comes
# from the BLAS test artifacts.
export ROCM_PATH
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:${ROCM_PATH}/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TENSILELITE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "ROCM_PATH=${ROCM_PATH}"
echo "ROCJITSU_SOURCE_DIR=${ROCJITSU_SOURCE_DIR}"
echo "ROCJITSU_BUILD_DIR=${ROCJITSU_BUILD_DIR}"
echo "ROCJITSU_CONFIG=${ROCJITSU_CONFIG}"
echo "HIPBLASLT_BENCH=${HIPBLASLT_BENCH}"
echo "TENSILELITE_ROOT=${TENSILELITE_ROOT}"
echo "TENSILE_DRIVER=${TENSILE_DRIVER}"
echo "TENSILELITE_CLIENT=${TENSILELITE_CLIENT}"
echo "RACE_REPORT_DIR=${RACE_REPORT_DIR}"
echo "PATH=${PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "PYTHONPATH=${PYTHONPATH}"

# rocjitsu is still consumed from a source checkout in this workflow. Build only
# the CLI and optional runtime/shim targets needed to launch the smoke workloads;
# this keeps the job independent of full rocm-systems packaging.
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

if command -v amdclang >/dev/null 2>&1 && command -v amdclang++ >/dev/null 2>&1; then
  cmake_args+=(
    -DCMAKE_C_COMPILER="$(command -v amdclang)"
    -DCMAKE_CXX_COMPILER="$(command -v amdclang++)"
  )
fi

run_timed "configure rocjitsu" cmake "${cmake_args[@]}"

# rocm-systems has been renaming/splitting rocjitsu build targets while this
# bridge job is being developed. Probe for optional targets instead of assuming
# one exact target set, so the CI job survives small target-layout changes.
cmake_target_exists() {
  local target="$1"
  cmake --build "${ROCJITSU_BUILD_DIR}" --target help \
    | grep -Eq "(^|[[:space:]])${target}([:[:space:]]|$)"
}

build_targets=(rocjitsu_bin)
if cmake_target_exists rocjitsu_shared; then
  build_targets+=(rocjitsu_shared)
fi
if cmake_target_exists rocjitsu_kmd_shim; then
  build_targets+=(rocjitsu_kmd_shim)
fi
run_timed "build rocjitsu" cmake --build "${ROCJITSU_BUILD_DIR}" --target "${build_targets[@]}"

ROCJITSU_BIN="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"
if [[ ! -x "${ROCJITSU_BIN}" ]]; then
  echo "rocjitsu binary not found after build: ${ROCJITSU_BIN}" >&2
  exit 1
fi

rm -f "${RACE_REPORT_DIR}/race.log"

show_rocjitsu_version() {
  echo "rocjitsu version:"
  "${ROCJITSU_BIN}" --version
}

# This is a cheap launch-path check for rocjitsu itself. Treat it as diagnostic
# rather than authoritative: the real signal is whether HIP/HSA workloads from
# the fetched ROCm payload run and whether their race logs contain reports.
run_sanity_check() {
  echo "rocjitsu /bin/true sanity check:"
  mkdir -p "${RACE_REPORT_DIR}/sanity"
  timeout 30 \
    env \
      RJ_LOG=1 \
      RJ_SINKS=stderr,file \
      RJ_SINK_DIR="${RACE_REPORT_DIR}/sanity" \
      "${ROCJITSU_BIN}" \
        --config "${ROCJITSU_CONFIG}" \
        -- /bin/true \
    2>&1 | tee "${RACE_REPORT_DIR}/sanity.log"
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "rocjitsu /bin/true sanity check failed with status ${status}" >&2
    return "${status}"
  fi
}

run_hipblaslt_bench_smoke() {
  mkdir -p "${RACE_REPORT_DIR}/hipblaslt-bench"
  echo "running hipblaslt-bench under rocjitsu race detection"
  # Use zero initialization so this smoke measures the hipBLASLt GEMM dispatch
  # path rather than spending most of its signal on device-side fill kernels.
  # The default HPL initialization can generate separate fill kernels that are
  # useful for compiler/detector investigation, but they obscure this CI gate.
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
          -m 128 \
          -n 128 \
          -k 128 \
          --iters 1 \
          --cold_iters 0 \
    2>&1 | tee "${RACE_REPORT_DIR}/hipblaslt-bench.log"
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "hipblaslt-bench smoke command failed with status ${status}" >&2
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

write_tensilelite_smoke_yaml() {
  local yaml="$1"
  # This reduced TensileLite config is intentionally embedded instead of checked
  # in as a test-data file. The CI smoke only needs one small f32 assembly GEMM
  # to exercise the Python driver, generated code object, prebuilt
  # tensilelite-client, and rocjitsu race detector together.
  cat >"${yaml}" <<'YAML'
GlobalParameters:
  NumElementsToValidate: -1
  DataInitTypeBeta: 0
  DataInitTypeAlpha: 1
  Device: 0
  CpuThreads: 1

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

run_tensilelite_client_smoke() {
  local test_dir="${RACE_REPORT_DIR}/tensilelite"
  local output_dir="${test_dir}/output"
  local sink_dir="${test_dir}/sinks"
  local yaml="${test_dir}/f32_nt_32x32.yaml"
  mkdir -p "${output_dir}" "${sink_dir}"
  write_tensilelite_smoke_yaml "${yaml}"

  local tensile_args=(
    "${yaml}"
    "${output_dir}"
    --prebuilt-client "${TENSILELITE_CLIENT}"
    --gpu-targets gfx950
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
    echo "tensilelite-client smoke command failed with status ${status}" >&2
    return "${status}"
  fi

  if ! grep -q "PASSED" "${RACE_REPORT_DIR}/tensilelite-client.log"; then
    echo "tensilelite-client smoke did not report validation success" >&2
    return 1
  fi

  # As above, validation success and race-detector success are distinct signals.
  # Require both: the client must pass numerics, and rocjitsu must leave the race
  # sink free of RACE records.
  if [[ -f "${sink_dir}/race.log" ]] && grep -q '^RACE ' "${sink_dir}/race.log"; then
    echo "rocjitsu race detector reported a race in tensilelite-client:" >&2
    cat "${sink_dir}/race.log" >&2
    return 1
  fi
}

run_timed "rocjitsu version" show_rocjitsu_version

set +e
run_timed "rocjitsu sanity check" run_sanity_check
sanity_status=$?
set -e
if [[ "${sanity_status}" -ne 0 ]]; then
  echo "WARNING: rocjitsu /bin/true sanity check failed; continuing to GPU workload smokes" >&2
fi

smoke_status=0

# Run both workload smokes even if the first one fails. That gives the uploaded
# race-reports artifact a complete picture for the failing CI attempt instead of
# forcing a second long run just to learn whether the other workload also broke.
set +e
run_timed "hipblaslt-bench race smoke" run_hipblaslt_bench_smoke
hipblaslt_status=$?
run_timed "tensilelite-client race smoke" run_tensilelite_client_smoke
tensilelite_status=$?
set -e

if [[ "${hipblaslt_status}" -ne 0 ]]; then
  echo "hipblaslt-bench race smoke failed with status ${hipblaslt_status}" >&2
  smoke_status=1
fi

if [[ "${tensilelite_status}" -ne 0 ]]; then
  echo "tensilelite-client race smoke failed with status ${tensilelite_status}" >&2
  smoke_status=1
fi

if [[ "${smoke_status}" -ne 0 ]]; then
  echo "one or more rocjitsu race smokes failed" >&2
  exit "${smoke_status}"
fi

echo "rocjitsu race smokes completed without race reports"
