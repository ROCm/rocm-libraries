#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configure the supported hipBLASLt / TensileLite / rocISA / hipSPARSELt
# topologies from one source checkout. Run this inside a prepared ROCm
# environment; see scripts/README.md for a Docker invocation.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hipblaslt_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${hipblaslt_root}/../.." && pwd)"
hipsparselt_root="${repo_root}/projects/hipsparselt"
rocisa_root="${hipblaslt_root}/tensilelite/rocisa"

cmake_bin="${CMAKE_BIN:-cmake}"
python_bin="${PYTHON_BIN:-python3}"
generator="${CMAKE_GENERATOR:-Ninja}"
rocm_prefix="${ROCM_PATH:-/opt/rocm}"
gpu_targets="${GPU_TARGETS:-gfx950}"
results_dir="${RESULTS_DIR:-${repo_root}/.cmake-configure-matrix}"
nanobind_source="${NANOBIND_SOURCE:-}"
stage_prefix="${STAGED_HIPBLASLT_PREFIX:-}"
prepare_stage=0
build_device=0
keep_results=0
run_all=0
with_yaml=0
declare -a requested_cells=()
declare -a last_configure_command=()
declare -a current_assertions=()

snapshot_helper="${script_dir}/cmake_configure_matrix_result.py"
matrix_result=""
last_configure_source_dir=""
last_configure_build_dir=""
last_configure_stage_dir=""
last_configure_log_path=""
active_cell=""
active_cell_recorded=0
active_error=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Run a local configure matrix inside a prepared ROCm environment.

Options:
  --cell NAME             Run one cell (repeatable). Default: core local cells.
  --prepare-stage         Build/install a minimal staged hipBLASLt provider for
                          the staged-TheRock cell.
  --all                   Run every cell, including staged and YAML cells.
  --with-yaml             Add the YAML static-TensileLite cell.
  --stage-prefix PATH     Use an existing staged hipBLASLt provider.
  --nanobind-source PATH  Offline nanobind source for FetchContent cells.
  --results-dir PATH      Directory for isolated build/install trees.
  --gpu-targets LIST      GPU targets (default: ${gpu_targets}).
  --rocm-prefix PATH      ROCm prefix (default: ${rocm_prefix}).
  --build-device          Build device libraries for source/staged hipSPARSELt.
  --keep-results          Do not delete a cell's previous build directory.
  --list                  List cells and exit.
  -h, --help              Show this help.

Cells:
  hipblaslt-shared         Default shared producer with bundled codegen.
  hipblaslt-library-only   Shared library package without bundled codegen.
  hipblaslt-static         All-owner static msgpack package.
  hipblaslt-mixed          Shared hipBLASLt with static TensileLite (M1).
  hipblaslt-tl-yaml        TensileLite-only static YAML export.
  hipblaslt-device-only    Device generation with both host products disabled.
  rocisa-shared            Standalone rocISA with source StinkyTofu.
  rocisa-static            Standalone static rocISA/plugin topology.
  hipsparselt-source       Source-provider hipSPARSELt.
  hipsparselt-static       Static source-provider hipSPARSELt.
  hipsparselt-staged       Staged/TheRock hipSPARSELt provider.

The staged cell requires --stage-prefix or --prepare-stage. The YAML cell
requires a ROCm LLVM package with its zstd dependency closure available. Each
run writes <results-dir>/cmake-configure-matrix.json with the exact configure
argv, parsed CMake cache, compile commands, and other build-tree metadata.
EOF
}

list_cells() {
    cat <<'EOF'
hipblaslt-shared        shared producer / bundled codegen package
hipblaslt-library-only  BUNDLE=OFF library-only package
hipblaslt-static        all-owner static msgpack package
hipblaslt-mixed         shared hipBLASLt + static TensileLite export
hipblaslt-tl-yaml       TensileLite-only static YAML export
hipblaslt-device-only   device-only generation topology
rocisa-shared           standalone rocISA shared-StinkyTofu topology
rocisa-static           standalone rocISA static-StinkyTofu topology
hipsparselt-source      source-provider hipSPARSELt
hipsparselt-static      static source-provider hipSPARSELt
hipsparselt-staged      staged/TheRock hipSPARSELt provider
EOF
}

die() {
    if [[ -n "${active_cell}" ]]; then
        active_error="error: $*"
    fi
    echo "error: $*" >&2
    exit 1
}

require_path() {
    [[ -e "$1" ]] || die "required path does not exist: $1"
}

require_command() {
    command -v "$1" >/dev/null || die "required command not found: $1"
}

assert_contains() {
    local file="$1"
    local text="$2"
    grep -Fq -- "$text" "$file" || die "expected '$text' in $file"
    current_assertions+=("contains|${file}|${text}")
}

assert_not_contains() {
    local file="$1"
    local text="$2"
    if grep -Fq -- "$text" "$file"; then
        die "did not expect '$text' in $file"
    fi
    current_assertions+=("not_contains|${file}|${text}")
}

clean_cell() {
    local cell="$1"
    local file_api_query_dir
    cell_build="${results_dir}/${cell}/build"
    cell_stage="${results_dir}/${cell}/stage"
    if [[ "${keep_results}" -eq 0 ]]; then
        rm -rf "${cell_build}" "${cell_stage}"
    fi
    mkdir -p "${cell_build}" "${cell_stage}"
    file_api_query_dir="${cell_build}/.cmake/api/v1/query/"
    file_api_query_dir+="client-hipblaslt-configure-matrix"
    mkdir -p "${file_api_query_dir}"
    : > "${file_api_query_dir}/codemodel-v2"
    : > "${file_api_query_dir}/cache-v2"
}

run_configure_command() {
    last_configure_log_path="${last_configure_build_dir}/configure.log"
    "${last_configure_command[@]}" 2>&1 | tee "${last_configure_log_path}"
}

configure_hipblaslt() {
    local cell="$1"
    shift
    clean_cell "${cell}"
    last_configure_source_dir="${hipblaslt_root}"
    last_configure_build_dir="${cell_build}"
    last_configure_stage_dir="${cell_stage}"
    last_configure_command=("${cmake_bin}" -S "${hipblaslt_root}" -B "${cell_build}" -G "${generator}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${rocm_prefix}/bin/amdclang++" \
        -DCMAKE_C_COMPILER="${rocm_prefix}/bin/amdclang" \
        -DCMAKE_PREFIX_PATH="${rocm_prefix}" \
        -DCMAKE_INSTALL_PREFIX="${cell_stage}" \
        -DGPU_TARGETS="${gpu_targets}" \
        "$@" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
    run_configure_command
}

configure_hipsparselt() {
    local cell="$1"
    shift
    clean_cell "${cell}"
    last_configure_source_dir="${hipsparselt_root}"
    last_configure_build_dir="${cell_build}"
    last_configure_stage_dir="${cell_stage}"
    last_configure_command=("${cmake_bin}" -S "${hipsparselt_root}" -B "${cell_build}" -G "${generator}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${rocm_prefix}/bin/amdclang++" \
        -DCMAKE_C_COMPILER="${rocm_prefix}/bin/amdclang" \
        -DCMAKE_PREFIX_PATH="${matrix_cmake_prefix:-${rocm_prefix}}" \
        -DCMAKE_INSTALL_PREFIX="${cell_stage}" \
        -DGPU_TARGETS="${gpu_targets}" \
        -DHIPSPARSELT_ENABLE_CLIENT=OFF \
        -DHIPSPARSELT_BUILD_TESTING=OFF \
        -DHIPSPARSELT_ENABLE_MARKER=OFF \
        "$@" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
    run_configure_command
}

configure_rocisa() {
    local cell="$1"
    shift
    clean_cell "${cell}"
    last_configure_source_dir="${rocisa_root}"
    last_configure_build_dir="${cell_build}"
    last_configure_stage_dir="${cell_stage}"
    last_configure_command=("${cmake_bin}" -S "${rocisa_root}" -B "${cell_build}" -G "${generator}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${rocm_prefix}/bin/amdclang++" \
        -DROCM_PATH="${rocm_prefix}" \
        -DCMAKE_PREFIX_PATH="${rocm_prefix}" \
        -DGPU_TARGETS="${gpu_targets}" \
        "$@" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
    run_configure_command
}

nanobind_args=()
if [[ -n "${nanobind_source}" ]]; then
    require_path "${nanobind_source}/CMakeLists.txt"
    nanobind_args=("-DFETCHCONTENT_SOURCE_DIR_NANOBIND=${nanobind_source}")
fi

prepare_staged_provider() {
    local cell="staged-provider"
    local consumer_cell="${active_cell}"
    local consumer_error="${active_error}"
    local -a consumer_assertions=("${current_assertions[@]}")

    # Preserve an explicit consumer record if this prerequisite fails. The
    # staged provider is a real configure/build/install topology, so it gets
    # its own snapshot rather than being mislabeled as the consumer's tree.
    active_error="staged-provider prerequisite was not completed; consumer was not configured"
    snapshot_cell "${consumer_cell}" skipped 0
    active_cell="${cell}"
    active_cell_recorded=0
    active_error=""
    current_assertions=()
    last_configure_command=()
    last_configure_source_dir=""
    last_configure_build_dir=""
    last_configure_stage_dir=""
    last_configure_log_path=""
    configure_hipblaslt "${cell}" \
        -DHIPBLASLT_ENABLE_HOST=OFF \
        -DHIPBLASLT_ENABLE_DEVICE=OFF \
        -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
        -DHIPBLASLT_ENABLE_EXTOPS=OFF \
        -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF \
        -DTENSILELITE_ENABLE_HOST=ON \
        -DTENSILELITE_ENABLE_CLIENT=OFF \
        -DTENSILELITE_BUILD_TESTING=OFF \
        -DHIPBLASLT_BUNDLE_PYTHON_DEPS=ON \
        "${nanobind_args[@]}"
    "${cmake_bin}" --build "${cell_build}" --target tensilelite-host _rocisa 2>&1 \
        | tee -a "${last_configure_log_path}"
    "${cmake_bin}" --install "${cell_build}" 2>&1 | tee -a "${last_configure_log_path}"
    stage_prefix="${cell_stage}"
    snapshot_cell "${cell}" passed 0
    active_cell_recorded=1

    active_cell="${consumer_cell}"
    active_cell_recorded=0
    active_error="${consumer_error}"
    current_assertions=("${consumer_assertions[@]}")
    last_configure_command=()
    last_configure_source_dir=""
    last_configure_build_dir=""
    last_configure_stage_dir=""
    last_configure_log_path=""
}

cell_hipblaslt_shared() {
    configure_hipblaslt hipblaslt-shared \
        -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
        -DHIPBLASLT_BUNDLE_PYTHON_DEPS=ON \
        "${nanobind_args[@]}"
    assert_contains "${cell_build}/CMakeCache.txt" "HIPBLASLT_BUILD_SHARED_LIBS:BOOL=ON"
    assert_contains "${cell_build}/CMakeCache.txt" "TENSILELITE_BUILD_SHARED_LIBS:BOOL=ON"
}

cell_hipblaslt_library_only() {
    configure_hipblaslt hipblaslt-library-only \
        -DHIPBLASLT_ENABLE_DEVICE=OFF \
        -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
        -DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF
    assert_not_contains "${cell_build}/cmake_install.cmake" "hipblaslt_codegen.cmake"
}

cell_hipblaslt_static() {
    configure_hipblaslt hipblaslt-static \
        -DHIPBLASLT_ENABLE_DEVICE=OFF \
        -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
        -DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF \
        -DHIPBLASLT_BUILD_SHARED_LIBS=OFF \
        -DTENSILELITE_BUILD_SHARED_LIBS=OFF \
        -DORIGAMI_BUILD_SHARED_LIBS=OFF
    assert_contains "${cell_build}/CMakeCache.txt" "HIPBLASLT_BUILD_SHARED_LIBS:BOOL=OFF"
    assert_contains "${cell_build}/CMakeCache.txt" "TENSILELITE_BUILD_SHARED_LIBS:BOOL=OFF"
    assert_contains "${cell_build}/CMakeCache.txt" "ORIGAMI_BUILD_SHARED_LIBS:BOOL=OFF"
}

cell_hipblaslt_mixed() {
    configure_hipblaslt hipblaslt-mixed \
        -DHIPBLASLT_ENABLE_DEVICE=OFF \
        -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
        -DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF \
        -DTENSILELITE_BUILD_SHARED_LIBS=OFF
    assert_contains "${cell_build}/hipblaslt-config.cmake" "find_dependency(ZLIB)"
}

cell_hipblaslt_tl_yaml() {
    configure_hipblaslt hipblaslt-tl-yaml \
        -DHIPBLASLT_ENABLE_HOST=OFF \
        -DHIPBLASLT_ENABLE_DEVICE=OFF \
        -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
        -DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF \
        -DTENSILELITE_BUILD_SHARED_LIBS=OFF \
        -DHIPBLASLT_ENABLE_YAML=ON
    assert_contains "${cell_build}/hipblaslt-config.cmake" "find_dependency(LLVM)"
    assert_not_contains "${cell_build}/hipblaslt-config.cmake" "find_dependency(msgpack"
}

cell_hipblaslt_device_only() {
    configure_hipblaslt hipblaslt-device-only \
        -DHIPBLASLT_ENABLE_HOST=OFF \
        -DTENSILELITE_ENABLE_HOST=OFF \
        -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
        -DHIPBLASLT_BUNDLE_PYTHON_DEPS=ON \
        "${nanobind_args[@]}"
    assert_contains "${cell_build}/build.ninja" "tensilelite-device-libraries"
}

cell_rocisa_shared() {
    configure_rocisa rocisa-shared \
        -Dnanobind_DIR="${nanobind_source}/cmake"
    assert_contains "${cell_build}/CMakeCache.txt" "BUILD_SHARED_LIBS:BOOL=ON"
}

cell_rocisa_static() {
    configure_rocisa rocisa-static \
        -DBUILD_SHARED_LIBS=OFF \
        -DROCISA_BUILD_HELLOWORLD_STATIC_PLUGIN=ON \
        -Dnanobind_DIR="${nanobind_source}/cmake"
    assert_contains "${cell_build}/CMakeCache.txt" "BUILD_SHARED_LIBS:BOOL=OFF"
}

cell_hipsparselt_source() {
    configure_hipsparselt hipsparselt-source \
        -DHIPSPARSELT_ENABLE_THEROCK=OFF \
        -DHIPSPARSELT_ENABLE_DEVICE=ON \
        "${nanobind_args[@]}"
    assert_contains "${cell_build}/build.ninja" "--use-bundled-known-bugs"
    assert_not_contains "${cell_build}/hipsparselt-config.cmake" "find_dependency(hipblaslt)"
    if [[ "${build_device}" -eq 1 ]]; then
        "${cmake_bin}" --build "${cell_build}" --target tensilelite-device-libraries
    fi
}

cell_hipsparselt_static() {
    configure_hipsparselt hipsparselt-static \
        -DHIPSPARSELT_ENABLE_THEROCK=OFF \
        -DHIPSPARSELT_ENABLE_DEVICE=OFF \
        -DHIPSPARSELT_BUILD_SHARED_LIBS=OFF \
        "${nanobind_args[@]}"
    assert_contains "${cell_build}/hipsparselt-config.cmake" "find_dependency(hipblaslt)"
    assert_contains "${cell_build}/CPackConfig.cmake" "hipblaslt-static-dev"
}

cell_hipsparselt_staged() {
    if [[ -z "${stage_prefix}" ]]; then
        if [[ "${prepare_stage}" -eq 1 ]]; then
            prepare_staged_provider
        else
            die "hipsparselt-staged requires --stage-prefix or --prepare-stage"
        fi
    fi
    require_path "${stage_prefix}"
    local matrix_cmake_prefix="${stage_prefix};${rocm_prefix}"
    configure_hipsparselt hipsparselt-staged \
        -DHIPSPARSELT_ENABLE_THEROCK=ON \
        -DHIPSPARSELT_ENABLE_DEVICE=ON \
        -DHIPBLASLT_TENSILELITE_PATH="${hipblaslt_root}/tensilelite"
    assert_contains "${cell_build}/CMakeCache.txt" "hipblaslt_DIR:PATH="
    assert_contains "${cell_build}/build.ninja" "--use-bundled-known-bugs"
    if [[ "${build_device}" -eq 1 ]]; then
        "${cmake_bin}" --build "${cell_build}" --target tensilelite-device-libraries
    fi
}

source_for_cell() {
    case "$1" in
        hipblaslt-*|staged-provider) echo "${hipblaslt_root}" ;;
        rocisa-*) echo "${rocisa_root}" ;;
        hipsparselt-*) echo "${hipsparselt_root}" ;;
        *) echo "${repo_root}" ;;
    esac
}

snapshot_cell() {
    local cell="$1"
    local status="$2"
    local exit_code="$3"
    local source_dir="${last_configure_source_dir:-$(source_for_cell "${cell}")}"
    local build_dir="${last_configure_build_dir:-${results_dir}/${cell}/build}"
    local stage_dir="${last_configure_stage_dir:-${results_dir}/${cell}/stage}"
    local -a snapshot_args=(
        --output "${matrix_result}"
        --runner "${script_dir}/run_cmake_configure_matrix.sh"
        --cell "${cell}"
        --status "${status}"
        --exit-code "${exit_code}"
        --source-dir "${source_dir}"
        --build-dir "${build_dir}"
        --stage-dir "${stage_dir}"
        --context "cmake_bin=${cmake_bin}"
        --context "generator=${generator}"
        --context "rocm_prefix=${rocm_prefix}"
        --context "gpu_targets=${gpu_targets}"
        --context "nanobind_source=${nanobind_source}"
        --context "stage_prefix=${stage_prefix}"
        --context "prepare_stage=${prepare_stage}"
        --context "build_device=${build_device}"
        --context "keep_results=${keep_results}"
    )
    local assertion
    for assertion in "${current_assertions[@]}"; do
        snapshot_args+=(--assertion "${assertion}")
    done
    if [[ -n "${MATRIX_IMAGE:-}" ]]; then
        snapshot_args+=(--context "image=${MATRIX_IMAGE}")
    fi
    if [[ -n "${last_configure_log_path}" ]]; then
        snapshot_args+=(--log-file "${last_configure_log_path}")
    fi
    if [[ -n "${active_error}" ]]; then
        snapshot_args+=(--error "${active_error}")
    fi
    snapshot_args+=(--configure-command "${last_configure_command[@]}")
    "${python_bin}" "${snapshot_helper}" "${snapshot_args[@]}"
}

on_matrix_exit() {
    local exit_code=$?
    trap - EXIT
    if [[ -n "${active_cell}" && "${active_cell_recorded}" -eq 0 ]]; then
        if [[ -z "${active_error}" ]]; then
            active_error="matrix runner exited with status ${exit_code}; see configure_log for CMake output"
        fi
        set +e
        snapshot_cell "${active_cell}" failed "${exit_code}"
    fi
    exit "${exit_code}"
}

run_cell() {
    local cell="$1"
    echo "==> ${cell}"
    active_cell="${cell}"
    active_cell_recorded=0
    active_error=""
    current_assertions=()
    last_configure_command=()
    last_configure_source_dir=""
    last_configure_build_dir=""
    last_configure_stage_dir=""
    last_configure_log_path=""
    case "${cell}" in
        hipblaslt-shared) cell_hipblaslt_shared ;;
        hipblaslt-library-only) cell_hipblaslt_library_only ;;
        hipblaslt-static) cell_hipblaslt_static ;;
        hipblaslt-mixed) cell_hipblaslt_mixed ;;
        hipblaslt-tl-yaml) cell_hipblaslt_tl_yaml ;;
        hipblaslt-device-only) cell_hipblaslt_device_only ;;
        rocisa-shared) cell_rocisa_shared ;;
        rocisa-static) cell_rocisa_static ;;
        hipsparselt-source) cell_hipsparselt_source ;;
        hipsparselt-static) cell_hipsparselt_static ;;
        hipsparselt-staged) cell_hipsparselt_staged ;;
        *) die "unknown cell: ${cell}" ;;
    esac
    snapshot_cell "${cell}" passed 0
    active_cell_recorded=1
    active_cell=""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cell) requested_cells+=("${2:?missing cell name}"); shift 2 ;;
        --prepare-stage) prepare_stage=1; shift ;;
        --all) run_all=1; shift ;;
        --with-yaml) with_yaml=1; shift ;;
        --stage-prefix) stage_prefix="${2:?missing stage prefix}"; shift 2 ;;
        --nanobind-source) nanobind_source="${2:?missing nanobind path}"; nanobind_args=("-DFETCHCONTENT_SOURCE_DIR_NANOBIND=${nanobind_source}"); shift 2 ;;
        --results-dir) results_dir="${2:?missing results path}"; shift 2 ;;
        --gpu-targets) gpu_targets="${2:?missing GPU targets}"; shift 2 ;;
        --rocm-prefix) rocm_prefix="${2:?missing ROCm prefix}"; shift 2 ;;
        --build-device) build_device=1; shift ;;
        --keep-results) keep_results=1; shift ;;
        --list) list_cells; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

require_command "${cmake_bin}"
require_command "${python_bin}"
require_path "${snapshot_helper}"
require_path "${rocm_prefix}/bin/amdclang++"
mkdir -p "${results_dir}"
matrix_result="${results_dir}/cmake-configure-matrix.json"
rm -f "${matrix_result}"
trap on_matrix_exit EXIT

if [[ ${#requested_cells[@]} -eq 0 && "${run_all}" -eq 0 ]]; then
    requested_cells=(
        hipblaslt-shared
        hipblaslt-library-only
        hipblaslt-static
        hipblaslt-mixed
        hipblaslt-device-only
        rocisa-shared
        rocisa-static
        hipsparselt-source
        hipsparselt-static
    )
    if [[ "${with_yaml}" -eq 1 ]]; then
        requested_cells+=(hipblaslt-tl-yaml)
    fi
    if [[ "${prepare_stage}" -eq 1 || -n "${stage_prefix}" ]]; then
        requested_cells+=(hipsparselt-staged)
    fi
elif [[ "${run_all}" -eq 1 ]]; then
    requested_cells=(
        hipblaslt-shared
        hipblaslt-library-only
        hipblaslt-static
        hipblaslt-mixed
        hipblaslt-tl-yaml
        hipblaslt-device-only
        rocisa-shared
        rocisa-static
        hipsparselt-source
        hipsparselt-static
        hipsparselt-staged
    )
fi

for cell in "${requested_cells[@]}"; do
    case "${cell}" in
        hipblaslt-shared|hipblaslt-device-only|hipsparselt-source|hipsparselt-static)
            [[ -n "${nanobind_source}" ]] || die "${cell} requires --nanobind-source for offline FetchContent"
            ;;
        rocisa-shared|rocisa-static)
            [[ -n "${nanobind_source}" ]] || die "${cell} requires --nanobind-source for nanobind_DIR"
            ;;
    esac
    run_cell "${cell}"
done

echo "Configure matrix passed. Results: ${results_dir}"
echo "Matrix JSON: ${matrix_result}"
