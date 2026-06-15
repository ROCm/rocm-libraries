#!/usr/bin/env bash
# heuristics/sweep/build.sh
#
# Builds conv_candidate_sweep inside a ROCm build container.
#
# The sweep uses the pure-C ck_dsl engine (libckc_core.a) to JIT-compile and
# time every (tile, pipeline) candidate. No Python, no pybind11, no hipdnn SDK.
#
# SWEEP_SRC and ROCM_LIBS are derived from this script's location in the repo.
#
# Overridable via environment:
#   ARCH          GPU architecture              (default: gfx942)
#   BUILD_DIR     CMake build output dir        (default: $HOME/ckdsl_sweep_build)
#   CKC_CORE_LIB  Path to libckc_core.a         (auto-detected from /opt/rocm if unset)
#   CKC_INCLUDE   Path to ckc/ headers          (default: derived from ROCM_LIBS)

set -euo pipefail

SWEEP_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_LIBS="$(cd "${SWEEP_SRC}/../../../.." && pwd)"

: "${ARCH:=gfx942}"
: "${BUILD_DIR:=${HOME}/ckdsl_sweep_build}"
: "${CKC_INCLUDE:=${ROCM_LIBS}/projects/composablekernel/python/ck_dsl_c/include}"

CK_DSL_RUNTIME_INCLUDE="${ROCM_LIBS}/dnn-providers/ck-dsl-provider/runtime/include"

echo "=== conv_candidate_sweep build ===" >&2
echo "  SWEEP_SRC             : ${SWEEP_SRC}" >&2
echo "  ROCM_LIBS             : ${ROCM_LIBS}" >&2
echo "  ARCH                  : ${ARCH}" >&2
echo "  BUILD_DIR             : ${BUILD_DIR}" >&2
echo "  CK_DSL_RUNTIME_INCLUDE: ${CK_DSL_RUNTIME_INCLUDE}" >&2
echo "  CKC_INCLUDE           : ${CKC_INCLUDE}" >&2

mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=(
    -S "${SWEEP_SRC}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_HIP_ARCHITECTURES="${ARCH}"
    -DCMAKE_CXX_COMPILER=hipcc
    -DCK_DSL_RUNTIME_INCLUDE_DIR="${CK_DSL_RUNTIME_INCLUDE}"
    -DCKC_INCLUDE_DIR="${CKC_INCLUDE}"
)

if [[ -n "${CKC_CORE_LIB:-}" ]]; then
    CMAKE_ARGS+=(-DCKC_CORE_LIB="${CKC_CORE_LIB}")
fi

echo "=== Configuring ===" >&2
cmake "${CMAKE_ARGS[@]}"

echo "=== Building conv_candidate_sweep ===" >&2
cmake --build "${BUILD_DIR}" --target conv_candidate_sweep -j"$(nproc)"

echo "" >&2
echo "Build complete: ${BUILD_DIR}/conv_candidate_sweep" >&2
