#!/usr/bin/env bash
# heuristics/sweep/build.sh
#
# Builds conv_candidate_sweep inside a ROCm build container.
#
# Strategy: wrap the rocm-libraries superbuild in a thin top-level
# CMakeLists that (a) configures hipdnn + ck-dsl-provider via the real
# superbuild and (b) adds the sweep directory after provider targets are defined.
# This gives the sweep access to ck_dsl_provider_private (build-tree STATIC
# target) without an install step.
#
# SWEEP_SRC and ROCM_LIBS are derived from this script's location in the repo.
#
# Overridable via environment:
#   ARCH        GPU architecture          (default: gfx942)
#   BUILD_DIR   CMake build output dir    (default: $HOME/ckdsl_sweep_build)
#   VENV_DIR    Python venv for deps      (default: $HOME/ckdsl_sweep_venv)

set -euo pipefail

SWEEP_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_LIBS="$(cd "${SWEEP_SRC}/../../../.." && pwd)"

: "${ARCH:=gfx942}"
: "${BUILD_DIR:=${HOME}/ckdsl_sweep_build}"
: "${VENV_DIR:=${HOME}/ckdsl_sweep_venv}"

echo "=== conv_candidate_sweep build ===" >&2
echo "  SWEEP_SRC : ${SWEEP_SRC}" >&2
echo "  ROCM_LIBS : ${ROCM_LIBS}" >&2
echo "  ARCH      : ${ARCH}" >&2
echo "  BUILD_DIR : ${BUILD_DIR}" >&2
echo "  VENV_DIR  : ${VENV_DIR}" >&2

# ── Python deps (lightgbm + pybind11) ────────────────────────────────────────
if ! "${VENV_DIR}/bin/python3" -c "import lightgbm; import pybind11" 2>/dev/null; then
    echo "=== Creating/updating venv in ${VENV_DIR} ===" >&2
    python3 -m venv --system-site-packages "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --quiet lightgbm pybind11
fi
export PATH="${VENV_DIR}/bin:${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"
PYBIND11_CMAKE_DIR="$("${VENV_DIR}/bin/python3" -m pybind11 --cmakedir)"
LGBM_SO="$("${VENV_DIR}/bin/python3" -c \
    "import lightgbm, os; print(os.path.join(os.path.dirname(lightgbm.__file__), 'lib', 'lib_lightgbm.so'))")"

# ── Generate a thin wrapper CMakeLists that drives the real superbuild ────────
SUPER_SRC="${BUILD_DIR}/super_src"
mkdir -p "${SUPER_SRC}" "${BUILD_DIR}"

cat > "${SUPER_SRC}/CMakeLists.txt" << 'SUPER_EOF'
cmake_minimum_required(VERSION 3.21)
project(ConvCandidateSweepSuper LANGUAGES CXX HIP)

set(ROCM_LIBS_ENABLE_COMPONENTS "hipdnn;ck-dsl-provider")
set(BUILD_TESTING OFF)

# Pull in the real superbuild (defines hipdnn_data_sdk, ck_dsl_provider_private, etc.)
add_subdirectory("$ENV{ROCM_LIBS}" rocm_libs_build)

# After all provider targets exist, add the candidate sweep.
add_subdirectory("$ENV{SWEEP_SRC}" oracle_sweep)
SUPER_EOF

export ROCM_LIBS SWEEP_SRC

# ── Decompress the in-tree model so the CMake resolver finds it ───────────────
MODEL_GZ="${ROCM_LIBS}/dnn-providers/ck-dsl-provider/heuristics/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm.gz"
MODEL_LGB="${MODEL_GZ%.gz}"
if [[ ! -f "${MODEL_LGB}" ]]; then
    echo "=== Decompressing model ===" >&2
    gunzip -k "${MODEL_GZ}"
fi

echo "=== Configuring ===" >&2

cmake -S "${SUPER_SRC}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_HIP_ARCHITECTURES="${ARCH}" \
    -DGPU_TARGETS="${ARCH}" \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;ck-dsl-provider" \
    -DBUILD_TESTING=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_ClangTidy=ON \
    -DENABLE_CLANG_TIDY=OFF \
    -DLIGHTGBM_LIB="${LGBM_SO}" \
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
    -DPython3_EXECUTABLE="${VENV_DIR}/bin/python3" \
    -DCK_DSL_PROVIDER_SOURCE_DIR="${ROCM_LIBS}/dnn-providers/ck-dsl-provider/src" \
    -DCK_DSL_PROVIDER_BINARY_DIR="${BUILD_DIR}/rocm_libs_build/dnn-providers/ck-dsl-provider"

echo "=== Building conv_candidate_sweep ===" >&2
cmake --build "${BUILD_DIR}" --target conv_candidate_sweep -j"$(nproc)"

echo "" >&2
echo "Build complete: ${BUILD_DIR}/oracle_sweep/conv_candidate_sweep" >&2
