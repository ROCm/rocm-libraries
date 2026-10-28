#!/usr/bin/env bash
# Install header-only hipblas-common from this checkout into ${JIT_DEPS_DIR}.
# Locally: bash projects/hipblaslt/scripts/jit/stage-hipblas-common.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cmake -S "${ROOT}/projects/hipblas-common" -B "${ROOT}/build/hipblas-common" -G Ninja \
    -DCMAKE_CXX_COMPILER="${ROCM_PATH}/bin/amdclang++" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}" -DCMAKE_INSTALL_PREFIX="${JIT_DEPS_DIR}"
cmake --install "${ROOT}/build/hipblas-common"
