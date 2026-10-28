#!/usr/bin/env bash
# Configure and build the host hipBLASLt library with JIT and its tests.
# Locally, with a ROCm prefix and the virtualenv from install-deps.sh:
#   export ROCM_PATH=... JIT_ARCHITECTURE=gfx950
#   bash projects/hipblaslt/scripts/jit/configure-build.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

jobs="${KUBE_CPU_REQUEST:-8}"
jobs="${jobs%.*}"
(( jobs > 16 )) && jobs=16
(( jobs < 1 )) && jobs=1

mkdir -p "${JIT_LOG_DIR}"
# Host library only.
cmake -S "${ROOT}/projects/hipblaslt" -B "${JIT_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${ROCM_PATH}/bin/amdclang" \
    -DCMAKE_CXX_COMPILER="${ROCM_PATH}/bin/amdclang++" \
    -DCMAKE_PREFIX_PATH="${JIT_DEPS_DIR};${ROCM_PATH}" \
    -DPython_EXECUTABLE="${VENV_DIR}/bin/python" \
    -DPython3_EXECUTABLE="${VENV_DIR}/bin/python" \
    -DGPU_TARGETS="${JIT_ARCHITECTURE}" \
    -DHIPBLASLT_ENABLE_JIT=ON \
    -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_ENABLE_DEVICE=OFF \
    -DHIPBLASLT_ENABLE_CLIENT=OFF -DHIPBLASLT_BUILD_TESTING=ON \
    -DHIPBLASLT_ENABLE_ROCROLLER=OFF -DHIPBLASLT_ENABLE_MARKER=OFF \
    -DHIPBLASLT_ENABLE_EXTOPS=OFF -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF \
    -DTENSILELITE_BUILD_TESTING=OFF -DTENSILELITE_ENABLE_CLIENT=OFF \
    2>&1 | tee "${JIT_LOG_DIR}/configure.log"
cmake --build "${JIT_BUILD_DIR}" --parallel "${jobs}" 2>&1 | tee "${JIT_LOG_DIR}/build.log"
