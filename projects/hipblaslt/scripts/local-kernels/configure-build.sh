#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configure the host hipBLASLt library with local kernel registration and build
# hipblaslt-test. No device kernel library is built: the local kernel tests run
# over a stub shipped library.
# Locally, with a ROCm prefix and the virtualenv from install-deps.sh:
#   export ROCM_PATH=... LOCAL_KERNELS_ARCHITECTURE=gfx942
#   bash projects/hipblaslt/scripts/local-kernels/configure-build.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

jobs="${KUBE_CPU_REQUEST:-8}"
jobs="${jobs%.*}"
(( jobs > 16 )) && jobs=16
(( jobs < 1 )) && jobs=1

cmake -S "${ROOT}/projects/hipblaslt" -B "${LOCAL_KERNELS_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${ROCM_PATH}/bin/amdclang" \
    -DCMAKE_CXX_COMPILER="${ROCM_PATH}/bin/amdclang++" \
    -DCMAKE_PREFIX_PATH="${LOCAL_KERNELS_DEPS_DIR};${ROCM_PATH}" \
    -DPython_EXECUTABLE="${VENV_DIR}/bin/python" \
    -DPython3_EXECUTABLE="${VENV_DIR}/bin/python" \
    -DGPU_TARGETS="${LOCAL_KERNELS_ARCHITECTURE}" \
    -DHIPBLASLT_ENABLE_LOCAL_KERNELS=ON \
    -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_ENABLE_DEVICE=OFF \
    -DHIPBLASLT_ENABLE_EXTOPS=OFF -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF \
    -DHIPBLASLT_ENABLE_CLIENT=ON -DHIPBLASLT_BUILD_TESTING=ON -DHIPBLASLT_ENABLE_SAMPLES=OFF \
    -DHIPBLASLT_ENABLE_BLIS=OFF -DHIPBLASLT_ENABLE_AMD_SMI=OFF \
    -DHIPBLASLT_ENABLE_ROCROLLER=OFF -DHIPBLASLT_ENABLE_MARKER=OFF \
    -DTENSILELITE_BUILD_TESTING=OFF -DTENSILELITE_ENABLE_CLIENT=OFF \
    2>&1 | tee "${LOCAL_KERNELS_LOG_DIR}/configure.log"
cmake --build "${LOCAL_KERNELS_BUILD_DIR}" --target hipblaslt-test --parallel "${jobs}" \
    2>&1 | tee "${LOCAL_KERNELS_LOG_DIR}/build.log"
