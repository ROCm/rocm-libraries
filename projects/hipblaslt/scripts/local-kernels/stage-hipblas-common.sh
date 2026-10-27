#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Install header-only hipblas-common from this checkout into
# ${LOCAL_KERNELS_DEPS_DIR}.
# Locally: bash projects/hipblaslt/scripts/local-kernels/stage-hipblas-common.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cmake -S "${ROOT}/projects/hipblas-common" -B "${GITHUB_WORKSPACE}/build/hipblas-common" -G Ninja \
    -DCMAKE_CXX_COMPILER="${ROCM_PATH}/bin/amdclang++" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}" -DCMAKE_INSTALL_PREFIX="${LOCAL_KERNELS_DEPS_DIR}"
cmake --install "${GITHUB_WORKSPACE}/build/hipblas-common"
