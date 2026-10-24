#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Fetch the compiler, runtime and development pieces of a ROCm SDK, without a
# prebuilt hipBLASLt. Requires TheRock checked out at ${ROOT}/TheRock.
# Locally, export SDK_RUN_ID, AMDGPU_FAMILY and LOCAL_KERNELS_ARCHITECTURE first.
#   bash projects/hipblaslt/scripts/local-kernels/fetch-sdk.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${SDK_RUN_ID:?Set SDK_RUN_ID to the baseline TheRock workflow run}"
: "${AMDGPU_FAMILY:?Set AMDGPU_FAMILY to the artifact family, such as gfx94X-dcgpu}"

# A complete release tarball or blas artifact contains a prebuilt hipBLASLt,
# which the build could find instead of its own. Fetch only these compiler,
# runtime and development components.
"${VENV_DIR}/bin/python" "${ROOT}/TheRock/build_tools/fetch_artifacts.py" \
    --run-id "${SDK_RUN_ID}" --run-github-repo ROCm/TheRock \
    --artifact-group "${AMDGPU_FAMILY}" --amdgpu-targets "${LOCAL_KERNELS_ARCHITECTURE}" \
    --output-dir "${ROCM_PATH}" --flatten \
    '^(amd-llvm|base|sysdeps|core-(runtime|hip|kpack|ocl|amdsmi)|rocjitsu-hotswap|rocprofiler-sdk|host-suite-sparse)_(run|lib|dev)_' \
    2>&1 | tee "${LOCAL_KERNELS_LOG_DIR}/sdk-dependencies.log"
if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "${ROCM_PATH}/bin" >> "${GITHUB_PATH}"
    echo "${ROCM_PATH}/llvm/bin" >> "${GITHUB_PATH}"
fi
echo "ROCm prefix: ${ROCM_PATH}"
