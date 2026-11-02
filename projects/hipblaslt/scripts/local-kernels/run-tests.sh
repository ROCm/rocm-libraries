#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Run the local kernel tests in hipblaslt-test. Fails if the binary has none,
# which means HIPBLASLT_ENABLE_LOCAL_KERNELS did not reach the build.
# Locally: bash projects/hipblaslt/scripts/local-kernels/run-tests.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

test_binary="${LOCAL_KERNELS_BUILD_DIR}/clients/hipblaslt-test"
listed="$("${test_binary}" --gtest_list_tests --gtest_filter='LocalKernel*')"
if ! grep -q '^  ' <<< "${listed}"; then
    echo "hipblaslt-test has no LocalKernel* tests; HIPBLASLT_ENABLE_LOCAL_KERNELS is off" >&2
    exit 1
fi
"${test_binary}" --gtest_filter='LocalKernel*' 2>&1 | tee "${LOCAL_KERNELS_LOG_DIR}/hipblaslt-test.log"
