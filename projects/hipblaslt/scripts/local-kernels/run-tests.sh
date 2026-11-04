#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Run the local kernel tests in hipblaslt-test, and the code object loader
# tests that build only with them. Fails if the binary has no local kernel
# tests, which means HIPBLASLT_ENABLE_LOCAL_KERNELS did not reach the build.
# With LOCAL_KERNELS_GPUS above 1, that many GPUs are made visible and any
# skipped test fails the run, since only the tests that need two devices skip.
# Locally: bash projects/hipblaslt/scripts/local-kernels/run-tests.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

gpus="${LOCAL_KERNELS_GPUS:-1}"
if (( gpus > 1 )); then
    HIP_VISIBLE_DEVICES="$(seq -s, 0 $(( gpus - 1 )))"
    export HIP_VISIBLE_DEVICES
fi

test_binary="${LOCAL_KERNELS_BUILD_DIR}/clients/hipblaslt-test"
log="${LOCAL_KERNELS_LOG_DIR}/hipblaslt-test.log"
listed="$("${test_binary}" --gtest_list_tests --gtest_filter='LocalKernel*')"
if ! grep -q '^  ' <<< "${listed}"; then
    echo "hipblaslt-test has no LocalKernel* tests; HIPBLASLT_ENABLE_LOCAL_KERNELS is off" >&2
    exit 1
fi
"${test_binary}" --gtest_filter='LocalKernel*:HipSolutionAdapter*' 2>&1 | tee "${log}"
if (( gpus > 1 )) && grep -q '^\[  SKIPPED \]' "${log}"; then
    echo "tests were skipped with ${gpus} GPUs visible (HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES})" >&2
    exit 1
fi
