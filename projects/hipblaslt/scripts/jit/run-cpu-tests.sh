#!/usr/bin/env bash
# Run the JIT tests that need no GPU. Hide every GPU before invoking this script
# when the machine has one (HIP_VISIBLE_DEVICES=-1 and ROCR_VISIBLE_DEVICES=-1).
# Locally: bash projects/hipblaslt/scripts/jit/run-cpu-tests.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${JIT_LOG_DIR}"
ctest --test-dir "${JIT_BUILD_DIR}/clients/tests/jit" -L jit-cpu --no-tests=error \
    --output-on-failure 2>&1 | tee "${JIT_LOG_DIR}/ctest-cpu.log"
