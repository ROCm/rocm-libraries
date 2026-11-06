#!/usr/bin/env bash
# Run the JIT tests that need a GPU. Device 0 must be gfx90a, gfx942 or gfx950.
# Locally: bash projects/hipblaslt/scripts/jit/run-gpu-tests.sh

set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${JIT_LOG_DIR}"
ctest --test-dir "${JIT_BUILD_DIR}/clients/tests/jit" -L jit-gpu --no-tests=error \
    --output-on-failure 2>&1 | tee "${JIT_LOG_DIR}/ctest-gpu.log"
