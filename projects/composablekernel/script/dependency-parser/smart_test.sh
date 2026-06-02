#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Smart Test Script (test execution only)
#
# Phase 2 of the decoupled smart-build pipeline. Consumes the build/ directory
# and selection artifacts produced by smart_build.sh and runs the tests:
#   - selective: runs only the affected tests (ctest -R over regex_chunks)
#   - full:      runs the whole ctest suite
#   - none:      runs nothing (no CK code affected)
# The build mode is read from build_mode.env, so smart_build.sh MUST have run
# first (same workspace, or the build/ dir carried over for a cross-node split).
#
# Exit codes:
#   0 = Success (tests passed, or nothing to test)
#   1 = Test failure, or build phase did not run (missing build_mode.env)
#
# Environment: see lib_env.sh for the shared variables and defaults (this script
# uses BUILD_DIR and CTEST_PARALLEL).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_env.sh
source "${SCRIPT_DIR}/lib_env.sh"
init_smart_build_env
LOG_FILE="${BUILD_DIR}/smart_test.log"

# shellcheck source=lib_logging.sh
source "${SCRIPT_DIR}/lib_logging.sh"
start_tee_log "${LOG_FILE}"

cd "${BUILD_DIR}"

echo "========================================="
echo "Smart Test (test execution)"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "CTEST_PARALLEL: ${CTEST_PARALLEL}"
echo "-----------------------------------------"

# The build phase records the mode in build_mode.env. Its absence means
# smart_build.sh never ran in this workspace - fail loudly rather than silently
# testing nothing.
if [ ! -f build_mode.env ]; then
    echo "Error: build_mode.env not found in ${BUILD_DIR}"
    echo "smart_build.sh must run before smart_test.sh (same workspace / carried-over build dir)."
    exit 1
fi

# build_mode.env contains SMART_BUILD_MODE=selective|full|none (sourcing handles
# an optional 'export ' prefix too).
# shellcheck disable=SC1091
source build_mode.env
MODE="${SMART_BUILD_MODE:-unknown}"
echo "SMART_BUILD_MODE: ${MODE}"

case "${MODE}" in
    none)
        echo "[OK] No tests affected by changes - skipping test execution"
        exit 0
        ;;
    full)
        echo ""
        echo "Full mode - running the complete ctest suite..."
        CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL} ctest --output-on-failure
        echo ""
        echo "[OK] Smart test complete (full mode)"
        exit 0
        ;;
    selective)
        if [ ! -f tests_to_run.json ]; then
            echo "Error: tests_to_run.json not found (selective mode expects it from smart_build.sh)"
            exit 1
        fi
        echo ""
        echo "Selective mode - running affected tests..."
        NUM_CHUNKS=$(jq -r '.regex_chunks | length' tests_to_run.json)
        echo "Running ${NUM_CHUNKS} test chunk(s)"

        if [ "$NUM_CHUNKS" -eq 1 ]; then
            TEST_REGEX=$(jq -r '.regex_chunks[0]' tests_to_run.json)
            CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL} ctest --output-on-failure -R "${TEST_REGEX}"
        else
            for ((i=0; i<NUM_CHUNKS; i++)); do
                TEST_REGEX=$(jq -r ".regex_chunks[$i]" tests_to_run.json)
                echo "Running test chunk $((i+1))/${NUM_CHUNKS}"
                CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL} ctest --output-on-failure -R "${TEST_REGEX}"
            done
        fi
        echo ""
        echo "[OK] Smart test complete (selective mode)"
        exit 0
        ;;
    *)
        echo "Error: unrecognized SMART_BUILD_MODE='${MODE}' in build_mode.env"
        exit 1
        ;;
esac
