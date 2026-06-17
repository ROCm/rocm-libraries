#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Smart Test Script (test execution only)
#
# Phase 2 of the decoupled smart-build pipeline. Consumes the build/ directory
# and selection artifacts produced by smart_build.sh and runs the tests:
#   - selective: runs only the affected tests (ctest -R over regex_chunks)
#   - full:      runs the whole ctest suite (equivalent to the `check` target run)
#   - none:      runs nothing (no CK code affected)
# The build mode is read from build_mode.env, so smart_build.sh MUST have run
# first (same workspace, or the build/ dir carried over for a cross-node split).
#
# Exit codes:
#   0 = Success (tests passed, or nothing to test)
#   1 = Test failure, or build phase did not run (missing build_mode.env)
#
# Environment variables:
#   BUILD_DIR - Build directory (defaults to current directory)
#   CTEST_PARALLEL - ctest parallel level (default: 4)

set -e

BUILD_DIR="${BUILD_DIR:-$(pwd)}"
CTEST_PARALLEL="${CTEST_PARALLEL:-4}"

# Tee all output to a per-phase log so the test stage can archive it
# independently of the build stage. Process substitution (not a pipe) keeps the
# exit status of the commands below intact.
LOG_FILE="${BUILD_DIR}/smart_test.log"
exec > >(tee "${LOG_FILE}") 2>&1

cd "${BUILD_DIR}"

echo "========================================="
echo "Smart Test (test execution)"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "CTEST_PARALLEL: ${CTEST_PARALLEL}"
echo "-----------------------------------------"

# The build phase records the mode in build_mode.env; its presence confirms
# smart_build.sh ran in this workspace. Require it so a skipped build surfaces
# loudly here instead of silently testing nothing.
if [ ! -f build_mode.env ]; then
    echo "Error: build_mode.env missing in ${BUILD_DIR}; run smart_build.sh first"
    echo "(smart_test.sh consumes the build/ dir + selection artifacts it produces.)"
    exit 1
fi

# build_mode.env contains SMART_BUILD_MODE=selective|full|none.
# shellcheck disable=SC1091
source build_mode.env
MODE="${SMART_BUILD_MODE:-unknown}"
echo "SMART_BUILD_MODE: ${MODE}"

case "${MODE}" in
    none)
        echo "[OK] No tests affected by changes - nothing to test"
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
            echo "Error: tests_to_run.json missing (selective mode expects it from smart_build.sh)"
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
