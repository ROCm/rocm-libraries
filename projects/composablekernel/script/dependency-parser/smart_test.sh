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
#   JUNIT_OUTPUT - JUnit XML report path written by ctest (default: junit.xml).
#                  CI sets this to a globally-unique name (job/run/arch/stage).

set -e

BUILD_DIR="${BUILD_DIR:-$(pwd)}"
CTEST_PARALLEL="${CTEST_PARALLEL:-4}"
JUNIT_OUTPUT="${JUNIT_OUTPUT:-junit.xml}"

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
echo "JUNIT_OUTPUT: ${JUNIT_OUTPUT}"
echo "-----------------------------------------"

# The build phase records the mode in build_mode.env; its presence confirms
# smart_build.sh ran in this workspace. Require it so a skipped build surfaces
# loudly here instead of silently testing nothing.
if [ ! -f build_mode.env ]; then
    echo "Error: build_mode.env missing in ${BUILD_DIR}; run smart_build.sh first"
    echo "(smart_test.sh consumes the build/ dir + selection artifacts it produces.)"
    exit 1
fi

# build_mode.env contains SMART_BUILD_MODE=selective|full|none. Parse the value
# directly instead of sourcing the file: sourcing would execute arbitrary shell
# if the file were ever corrupted or tampered with. Take the last assignment so a
# repeated key resolves deterministically, and require it to be present.
MODE=$(sed -n 's/^SMART_BUILD_MODE=//p' build_mode.env | tail -n 1)
if [ -z "${MODE}" ]; then
    echo "Error: SMART_BUILD_MODE not set in build_mode.env"
    exit 1
fi
echo "SMART_BUILD_MODE: ${MODE}"

case "${MODE}" in
    none)
        echo "[OK] No tests affected by changes - nothing to test"
        exit 0
        ;;
    full)
        echo ""
        echo "Full mode - running the complete ctest suite..."
        CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL} ctest --output-on-failure --output-junit "${JUNIT_OUTPUT}"
        echo ""
        echo "[OK] Smart test complete (full mode)"
        exit 0
        ;;
    selective)
        # tests_to_run.txt is the selection list (one ctest test name per line)
        # emitted by the parser. `ctest --tests-from-file` runs all of them in a
        # single invocation with exact-name matching, so there is no `-R` regex
        # length limit and no chunking. (The JSON still carries regex_chunks for
        # other consumers; the test phase no longer reads them.)
        if [ ! -f tests_to_run.txt ]; then
            echo "Error: tests_to_run.txt missing (selective mode expects it from smart_build.sh)"
            exit 1
        fi
        # Require a non-empty list. An empty file would run zero tests yet exit 0,
        # silently skipping the test phase in the mode meant to run tests - the
        # 'none' mode is the only sanctioned no-op. Fail loudly instead.
        if [ ! -s tests_to_run.txt ] || ! grep -q '[^[:space:]]' tests_to_run.txt; then
            echo "Error: tests_to_run.txt is empty (selective mode expects >=1 test)"
            exit 1
        fi
        NUM_TESTS=$(grep -c '[^[:space:]]' tests_to_run.txt)
        echo ""
        echo "Selective mode - running ${NUM_TESTS} affected test(s)..."
        CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL} ctest --output-on-failure --tests-from-file tests_to_run.txt --output-junit "${JUNIT_OUTPUT}"
        echo ""
        echo "[OK] Smart test complete (selective mode)"
        exit 0
        ;;
    *)
        echo "Error: unrecognized SMART_BUILD_MODE='${MODE}' in build_mode.env"
        exit 1
        ;;
esac
