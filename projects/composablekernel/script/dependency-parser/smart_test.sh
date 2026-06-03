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

# --dry-run prints the ctest command(s) that would run, without executing them
# (GPU-free verification of the full/selective/none dispatch).
DRY_RUN="${DRY_RUN:-false}"
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done
# run_ctest <args...>: execute ctest, or just print it under DRY_RUN.
run_ctest() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY RUN would run: CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL} ctest $*"
    else
        CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL} ctest "$@"
    fi
}

# run_always_run_class: run the "always-run" tests the selector can never map.
# The reachability guardrail classifies ctest tests with no compiled bin/ target
# (python scripts, try_compile tests) as `non_compiled` in reachability_result.json.
# Since no source file maps to them, a selective/none build would otherwise skip
# them, so run them every non-full build. The set is auto-derived (no whitelist):
# a newly added python test lands here automatically on the next build.
run_always_run_class() {
    local report="reachability_result.json"
    if [ ! -f "$report" ]; then
        echo "Always-run class: ${report} not found (reachability guardrail skipped) - none to run"
        return 0
    fi
    # Anchor each name so -R matches it exactly; ctest test names are CMake
    # identifiers, so no regex escaping is needed.
    local names
    names=$(jq -r '.non_compiled[]?' "$report" | paste -sd'|' -)
    if [ -z "$names" ]; then
        echo "Always-run class: none"
        return 0
    fi
    echo "Always-run class: $(jq -r '.non_compiled | length' "$report") non-compiled test(s)"
    run_ctest --output-on-failure -R "^(${names})\$"
}

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

# The build phase records the mode in build_mode.env; its presence confirms
# smart_build.sh ran in this workspace. Require it so a skipped build surfaces
# loudly here instead of silently testing nothing.
if [ ! -f build_mode.env ]; then
    echo "Error: build_mode.env missing in ${BUILD_DIR}; run smart_build.sh first"
    echo "(smart_test.sh consumes the build/ dir + selection artifacts it produces.)"
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
        echo "No compiled tests affected by changes"
        # The always-run class (python/try_compile) maps to no source file, so a
        # change scoped purely to one yields mode=none; run them here so it isn't skipped.
        run_always_run_class
        echo "[OK] Smart test complete (none mode - always-run class only)"
        exit 0
        ;;
    full)
        echo ""
        echo "Full mode - running the complete ctest suite..."
        # Exclude the separate flag-gated suites (rocm_ck, builder): they are
        # registered with ctest but built by their own targets (check-rocm-ck /
        # check-builder), not by `ninja tests examples`, so a bare ctest reports
        # them "Not Run" and fails. They are owned by their own stages (D13).
        # Override/disable via CTEST_FULL_EXCLUDE_LABELS (empty = no exclusion).
        CTEST_FULL_EXCLUDE_LABELS="${CTEST_FULL_EXCLUDE_LABELS-ROCM_CK_|BUILDER_SMOKE}"
        EXCLUDE_ARGS=()
        if [ -n "${CTEST_FULL_EXCLUDE_LABELS}" ]; then
            EXCLUDE_ARGS=(-LE "${CTEST_FULL_EXCLUDE_LABELS}")
        fi
        run_ctest --output-on-failure "${EXCLUDE_ARGS[@]}"
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
            run_ctest --output-on-failure -R "${TEST_REGEX}"
        else
            for ((i=0; i<NUM_CHUNKS; i++)); do
                TEST_REGEX=$(jq -r ".regex_chunks[$i]" tests_to_run.json)
                echo "Running test chunk $((i+1))/${NUM_CHUNKS}"
                run_ctest --output-on-failure -R "${TEST_REGEX}"
            done
        fi
        # The selector can't map non-compiled tests (python/try_compile), so run
        # that always-run class alongside the selected chunks.
        run_always_run_class
        echo ""
        echo "[OK] Smart test complete (selective mode)"
        exit 0
        ;;
    *)
        echo "Error: unrecognized SMART_BUILD_MODE='${MODE}' in build_mode.env"
        exit 1
        ;;
esac
