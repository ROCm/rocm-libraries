#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Smart Build CI Script
#
# This script orchestrates the smart-build process:
# 1. Runs ci_safety_check.sh to determine if selective build is safe
# 2. Generates dependency map using cmake-parse
# 3. Selects affected tests
# 4. Outputs build targets to a file for Jenkins to consume
#
# Exit codes:
#   0 = Success (selective build targets generated)
#   1 = Full build required (run ninja check)
#
# Output files:
#   tests_to_run.json     - Selected tests and executables
#   build_targets.txt     - Space-separated list of ninja targets to build
#   build_mode.env        - Environment variables (SMART_BUILD_MODE=selective|full|none)
#   smart_build_ci.log    - Full run log (also printed to stdout via tee)
#   reachability_result.json - Guardrail: tests unreachable from any changed file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$(pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd ${BUILD_DIR}/.. && pwd)}"
PARALLEL="${PARALLEL:-32}"
BASE_BRANCH="${BASE_BRANCH:-develop}"
LOG_FILE="${BUILD_DIR}/smart_build_ci.log"

# Stream output to a log file (for CI artifact archiving) as well as the console.
# When invoked by smart_build_and_test.sh the parent already tees a combined log,
# so we skip our own tee to avoid double-logging and out-of-order interleaving
# (smart_build_ci.log is only produced when this script is run standalone).
# A backgrounded tee draining a FIFO (whose PID we wait on at exit) is used
# instead of `exec > >(tee)` so the log is fully flushed before the script exits:
# the bare process-substitution form does not wait for tee and can lose the tail
# (including the final verdict banner).
if [ -z "${_SMART_BUILD_NESTED:-}" ]; then
    _LOG_FIFO="$(mktemp -u)"
    mkfifo "${_LOG_FIFO}"
    tee "${LOG_FILE}" < "${_LOG_FIFO}" &
    _TEE_PID=$!
    exec > "${_LOG_FIFO}" 2>&1
    rm -f "${_LOG_FIFO}"
    trap '_rc=$?; exec 1>&- 2>&-; wait "${_TEE_PID}" 2>/dev/null || true; exit ${_rc}' EXIT
fi

echo "========================================="
echo "Smart Build CI"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "WORKSPACE_ROOT: ${WORKSPACE_ROOT}"
echo "BASE_BRANCH: ${BASE_BRANCH}"
echo "PARALLEL: ${PARALLEL}"
echo "-----------------------------------------"

# Step 1: Run CI safety check
echo "Step 1: Running CI safety check..."
cd "${BUILD_DIR}"

if ! bash "${SCRIPT_DIR}/ci_safety_check.sh"; then
    echo "CI safety check failed - full build required"
    echo "full" > build_targets.txt
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi

echo "[OK] CI safety check passed - selective build enabled"

# Step 2: Generate dependency map
echo ""
echo "Step 2: Generating dependency map..."
if [ ! -f "compile_commands.json" ]; then
    echo "Error: compile_commands.json not found in ${BUILD_DIR}"
    echo "Make sure cmake configure has been run with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi

if [ ! -f "build.ninja" ]; then
    echo "Error: build.ninja not found in ${BUILD_DIR}"
    echo "Make sure cmake configure has been run with -G Ninja"
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi

python3 "${SCRIPT_DIR}/main.py" cmake-parse \
    compile_commands.json \
    build.ninja \
    --workspace-root "${WORKSPACE_ROOT}" \
    --parallel ${PARALLEL} \
    --output enhanced_dependency_mapping.json

if [ ! -f "enhanced_dependency_mapping.json" ]; then
    echo "Error: Failed to generate enhanced_dependency_mapping.json"
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi

echo "[OK] Dependency map generated"

# Step 2b: Reachability guardrail (observability, non-fatal).
# Flags ctest tests that no file maps to - the filter can never select them, i.e.
# guaranteed false negatives (usually a dependency-extraction gap). Emits
# reachability_result.json for CI to archive; does NOT fail the build.
echo ""
echo "Step 2b: Reachability guardrail (non-fatal)..."
ctest -N > ctest_list.txt 2>/dev/null || true
# Guard: if ctest -N produced no test lines the guardrail would trivially pass
# (empty intersection), giving a false green. Skip it and warn instead.
if ! grep -q "Test #" ctest_list.txt 2>/dev/null; then
    echo "WARNING: ctest -N returned no tests (not yet configured or wrong CWD?) - skipping reachability guardrail"
else
    python3 "${SCRIPT_DIR}/filter_oracle.py" reachability \
        --depmap enhanced_dependency_mapping.json \
        --ctest ctest_list.txt \
        --ninja build.ninja \
        --output reachability_result.json \
        || echo "WARNING: reachability guardrail found unreachable compiled tests (see reachability_result.json) - continuing"
fi

# Step 3: Select affected tests
echo ""
echo "Step 3: Selecting affected tests..."
python3 "${SCRIPT_DIR}/main.py" select \
    enhanced_dependency_mapping.json \
    origin/${BASE_BRANCH} \
    HEAD \
    --ctest-only \
    --output tests_to_run.json

if [ ! -f "tests_to_run.json" ]; then
    echo "Error: Failed to generate tests_to_run.json"
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi

# Step 4: Check if any tests were selected
num_tests=$(jq -r '.tests_to_run | length' tests_to_run.json 2>/dev/null || echo "0")
echo "[OK] Selected ${num_tests} tests"

if [ "${num_tests}" -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Result: No tests affected by changes"
    echo "========================================="
    echo "none" > build_targets.txt
    echo "SMART_BUILD_MODE=none" > build_mode.env
    exit 0
fi

# Step 5: Extract build targets (executables)
echo ""
echo "Step 5: Extracting build targets..."
jq -r '.executables[]' tests_to_run.json | tr '\n' ' ' > build_targets.txt

num_targets=$(jq -r '.executables | length' tests_to_run.json)
echo "[OK] Generated ${num_targets} build targets"

echo "SMART_BUILD_MODE=selective" > build_mode.env

# Display summary
echo ""
echo "========================================="
echo "Smart Build Summary"
echo "========================================="
echo "Tests to run: ${num_tests}"
echo "Build targets: ${num_targets}"
echo "Output files:"
echo "  - tests_to_run.json (test selection)"
echo "  - build_targets.txt (ninja targets)"
echo "  - build_mode.env (SMART_BUILD_MODE=selective)"
echo "  - smart_build_ci.log (full run log)"
echo "========================================="

# Show first few targets for verification
echo ""
echo "Sample build targets (first 5):"
# Use tr+awk rather than nested head to avoid SIGPIPE under set -e
tr ' ' '\n' < build_targets.txt | awk 'NR<=5'

echo ""
echo "[OK] Smart build preparation complete"
exit 0
