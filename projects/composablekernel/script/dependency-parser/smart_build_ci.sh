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
#   tests_to_run.json - Selected tests and executables
#   tests_to_run.txt - One ctest test name per line (emitted by the selector;
#                      consumed by smart_test.sh via `ctest --tests-from-file`)
#   build_targets.txt - Space-separated list of ninja targets to build
# (build_mode.env is written by smart_build.sh, not this script.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$(pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd ${BUILD_DIR}/.. && pwd)}"
PARALLEL="${PARALLEL:-32}"
BASE_BRANCH="${BASE_BRANCH:-develop}"

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
    exit 1
fi

echo "[OK] CI safety check passed - selective build enabled"

# Step 2: Generate dependency map
echo ""
echo "Step 2: Generating dependency map..."
if [ ! -f "compile_commands.json" ]; then
    echo "Error: compile_commands.json not found in ${BUILD_DIR}"
    echo "Make sure cmake configure has been run with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    exit 1
fi

if [ ! -f "build.ninja" ]; then
    echo "Error: build.ninja not found in ${BUILD_DIR}"
    echo "Make sure cmake configure has been run with -G Ninja"
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
    exit 1
fi

echo "[OK] Dependency map generated"

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
    exit 1
fi

# Step 4: Check if any tests were selected.
# Validate the selection file first: a parse error, a non-object, or a
# tests_to_run that is missing or not an array must NOT be masked as "0 tests" -
# that would skip testing entirely (the downstream stage trusts
# build_mode.env=none as authoritative). Checking the type (not just key
# presence) rejects "tests_to_run": null and other unexpected shapes, since
# `jq '... | length'` would silently report 0 for those. Any selector
# uncertainty falls back to a full build, never a silent skip.
if ! jq -e '.tests_to_run | type == "array"' tests_to_run.json >/dev/null 2>&1; then
    echo "Error: tests_to_run.json is malformed or tests_to_run is not an array - forcing full build"
    echo "full" > build_targets.txt
    exit 1
fi

num_tests=$(jq -r '.tests_to_run | length' tests_to_run.json)
echo "[OK] Selected ${num_tests} tests"

if [ "${num_tests}" -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Result: No tests affected by changes"
    echo "========================================="
    echo "none" > build_targets.txt
    exit 0
fi

# Step 5: Extract build targets (executables)
echo ""
echo "Step 5: Extracting build targets..."
# Validate .executables the same way as .tests_to_run above: it must be a
# non-empty array here (num_tests>0 was just confirmed). A missing/null/wrong
# shape would otherwise slip through and, combined with the jq extraction below,
# yield an empty build_targets.txt - which makes smart_build.sh run `ninja` with
# no explicit targets, silently building the default set while reporting
# selective mode. Any selector uncertainty falls back to a full build.
if ! jq -e '.executables | type == "array" and length > 0' tests_to_run.json >/dev/null 2>&1; then
    echo "Error: executables missing, empty, or not an array in tests_to_run.json - forcing full build"
    echo "full" > build_targets.txt
    exit 1
fi

# Build the target list with a single jq join() rather than `jq ... | tr`. The
# pipeline form hides a jq failure behind tr's exit status (no pipefail here), so
# a broken extraction could masquerade as an empty-but-successful target list.
jq -r '.executables | join(" ")' tests_to_run.json > build_targets.txt

num_targets=$(jq -r '.executables | length' tests_to_run.json)
echo "[OK] Generated ${num_targets} build targets"

# Display summary
echo ""
echo "========================================="
echo "Smart Build Summary"
echo "========================================="
echo "Tests to run: ${num_tests}"
echo "Build targets: ${num_targets}"
echo "Output files:"
echo "  - tests_to_run.json (test selection)"
echo "  - tests_to_run.txt (ctest --tests-from-file list)"
echo "  - build_targets.txt (ninja targets)"
echo "========================================="

# Show first few targets for verification
echo ""
echo "Sample build targets (first 5):"
head -1 build_targets.txt | tr ' ' '\n' | head -5

echo ""
echo "[OK] Smart build preparation complete"
exit 0
