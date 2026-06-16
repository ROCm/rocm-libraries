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
# shellcheck source=lib_env.sh
source "${SCRIPT_DIR}/lib_env.sh"
init_smart_build_env
LOG_FILE="${BUILD_DIR}/smart_build_ci.log"

# Stream output to a log file (for CI artifact archiving) as well as the console.
# When invoked by smart_build.sh the parent already tees a combined log, so
# start_tee_log honors _SMART_BUILD_NESTED and skips its own tee (smart_build_ci.log
# is only produced when this script is run standalone).
# shellcheck source=lib_logging.sh
source "${SCRIPT_DIR}/lib_logging.sh"
start_tee_log "${LOG_FILE}"

echo "========================================="
echo "Smart Build CI"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "WORKSPACE_ROOT: ${WORKSPACE_ROOT}"
echo "BASE_BRANCH: ${BASE_BRANCH}"
echo "PARALLEL: ${PARALLEL}"
echo "-----------------------------------------"

# Step 1: CI safety check decides whether a *selective* build is safe. We capture
# the decision but do NOT exit yet: the selection pipeline below runs on EVERY
# build (an advisory "as-if" computation), so the selective path + JUnit are
# exercised and published even when the actual build will be full. The exit code
# at the end still drives the real build (1 = full, 0 = selective/none).
echo "Step 1: Running CI safety check..."
cd "${BUILD_DIR}"

FULL_REQUIRED=0
if ! bash "${SCRIPT_DIR}/ci_safety_check.sh"; then
    FULL_REQUIRED=1
    echo "CI safety check: full build required"
else
    echo "[OK] CI safety check: selective build eligible"
fi

# Inputs needed for both the as-if selection and the build itself.
if [ ! -f "compile_commands.json" ] || [ ! -f "build.ninja" ]; then
    echo "Error: compile_commands.json / build.ninja not found in ${BUILD_DIR}"
    echo "Make sure cmake configured with -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    echo "full" > build_targets.txt
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi

# Step 2: Generate dependency map (always, for the as-if selection)
echo ""
echo "Step 2: Generating dependency map..."
# Key the decision off cmake-parse's EXIT CODE, not file existence: a non-zero
# exit means the depmap could not be (re)generated, so fall back to full. Because
# success is never inferred from "a file is present", a leftover
# enhanced_dependency_mapping.json from a prior run on a reused build dir can never
# be mistaken for a fresh one - there is no stale-reuse window to guard against.
if ! python3 "${SCRIPT_DIR}/main.py" cmake-parse \
    compile_commands.json \
    build.ninja \
    --workspace-root "${WORKSPACE_ROOT}" \
    --parallel ${PARALLEL} \
    --output enhanced_dependency_mapping.json; then
    echo "cmake-parse failed - full build"
    echo "full" > build_targets.txt
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi
echo "[OK] Dependency map generated"

# Step 2b: Reachability guardrail (advisory).
# ASIF_MODE and LABEL_ARGS are computed here (before step 3 where num_tests is
# known) so that both the reachability and validate JUnit share the same tags.
# ASIF_MODE defaults to full until select runs; it is refined after step 3.
ASIF_MODE=full
LABEL_ARGS=()
if [ -n "${ARCH_NAME:-}" ]; then LABEL_ARGS=(--label "${ARCH_NAME}"); fi

echo ""
echo "Step 2b: Reachability guardrail (non-fatal)..."
ctest -N > ctest_list.txt 2>/dev/null || true
if ! grep -q "Test #" ctest_list.txt 2>/dev/null; then
    echo "WARNING: ctest -N returned no tests (not yet configured or wrong CWD?) - skipping reachability guardrail"
else
    python3 "${SCRIPT_DIR}/filter_oracle.py" reachability \
        --depmap enhanced_dependency_mapping.json \
        --ctest ctest_list.txt \
        --ninja build.ninja \
        --codegen-inventory "${SCRIPT_DIR}/codegen_blindspots.json" \
        --output reachability_result.json \
        --junit reachability_result.xml \
        --mode "${ASIF_MODE}" \
        "${LABEL_ARGS[@]}" \
        || echo "WARNING: reachability guardrail found unreachable compiled tests (see reachability_result.json) - continuing"
fi

# Step 3: Select affected tests (the as-if selection)
echo ""
echo "Step 3: Selecting affected tests..."
# Same exit-code contract as cmake-parse above: a non-zero select exit -> full, so
# a leftover tests_to_run.json from a prior run is never reused. (A present-but-
# malformed file written on a zero exit is still caught by the jq guard below.)
if ! python3 "${SCRIPT_DIR}/main.py" select \
    enhanced_dependency_mapping.json \
    origin/${BASE_BRANCH} \
    HEAD \
    --ctest-only \
    --output tests_to_run.json; then
    echo "select failed - full build"
    echo "full" > build_targets.txt
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi
# A jq parse failure here means tests_to_run.json is present but malformed.
# Fall back to full (safe) - never let a corrupt selection collapse to `num_tests=0`,
# which would be read as `none` below and silently skip every test. A valid file
# reporting 0 is a legitimate "no CK files changed" and still maps to none.
if ! num_tests=$(jq -r '.tests_to_run | length' tests_to_run.json 2>/dev/null); then
    echo "Selection file malformed (jq parse failed) - full build"
    echo "full" > build_targets.txt
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi
jq -r '.executables[]' tests_to_run.json 2>/dev/null | paste -sd' ' - > selected_targets.txt
echo "[OK] As-if selection: ${num_tests} tests"

# Step 3b: Selection-validity smoke (advisory) - exercises the validate gate and
# publishes JUnit on every build, full or selective. Tag with the mode this
# selection will be used in (full/none = advisory as-if; selective = real) so the
# JUnit trend keeps them distinct.
if [ "${FULL_REQUIRED}" -eq 1 ]; then
    ASIF_MODE=full
elif [ "${num_tests}" -eq 0 ]; then
    ASIF_MODE=none
else
    ASIF_MODE=selective
fi
echo ""
echo "Step 3b: Selection-validity smoke (mode=${ASIF_MODE}, non-fatal)..."
ninja -t targets all > ninja_targets.txt 2>/dev/null || true
python3 "${SCRIPT_DIR}/main.py" validate \
    tests_to_run.json \
    --ninja-targets ninja_targets.txt \
    --ctest ctest_list.txt \
    --output smoke_result.json \
    --junit smoke_result.xml \
    --mode "${ASIF_MODE}" \
    "${LABEL_ARGS[@]}" \
    || echo "WARNING: selection validation flagged issues (see smoke_result.json) - continuing"

# Step 4: Decide the actual build mode (the as-if artifacts above are produced
# regardless; only the build target set depends on this).
echo ""
if [ "${FULL_REQUIRED}" -eq 1 ]; then
    echo "Result: FULL build (safety check) - as-if selection above is advisory only"
    echo "full" > build_targets.txt
    echo "SMART_BUILD_MODE=full" > build_mode.env
    exit 1
fi

if [ "${num_tests}" -eq 0 ]; then
    echo "Result: No tests affected by changes - nothing to build"
    echo "none" > build_targets.txt
    echo "SMART_BUILD_MODE=none" > build_mode.env
    exit 0
fi

cp selected_targets.txt build_targets.txt
num_targets=$(jq -r '.executables | length' tests_to_run.json)
echo "SMART_BUILD_MODE=selective" > build_mode.env
echo "Result: SELECTIVE build - ${num_targets} targets"
echo "Sample build targets (first 5):"
tr ' ' '\n' < build_targets.txt | awk 'NR<=5'
echo ""
echo "[OK] Smart build preparation complete"
exit 0
