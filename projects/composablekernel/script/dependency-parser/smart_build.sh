#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Smart Build Script (selection + build; no test execution)
#
# Phase 1 of the decoupled smart-build pipeline. Determines the build mode and
# targets (via smart_build_ci.sh), then builds:
#   - selective: builds only the affected test executables
#   - full:      builds the whole test/example closure (ninja check_prebuild, no run)
#   - none:      nothing to build (no CK code affected)
# Test execution is a separate phase - see smart_test.sh, which consumes the
# build/ directory and the selection artifacts produced here.
#
# The chosen mode is recorded in build_mode.env (SMART_BUILD_MODE=...) so the
# test phase knows what to run; its presence also tells smart_test.sh the build
# phase actually ran.
#
# Exit codes:
#   0 = Success (build complete, or nothing to build)
#   1 = Build failure
#
# Environment variables:
#   WORKSPACE_ROOT - Path to workspace root
#   BUILD_DIR - Build directory (defaults to current directory)
#   PARALLEL - Number of parallel jobs for dependency analysis (default: 32)
#   NINJA_JOBS - Number of ninja parallel jobs (required)
#   ARCH_NAME - Architecture name for trace files (required if PROCESS_NINJA_TRACE=true)
#   PROCESS_NINJA_TRACE - Set to "true" to process ninja build traces (default: false)
#   NINJA_FTIME_TRACE - Set to "true" to run ClangBuildAnalyzer (default: false)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$(pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${BUILD_DIR}/.." && pwd)}"
PARALLEL="${PARALLEL:-32}"
PROCESS_NINJA_TRACE="${PROCESS_NINJA_TRACE:-false}"
NINJA_FTIME_TRACE="${NINJA_FTIME_TRACE:-false}"

# Tee all output to a per-phase log so the build stage can archive it
# independently of the test stage. Process substitution (not a pipe) keeps the
# exit status of the commands below intact.
LOG_FILE="${BUILD_DIR}/smart_build.log"
exec > >(tee "${LOG_FILE}") 2>&1

# Validate required parameters
if [ -z "$NINJA_JOBS" ]; then
    echo "Error: NINJA_JOBS environment variable is required"
    exit 1
fi

if [ "$PROCESS_NINJA_TRACE" = "true" ] && [ -z "$ARCH_NAME" ]; then
    echo "Error: ARCH_NAME environment variable is required when PROCESS_NINJA_TRACE=true"
    exit 1
fi

echo "========================================="
echo "Smart Build (selection + build)"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "WORKSPACE_ROOT: ${WORKSPACE_ROOT}"
echo "NINJA_JOBS: ${NINJA_JOBS}"
echo "PROCESS_NINJA_TRACE: ${PROCESS_NINJA_TRACE}"
echo "NINJA_FTIME_TRACE: ${NINJA_FTIME_TRACE}"
echo "-----------------------------------------"

cd "${BUILD_DIR}"

# Process the ninja build trace if requested (shared by full + selective paths).
process_ninja_trace() {
    [ "$PROCESS_NINJA_TRACE" = "true" ] || return 0
    echo ""
    echo "Processing ninja build trace..."
    python3 ../script/ninja_json_converter.py .ninja_log --legacy-format --output ck_build_trace_${ARCH_NAME}.json
    python3 ../script/parse_ninja_trace.py ck_build_trace_${ARCH_NAME}.json
    if [ "$NINJA_FTIME_TRACE" = "true" ]; then
        echo "Running ClangBuildAnalyzer..."
        /ClangBuildAnalyzer/build/ClangBuildAnalyzer --all . clang_build.log
        /ClangBuildAnalyzer/build/ClangBuildAnalyzer --analyze clang_build.log > clang_build_analysis_${ARCH_NAME}.log
    fi
}

# Step 1: Run smart-build CI script (selection).
echo "Using Smart Build System"
echo ""

export WORKSPACE_ROOT
export PARALLEL

if ! bash "${SCRIPT_DIR}/smart_build_ci.sh"; then
    # Full build required (exit code 1 from smart_build_ci.sh).
    echo "SMART_BUILD_MODE=full" > build_mode.env
    echo "Full build mode - building the complete test/example closure (no run)"
    # check_prebuild is the build-only half of the CMake `check` target: a
    # command-less aggregate of the entire test/example build closure. It builds
    # exactly the set the full ctest run in smart_test.sh executes, with no run.
    ninja -j"${NINJA_JOBS}" check_prebuild
    process_ninja_trace
    echo ""
    echo "[OK] Smart build complete (full mode - all tests built)"
    exit 0
fi

# Step 2: Selective build mode - read targets.
BUILD_TARGETS=$(cat build_targets.txt)

if [ "$BUILD_TARGETS" = "none" ]; then
    echo "SMART_BUILD_MODE=none" > build_mode.env
    echo "[OK] No tests affected by changes - nothing to build"
    exit 0
fi

# Step 3: Build only affected targets.
echo "SMART_BUILD_MODE=selective" > build_mode.env
echo "[OK] Selective build - building only affected targets"
echo "Building targets: ${BUILD_TARGETS}"
# Word-split BUILD_TARGETS intentionally: targets are space-separated basenames
# that never contain spaces (ninja target naming convention).
# shellcheck disable=SC2086
ninja -j"${NINJA_JOBS}" ${BUILD_TARGETS}

process_ninja_trace

echo ""
echo "[OK] Smart build complete (selective mode)"
exit 0
