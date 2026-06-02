#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Smart Build and Test Execution Script
#
# This script handles the complete smart-build workflow:
# 1. Runs smart_build_ci.sh to determine build mode and targets
# 2. Builds only affected targets (selective mode) or everything (full mode)
# 3. Runs affected tests using ctest with regex filtering
# 4. Optionally processes ninja build traces
#
# Dry-run / smoke mode (DRY_RUN=true or --dry-run/--smoke):
#   Instead of compiling and testing, validates the selected executables against
#   ninja's real target namespace (`ninja -t targets all`) via main.py validate,
#   writing a structured smoke_result.json verdict. This proves every selected
#   target is one ninja actually knows about - without invoking the compiler or a
#   GPU - a fast, GPU-free gate that catches test-selection bugs (e.g. a target
#   name ninja does not know) before committing to a real build.
#   (`ninja -n` is deliberately NOT used: CK's GLOB CONFIGURE_DEPENDS makes every
#   ninja call regenerate build.ninja, so `ninja -n` exits 0 for any target.)
#
# Exit codes:
#   0 = Success
#   1 = Build or test failure (or, in dry-run, an unresolvable target)
#
# Environment variables:
#   WORKSPACE_ROOT - Path to workspace root
#   BUILD_DIR - Build directory (defaults to current directory)
#   PARALLEL - Number of parallel jobs for dependency analysis (default: 32)
#   NINJA_JOBS - Number of ninja parallel jobs (required unless DRY_RUN=true)
#   ARCH_NAME - Architecture name for trace files (required if PROCESS_NINJA_TRACE=true)
#   PROCESS_NINJA_TRACE - Set to "true" to process ninja build traces (default: false)
#   NINJA_FTIME_TRACE - Set to "true" to run ClangBuildAnalyzer (default: false)
#   DRY_RUN - Set to "true" to validate the build graph without building/testing (default: false)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$(pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd ${BUILD_DIR}/.. && pwd)}"
PARALLEL="${PARALLEL:-32}"
PROCESS_NINJA_TRACE="${PROCESS_NINJA_TRACE:-false}"
NINJA_FTIME_TRACE="${NINJA_FTIME_TRACE:-false}"
DRY_RUN="${DRY_RUN:-false}"
LOG_FILE="${BUILD_DIR}/smart_build.log"

# Allow --dry-run / --smoke as a CLI alternative to DRY_RUN=true
for arg in "$@"; do
    case "$arg" in
        --dry-run|--smoke) DRY_RUN=true ;;
    esac
done

# Stream output to a combined log file (for CI artifact archiving) as well as the
# console. This is the top-level entry point, so it always tees; it exports
# _SMART_BUILD_NESTED before calling smart_build_ci.sh so the child skips its own
# tee and its output flows into this single combined log in order. A backgrounded
# tee draining a FIFO (whose PID we wait on at exit) is used instead of
# `exec > >(tee)` so the log is fully flushed before exit (the bare form can lose
# the tail, including the final pass/fail banner).
_LOG_FIFO="$(mktemp -u)"
mkfifo "${_LOG_FIFO}"
tee "${LOG_FILE}" < "${_LOG_FIFO}" &
_TEE_PID=$!
exec > "${_LOG_FIFO}" 2>&1
rm -f "${_LOG_FIFO}"
trap '_rc=$?; exec 1>&- 2>&-; wait "${_TEE_PID}" 2>/dev/null || true; exit ${_rc}' EXIT

# Validate required parameters
# NINJA_JOBS is not needed in dry-run mode (no compilation; uses ninja -t targets all).
if [ "$DRY_RUN" != "true" ] && [ -z "$NINJA_JOBS" ]; then
    echo "Error: NINJA_JOBS environment variable is required"
    exit 1
fi

if [ "$PROCESS_NINJA_TRACE" = "true" ] && [ -z "$ARCH_NAME" ]; then
    echo "Error: ARCH_NAME environment variable is required when PROCESS_NINJA_TRACE=true"
    exit 1
fi

echo "========================================="
echo "Smart Build and Test Execution"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "WORKSPACE_ROOT: ${WORKSPACE_ROOT}"
echo "NINJA_JOBS: ${NINJA_JOBS}"
echo "PROCESS_NINJA_TRACE: ${PROCESS_NINJA_TRACE}"
echo "NINJA_FTIME_TRACE: ${NINJA_FTIME_TRACE}"
echo "DRY_RUN: ${DRY_RUN}"
echo "-----------------------------------------"

cd "${BUILD_DIR}"

# Step 1: Run smart-build CI script
echo "🚀 Using Smart Build System"
echo ""

export WORKSPACE_ROOT
export PARALLEL
# Tell the child to skip its own tee; its output flows into our combined log.
export _SMART_BUILD_NESTED=1

if ! bash "${SCRIPT_DIR}/smart_build_ci.sh"; then
    # Full build required (exit code 1 from smart_build_ci.sh)
    if [ "$DRY_RUN" = "true" ]; then
        echo "🧪 DRY RUN - full build mode: no selection to validate (everything is built)"
        echo "✓ Dry run complete (full build mode)"
        exit 0
    fi

    echo "⚠ Full build mode - building and testing everything"
    ninja -j${NINJA_JOBS} check

    # Process ninja build trace if requested
    if [ "$PROCESS_NINJA_TRACE" = "true" ]; then
        echo ""
        echo "Processing ninja build trace..."
        python3 ../script/ninja_json_converter.py .ninja_log --legacy-format --output ck_build_trace_${ARCH_NAME}.json
        python3 ../script/parse_ninja_trace.py ck_build_trace_${ARCH_NAME}.json

        if [ "$NINJA_FTIME_TRACE" = "true" ]; then
            echo "Running ClangBuildAnalyzer..."
            /ClangBuildAnalyzer/build/ClangBuildAnalyzer --all . clang_build.log
            /ClangBuildAnalyzer/build/ClangBuildAnalyzer --analyze clang_build.log > clang_build_analysis_${ARCH_NAME}.log
        fi
    fi

    exit 0
fi

# Step 2: Selective build mode - read targets
BUILD_TARGETS=$(cat build_targets.txt)

if [ "$BUILD_TARGETS" = "none" ]; then
    echo "✓ No tests affected by changes - skipping build and test execution"
    exit 0
fi

# Step 3: Build only affected targets
if [ "$DRY_RUN" = "true" ]; then
    NUM_TARGETS=$(echo "${BUILD_TARGETS}" | wc -w)
    echo "🧪 DRY RUN - validating ${NUM_TARGETS} selected target(s), no compilation, no tests"
    # Validate the selection against ninja's real target namespace.
    # NOTE: `ninja -n <target>` is NOT used as the oracle: CK uses CMake GLOB
    # CONFIGURE_DEPENDS, so every ninja call regenerates build.ninja and
    # `ninja -n` then exits 0 for any target (real or bogus). The reliable
    # oracle is the target list from `ninja -t targets all`.
    ninja -t targets all > ninja_targets.txt 2>/dev/null || { echo "⚠ ninja -t targets all failed; cannot validate target namespace"; exit 1; }
    python3 "${SCRIPT_DIR}/main.py" validate \
        tests_to_run.json \
        --ninja-targets ninja_targets.txt \
        --output smoke_result.json
    echo "✓ Dry run complete - selection validated against ninja target namespace"
    exit 0
fi

# Observability (non-fatal): record a structured verdict on whether the selection
# maps to real ninja targets. This does NOT change build/test behavior - it only
# emits smoke_result.json / smoke_result.xml for CI to archive. The real build
# below proceeds regardless of the verdict.
echo ""
echo "Recording selection validation (observability, non-fatal)..."
ninja -t targets all > ninja_targets.txt 2>/dev/null || true
python3 "${SCRIPT_DIR}/main.py" validate \
    tests_to_run.json \
    --ninja-targets ninja_targets.txt \
    --output smoke_result.json \
    --junit smoke_result.xml \
    || echo "⚠ selection validation flagged issues (see smoke_result.json) - continuing with build"

echo "✓ Selective build - building only affected targets"
echo "Building targets: ${BUILD_TARGETS}"
# Word-split BUILD_TARGETS intentionally: targets are space-separated basenames
# that never contain spaces (ninja target naming convention).
# shellcheck disable=SC2086
ninja -j"${NINJA_JOBS}" ${BUILD_TARGETS}

# Process ninja build trace if requested
if [ "$PROCESS_NINJA_TRACE" = "true" ]; then
    echo ""
    echo "Processing ninja build trace..."
    python3 ../script/ninja_json_converter.py .ninja_log --legacy-format --output ck_build_trace_${ARCH_NAME}.json
    python3 ../script/parse_ninja_trace.py ck_build_trace_${ARCH_NAME}.json

    if [ "$NINJA_FTIME_TRACE" = "true" ]; then
        echo "Running ClangBuildAnalyzer..."
        /ClangBuildAnalyzer/build/ClangBuildAnalyzer --all . clang_build.log
        /ClangBuildAnalyzer/build/ClangBuildAnalyzer --analyze clang_build.log > clang_build_analysis_${ARCH_NAME}.log
    fi
fi

# Step 4: Run affected tests using regex_chunks
echo ""
echo "Running affected tests..."

NUM_CHUNKS=$(jq -r '.regex_chunks | length' tests_to_run.json)
echo "Running ${NUM_CHUNKS} test chunk(s)"

if [ "$NUM_CHUNKS" -eq 1 ]; then
    TEST_REGEX=$(jq -r '.regex_chunks[0]' tests_to_run.json)
    CTEST_PARALLEL_LEVEL=4 ctest --output-on-failure -R "${TEST_REGEX}"
else
    for ((i=0; i<NUM_CHUNKS; i++)); do
        TEST_REGEX=$(jq -r ".regex_chunks[$i]" tests_to_run.json)
        echo "Running test chunk $((i+1))/${NUM_CHUNKS}"
        CTEST_PARALLEL_LEVEL=4 ctest --output-on-failure -R "${TEST_REGEX}"
    done
fi

echo ""
echo "✓ Smart build and test execution complete"
exit 0
