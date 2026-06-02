#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Smart Build Script (selection + build; no test execution)
#
# Phase 1 of the decoupled smart-build pipeline. Determines the build mode and
# targets (via smart_build_ci.sh), then builds:
#   - selective: builds only the affected test executables
#   - full:      builds all test executables (`ninja tests`, no run)
#   - none:      nothing to build (no CK code affected)
# Test execution is a separate phase - see smart_test.sh, which consumes the
# build/ directory and the selection artifacts produced here.
#
# Dry-run / smoke mode (DRY_RUN=true or --dry-run/--smoke):
#   Validates the selected executables against ninja's real target namespace
#   (`ninja -t targets all`) via main.py validate, writing smoke_result.json -
#   without invoking the compiler. Caveat: `ninja -t targets all` is the oracle
#   because CK's GLOB CONFIGURE_DEPENDS regenerates build.ninja on every call, so
#   `ninja -n` exits 0 for any name.
#
# Exit codes:
#   0 = Success (build complete, or dry-run validated, or nothing to build)
#   1 = Build failure (or, in dry-run, an unresolvable target)
#
# Environment: see lib_env.sh for the shared variables and defaults. This script
# also requires NINJA_JOBS (unless DRY_RUN) and ARCH_NAME (if PROCESS_NINJA_TRACE=true).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_env.sh
source "${SCRIPT_DIR}/lib_env.sh"
init_smart_build_env
LOG_FILE="${BUILD_DIR}/smart_build.log"

# Allow --dry-run / --smoke as a CLI alternative to DRY_RUN=true
for arg in "$@"; do
    case "$arg" in
        --dry-run|--smoke) DRY_RUN=true ;;
    esac
done

# shellcheck source=lib_logging.sh
source "${SCRIPT_DIR}/lib_logging.sh"
start_tee_log "${LOG_FILE}"

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
echo "Smart Build (selection + build)"
echo "========================================="
echo "BUILD_DIR: ${BUILD_DIR}"
echo "WORKSPACE_ROOT: ${WORKSPACE_ROOT}"
echo "NINJA_JOBS: ${NINJA_JOBS}"
echo "PROCESS_NINJA_TRACE: ${PROCESS_NINJA_TRACE}"
echo "NINJA_FTIME_TRACE: ${NINJA_FTIME_TRACE}"
echo "DRY_RUN: ${DRY_RUN}"
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

# Step 1: Run smart-build CI script (selection)
echo "Using Smart Build System"
echo ""

export WORKSPACE_ROOT
export PARALLEL
# Tell the child to skip its own tee; its output flows into our combined log.
export _SMART_BUILD_NESTED=1

if ! bash "${SCRIPT_DIR}/smart_build_ci.sh"; then
    # Full build required (exit code 1 from smart_build_ci.sh)
    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY RUN - full build mode: no selection to validate (everything is built)"
        echo "[OK] Dry run complete (full build mode)"
        exit 0
    fi

    echo "WARNING: Full build mode - building all test executables"
    # Build only (no run): the `tests` target aggregates every test executable
    # (add_dependencies(tests <test>)). Test execution happens in smart_test.sh.
    ninja -j${NINJA_JOBS} tests
    process_ninja_trace
    echo ""
    echo "[OK] Smart build complete (full mode - all tests built)"
    exit 0
fi

# Step 2: Selective build mode - read targets
BUILD_TARGETS=$(cat build_targets.txt)

if [ "$BUILD_TARGETS" = "none" ]; then
    echo "[OK] No tests affected by changes - nothing to build"
    exit 0
fi

# Step 3: Build only affected targets
if [ "$DRY_RUN" = "true" ]; then
    NUM_TARGETS=$(echo "${BUILD_TARGETS}" | wc -w)
    echo "DRY RUN - validating ${NUM_TARGETS} selected target(s), no compilation"
    # Validate the selection against ninja's real target namespace.
    # Caveat: `ninja -t targets all` is the oracle because CK's GLOB
    # CONFIGURE_DEPENDS regenerates build.ninja on every call, so `ninja -n`
    # exits 0 for any name and can't test target existence.
    ninja -t targets all > ninja_targets.txt 2>/dev/null || { echo "WARNING: ninja -t targets all failed; skipping target-namespace validation"; exit 1; }
    python3 "${SCRIPT_DIR}/main.py" validate \
        tests_to_run.json \
        --ninja-targets ninja_targets.txt \
        --output smoke_result.json
    echo "[OK] Dry run complete - selection validated against ninja target namespace"
    exit 0
fi

# Observability (advisory): record a structured verdict on whether the selection
# maps to real ninja targets. This emits smoke_result.json / smoke_result.xml for
# CI to archive; the build below proceeds regardless of the verdict.
echo ""
echo "Recording selection validation (observability, non-fatal)..."
ninja -t targets all > ninja_targets.txt 2>/dev/null || true
python3 "${SCRIPT_DIR}/main.py" validate \
    tests_to_run.json \
    --ninja-targets ninja_targets.txt \
    --output smoke_result.json \
    --junit smoke_result.xml \
    || echo "WARNING: selection validation flagged issues (see smoke_result.json) - continuing with build"

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
