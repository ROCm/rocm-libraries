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
        echo "DRY RUN - full build mode (actual build skipped); as-if selection + smoke computed by smart_build_ci"
        echo "[OK] Dry run complete (full build mode)"
        exit 0
    fi

    echo "Full build mode - building all test + example executables"
    # Build only (no run): `tests` + `examples` together cover every ctest-run
    # executable. Examples are EXCLUDE_FROM_ALL and attached to `check`/`examples`
    # (not to `tests` or the default `all`), so `tests` alone leaves them unbuilt
    # and the full ctest run in smart_test.sh would mark them "Not Run". This
    # matches what the old `ninja check` built (check's deps = tests + examples).
    ninja -j${NINJA_JOBS} tests examples
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

# Step 3: Build only affected targets.
# smart_build_ci.sh already ran the as-if selection + the selection-validity smoke
# (smoke_result.json/.xml) on this build, so we just build here.
if [ "$DRY_RUN" = "true" ]; then
    NUM_TARGETS=$(echo "${BUILD_TARGETS}" | wc -w)
    echo "[OK] Dry run complete - ${NUM_TARGETS} target(s) selected; smoke verdict in smoke_result.json (no compilation)"
    exit 0
fi

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
