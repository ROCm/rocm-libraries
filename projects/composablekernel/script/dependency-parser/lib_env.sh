#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Shared environment defaults for the smart-build scripts.
#
# init_smart_build_env
#   Sets the shared environment defaults used across smart_build.sh,
#   smart_build_ci.sh, and smart_test.sh. Each value is taken from the
#   environment when set, otherwise a default is applied - so CI (which exports
#   these via vars/ck.groovy) always wins, and standalone runs get sane defaults.
#   Unused variables are harmless: each script reads only the subset it needs.
#
# Canonical environment-variable reference for the pipeline:
#   BUILD_DIR            - Build directory                       (default: $(pwd))
#   WORKSPACE_ROOT       - Workspace root                        (default: BUILD_DIR/..)
#   PARALLEL             - Parallel jobs for dependency analysis (default: 32)
#   BASE_BRANCH          - Base branch for the change diff       (default: develop)
#   PROCESS_NINJA_TRACE  - "true" to process ninja build traces  (default: false)
#   NINJA_FTIME_TRACE    - "true" to run ClangBuildAnalyzer      (default: false)
#   DRY_RUN              - "true" to validate without building   (default: false)
#   CTEST_PARALLEL       - ctest parallel level                  (default: 4)
#
# Set per script (no shared default):
#   SCRIPT_DIR   - each script computes it from its own BASH_SOURCE
#   LOG_FILE     - per-script log name (smart_build.log / smart_test.log / ...)
#   NINJA_JOBS   - required by smart_build.sh (unless DRY_RUN); no default
#   ARCH_NAME    - required by smart_build.sh when PROCESS_NINJA_TRACE=true
init_smart_build_env() {
    BUILD_DIR="${BUILD_DIR:-$(pwd)}"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${BUILD_DIR}/.." && pwd)}"
    PARALLEL="${PARALLEL:-32}"
    BASE_BRANCH="${BASE_BRANCH:-develop}"
    PROCESS_NINJA_TRACE="${PROCESS_NINJA_TRACE:-false}"
    NINJA_FTIME_TRACE="${NINJA_FTIME_TRACE:-false}"
    DRY_RUN="${DRY_RUN:-false}"
    CTEST_PARALLEL="${CTEST_PARALLEL:-4}"
}
