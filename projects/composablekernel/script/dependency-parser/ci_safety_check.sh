#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# CI Safety Check for Smart Build System
#
# This script determines when to force full builds vs selective builds.
# Integrates with existing Jenkins infrastructure (FORCE_CI, BRANCH_NAME, etc.)
#
# Exit codes:
#   0 = Selective build OK (use smart build)
#   1 = Full build required
#
# Environment variables (set by Jenkins):
#   FORCE_CI - Set to "true" for nightly/scheduled builds
#   BRANCH_NAME - Git branch name
#   GIT_PREVIOUS_COMMIT, GIT_COMMIT - For detecting changes
#
# Manual override (set by developer/admin if needed):
#   DISABLE_SMART_BUILD - Set to "true" to force full build

set -e

# Configuration
FORCE_FULL_BUILD=false
REASON=""

# 1. Check if this is a nightly/scheduled build
# Existing Jenkins infrastructure sets FORCE_CI=true for cron-triggered builds
if [ "$FORCE_CI" = "true" ]; then
    FORCE_FULL_BUILD=true
    REASON="nightly/scheduled build (FORCE_CI=true from Jenkins cron)"
fi

# 2. Manual override to disable smart build
# Set DISABLE_SMART_BUILD=true in Jenkins job parameters if you want to force a full build
if [ "$DISABLE_SMART_BUILD" = "true" ]; then
    FORCE_FULL_BUILD=true
    REASON="manual override (DISABLE_SMART_BUILD=true)"
fi

# 3. Force full build if CMakeLists.txt or cmake/ configuration changed
if [ -n "$GIT_PREVIOUS_COMMIT" ] && [ -n "$GIT_COMMIT" ]; then
    CHANGED_FILES=$(git diff --name-only $GIT_PREVIOUS_COMMIT..$GIT_COMMIT 2>/dev/null || echo "")
else
    # Fallback to comparing with develop
    CHANGED_FILES=$(git diff --name-only origin/develop...HEAD 2>/dev/null || echo "")
fi

if echo "$CHANGED_FILES" | grep -qE "(CMakeLists\.txt|cmake/.*\.cmake)"; then
    FORCE_FULL_BUILD=true
    REASON="build system configuration changed (CMakeLists.txt or cmake/*.cmake)"
fi

# 4. Force full build if dependency cache is older than 7 days
CACHE_FILE="cmake_dependency_mapping.json"
if [ -f "$CACHE_FILE" ]; then
    # Different stat command for Linux vs macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CACHE_MTIME=$(stat -f %m "$CACHE_FILE")
    else
        CACHE_MTIME=$(stat -c %Y "$CACHE_FILE")
    fi
    CURRENT_TIME=$(date +%s)
    CACHE_AGE_DAYS=$(( ($CURRENT_TIME - $CACHE_MTIME) / 86400 ))

    if [ $CACHE_AGE_DAYS -gt 7 ]; then
        FORCE_FULL_BUILD=true
        REASON="dependency cache older than 7 days"
    fi
fi

# Output decision
echo "========================================="
echo "Smart Build Safety Check"
echo "========================================="
echo "FORCE_CI: ${FORCE_CI:-false}"
echo "BRANCH_NAME: ${BRANCH_NAME:-unknown}"
echo "DISABLE_SMART_BUILD: ${DISABLE_SMART_BUILD:-false}"
echo "-----------------------------------------"

if [ "$FORCE_FULL_BUILD" = true ]; then
    echo "Decision: 🔴 FULL BUILD REQUIRED"
    echo "Reason: $REASON"
    echo "========================================="
    echo "export SMART_BUILD_MODE=full" > build_mode.env
    exit 1  # Exit with error to signal full build needed
else
    echo "Decision: 🟢 SELECTIVE BUILD ENABLED"
    echo "Using smart build for faster CI"
    echo "========================================="
    echo "export SMART_BUILD_MODE=selective" > build_mode.env
    exit 0  # Exit success to signal selective build OK
fi
