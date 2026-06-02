#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Validate Smart Build vs Legacy Method for a PR
#
# This script compares smart build and legacy dependency analysis
# to ensure both methods produce the same test selection results.

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PR_NUMBER=""
BASE_BRANCH="origin/develop"
# The smart-build tooling now lives on develop, so rebasing the PR onto
# origin/develop is the default. Override with -s for a feature branch.
SMART_BUILD_BRANCH="origin/develop"
BUILD_DIR="../../build"
SKIP_BUILD=false
SKIP_LEGACY=false

print_help() {
    cat << 'HELP'
Usage: validate_pr.sh -p PR_NUMBER [OPTIONS]

Validates that smart build and legacy methods select the same tests for a PR.

Required:
  -p, --pr PR_NUMBER          PR number to validate

Options:
  -b, --base BRANCH           Base branch (default: origin/develop)
  -s, --smart-build BRANCH    Smart build branch (default: origin/develop)
  --skip-build                Skip full build (use existing build artifacts)
  --skip-legacy               Skip legacy analysis (only run smart build)
  -h, --help                  Show this help

Examples:
  # Validate PR 5324
  ./validate_pr.sh -p 5324

  # Validate PR 5168 with custom base
  ./validate_pr.sh -p 5168 -b origin/main

  # Quick validation (skip build, only smart build)
  ./validate_pr.sh -p 5324 --skip-build --skip-legacy

Output:
  Results saved to build/prXXXX_validation_results.txt
HELP
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pr)
            PR_NUMBER="$2"
            shift 2
            ;;
        -b|--base)
            BASE_BRANCH="$2"
            shift 2
            ;;
        -s|--smart-build)
            SMART_BUILD_BRANCH="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-legacy)
            SKIP_LEGACY=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$PR_NUMBER" ]; then
    log_error "PR number is required"
    print_help
    exit 1
fi

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUTPUT_FILE="$BUILD_DIR/pr${PR_NUMBER}_validation_results.txt"

log_section "Validation Configuration"
echo "PR Number: $PR_NUMBER"
echo "Base Branch: $BASE_BRANCH"
echo "Smart Build Branch: $SMART_BUILD_BRANCH"
echo "Skip Build: $SKIP_BUILD"
echo "Skip Legacy: $SKIP_LEGACY"
echo "Output File: $OUTPUT_FILE"

# Start validation log
exec > >(tee "$OUTPUT_FILE") 2>&1

log_section "Step 1: Fetch PR $PR_NUMBER"
cd "$PROJECT_ROOT" || exit 1

# This script checks out and rebases in place. Refuse to run on a dirty tree
# (those changes would be lost) and restore the caller's original branch on exit.
if ! git diff --quiet || ! git diff --cached --quiet; then
    log_error "Working tree has uncommitted changes; refusing to run (checkout/rebase would discard them)."
    exit 1
fi

ORIG_REF="$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)"
restore_git() {
    log_info "Restoring original git state (${ORIG_REF})..."
    git rebase --abort 2>/dev/null || true
    git checkout -q "$ORIG_REF" 2>/dev/null || true
    git branch -D "pr-${PR_NUMBER}" 2>/dev/null || true
}
trap restore_git EXIT

log_info "Fetching PR #$PR_NUMBER..."
git fetch origin pull/${PR_NUMBER}/head:pr-${PR_NUMBER}

log_info "Checking out PR branch..."
git checkout pr-${PR_NUMBER}

log_info "PR commit:"
git log --oneline -1

log_section "Step 2: Rebase on Smart Build Branch"
log_info "Rebasing pr-${PR_NUMBER} on $SMART_BUILD_BRANCH..."

# Attempt rebase, handling conflicts by accepting PR changes
if ! git rebase "${SMART_BUILD_BRANCH}"; then
    log_warn "Rebase conflicts detected, resolving by accepting PR changes..."

    # Loop to handle multiple conflicts during rebase
    while true; do
        # Get list of conflicted files
        CONFLICTED_FILES=$(git diff --name-only --diff-filter=U)

        if [ -z "$CONFLICTED_FILES" ]; then
            log_info "No more conflicts, rebase complete"
            break
        fi

        log_info "Conflicted files:"
        echo "$CONFLICTED_FILES"

        # For each conflicted file, accept the PR's version (theirs)
        while IFS= read -r file; do
            if [ -f "$file" ]; then
                log_info "Accepting PR changes for: $file"
                git checkout --theirs "$file"
                git add "$file"
            fi
        done <<< "$CONFLICTED_FILES"

        # Continue the rebase
        log_info "Continuing rebase..."
        if git -c core.editor=true rebase --continue 2>&1 | grep -q "No changes"; then
            log_warn "No changes after conflict resolution, skipping commit"
            git rebase --skip
        elif git rebase --show-current-patch &>/dev/null; then
            # Still in rebase, continue loop
            continue
        else
            # Rebase complete
            log_info "Rebase completed"
            break
        fi
    done
fi

log_info "Rebased commits:"
git log --oneline -5

log_section "Step 3: Analyze Changed Files"
log_info "Files changed vs $BASE_BRANCH:"
CHANGED_FILES=$(git diff --name-only "${BASE_BRANCH}"...HEAD -- projects/composablekernel)
# `echo "" | wc -l` is 1, so count non-empty lines to report 0 for an empty diff.
if [ -z "$CHANGED_FILES" ]; then
    NUM_FILES=0
else
    NUM_FILES=$(printf '%s\n' "$CHANGED_FILES" | grep -c .)
fi
echo "$CHANGED_FILES" | head -20
if [ "$NUM_FILES" -gt 20 ]; then
    echo "... (showing first 20 of $NUM_FILES files)"
fi
echo ""
echo "Total changed files: $NUM_FILES"

log_section "Step 4: Generate Fresh Dependency Map"
cd "$BUILD_DIR" || exit 1

log_info "Configuring CMake to generate compile_commands.json..."
# CMAKE_EXTRA_ARGS lets callers inject environment-specific flags, e.g. a CPU-only
# cluster node: CMAKE_EXTRA_ARGS="-DCMAKE_CXX_COMPILER=amdclang++ -DGPU_TARGETS=gfx942"
# Suppress the verbose "-- Configuring..." progress lines; propagate cmake errors.
cmake .. -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ${CMAKE_EXTRA_ARGS:-} 2>&1 \
    | { grep -v "^-- " || true; }

if [ ! -f "compile_commands.json" ]; then
    log_error "CMake configuration failed - compile_commands.json not generated"
    exit 1
fi

if [ ! -f "build.ninja" ]; then
    log_error "build.ninja not found - CMake should have generated it"
    exit 1
fi

log_info "Generating fresh dependency map for PR validation..."
START_TIME=$(date +%s)
python3 ../script/dependency-parser/main.py cmake-parse \
    compile_commands.json \
    build.ninja \
    --workspace-root .. \
    --output enhanced_dependency_mapping.json

if [ ! -f "enhanced_dependency_mapping.json" ]; then
    log_error "Dependency map generation failed"
    exit 1
fi

END_TIME=$(date +%s)
DEP_TIME=$((END_TIME - START_TIME))
log_info "Dependency map generated in ${DEP_TIME} seconds"

SMART_MAP="enhanced_dependency_mapping.json"
SMART_FILES=$(jq '.file_to_executables | length' $SMART_MAP)
log_info "Dependency map tracks $SMART_FILES files"

log_section "Step 5: Smart Build Test Selection"

log_info "Running smart build test selection..."
python3 ../script/dependency-parser/main.py select \
    "$SMART_MAP" \
    "${BASE_BRANCH}" \
    HEAD \
    --ctest-only \
    --output pr${PR_NUMBER}_smart_build.json

SMART_TESTS=$(jq -r '.tests_to_run | length' pr${PR_NUMBER}_smart_build.json)
log_info "Smart build selected: $SMART_TESTS tests"

# Show statistics
echo ""
echo "Smart Build Results:"
jq '{changed_files: .changed_files | length, tests_selected: .tests_to_run | length, statistics}' pr${PR_NUMBER}_smart_build.json

if [ "$SKIP_LEGACY" = true ]; then
    log_section "Validation Complete (Legacy Skipped)"
    echo ""
    echo "Smart Build: $SMART_TESTS tests selected"
    echo "Legacy: Skipped"
    exit 0
fi

log_section "Step 6: Full Build (for Legacy Method)"
if [ "$SKIP_BUILD" = true ]; then
    log_warn "Skipping build (--skip-build specified)"
    log_info "Using existing build artifacts..."
else
    log_info "Running full build (this takes ~60 minutes)..."
    START_TIME=$(date +%s)
    
    if ninja 2>&1 | tee pr${PR_NUMBER}_build.log; then
        END_TIME=$(date +%s)
        BUILD_TIME=$((END_TIME - START_TIME))
        log_info "Build completed in $((BUILD_TIME / 60)) minutes"
    else
        log_error "Build failed. Check pr${PR_NUMBER}_build.log for details."
        exit 1
    fi
fi

log_section "Step 7: Legacy Dependency Analysis"
log_info "Generating legacy dependency map (ninja -t deps)..."
python3 ../script/dependency-parser/main.py parse build.ninja

if [ ! -f "enhanced_dependency_mapping.json" ]; then
    log_error "Legacy dependency map generation failed"
    exit 1
fi

LEGACY_FILES=$(jq '.file_to_executables | length' enhanced_dependency_mapping.json)
log_info "Legacy map tracks $LEGACY_FILES files"

log_section "Step 8: Legacy Test Selection"
log_info "Running legacy test selection..."
python3 ../script/dependency-parser/main.py select \
    enhanced_dependency_mapping.json \
    "${BASE_BRANCH}" \
    HEAD \
    --ctest-only \
    --output pr${PR_NUMBER}_legacy_tests.json

LEGACY_TESTS=$(jq -r '.tests_to_run | length' pr${PR_NUMBER}_legacy_tests.json)
log_info "Legacy method selected: $LEGACY_TESTS tests"

# Show statistics
echo ""
echo "Legacy Method Results:"
jq '{changed_files: .changed_files | length, tests_selected: .tests_to_run | length, statistics}' pr${PR_NUMBER}_legacy_tests.json

log_section "Step 9: Compare Results"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    VALIDATION RESULTS                          ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║ PR Number:          #${PR_NUMBER}                              "
echo "║ Changed Files:      $NUM_FILES                                 "
echo "║ Smart Build Tests:  $SMART_TESTS                               "
echo "║ Legacy Tests:       $LEGACY_TESTS                              "
echo "╠════════════════════════════════════════════════════════════════╣"

# Verdict is by SET equality, not count. Equal counts with different members is
# a real selection bug, not a pass - comparing counts only would hide it.
SMART_LIST=$(jq -r '.tests_to_run | sort | .[]' pr${PR_NUMBER}_smart_build.json)
LEGACY_LIST=$(jq -r '.tests_to_run | sort | .[]' pr${PR_NUMBER}_legacy_tests.json)

if [ "$SMART_LIST" = "$LEGACY_LIST" ]; then
    echo "║ Result:             ✅ MATCH (identical test sets)            "
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    log_info "VALIDATION PASSED: both methods selected the same $SMART_TESTS tests"
    EXIT_CODE=0
else
    echo "║ Result:             ❌ MISMATCH                                "
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    log_error "VALIDATION FAILED: smart build and legacy selected different test sets"
    if [ "$SMART_TESTS" -eq "$LEGACY_TESTS" ]; then
        log_warn "Counts are equal ($SMART_TESTS) but the SETS differ - a count-only check would have falsely PASSED this."
    else
        log_warn "Counts differ: smart=$SMART_TESTS legacy=$LEGACY_TESTS"
    fi

    log_warn "Analyzing differences..."
    SMART_ONLY=$(comm -23 <(printf '%s\n' "$SMART_LIST") <(printf '%s\n' "$LEGACY_LIST") | grep -c . || true)
    LEGACY_ONLY=$(comm -13 <(printf '%s\n' "$SMART_LIST") <(printf '%s\n' "$LEGACY_LIST") | grep -c . || true)
    echo "Tests only in smart build: $SMART_ONLY"
    echo "Tests only in legacy: $LEGACY_ONLY"
    echo "--- unified diff (smart vs legacy) ---"
    diff <(printf '%s\n' "$SMART_LIST") <(printf '%s\n' "$LEGACY_LIST") || true

    EXIT_CODE=1
fi

log_section "Summary"
echo "Validation complete for PR #$PR_NUMBER"
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Smart build JSON: pr${PR_NUMBER}_smart_build.json"
if [ "$SKIP_LEGACY" = false ]; then
    echo "Legacy JSON: pr${PR_NUMBER}_legacy_tests.json"
fi

exit $EXIT_CODE
