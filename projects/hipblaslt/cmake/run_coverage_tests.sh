#!/bin/bash
# Wrapper script to run hipblaslt-test with GTEST_FILTER support
# Usage: run_coverage_tests.sh <test-binary> <coverage-dir>

TEST_BINARY="$1"
COVERAGE_DIR="$2"

# Use GTEST_FILTER from environment, or default to "*" (all tests)
FILTER="${GTEST_FILTER:-*}"

# Math CI narrows coverage to the pre_checkin data-driven cases. Keep focused
# adapter and ownership tests in that run without changing unrestricted runs
# or existing negative exclusions.
POSITIVE_FILTER="${FILTER%%-*}"
if [[ -n "${POSITIVE_FILTER}" && "${POSITIVE_FILTER}" != "*" ]]; then
    for REQUIRED_FILTER in "*HostNumerics*" "HipBuffer.smoke_Move*"; do
        if [[ "${POSITIVE_FILTER}" != *"${REQUIRED_FILTER}"* ]]; then
            POSITIVE_FILTER="${POSITIVE_FILTER}:${REQUIRED_FILTER}"
        fi
    done

    if [[ "${FILTER}" == *-* ]]; then
        FILTER="${POSITIVE_FILTER}-${FILTER#*-}"
    else
        FILTER="${POSITIVE_FILTER}"
    fi
fi

echo "Running coverage with GTEST_FILTER: $FILTER"

# Run the test with profiling
LLVM_PROFILE_FILE="${COVERAGE_DIR}/profraw/hipblaslt-coverage_%p.profraw" \
GTEST_LISTENER=NO_PASS_LINE_IN_LOG \
"${TEST_BINARY}" --gtest_filter="$FILTER" --precompile=hipblaslt-test-precompile.db
