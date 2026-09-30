#!/bin/bash
# Wrapper script to run hipblaslt-test with GTEST_FILTER support
# Usage: run_coverage_tests.sh <test-binary> <coverage-dir>

TEST_BINARY="$1"
COVERAGE_DIR="$2"

# Use GTEST_FILTER from environment, or default to "*" (all tests)
FILTER="${GTEST_FILTER:-*}"

echo "Running coverage with GTEST_FILTER: $FILTER"

# Run the test with profiling
LLVM_PROFILE_FILE="${COVERAGE_DIR}/profraw/hipblaslt-coverage_%p.profraw" \
GTEST_LISTENER=NO_PASS_LINE_IN_LOG \
"${TEST_BINARY}" --gtest_filter="$FILTER" --precompile=hipblaslt-test-precompile.db
