#!/bin/bash
# Wrapper script to run hipblaslt-test with GTEST_FILTER support
# Usage: run_coverage_tests.sh <test-binary> <coverage-dir>

TEST_BINARY="$1"
COVERAGE_DIR="$2"

# Use GTEST_FILTER from environment, or default to "*" (all tests)
FILTER="${GTEST_FILTER:-*}"

echo "Running coverage with GTEST_FILTER: $FILTER"

# Enable full logging to increase code coverage for utility.cpp and tensile_host.cpp
# Layer mode flags: error=1, trace=2, hints=4, info=8, api=16, bench=32, profile=64, extended_profile=128
# 255 = All logging modes enabled (triggers all logging/profiling code paths)
# This covers:
#   - utility.cpp: String conversion functions (hipDataType_to_string, rocblaslt_epilogue_to_string, etc.)
#   - tensile_host.cpp: Profiling and logging functions
# Previous: 232 (bench + profile + extended_profile + info only)
export HIPBLASLT_LOG_MASK=255

# Enable check numerics to increase code coverage for check_numerics_matrix.hpp
# This triggers numerical checking code paths (NaN/Inf detection, validation)
# Expected coverage gain: ~130 lines in check_numerics_matrix.hpp (20% → 76%)
# DISABLED: Causes illegal memory access errors during hipModuleUnload
# export HIPBLASLT_CHECK_NUMERICS=1

# Run all tests with profiling
# The filter "*" includes:
#   - Main matmul tests (matmul/*, grouped_gemm/*, etc.)
#   - Auxiliary tests (aux_handle_test/*, aux_ext_test/*, aux_attr_test/*)
#   - Extension operation tests (ExtOpTest/*)
#   - Ext API tests (tests with use_ext flag, including *APIExt* pattern)
# All tests write to the same profraw pattern for proper coverage merge
# Continue even if tests fail so we can generate coverage report
LLVM_PROFILE_FILE="${COVERAGE_DIR}/profraw/hipblaslt-coverage_%p.profraw" \
GTEST_LISTENER=NO_PASS_LINE_IN_LOG \
"${TEST_BINARY}" --gtest_filter="$FILTER" --precompile=hipblaslt-test-precompile.db || {
    TEST_EXIT_CODE=$?
    echo "WARNING: Tests failed with exit code ${TEST_EXIT_CODE}, but continuing to generate coverage report"
    # Save the exit code to a file so we can report it at the end
    echo "${TEST_EXIT_CODE}" > "${COVERAGE_DIR}/test_exit_code.txt"
}

# Always exit successfully so coverage report generation continues
exit 0
