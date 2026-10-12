#!/bin/bash
# Run hipblaslt-test under coverage profiling.
# Usage: run_coverage_tests.sh <test-binary> <coverage-dir>
#
# Pass 1 runs GTEST_FILTER (default "*"); its result is the script's exit code.
# Unless HIPBLASLT_COVERAGE_EXTRA_PASSES=0, three more passes add coverage for
# code the CI filter does not reach:
#   2. tests whose names carry no tier category, which "*smoke*:*pre_checkin*"
#      never selects;
#   3. smoke matmul tests with all logging enabled;
#   4. smoke matmul tests with HIPBLASLT_CHECK_NUMERICS enabled.
# Failures in passes 2-4 are reported but do not change the exit code. These
# tests gate in the regular test jobs; here they only contribute coverage.

set -uo pipefail

TEST_BINARY="$1"
COVERAGE_DIR="$2"

FILTER="${GTEST_FILTER:-*}"
EXTRA_PASSES="${HIPBLASLT_COVERAGE_EXTRA_PASSES:-1}"

UNTIERED_FILTER="*-*smoke*:*pre_checkin*:*quick*:*nightly*:*stress*:*HMM*:*sdc_hunt*:*known_bug*:*multi_gpu*"
SMOKE_MATMUL_FILTER="*matmul_test.matmul/smoke_*"

export LLVM_PROFILE_FILE="${COVERAGE_DIR}/profraw/hipblaslt-coverage_%p.profraw"
export GTEST_LISTENER=NO_PASS_LINE_IN_LOG

summary=()

# run_pass <name> <gtest-filter> [ENV=VALUE ...]
run_pass() {
    local name="$1"
    local filter="$2"
    shift 2
    echo "Coverage pass '${name}': GTEST_FILTER=${filter} $*"
    env "$@" "${TEST_BINARY}" --gtest_filter="${filter}" \
        --precompile=hipblaslt-test-precompile.db
    local status=$?
    summary+=("${name}: exit ${status}")
    return "${status}"
}

run_pass "ci-filter" "${FILTER}"
primary_status=$?

if [[ "${EXTRA_PASSES}" != "0" ]]; then
    mkdir -p "${COVERAGE_DIR}/logs"
    run_pass "untiered" "${UNTIERED_FILTER}"
    run_pass "logging" "${SMOKE_MATMUL_FILTER}" \
        -u HIPBLASLT_LOG_LEVEL \
        HIPBLASLT_LOG_MASK=255 \
        HIPBLASLT_LOG_FILE="${COVERAGE_DIR}/logs/hipblaslt-log.txt"
    run_pass "check-numerics" "${SMOKE_MATMUL_FILTER}" \
        HIPBLASLT_CHECK_NUMERICS=2
fi

echo "Coverage pass summary (only ci-filter sets the exit code):"
printf '  %s\n' "${summary[@]}"
exit "${primary_status}"
