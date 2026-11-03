#!/bin/bash
# Run hipblaslt-test under coverage profiling.
# Usage: run_coverage_tests.sh <test-binary> <coverage-dir> [tier-categories]
#
# tier-categories is a comma-separated list of the test categories that carry a
# tier token in their names; CMake derives it from clients/tests/data. Pass 2
# selects everything outside that list.
#
# Pass 1 runs GTEST_FILTER (default "*"). Unless
# HIPBLASLT_COVERAGE_EXTRA_PASSES=0, three more passes add coverage for code the
# CI filter does not reach:
#   2. tests whose names carry no tier category, which "*smoke*:*pre_checkin*"
#      never selects;
#   3. smoke matmul tests with all logging enabled;
#   4. smoke matmul tests with HIPBLASLT_CHECK_NUMERICS enabled.
#
# A pass gates (its failure fails this script) when it is the only place that
# configuration runs. Passes 3 and 4 are the only runs of HIPBLASLT_LOG_MASK and
# HIPBLASLT_CHECK_NUMERICS over the matmul suite, so they gate. Pass 2 does not:
# its tests all run unfiltered in the Math CI precheckin job, which gates there.

set -uo pipefail

TEST_BINARY="$1"
COVERAGE_DIR="$2"
TIER_CATEGORIES="${3:-smoke,quick,pre_checkin,nightly,stress,HMM,sdc_hunt,known_bug,multi_gpu}"

FILTER="${GTEST_FILTER:-*}"
EXTRA_PASSES="${HIPBLASLT_COVERAGE_EXTRA_PASSES:-1}"

# A crash discards the whole pass's profile, so tests that crash only under
# coverage are excluded here, each with its tracking ticket.
#   StreamKNullStream: segfaults in the coverage build only (AIHPBLAS-5137).
#   *pointer_array*: HIPBLASLT_CHECK_NUMERICS faults the GPU on pointer-array
#   batched matmuls (AIHPBLAS-5136).
# Both still run in the unfiltered precheckin job; see TESTING.md, "Known Bugs
# and Expected Failures".
UNTIERED_FILTER="*-"
while IFS= read -r category; do
    [[ -n "${category}" ]] && UNTIERED_FILTER+="*${category}*:"
done <<< "${TIER_CATEGORIES//,/$'\n'}"
UNTIERED_FILTER+="StreamKNullStream.*"

SMOKE_MATMUL_FILTER="*matmul_test.matmul/smoke_*"
CHECK_NUMERICS_FILTER="${SMOKE_MATMUL_FILTER}-*pointer_array*"

export LLVM_PROFILE_FILE="${COVERAGE_DIR}/profraw/hipblaslt-coverage_%p.profraw"
export GTEST_LISTENER=NO_PASS_LINE_IN_LOG

summary=()
exit_status=0

# run_pass <name> <gating|advisory> <gtest-filter> [ENV=VALUE ...]
run_pass() {
    local name="$1"
    local gating="$2"
    local filter="$3"
    shift 3
    echo "Coverage pass '${name}' (${gating}): GTEST_FILTER=${filter} $*"
    env "$@" "${TEST_BINARY}" --gtest_filter="${filter}" \
        --precompile=hipblaslt-test-precompile.db
    local status=$?
    summary+=("${name} (${gating}): exit ${status}")
    if [[ "${gating}" == "gating" ]] && (( status != 0 && exit_status == 0 )); then
        exit_status="${status}"
    fi
}

run_pass "ci-filter" gating "${FILTER}"

if [[ "${EXTRA_PASSES}" != "0" ]]; then
    mkdir -p "${COVERAGE_DIR}/logs"
    run_pass "untiered" advisory "${UNTIERED_FILTER}"
    run_pass "logging" gating "${SMOKE_MATMUL_FILTER}" \
        -u HIPBLASLT_LOG_LEVEL \
        HIPBLASLT_LOG_MASK=255 \
        HIPBLASLT_LOG_FILE="${COVERAGE_DIR}/logs/hipblaslt-log.txt"
    run_pass "check-numerics" gating "${CHECK_NUMERICS_FILTER}" \
        HIPBLASLT_CHECK_NUMERICS=2
fi

echo "Coverage pass summary (a gating pass's failure fails this script):"
printf '  %s\n' "${summary[@]}"
exit "${exit_status}"
