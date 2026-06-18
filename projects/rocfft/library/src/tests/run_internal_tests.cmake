# Driver for the build-time internal-test run.  Runs the test executable
# and fails if the test binary fails OR if no tests were actually run
# (e.g. a filter that matched nothing) - gtest exits 0 in that case, which
# would otherwise let an empty selection silently "pass".
#
# Expected -D arguments:
#   TEST_EXE    : path to the test executable
#   TEST_FILTER : GoogleTest filter string (may be empty = run all)

set( run_args "" )
if( TEST_FILTER )
  set( run_args --gtest_filter=${TEST_FILTER} )
endif()

execute_process(
  COMMAND ${TEST_EXE} ${run_args}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE  output
)

# Echo the gtest output so it still appears in the build log.
message( "${output}" )

if( NOT result EQUAL 0 )
  message( FATAL_ERROR "Internal tests failed (exit code ${result})." )
endif()

# gtest prints "Running N tests" - treat zero as a failure so a filter
# that selects nothing does not silently succeed.
if( output MATCHES "Running 0 tests from 0 test suites" )
  message( FATAL_ERROR
    "No internal tests were selected - filter \"${TEST_FILTER}\" matched nothing." )
endif()
