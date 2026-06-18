# Runs the build-time internal tests, failing if the binary fails or if
# the filter selects nothing.  Args: TEST_EXE, TEST_FILTER (may be empty).

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

# Echo gtest output into the build log.
message( "${output}" )

if( NOT result EQUAL 0 )
  message( FATAL_ERROR "Internal tests failed (exit code ${result})." )
endif()

# An empty selection makes gtest exit 0; treat it as a failure.
if( output MATCHES "Running 0 tests from 0 test suites" )
  message( FATAL_ERROR
    "No internal tests were selected - filter \"${TEST_FILTER}\" matched nothing." )
endif()
