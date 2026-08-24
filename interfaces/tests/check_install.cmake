file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${PROJECT_BINARY_DIR}" --prefix "${TEST_ROOT}/prefix"
    --component rocm-interfaces
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "shadow install failed:\n${install_output}\n${install_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${CONSUMER_SOURCE_DIR}"
    -B "${TEST_ROOT}/build"
    -GNinja
    "-DCMAKE_PREFIX_PATH=${TEST_ROOT}/prefix;${DEPENDENCY_PREFIX}"
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "installed consumer configure failed:\n${configure_output}\n${configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${TEST_ROOT}/build"
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "installed consumer build failed:\n${build_output}\n${build_error}")
endif()

if(REAL_ROCBLAS_LIBRARY)
  execute_process(
    COMMAND "${TEST_ROOT}/build/rocm_interfaces_install_consumer"
      "${TEST_ROOT}/prefix/${INSTALL_LIBDIR}/rocm/interfaces/providers/rocblas-system.json"
      "${REAL_ROCBLAS_LIBRARY}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error)
  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "installed manifest consumer failed:\n${run_output}\n${run_error}")
  endif()
endif()
