file(REMOVE_RECURSE "${TEST_ROOT}")

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${SOURCE_ROOT}"
    -B "${TEST_ROOT}/off"
    -GNinja
    "-DCMAKE_C_COMPILER=${C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${DEPENDENCY_PREFIX}"
    -DROCM_LIBS_ENABLE_COMPONENTS=hipblas-common
    -DROCM_LIBS_ENABLE_ROOT_CTEST=OFF
    -DBUILD_TESTING=OFF
    -DROCM_INTERFACES_BUILD_TOOLS=OFF
    -DROCM_INTERFACES_CHECK_API_DRIFT=OFF
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR
    "root default-off configure failed:\n${configure_output}\n${configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${TEST_ROOT}/off" --target rocblas_loader_shadow
  RESULT_VARIABLE default_off_result
  OUTPUT_QUIET
  ERROR_QUIET)
if(default_off_result EQUAL 0)
  message(FATAL_ERROR "root interfaces target exists without the opt-in switch")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${SOURCE_ROOT}"
    -B "${TEST_ROOT}/on"
    -GNinja
    "-DCMAKE_C_COMPILER=${C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${DEPENDENCY_PREFIX}"
    -DROCM_LIBS_ENABLE_COMPONENTS=hipblas-common
    -DROCM_LIBS_ENABLE_INTERFACES=ON
    -DROCM_LIBS_ENABLE_ROOT_CTEST=OFF
    -DBUILD_TESTING=OFF
    -DROCM_INTERFACES_BUILD_TOOLS=OFF
    -DROCM_INTERFACES_CHECK_API_DRIFT=OFF
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR
    "root interfaces opt-in configure failed:\n${configure_output}\n${configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${TEST_ROOT}/on" --target
    rocblas_loader_shadow
    rocm_rocblas_bridge_provider_system
    rocm_blas_narrow_v2_provider_system
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "root interfaces build failed:\n${build_output}\n${build_error}")
endif()
