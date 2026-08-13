# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

foreach(required_variable
        COMPONENT_BINARY_DIR
        PACKAGE_SOURCE_DIR
        PACKAGE_WORK_DIR
        TEST_GENERATOR
        CTEST_COMMAND)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

set(install_dir "${PACKAGE_WORK_DIR}/install")
set(build_dir "${PACKAGE_WORK_DIR}/build")
file(REMOVE_RECURSE "${PACKAGE_WORK_DIR}")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${COMPONENT_BINARY_DIR}" --prefix "${install_dir}"
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "ROCMxLayoutTransforms package installation failed.")
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        -S "${PACKAGE_SOURCE_DIR}"
        -B "${build_dir}"
        -G "${TEST_GENERATOR}"
        "-DCMAKE_PREFIX_PATH=${install_dir}"
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "ROCMxLayoutTransforms package consumer configuration failed.")
endif()

set(build_command "${CMAKE_COMMAND}" --build "${build_dir}")
if(TEST_CONFIG)
    list(APPEND build_command --config "${TEST_CONFIG}")
endif()
execute_process(COMMAND ${build_command} RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "ROCMxLayoutTransforms package consumer build failed.")
endif()

set(test_command "${CTEST_COMMAND}" --test-dir "${build_dir}" --output-on-failure)
if(TEST_CONFIG)
    list(APPEND test_command --build-config "${TEST_CONFIG}")
endif()
execute_process(COMMAND ${test_command} RESULT_VARIABLE test_result)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "ROCMxLayoutTransforms package consumer test failed.")
endif()
