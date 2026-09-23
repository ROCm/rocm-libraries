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

set(configure_arguments)
if(TEST_CXX_COMPILER)
    list(APPEND configure_arguments "-DCMAKE_CXX_COMPILER=${TEST_CXX_COMPILER}")
endif()
if(TEST_MAKE_PROGRAM)
    list(APPEND configure_arguments "-DCMAKE_MAKE_PROGRAM=${TEST_MAKE_PROGRAM}")
endif()
if(DEFINED TEST_CXX_FLAGS)
    list(APPEND configure_arguments "-DCMAKE_CXX_FLAGS=${TEST_CXX_FLAGS}")
endif()
if(DEFINED TEST_EXE_LINKER_FLAGS)
    list(APPEND configure_arguments "-DCMAKE_EXE_LINKER_FLAGS=${TEST_EXE_LINKER_FLAGS}")
endif()
if(TEST_CONFIG)
    list(APPEND configure_arguments "-DCMAKE_BUILD_TYPE=${TEST_CONFIG}")
endif()

set(component_build_command "${CMAKE_COMMAND}" --build "${COMPONENT_BINARY_DIR}")
if(TEST_CONFIG)
    list(APPEND component_build_command --config "${TEST_CONFIG}")
endif()
execute_process(
    COMMAND ${component_build_command}
    RESULT_VARIABLE component_build_result
)
if(NOT component_build_result EQUAL 0)
    message(FATAL_ERROR "ROCHostNumerics package artifacts failed to build.")
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${COMPONENT_BINARY_DIR}" --prefix "${install_dir}"
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "ROCHostNumerics package installation failed.")
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        -S "${PACKAGE_SOURCE_DIR}"
        -B "${build_dir}"
        -G "${TEST_GENERATOR}"
        "-DCMAKE_PREFIX_PATH=${install_dir}"
        ${configure_arguments}
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "ROCHostNumerics package consumer configuration failed.")
endif()

set(
    build_command
    "${CMAKE_COMMAND}" --build "${build_dir}"
    --target host-numerics-amd-gpu-layout-package-test
)
if(TEST_CONFIG)
    list(APPEND build_command --config "${TEST_CONFIG}")
endif()
execute_process(COMMAND ${build_command} RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "ROCHostNumerics AMD GPU layout package consumer build failed.")
endif()

set(
    test_command
    "${CTEST_COMMAND}" --test-dir "${build_dir}" --output-on-failure
    --tests-regex host-numerics-amd-gpu-layout-package-consumer
)
if(TEST_CONFIG)
    list(APPEND test_command --build-config "${TEST_CONFIG}")
endif()
execute_process(COMMAND ${test_command} RESULT_VARIABLE test_result)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "ROCHostNumerics AMD GPU layout package consumer test failed.")
endif()
