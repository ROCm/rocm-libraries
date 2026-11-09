# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Match the component and monorepo minimum: CMake 3.25.2, released on 2023-01-19.
cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsPackageTest LANGUAGES CXX)

find_package(HostNumerics CONFIG REQUIRED)

if(NOT HOSTNUMERICS_SOURCE_DIR)
    message(FATAL_ERROR "Set HOSTNUMERICS_SOURCE_DIR to the HostNumerics source tree.")
endif()

list(APPEND CMAKE_MODULE_PATH "${HOSTNUMERICS_SOURCE_DIR}/cmake")
include(HostNumericsTestDependencies)
add_executable(hostnumerics-package-test "${HOSTNUMERICS_SOURCE_DIR}/tests/core_test.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/tests/expected_test.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/tests/integer_conversion_test.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/tests/scalar_storage_test.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/tests/reshape_permutation_test.cpp")
target_link_libraries(hostnumerics-package-test PRIVATE roc::hostnumerics GTest::gtest_main)
add_executable(hostnumerics-amd-gpu-layout-package-test
    "${HOSTNUMERICS_SOURCE_DIR}/tests/amd_gpu_mx_layout_test.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/tests/amd_gpu_operand_layout_test.cpp")
target_link_libraries(hostnumerics-amd-gpu-layout-package-test
    PRIVATE roc::hostnumerics-amd-gpu-layout GTest::gtest_main)

enable_testing()
include(GoogleTest)
gtest_discover_tests(
    hostnumerics-package-test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-package-test."
)
gtest_discover_tests(
    hostnumerics-amd-gpu-layout-package-test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-amd-gpu-layout-package-test."
)
