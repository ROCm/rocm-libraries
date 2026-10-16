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
add_executable(hostnumerics-package-test "${HOSTNUMERICS_SOURCE_DIR}/tests/version_test.cpp")
target_link_libraries(hostnumerics-package-test PRIVATE roc::hostnumerics GTest::gtest_main)

enable_testing()
add_test(NAME hostnumerics-package-test COMMAND hostnumerics-package-test)

# Internal C++ result contracts cannot be exercised through the Python bindings.
add_executable(hostnumerics-result-tests "${HOSTNUMERICS_SOURCE_DIR}/tests/result_test.cpp")
target_link_libraries(hostnumerics-result-tests PRIVATE roc::hostnumerics-core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(
    hostnumerics-result-tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-result."
)
