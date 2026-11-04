# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Match the component and monorepo minimum: CMake 3.25.2, released on 2023-01-19.
cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsEmbeddedTest LANGUAGES CXX)

if(NOT HOSTNUMERICS_SOURCE_DIR)
    message(FATAL_ERROR "Set HOSTNUMERICS_SOURCE_DIR to the HostNumerics source tree.")
endif()

set(CMAKE_DISABLE_FIND_PACKAGE_Python TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_nanobind TRUE)
add_subdirectory("${HOSTNUMERICS_SOURCE_DIR}" hostnumerics)
get_directory_property(_hostnumerics_test_dependencies_loaded
    DIRECTORY "${HOSTNUMERICS_SOURCE_DIR}" HOSTNUMERICS_TEST_DEPENDENCIES_LOADED)
if(HOSTNUMERICS_BUILD_TESTING OR HOSTNUMERICS_BUILD_PYTHON OR TARGET GTest::gtest
   OR _hostnumerics_test_dependencies_loaded)
    message(FATAL_ERROR "Embedded HostNumerics should default to no tests or Python.")
endif()

list(APPEND CMAKE_MODULE_PATH "${HOSTNUMERICS_SOURCE_DIR}/cmake")
include(HostNumericsTestDependencies)
add_executable(hostnumerics-embedded-test "${HOSTNUMERICS_SOURCE_DIR}/tests/core_test.cpp"
    "${HOSTNUMERICS_SOURCE_DIR}/tests/expected_test.cpp")
target_link_libraries(hostnumerics-embedded-test PRIVATE roc::hostnumerics GTest::gtest_main)

enable_testing()
include(GoogleTest)
gtest_discover_tests(
    hostnumerics-embedded-test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-embedded-test."
)
