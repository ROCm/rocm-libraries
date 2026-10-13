# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Match the component and monorepo minimum: CMake 3.25.2, released on 2023-01-19.
cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsPackageTest LANGUAGES CXX)

find_package(HostNumerics CONFIG REQUIRED)

option(HOSTNUMERICS_ENABLE_FETCH "Allow test builds to fetch missing dependencies." ON)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/HostNumericsTestDependencies.cmake")
add_executable(hostnumerics-package-test ../version_test.cpp)
target_link_libraries(hostnumerics-package-test PRIVATE roc::hostnumerics GTest::gtest_main)

enable_testing()
add_test(NAME hostnumerics-package-test COMMAND hostnumerics-package-test)
