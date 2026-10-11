# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Match the component and monorepo minimum: CMake 3.25.2, released on 2023-01-19.
cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsEmbeddedTest LANGUAGES CXX)

set(CMAKE_DISABLE_FIND_PACKAGE_Python TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_nanobind TRUE)
add_subdirectory(../.. hostnumerics)
if(HOSTNUMERICS_BUILD_TESTING OR HOSTNUMERICS_BUILD_PYTHON)
    message(FATAL_ERROR "Embedded HostNumerics should default to no tests or Python.")
endif()

add_executable(hostnumerics-embedded-test ../version_test.cpp)
target_link_libraries(hostnumerics-embedded-test PRIVATE roc::hostnumerics)

enable_testing()
add_test(NAME hostnumerics-embedded-test COMMAND hostnumerics-embedded-test)
