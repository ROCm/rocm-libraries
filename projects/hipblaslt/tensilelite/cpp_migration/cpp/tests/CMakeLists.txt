# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Standalone gtest target for the tensile_writer C++ migration headers.
#
# The subtile geometry / tile_info / emit_leaves headers are pure integer/float
# math with no nanobind, Python, HIP, or rocisa dependency, so this test build
# is fully independent of the nanobind extension build (../../CMakeLists.txt)
# and of the heavy hipblaslt build. It needs only a C++20 compiler and GTest
# (libgtest-dev is pre-installed in hipblaslt-dev:latest).
#
# Build + run inside the container:
#   cd /workspace/rocm-libraries/projects/hipblaslt/tensilelite/cpp_migration
#   cmake -S cpp/tests -B build_tests -G Ninja
#   cmake --build build_tests
#   ctest --test-dir build_tests --output-on-failure

cmake_minimum_required(VERSION 3.20)

project(tensile_writer_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTest REQUIRED)
include(GoogleTest)
enable_testing()

add_executable(tensile_writer_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/subtile_geometry_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tile_info_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/emit_leaves_test.cpp"
)

target_include_directories(tensile_writer_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

target_link_libraries(tensile_writer_tests PRIVATE GTest::gtest GTest::gtest_main)

gtest_discover_tests(tensile_writer_tests)
