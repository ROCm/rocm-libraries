# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostNumericsPackageTest LANGUAGES CXX)

find_package(ROCHostNumerics CONFIG REQUIRED)

add_executable(host-numerics-package-test main.cpp)
target_link_libraries(host-numerics-package-test PRIVATE roc::host-numerics)

enable_testing()
add_test(NAME host-numerics-package-test COMMAND host-numerics-package-test)
