# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidationCorePackageTest LANGUAGES CXX)

find_package(ROCHostValidation REQUIRED CONFIG)

add_executable(host-validation-core-package-test main.cpp)
target_link_libraries(
    host-validation-core-package-test
    PRIVATE
        roc::host-validation-core
)
