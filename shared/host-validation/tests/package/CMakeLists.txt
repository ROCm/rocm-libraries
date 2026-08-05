# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidationPackageTest LANGUAGES CXX)

find_package(ROCHostValidation CONFIG REQUIRED)

add_executable(host-validation-package-test main.cpp)
target_link_libraries(host-validation-package-test PRIVATE roc::host-validation-core)

add_executable(host-validation-operations-package-test validation.cpp)
target_link_libraries(host-validation-operations-package-test PRIVATE roc::host-validation)
