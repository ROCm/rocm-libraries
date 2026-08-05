# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidationPackageTest LANGUAGES CXX)

find_package(ROCHostValidation CONFIG REQUIRED)

add_executable(host-validation-package-test main.cpp)
target_link_libraries(host-validation-package-test PRIVATE roc::host-validation-core)

add_executable(host-validation-operations-package-test validation.cpp)
target_link_libraries(host-validation-operations-package-test PRIVATE roc::host-validation)

add_executable(host-validation-tiled-package-test tiled.cpp)
target_link_libraries(host-validation-tiled-package-test PRIVATE roc::host-validation-tiled)

if(TARGET roc::host-validation-blas)
    add_executable(host-validation-blas-package-test blas.cpp)
    target_link_libraries(
        host-validation-blas-package-test
        PRIVATE
            roc::host-validation-blas
    )
endif()
