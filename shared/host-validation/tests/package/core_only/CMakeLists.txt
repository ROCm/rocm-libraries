# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidationCorePackageTest LANGUAGES CXX)

find_package(ROCHostValidation REQUIRED CONFIG COMPONENTS Core)

foreach(unexpected_target
        roc::host-validation
        roc::host-validation-tiled
        roc::host-validation-blas
        roc::host-validation-mx
        roc::host-validation-amd-gpu-layout)
    if(TARGET "${unexpected_target}")
        message(FATAL_ERROR
            "Core-only lookup unexpectedly loaded ${unexpected_target}."
        )
    endif()
endforeach()

add_executable(host-validation-core-package-test main.cpp)
target_link_libraries(
    host-validation-core-package-test
    PRIVATE
        roc::host-validation-core
)
