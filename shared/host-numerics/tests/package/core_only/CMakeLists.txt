# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostNumericsCorePackageTest LANGUAGES CXX)

find_package(ROCHostNumerics REQUIRED CONFIG COMPONENTS Core)

foreach(unexpected_target
        roc::host-numerics
        roc::host-numerics-blas
        roc::host-numerics-amd-gpu-layout)
    if(TARGET "${unexpected_target}")
        message(FATAL_ERROR
            "Core-only lookup unexpectedly loaded ${unexpected_target}."
        )
    endif()
endforeach()

add_executable(host-numerics-core-package-test main.cpp)
target_link_libraries(
    host-numerics-core-package-test
    PRIVATE
        roc::host-numerics-core
)
