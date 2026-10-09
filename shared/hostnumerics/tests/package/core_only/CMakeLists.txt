# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsCorePackageTest LANGUAGES CXX)

find_package(HostNumerics REQUIRED CONFIG COMPONENTS Core)

foreach(unexpected_target
        roc::hostnumerics
        roc::hostnumerics-blas
        roc::hostnumerics-amd-gpu-layout)
    if(TARGET "${unexpected_target}")
        message(FATAL_ERROR
            "Core-only lookup unexpectedly loaded ${unexpected_target}."
        )
    endif()
endforeach()

add_executable(hostnumerics-core-package-test main.cpp)
target_link_libraries(
    hostnumerics-core-package-test
    PRIVATE
        roc::hostnumerics-core
)
