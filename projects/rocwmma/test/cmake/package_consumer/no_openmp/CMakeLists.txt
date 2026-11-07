# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

# Compile device code without probing for a GPU; the test never launches it.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    if(DEFINED GPU_TARGETS)
        set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}" CACHE STRING "HIP compilation targets")
    else()
        set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "HIP compilation targets")
    endif()
endif()
set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "HIP package compilation targets")

project(rocwmma_package_consumer LANGUAGES CXX HIP)

# An installed rocWMMA consumer must not need either OpenMP package variant.
set(CMAKE_DISABLE_FIND_PACKAGE_OpenMP TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_openmp TRUE)
find_package(rocwmma CONFIG REQUIRED)

add_executable(rocwmma_package_consumer main.hip)
set_target_properties(rocwmma_package_consumer PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)
target_link_libraries(rocwmma_package_consumer PRIVATE roc::rocwmma)

enable_testing()
add_test(NAME rocwmma_package_consumer COMMAND rocwmma_package_consumer)
