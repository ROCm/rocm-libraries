# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostNumericsSourceBuild LANGUAGES CXX)

if(NOT DEFINED ROCHostNumerics_SOURCE_DIR)
    message(FATAL_ERROR "ROCHostNumerics_SOURCE_DIR is required.")
endif()

set(HOST_NUMERICS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HOST_NUMERICS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(HOST_NUMERICS_BUILD_BLAS_BACKEND ON CACHE BOOL "" FORCE)
list(PREPEND CMAKE_MODULE_PATH "${ROCHostNumerics_SOURCE_DIR}/cmake")
add_subdirectory("${ROCHostNumerics_SOURCE_DIR}" host-numerics)

foreach(required_target
        roc::host-numerics-core
        roc::host-numerics
        roc::host-numerics-amd-gpu-layout)
    if(NOT TARGET "${required_target}")
        message(FATAL_ERROR
            "Source build did not define required target ${required_target}."
        )
    endif()
endforeach()

if(NOT TARGET roc::host-numerics-blas)
    message(FATAL_ERROR
        "Source build did not define roc::host-numerics-blas."
    )
endif()
