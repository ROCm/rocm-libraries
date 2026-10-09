# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsSourceBuild LANGUAGES CXX)

if(NOT DEFINED HostNumerics_SOURCE_DIR)
    message(FATAL_ERROR "HostNumerics_SOURCE_DIR is required.")
endif()

set(HOSTNUMERICS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HOSTNUMERICS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(HOSTNUMERICS_BUILD_BLAS_BACKEND ON CACHE BOOL "" FORCE)
list(PREPEND CMAKE_MODULE_PATH "${HostNumerics_SOURCE_DIR}/cmake")
add_subdirectory("${HostNumerics_SOURCE_DIR}" hostnumerics)

foreach(required_target
        roc::hostnumerics-core
        roc::hostnumerics
        roc::hostnumerics-amd-gpu-layout)
    if(NOT TARGET "${required_target}")
        message(FATAL_ERROR
            "Source build did not define required target ${required_target}."
        )
    endif()
endforeach()

if(NOT TARGET roc::hostnumerics-blas)
    message(FATAL_ERROR
        "Source build did not define roc::hostnumerics-blas."
    )
endif()
