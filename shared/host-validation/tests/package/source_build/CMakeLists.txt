# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidationSourceBuildCompatibility LANGUAGES CXX)

if(NOT DEFINED ROCHostValidation_SOURCE_DIR)
    message(FATAL_ERROR "ROCHostValidation_SOURCE_DIR is required.")
endif()

set(HOST_VALIDATION_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HOST_VALIDATION_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(HOST_VALIDATION_BUILD_BLAS_BACKEND OFF CACHE BOOL "" FORCE)
add_subdirectory("${ROCHostValidation_SOURCE_DIR}" host-validation)

foreach(required_target
        roc::host-validation-core
        roc::host-validation
        roc::host-validation-tiled
        roc::host-validation-mx
        roc::host-validation-amd-gpu-layout)
    if(NOT TARGET "${required_target}")
        message(FATAL_ERROR
            "Source build did not define required target ${required_target}."
        )
    endif()
endforeach()

if(NOT COMMAND roc_host_validation_enable_mx_backend)
    message(FATAL_ERROR
        "roc_host_validation_enable_mx_backend is not available."
    )
endif()
roc_host_validation_enable_mx_backend()

if(NOT COMMAND roc_host_validation_enable_blas_backend)
    message(FATAL_ERROR
        "roc_host_validation_enable_blas_backend is not available."
    )
endif()
find_package(BLAS QUIET)
if(BLAS_FOUND)
    roc_host_validation_enable_blas_backend()
    if(NOT TARGET roc::host-validation-blas)
        message(FATAL_ERROR
            "roc_host_validation_enable_blas_backend did not define its target."
        )
    endif()
endif()
