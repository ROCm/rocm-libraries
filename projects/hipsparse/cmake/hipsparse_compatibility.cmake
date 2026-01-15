# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# hipSPARSE Build Option Compatibility Checks
# This file contains validation logic for incompatible option combinations

# Ensure CUDA and HIP backends are not enabled simultaneously
if(HIPSPARSE_ENABLE_CUDA AND HIPSPARSE_ENABLE_HIP)
    message(FATAL_ERROR "Cannot enable both CUDA and HIP backends simultaneously.")
endif()

# Ensure at least one backend is enabled
if(NOT HIPSPARSE_ENABLE_CUDA AND NOT HIPSPARSE_ENABLE_HIP)
    message(FATAL_ERROR 
        "At least one backend must be enabled.\n"
        "Set HIPSPARSE_ENABLE_CUDA=ON or HIPSPARSE_ENABLE_HIP=ON")
endif()

# Code coverage requires Debug or RelWithDebInfo build type
if(HIPSPARSE_ENABLE_COVERAGE)
    if((NOT CMAKE_BUILD_TYPE STREQUAL "Debug") AND (NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        message(FATAL_ERROR "Code coverage is only supported for CMAKE_BUILD_TYPE=Debug and CMAKE_BUILD_TYPE=RelWithDebInfo.")
    endif()
endif()

# Address sanitizer and Fortran clients are incompatible
if(HIPSPARSE_ENABLE_ASAN AND HIPSPARSE_ENABLE_FORTRAN)
    message(FATAL_ERROR "Cannot enable both address sanitizer and Fortran clients")
endif()
