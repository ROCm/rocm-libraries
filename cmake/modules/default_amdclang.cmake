# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Default to amdclang / amdclang++ for standalone builds when no compiler is
# specified.  Must be include()'d BEFORE the project() call.

if(NOT DEFINED ENV{ROCM_PATH})
    set(_ROCM_PATH "/opt/rocm")
else()
    set(_ROCM_PATH "$ENV{ROCM_PATH}")
endif()

# --- C++ compiler ---
if(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    set(_AMDCLANGXX "${_ROCM_PATH}/lib/llvm/bin/amdclang++")
    if(WIN32)
        set(_AMDCLANGXX "${_AMDCLANGXX}.exe")
    endif()
    if(EXISTS "${_AMDCLANGXX}")
        set(CMAKE_CXX_COMPILER "${_AMDCLANGXX}")
        message(STATUS "Using amdclang++ from ROCm: ${_AMDCLANGXX}")
    else()
        message(STATUS
            "amdclang++ not found at ${_AMDCLANGXX}; using default C++ compiler "
            "(set CXX/CMAKE_CXX_COMPILER or ROCM_PATH to override)")
    endif()
endif()

# --- C compiler ---
if(NOT CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    set(_AMDCLANG "${_ROCM_PATH}/lib/llvm/bin/amdclang")
    if(WIN32)
        set(_AMDCLANG "${_AMDCLANG}.exe")
    endif()
    if(EXISTS "${_AMDCLANG}")
        set(CMAKE_C_COMPILER "${_AMDCLANG}")
        message(STATUS "Using amdclang from ROCm: ${_AMDCLANG}")
    else()
        message(STATUS
            "amdclang not found at ${_AMDCLANG}; using default C compiler "
            "(set CC/CMAKE_C_COMPILER or ROCM_PATH to override)")
    endif()
endif()

unset(_ROCM_PATH)
unset(_AMDCLANGXX)
unset(_AMDCLANG)
