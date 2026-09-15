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
    if(NOT EXISTS "${_AMDCLANGXX}")
        message(FATAL_ERROR
            "amdclang++ not found at ${_AMDCLANGXX}\n"
            "Either install the ROCm SDK, set ROCM_PATH (e.g. ROCM_PATH=$(rocm-sdk path --root)),"
            " or set CXX/CMAKE_CXX_COMPILER to your compiler.")
    endif()
    set(CMAKE_CXX_COMPILER "${_AMDCLANGXX}")
endif()

# --- C compiler ---
if(NOT CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    set(_AMDCLANG "${_ROCM_PATH}/lib/llvm/bin/amdclang")
    if(WIN32)
        set(_AMDCLANG "${_AMDCLANG}.exe")
    endif()
    if(NOT EXISTS "${_AMDCLANG}")
        message(FATAL_ERROR
            "amdclang not found at ${_AMDCLANG}\n"
            "Either install the ROCm SDK, set ROCM_PATH (e.g. ROCM_PATH=$(rocm-sdk path --root)),"
            " or set CC/CMAKE_C_COMPILER to your compiler.")
    endif()
    set(CMAKE_C_COMPILER "${_AMDCLANG}")
endif()

unset(_ROCM_PATH)
unset(_AMDCLANGXX)
unset(_AMDCLANG)
