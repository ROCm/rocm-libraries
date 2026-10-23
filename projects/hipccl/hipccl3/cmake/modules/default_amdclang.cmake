# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################
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
