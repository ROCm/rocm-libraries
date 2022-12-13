# MIT License
#
# Copyright (c) 2018-2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
if(CMAKE_CXX_COMPILER MATCHES ".*nvcc$" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND CMAKE_MODULE_PATH /opt/rocm/hip/cmake)
    find_package(hip QUIET CONFIG PATHS /opt/rocm)
    if(NOT hip_FOUND)
        find_package(HIP REQUIRED)
    endif()
    if(HIP_COMPILER STREQUAL "clang")
       # TODO: The HIP package on NVIDIA platform is incorrect at few versions
       set(HIP_COMPILER "nvcc" CACHE STRING "HIP Compiler" FORCE)
    endif()
else()
  find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
endif()

if(HIP_COMPILER STREQUAL "nvcc")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        include(SetupNVCC)
    else()
        message(WARNING "On CUDA platform 'g++' is recommended C++ compiler.")
    endif()
elseif(HIP_COMPILER STREQUAL "clang")
    if(NOT (HIP_CXX_COMPILER MATCHES ".*hipcc" OR HIP_CXX_COMPILER MATCHES ".*clang\\+\\+"))
        message(FATAL_ERROR "On ROCm platform 'hipcc' or HIP-aware Clang must be used as C++ compiler.")
    endif()
else()
    message(FATAL_ERROR "HIP_COMPILER must be 'clang' (AMD ROCm platform) or `nvcc` (NVIDIA CUDA platform).")
endif()
