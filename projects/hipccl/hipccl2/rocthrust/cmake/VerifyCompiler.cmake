# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

# Find HIP package and verify that correct C++ compiler was selected for available
# platfrom. On ROCm platform host and device code is compiled by the same compiler:
# hcc. On CUDA host can be compiled by any C++ compiler while device code is compiled
# by nvcc compiler (CMake's CUDA package handles this).

# Find HIP package
find_package(HIP REQUIRED)

if(HIP_PLATFORM STREQUAL "nvcc")
  message(FATAL_ERROR "`nvcc` is not supported.")
elseif(HIP_PLATFORM STREQUAL "hcc")
  if(NOT (CMAKE_CXX_COMPILER MATCHES ".*/hcc$"))
    message(FATAL_ERROR "On ROCm platform 'hcc' must be used as C++ compiler.")
  else()
    # Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
    # (Thanks to rocBLAS devs for finding workaround for this problem.)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip)
    # Ignore hcc warning: argument unused during compilation: '-isystem /opt/rocm/hip/include'
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
    find_package(hcc REQUIRED CONFIG PATHS /opt/rocm)
    find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
  endif()
else()
  message(FATAL_ERROR "HIP_PLATFORM must be 'hcc' (AMD ROCm platform).")
endif()
