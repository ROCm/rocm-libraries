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
find_package(hip REQUIRED CONFIG PATHS /opt/rocm)

if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
    if(NOT (CMAKE_CXX_COMPILER MATCHES ".*/hcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
        message(FATAL_ERROR "On ROCm platform 'hcc' or 'clang' must be used as C++ compiler.")
    elseif(NOT CXX_VERSION_STRING MATCHES "clang")
        list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
    endif()

    if(HIP_COMPILER STREQUAL "hcc")
      list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc)
      find_package(hcc REQUIRED CONFIG PATHS /opt/rocm)
    endif()
else()
    message(FATAL_ERROR "HIP_COMPILER must be 'hcc' or 'clang' (AMD ROCm platform)")
endif()
