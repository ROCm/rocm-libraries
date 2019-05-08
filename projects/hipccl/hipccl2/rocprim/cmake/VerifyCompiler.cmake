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
# platfrom. On ROCm platform host and device code is compiled by the same compiler: hcc.

# Find HIP package
find_package(HIP 1.5.18263 REQUIRED) # 1.5.18263 is HIP version in ROCm 1.8.2

if(HIP_PLATFORM STREQUAL "hcc")
  if(NOT (CMAKE_CXX_COMPILER MATCHES ".*/hcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
    message(FATAL_ERROR "On ROCm platform 'hcc' or 'clang' must be used as C++ compiler.")
  else()
    # Determine if CXX Compiler is hcc, hip-clang or other
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
    string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})
    if(CXX_VERSION_STRING MATCHES "HCC")
        set(HIP_COMPILER "hcc" CACHE STRING "HIP Compiler")
    elseif(CXX_VERSION_STRING MATCHES "clang")
        set(HIP_COMPILER "clang" CACHE STRING "HIP Compiler")
    else()
        message(FATAL_ERROR "CXX Compiler version ${CXX_VERSION_STRING} unsupported.")
    endif()
    message(STATUS "HIP Compiler: " ${HIP_COMPILER})

    # Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
    # (Thanks to rocBLAS devs for finding workaround for this problem.)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip)
    # Ignore hcc warning: argument unused during compilation: '-isystem /opt/rocm/hip/include'
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
    if(HIP_COMPILER STREQUAL "hcc")
      find_package(hcc REQUIRED CONFIG PATHS /opt/rocm)
    else()
      find_package(hcc QUIET CONFIG PATHS /opt/rocm)
    endif()
    find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
  endif()
else()
  message(FATAL_ERROR "HIP_PLATFORM must be 'hcc' (AMD ROCm platform)")
endif()
