# MIT License
#
# Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

function(print_configuration_summary)
    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  HIP ROOT              : ${HIP_ROOT_DIR}")
    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
    string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS_STRIP)
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS_STRIP}")
if(HIP_COMPILER STREQUAL "nvcc")
    string(REPLACE ";" " " HIP_NVCC_FLAGS_STRIP "${HIP_NVCC_FLAGS}")
    string(STRIP "${HIP_NVCC_FLAGS_STRIP}" HIP_NVCC_FLAGS_STRIP)
    string(REPLACE ";" " " HIP_CPP_CONFIG_FLAGS_STRIP "${HIP_CPP_CONFIG_FLAGS}")
    string(STRIP "${HIP_CPP_CONFIG_FLAGS_STRIP}" HIP_CPP_CONFIG_FLAGS_STRIP)
    message(STATUS "  HIP flags             : ${HIP_CPP_CONFIG_FLAGS_STRIP}")
    message(STATUS "  NVCC flags            : ${HIP_NVCC_FLAGS_STRIP}")
endif()
    message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Install prefix        : ${CMAKE_INSTALL_PREFIX}")
if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
    message(STATUS "  Device targets        : ${AMDGPU_TARGETS}")
else()
    message(STATUS "  Device targets        : ${NVGPU_TARGETS}")
endif()
    message(STATUS "")
    message(STATUS "  DOWNLOAD_ROCPRIM      : ${DOWNLOAD_ROCPRIM}")
    message(STATUS "  BUILD_TEST            : ${BUILD_TEST}")
    message(STATUS "  BUILD_BENCHMARK       : ${BUILD_BENCHMARK}")
endfunction()
