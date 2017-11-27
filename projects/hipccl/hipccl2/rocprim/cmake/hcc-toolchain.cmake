# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

# Find HIP package
find_package(HIP REQUIRED)

# Select toolchain
if(HIP_PLATFORM STREQUAL "nvcc")
    message(FATAL_ERROR "CUDA platform is not supported.")
elseif(HIP_PLATFORM STREQUAL "hcc")
    # Find HCC executable
    find_program(
        HIP_HCC_EXECUTABLE
        NAMES hcc
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/hip
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
    )
    if(NOT HIP_HCC_EXECUTABLE)
        # Now search in default paths
        find_program(HIP_HCC_EXECUTABLE hcc)
    endif()
    mark_as_advanced(HIP_HCC_EXECUTABLE)
    set(CMAKE_CXX_COMPILER ${HIP_HCC_EXECUTABLE})
else()
    message(FATAL_ERROR "HIP_PLATFORM must be 'hcc' (AMD ROCm platform).")
endif()
