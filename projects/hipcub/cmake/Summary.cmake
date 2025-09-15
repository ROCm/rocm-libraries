# MIT License
#
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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
    find_package(Git)
    if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} show --format=%H --no-patch
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} show --format=%s --no-patch
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE COMMIT_SUBJECT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    endif()

    execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERBOSE_DETAILS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    find_program(UNAME_EXECUTABLE uname)
    if(UNAME_EXECUTABLE)
    execute_process(
        COMMAND ${UNAME_EXECUTABLE} -a
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE LINUX_KERNEL_DETAILS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    endif()

    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  HIP ROOT              : ${HIP_ROOT_DIR}")
if(USE_HIPCXX)
    message(STATUS "  HIP compiler          : ${CMAKE_HIP_COMPILER}")
    message(STATUS "  HIP compiler version  : ${CMAKE_HIP_COMPILER_VERSION}")
    string(STRIP "${CMAKE_HIP_FLAGS}" CMAKE_HIP_FLAGS_STRIP)
    message(STATUS "  HIP flags             : ${CMAKE_HIP_FLAGS_STRIP}")
else()
    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
    string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS_STRIP)
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS_STRIP}")
endif()
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
if(USE_HIPCXX)
    message(STATUS "  Device targets        : ${CMAKE_HIP_ARCHITECTURES}")
elseif(HIP_COMPILER STREQUAL "clang")
    message(STATUS "  Device targets        : ${AMDGPU_TARGETS}")
else()
    message(STATUS "  Device targets        : ${NVGPU_TARGETS}")
endif()
    message(STATUS "")
    message(STATUS "  EXTERNAL_DEPS_FORCE_DOWNLOAD: ${EXTERNAL_DEPS_FORCE_DOWNLOAD}")
    message(STATUS "  ROCPRIM_FETCH_METHOD        : ${ROCPRIM_FETCH_METHOD}")
    message(STATUS "  BUILD_TEST                  : ${BUILD_TEST}")
if(BUILD_TEST)
    message(STATUS "  CODE_COVERAGE               : ${CODE_COVERAGE}")
endif()
    message(STATUS "  BUILD_BENCHMARK             : ${BUILD_BENCHMARK}")
    message(STATUS "  BUILD_ADDRESS_SANITIZER     : ${BUILD_ADDRESS_SANITIZER}")
    message(STATUS "  BUILD_OFFLOAD_COMPRESS      : ${BUILD_OFFLOAD_COMPRESS}")
    message(STATUS "  USE_SYSTEM_LIB              : ${USE_SYSTEM_LIB}")     
    message(STATUS "")
    message(STATUS "Detailed:")
    message(STATUS "  C++ compiler details  : \n${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")
if(GIT_FOUND)
    message(STATUS "  Commit                : ${COMMIT_HASH}")
    message(STATUS "                          ${COMMIT_SUBJECT}")
endif()
if(UNAME_EXECUTABLE)
    message(STATUS "  Unix name             : ${LINUX_KERNEL_DETAILS}")
endif()
endfunction()
