# MIT License
#
# Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

function (print_configuration_summary)
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

    string(REPLACE "\n" ";" CMAKE_CXX_COMPILER_VERBOSE_DETAILS "${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")
    list(TRANSFORM CMAKE_CXX_COMPILER_VERBOSE_DETAILS PREPEND "--     ")
    string(REPLACE ";" "\n" CMAKE_CXX_COMPILER_VERBOSE_DETAILS "${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")

    # Joins CMAKE_CXX_FLAGS and COMPILE_OPTIONS
    string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS_STRIP)
    string(REPLACE " " ";" CMAKE_CXX_FLAGS_AND_OPTIONS_LIST "${CMAKE_CXX_FLAGS_STRIP}")
    list(APPEND CMAKE_CXX_FLAGS_AND_OPTIONS_LIST "${COMPILE_OPTIONS}")
    list(JOIN CMAKE_CXX_FLAGS_AND_OPTIONS_LIST " " CMAKE_CXX_FLAGS_AND_OPTIONS)

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
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS_AND_OPTIONS}")
endif()
    message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Install prefix        : ${CMAKE_INSTALL_PREFIX}")
if(HIP_COMPILER STREQUAL "clang")
if(USE_HIPCXX)
    message(STATUS "  Device targets        : ${CMAKE_HIP_ARCHITECTURES}")
else()
    message(STATUS "  Device targets        : ${GPU_TARGETS}")
endif()
endif()
    message(STATUS "")
    message(STATUS "  DISABLE_WERROR                  : ${DISABLE_WERROR}")
    message(STATUS "  DOWNLOAD_ROCPRIM                : ${DOWNLOAD_ROCPRIM}")
    message(STATUS "  BUILD_TEST                      : ${BUILD_TEST}")
    message(STATUS "  BUILD_HIPSTDPAR_TEST            : ${BUILD_HIPSTDPAR_TEST}")
    message(STATUS "  BUILD_HIPSTDPAR_TEST_WITH_TBB   : ${BUILD_HIPSTDPAR_TEST_WITH_TBB}")
    message(STATUS "  BUILD_EXAMPLES                  : ${BUILD_EXAMPLES}")
    message(STATUS "  BUILD_BENCHMARKS                : ${BUILD_BENCHMARKS}")
if(BUILD_BENCHMARKS)
    message(STATUS "  DOWNLOAD_ROCRAND                : ${DOWNLOAD_ROCRAND}")
endif()
    message(STATUS "  BUILD_ADDRESS_SANITIZER         : ${BUILD_ADDRESS_SANITIZER}")
    message(STATUS "")
    message(STATUS "Detailed:")
    message(STATUS "  C++ compiler details            : \n${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")
if(GIT_FOUND)
    message(STATUS "  Commit                          : ${COMMIT_HASH}")
    message(STATUS "                                    ${COMMIT_SUBJECT}")
endif()
if(UNAME_EXECUTABLE)
    message(STATUS "  Unix name                       : ${LINUX_KERNEL_DETAILS}")
endif()  
endfunction()
