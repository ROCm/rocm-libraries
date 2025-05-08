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

# ###########################
# hipCUB dependencies
# ###########################

# NOTE1: the reason we don't scope global state meddling using add_subdirectory
#        is because CMake < 3.24 lacks CMAKE_FIND_PACKAGE_TARGETS_GLOBAL which
#        would promote IMPORTED targets of find_package(CONFIG) to be visible
#        by other parts of the build. So we save and restore global state.
#
# NOTE2: We disable the ROCMChecks.cmake warning noting that we meddle with
#        global state. This is consequence of abusing the CMake CXX language
#        which HIP piggybacks on top of. This kind of HIP support has one chance
#        at observing the global flags, at the find_package(HIP) invocation.
#        The device compiler won't be able to pick up changes after that, hence
#        the warning.
#
# NOTE3: hipCUB and rocPRIM share CMake options for building tests, benchmarks
#        and examples. Until that's not fixed, we have to save/restore them.
set(USER_CXX_FLAGS ${CMAKE_CXX_FLAGS})
if(DEFINED BUILD_SHARED_LIBS)
  set(USER_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
endif()
set(USER_ROCM_WARN_TOOLCHAIN_VAR ${ROCM_WARN_TOOLCHAIN_VAR})

set(ROCM_WARN_TOOLCHAIN_VAR OFF CACHE BOOL "")
# Turn off warnings and errors for all warnings in dependencies
separate_arguments(CXX_FLAGS_LIST NATIVE_COMMAND ${CMAKE_CXX_FLAGS})
list(REMOVE_ITEM CXX_FLAGS_LIST /WX -Werror -Werror=pendantic -pedantic-errors)
if(MSVC)
  list(FILTER CXX_FLAGS_LIST EXCLUDE REGEX "/[Ww]([0-4]?)(all)?") # Remove MSVC warning flags
  list(APPEND CXX_FLAGS_LIST /w)
else()
  list(FILTER CXX_FLAGS_LIST EXCLUDE REGEX "-W(all|extra|everything)") # Remove GCC/LLVM flags
  list(APPEND CXX_FLAGS_LIST -w)
endif()
list(JOIN CXX_FLAGS_LIST " " CMAKE_CXX_FLAGS)
# Don't build client dependencies as shared
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Global flag to cause add_library() to create shared libraries if on." FORCE)

foreach(SHARED_OPTION BUILD_TEST BUILD_BENCHMARK BUILD_EXAMPLE)
  set(USER_${SHARED_OPTION} ${${SHARED_OPTION}})
  set(${SHARED_OPTION} OFF)
endforeach()

include(FetchContent)

# Test dependencies
if(USER_BUILD_TEST)
  # NOTE1: Google Test has created a mess with legacy FindGTest.cmake and newer GTestConfig.cmake
  #
  # FindGTest.cmake defines:   GTest::GTest, GTest::Main, GTEST_FOUND
  #
  # GTestConfig.cmake defines: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
  #
  # NOTE2: Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode, because targets
  #        will be duplicately defined.
  #
  # NOTE3: The following snippet first tries to find Google Test binary either in MODULE or CONFIG modes.
  #        If neither succeeds it goes on to import Google Test into this build either from a system
  #        source package (apt install googletest on Ubuntu 18.04 only) or GitHub and defines the MODULE
  #        mode targets. Otherwise if MODULE or CONFIG succeeded, then it prints the result to the
  #        console via a non-QUIET find_package call and if CONFIG succeeded, creates ALIAS targets
  #        with the MODULE IMPORTED names.
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(GTest QUIET)
  endif()
  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    option(BUILD_GTEST "Builds the googletest subproject" ON)
    option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
    option(INSTALL_GTEST "Enable installation of googletest." OFF)
    if(EXISTS /usr/src/googletest AND NOT DEPENDENCIES_FORCE_DOWNLOAD)
      FetchContent_Declare(
        googletest
        SOURCE_DIR /usr/src/googletest
      )
    else()
      message(STATUS "Google Test not found. Fetching...")
      FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.11.0
      )
    endif()
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main  ALIAS gtest_main)
  else()
    find_package(GTest REQUIRED)
    if(TARGET GTest::gtest_main AND NOT TARGET GTest::Main)
      add_library(GTest::GTest ALIAS GTest::gtest)
      add_library(GTest::Main  ALIAS GTest::gtest_main)
    endif()
  endif()
endif(USER_BUILD_TEST)

if(USER_BUILD_BENCHMARK)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(benchmark CONFIG QUIET)
  endif()
  if(NOT TARGET benchmark::benchmark)
    message(STATUS "Google Benchmark not found. Fetching...")
    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
    option(BENCHMARK_ENABLE_INSTALL "Enable installation of benchmark." OFF)
    FetchContent_Declare(
      googlebench
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.8.0
    )
    FetchContent_MakeAvailable(googlebench)
    if(NOT TARGET benchmark::benchmark)
      add_library(benchmark::benchmark ALIAS benchmark)
    endif()
  else()
    find_package(benchmark CONFIG REQUIRED)
  endif()
endif(USER_BUILD_BENCHMARK)

# CUB (only for CUDA platform)
if(HIP_COMPILER STREQUAL "nvcc")
  set(CCCL_MINIMUM_VERSION 2.6.0)
  if(NOT DOWNLOAD_CUB)
    find_package(CUB ${CCCL_MINIMUM_VERSION} CONFIG)
    find_package(Thrust ${CCCL_MINIMUM_VERSION} CONFIG)
    find_package(libcudacxx ${CCCL_MINIMUM_VERSION} CONFIG)
  endif()

  if (NOT CUB_FOUND OR NOT Thrust_FOUND OR NOT libcudacxx_FOUND)
    if(CUB_FOUND OR Thrust_FOUND OR libcudacxx_FOUND)
      message(WARNING "Found one of CUB, Thrust or libcu++, but not all of them.
                       This can lead to mixing different potentially incompatible versions.")
    endif()

    message(STATUS "CUB, Thrust or libcu++ not found, downloading and extracting CCCL ${CCCL_MINIMUM_VERSION}")
    file(DOWNLOAD https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_MINIMUM_VERSION}.zip
                  ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip
         STATUS cccl_download_status LOG cccl_download_log)

    list(GET cccl_download_status 0 cccl_download_error_code)
    if(cccl_download_error_code)
      message(FATAL_ERROR "Error: downloading "
              "https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_MINIMUM_VERSION}.zip failed "
              "error_code: ${cccl_download_error_code} "
              "log: ${cccl_download_log}")
    endif()

    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip)
    else()
      execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      RESULT_VARIABLE cccl_unpack_error_code)
      if(cccl_unpack_error_code)
        message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip failed")
      endif()
    endif()

    find_package(CUB ${CCCL_MINIMUM_VERSION} CONFIG REQUIRED NO_DEFAULT_PATH
                 PATHS ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}/cub)
    find_package(Thrust ${CCCL_MINIMUM_VERSION} CONFIG REQUIRED NO_DEFAULT_PATH
                 PATHS ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}/thrust)
    find_package(libcudacxx ${CCCL_MINIMUM_VERSION} CONFIG REQUIRED NO_DEFAULT_PATH
                 PATHS ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}/libcudacxx)
  endif()
else()
  # rocPRIM (only for ROCm platform)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Add default install location for WIN32 and non-WIN32 as hint
    find_package(rocprim CONFIG QUIET PATHS "${ROCM_ROOT}/lib/cmake/rocprim")
  endif()
  if(NOT TARGET roc::rocprim)
    message(STATUS "rocPRIM not found. Fetching...")
    FetchContent_Declare(
            prim
            GIT_REPOSITORY https://github.com/ROCm/rocPRIM.git
            GIT_TAG        develop
    )
    FetchContent_MakeAvailable(prim)
    if(NOT TARGET roc::rocprim)
      add_library(roc::rocprim ALIAS rocprim)
    endif()
    if(NOT TARGET roc::rocprim_hip)
      add_library(roc::rocprim_hip ALIAS rocprim_hip)
    endif()
  else()
    find_package(rocprim CONFIG REQUIRED)
  endif()
endif()

foreach(SHARED_OPTION BUILD_TEST BUILD_BENCHMARK BUILD_EXAMPLE)
  set(${SHARED_OPTION} ${USER_${SHARED_OPTION}})
endforeach()

# Restore user global state
set(CMAKE_CXX_FLAGS ${USER_CXX_FLAGS})
if(DEFINED USER_BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ${USER_BUILD_SHARED_LIBS})
else()
  unset(BUILD_SHARED_LIBS CACHE )
endif()
set(ROCM_WARN_TOOLCHAIN_VAR ${USER_ROCM_WARN_TOOLCHAIN_VAR} CACHE BOOL "")
