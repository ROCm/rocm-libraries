# MIT License
#
# Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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
# rocPRIM dependencies
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

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included (when not using HIP-CPU).

include(FetchContent)

if(USE_HIP_CPU)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(hip_cpu_rt QUIET)
  endif()
  if(NOT TARGET hip_cpu_rt::hip_cpu_rt)
    message(STATUS "HIP-CPU runtime not found. Fetching...")
    FetchContent_Declare(
      hip-cpu
      GIT_REPOSITORY https://github.com/ROCm-Developer-Tools/HIP-CPU.git
      GIT_TAG        56f559c93be210bb300dad3673c06d2bb0119d13 # master@2022.07.01
    )
    FetchContent_MakeAvailable(hip-cpu)
    if(NOT TARGET hip_cpu_rt::hip_cpu_rt)
      add_library(hip_cpu_rt::hip_cpu_rt ALIAS hip_cpu_rt)
    endif()
  else()
    find_package(hip_cpu_rt REQUIRED)
    # If we found HIP-CPU as binary, search for transitive dependencies
    find_package(Threads REQUIRED)
    set(CMAKE_REQUIRED_FLAGS "-std=c++17")
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(__GLIBCXX__ "cstddef" STL_IS_GLIBCXX)
    set(STL_DEPENDS_ON_TBB ${STL_IS_GLIBCXX})
    if(STL_DEPENDS_ON_TBB)
      find_package(TBB QUIET)
      if(NOT TARGET TBB::tbb AND NOT TARGET tbb)
        message(STATUS "Thread Building Blocks not found. Fetching...")
        FetchContent_Declare(
          thread-building-blocks
          GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
          GIT_TAG        3df08fe234f23e732a122809b40eb129ae22733f # v2021.5.0
        )
        FetchContent_MakeAvailable(thread-building-blocks)
      else()
        find_package(TBB REQUIRED)
      endif()
    endif(STL_DEPENDS_ON_TBB)
  endif()
endif(USE_HIP_CPU)

# Test dependencies
if(BUILD_TEST)
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
        GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
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
endif(BUILD_TEST)

if(BUILD_BENCHMARK)
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
      GIT_TAG        d17ea665515f0c54d100c6fc973632431379f64b # v1.6.1
    )
    set(HAVE_STD_REGEX ON)
    set(RUN_HAVE_STD_REGEX 1)
    FetchContent_MakeAvailable(googlebench)
    if(NOT TARGET benchmark::benchmark)
      add_library(benchmark::benchmark ALIAS benchmark)
    endif()
  else()
    find_package(benchmark CONFIG REQUIRED)
  endif()
endif(BUILD_BENCHMARK)

if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
  set(CMAKE_FIND_DEBUG_MODE TRUE)
  find_package(ROCM 0.7.3 CONFIG QUIET PATHS /opt/rocm)
  set(CMAKE_FIND_DEBUG_MODE FALSE)
endif()
if(NOT ROCM_FOUND)
  if(NOT EXISTS "${FETCHCONTENT_BASE_DIR}/rocm-cmake-src")
    message(STATUS "ROCm CMake not found. Fetching...")
    set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
    FetchContent_Declare(
      rocm-cmake
      URL  https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.tar.gz
    )
    FetchContent_MakeAvailable(rocm-cmake)
  endif()
  find_package(ROCM CONFIG REQUIRED NO_DEFAULT_PATH HINTS "${rocm-cmake_SOURCE_DIR}")
else()
  find_package(ROCM 0.7.3 CONFIG REQUIRED PATHS /opt/rocm)
endif()

# Restore user global state
set(CMAKE_CXX_FLAGS ${USER_CXX_FLAGS})
if(DEFINED USER_BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ${USER_BUILD_SHARED_LIBS})
else()
  unset(BUILD_SHARED_LIBS CACHE )
endif()
set(ROCM_WARN_TOOLCHAIN_VAR ${USER_ROCM_WARN_TOOLCHAIN_VAR} CACHE BOOL "")

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMHeaderWrapper)
include(ROCMCheckTargetIds)
include(ROCMClients)
