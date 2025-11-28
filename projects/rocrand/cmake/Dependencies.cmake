# MIT License
#
# Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Dependencies

# Save global state
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

# Change variables before configuring dependencies
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
# when VerifyCompiler.cmake is included.

# Fortran Wrapper
if(BUILD_FORTRAN_WRAPPER)
    enable_language(Fortran)
endif()

set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)

# Find or download/install rocm-cmake project
if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
  find_package(ROCmCMakeBuildTools 0.7.3 CONFIG QUIET PATHS "${ROCM_ROOT}")
endif()
if(NOT ROCmCMakeBuildTools_FOUND)
  message(STATUS "ROCm CMake not found. Fetching...")
  # We don't really want to consume the build and test targets of ROCm CMake.
  # CMake 3.18 allows omitting them, even though there's a CMakeLists.txt in source root.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(SOURCE_SUBDIR_ARG SOURCE_SUBDIR "DISABLE ADDING TO BUILD")
  else()
    set(SOURCE_SUBDIR_ARG)
  endif()
  include(FetchContent)
  FetchContent_Declare(
    rocm-cmake
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG        rocm-6.4.4
    ${SOURCE_SUBDIR_ARG}
  )
  FetchContent_MakeAvailable(rocm-cmake)
  find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
else()
  find_package(ROCmCMakeBuildTools 0.7.3 CONFIG REQUIRED PATHS "${ROCM_ROOT}")
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)

# For downloading and building required dependencies
include(FetchContent)
# Test dependencies
if(BUILD_TEST)
  # Google Test (https://github.com/google/googletest)
  # NOTE: Google Test has created a mess with legacy FindGTest.cmake and newer GTestConfig.cmake
  #
  # FindGTest.cmake defines:   GTest::GTest, GTest::Main, GTEST_FOUND
  #
  # GTestConfig.cmake defines: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
  #
  # NOTE2: Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode, because targets
  #        will be duplicately defined.
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(GTest QUIET)
  endif()

  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    message(STATUS "Google Test not found or force download on. Fetching...")
    option(BUILD_GTEST "Builds the googletest subproject" ON)
    option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
    option(INSTALL_GTEST "Enable installation of googletest" ON)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.15.2
    )
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  # Google Benchmark (https://github.com/google/benchmark)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(benchmark 1.9.1 QUIET)
  endif()

  if(NOT TARGET benchmark::benchmark)
    message(STATUS "Google Benchmark not found or force download on. Fetching...")
    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library" OFF)
    option(BENCHMARK_ENABLE_INSTALL "Enable installation of benchmark" OFF)
    FetchContent_Declare(
      googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.9.1
    )
    set(HAVE_STD_REGEX ON)
    set(RUN_HAVE_STD_REGEX 1)
    FetchContent_MakeAvailable(googlebenchmark)
  endif()
endif()

# Restore user global state
set(CMAKE_CXX_FLAGS ${USER_CXX_FLAGS})
if(DEFINED USER_BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ${USER_BUILD_SHARED_LIBS})
else()
  unset(BUILD_SHARED_LIBS CACHE )
endif()
set(ROCM_WARN_TOOLCHAIN_VAR ${USER_ROCM_WARN_TOOLCHAIN_VAR} CACHE BOOL "")
