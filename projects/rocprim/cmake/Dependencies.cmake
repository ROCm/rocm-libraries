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
# when VerifyCompiler.cmake is included.

include(FetchContent)

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

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
    if(WIN32)
      # Older versions of gtest on Windows does not support printing of 128-bit values,
      # Causing compilation errors.
      find_package(GTest 1.11.0 REQUIRED)
    else()
      find_package(GTest QUIET)
    endif()
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
      GIT_TAG        v1.8.0
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
  find_package(ROCM 0.11.0 CONFIG QUIET PATHS "${ROCM_ROOT}") # rocm-cmake
endif()
if(NOT ROCM_FOUND)
  message(STATUS "ROCm CMake not found. Fetching...")
  # We don't really want to consume the build and test targets of ROCm CMake.
  # CMake 3.18 allows omitting them, even though there's a CMakeLists.txt in source root.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(SOURCE_SUBDIR_ARG SOURCE_SUBDIR "DISABLE ADDING TO BUILD")
  else()
    set(SOURCE_SUBDIR_ARG)
  endif()
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  FetchContent_Declare(
    rocm-cmake
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG        rocm-6.1.2
    ${SOURCE_SUBDIR_ARG}
  )
  FetchContent_GetProperties(rocm-cmake)
  if(NOT rocm-cmake_POPULATED)
    # rocm-cmake 0.12.0 and higher needs to built from source
    FetchContent_Populate(rocm-cmake)
    message("Populated: ${rocm-cmake_SOURCE_DIR}")
    execute_process(
      WORKING_DIRECTORY ${rocm-cmake_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} ${rocm-cmake_SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=.
    )
    execute_process(
      WORKING_DIRECTORY ${rocm-cmake_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} --build ${rocm-cmake_SOURCE_DIR} --target install
    )
  endif()
  FetchContent_MakeAvailable(rocm-cmake)
  find_package(ROCM CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
else()
  find_package(ROCM 0.11.0 CONFIG REQUIRED PATHS "${ROCM_ROOT}")
endif()


# rocRAND (https://github.com/ROCmSoftwarePlatform/rocRAND)
if(WITH_ROCRAND)
  find_package(rocrand QUIET)
endif()
if(WITH_ROCRAND AND NOT rocrand_FOUND)
  message(STATUS "Downloading and building rocrand.")
  set(ROCRAND_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand CACHE PATH "")

  set(EXTRA_CMAKE_ARGS "-DGPU_TARGETS=${GPU_TARGETS}")
  # CMAKE_ARGS of download_project (or ExternalProject_Add) can't contain ; so another separator
  # is needed and LIST_SEPARATOR is passed to download_project()
  string(REPLACE ";" "|" EXTRA_CMAKE_ARGS "${EXTRA_CMAKE_ARGS}")
  # Pass launcher so sccache can be used to speed up building rocRAND
  if(CMAKE_CXX_COMPILER_LAUNCHER)
    set(EXTRA_CMAKE_ARGS "${EXTRA_CMAKE_ARGS} -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
  endif()
  download_project(
    PROJ                  rocrand
    GIT_REPOSITORY        https://github.com/ROCmSoftwarePlatform/rocRAND.git
    GIT_TAG               develop
    GIT_SHALLOW           TRUE
    INSTALL_DIR           ${ROCRAND_ROOT}
    LIST_SEPARATOR        |
    CMAKE_ARGS            -DCMAKE_CXX_COMPILER=hipcc -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm ${EXTRA_CMAKE_ARGS}
    LOG_DOWNLOAD          TRUE
    LOG_CONFIGURE         TRUE
    LOG_BUILD             TRUE
    LOG_INSTALL           TRUE
    LOG_OUTPUT_ON_FAILURE TRUE
    BUILD_PROJECT         TRUE
    UPDATE_DISCONNECTED   TRUE
  )
  find_package(rocrand REQUIRED CONFIG PATHS ${ROCRAND_ROOT})
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
include(ROCMCheckTargetIds)
include(ROCMClients)
if(BUILD_DOCS)
  include(ROCMSphinxDoc)
endif()
