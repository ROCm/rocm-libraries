# ########################################################################
# Copyright 2019-2024 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust dependencies
# ###########################

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

# rocPRIM (https://github.com/ROCmSoftwarePlatform/rocPRIM)
if(NOT DOWNLOAD_ROCPRIM)
  find_package(rocprim QUIET)
endif()
if(NOT rocprim_FOUND)
  message(STATUS "Downloading and building rocprim.")
  download_project(
    PROJ                rocprim
    GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
    GIT_TAG             develop
    INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim
    CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
    LOG_DOWNLOAD        TRUE
    LOG_CONFIGURE       TRUE
    LOG_BUILD           TRUE
    LOG_INSTALL         TRUE
    BUILD_PROJECT       TRUE
    UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
  )
  find_package(rocprim REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim NO_DEFAULT_PATH)
endif()

# Test dependencies
if(BUILD_TEST OR BUILD_HIPSTDPAR_TEST)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Test (https://github.com/google/googletest)
    find_package(GTest QUIET)
    find_package(TBB QUIET)
  else()
    message(STATUS "Force installing GTest.")
  endif()

  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/gtest CACHE PATH "")

    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.11.0
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE
    )
    find_package(GTest REQUIRED CONFIG PATHS ${GTEST_ROOT})
  endif()

  if (NOT TARGET TBB::tbb AND NOT TARGET tbb AND BUILD_HIPSTDPAR_TEST_WITH_TBB)
    message(STATUS "TBB not found or force download TBB on. Downloading and building TBB.")
    set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/tbb CACHE PATH "" FORCE)

    download_project(
      PROJ  TBB
      GIT_REPOSITORY      https://github.com/oneapi-src/oneTBB.git
      GIT_TAG             1c4c93fc5398c4a1acb3492c02db4699f3048dea # v2021.13.0
      INSTALL_DIR         ${TBB_ROOT}
      CMAKE_ARGS          -DCMAKE_CXX_COMPILER=g++ -DTBB_TEST=OFF -DTBB_BUILD=ON -DTBB_INSTALL=ON -DTBBMALLOC_PROXY_BUILD=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE
    )
    find_package(TBB REQUIRED CONFIG PATHS ${TBB_ROOT})
  
  endif()

  # SQlite (for run-to-run bitwise-reproducibility tests)
  # Note: SQLite 3.36.0 enabled the backup API by default, which we need
  # for cache serialization.  We also want to use a static SQLite,
  # and distro static libraries aren't typically built
  # position-independent.
  include( FetchContent )

  if(DEFINED ENV{SQLITE_3_43_2_SRC_URL})
    set(SQLITE_3_43_2_SRC_URL_INIT $ENV{SQLITE_3_43_2_SRC_URL})
  else()
    set(SQLITE_3_43_2_SRC_URL_INIT https://www.sqlite.org/2023/sqlite-amalgamation-3430200.zip)
  endif()
    set(SQLITE_3_43_2_SRC_URL ${SQLITE_3_43_2_SRC_URL_INIT} CACHE STRING "Location of SQLite source code")
    set(SQLITE_SRC_3_43_2_SHA3_256 af02b88cc922e7506c6659737560c0756deee24e4e7741d4b315af341edd8b40 CACHE STRING "SHA3-256 hash of SQLite source code")

    # embed SQLite
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      # use extract timestamp for fetched files instead of timestamps in the archive
      cmake_policy(SET CMP0135 NEW)
    endif()

  message("Downloading SQLite.")
  FetchContent_Declare(sqlite_local
    URL ${SQLITE_3_43_2_SRC_URL}
    URL_HASH SHA3_256=${SQLITE_SRC_3_43_2_SHA3_256}
  )
  FetchContent_MakeAvailable(sqlite_local)

  add_library(sqlite3 OBJECT ${sqlite_local_SOURCE_DIR}/sqlite3.c)
  target_include_directories(sqlite3 PUBLIC ${sqlite_local_SOURCE_DIR})
  set_target_properties( sqlite3 PROPERTIES
      C_VISIBILITY_PRESET "hidden"
      VISIBILITY_INLINES_HIDDEN ON
      POSITION_INDEPENDENT_CODE ON
      LINKER_LANGUAGE CXX
  )

  # We don't need extensions, and omitting them from SQLite removes the
  # need for dlopen/dlclose from within rocThrust.
  # We also don't need the shared cache, and omitting it yields some performance improvements.
  target_compile_options(
      sqlite3
      PRIVATE -DSQLITE_OMIT_LOAD_EXTENSION
      PRIVATE -DSQLITE_OMIT_SHARED_CACHE
  )
endif()

# Benchmark dependencies
if(BUILD_BENCHMARKS)
  set(BENCHMARK_VERSION 1.8.0)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Benchmark (https://github.com/google/benchmark.git)
    find_package(benchmark ${BENCHMARK_VERSION} QUIET)
  else()
    message(STATUS "Force installing Google Benchmark.")
  endif()

  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found or force download Google Benchmark on. Downloading and building Google Benchmark.")
    if(CMAKE_CONFIGURATION_TYPES)
      message(FATAL_ERROR "DownloadProject.cmake doesn't support multi-configuration generators.")
    endif()
    set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/googlebenchmark CACHE PATH "")
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    # hip-clang cannot compile googlebenchmark for some reason
      if(WIN32)
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=cl")
      else()
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
      endif()
    endif()

    download_project(
      PROJ                googlebenchmark
      GIT_REPOSITORY      https://github.com/google/benchmark.git
      GIT_TAG             v${BENCHMARK_VERSION}
      INSTALL_DIR         ${GOOGLEBENCHMARK_ROOT}
      CMAKE_ARGS          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=OFF -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_CXX_STANDARD=14 ${COMPILER_OVERRIDE}
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE
    )
    find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT} NO_DEFAULT_PATH)
  endif()
endif()
