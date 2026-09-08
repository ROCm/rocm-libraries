# ########################################################################
# Copyright 2019-2026 Advanced Micro Devices, Inc.
# ########################################################################

include(dependencies/monorepo)

# The option of using the SQLite provided by the system, instead of downloading a copy
option( SQLITE_USE_SYSTEM_PACKAGE "Use SQLite3 from find_package" ON )

if(${LINK_HIP_DEVICE_LIBS} AND NOT GRAFT_THRUST_ONTO_BINARIES)
  fetch_monorepo_dep(
    PACKAGE rocprim
    VERSION ${MIN_ROCPRIM_PACKAGE_VERSION}
  )
endif()

# Search for libhipcxx if requested (default: ON)
if(${ROCTHRUST_USE_LIBHIPCXX})
  find_package(libhipcxx)
  if (NOT TARGET libhipcxx::libhipcxx)
    message(STATUS "libhipcxx installation not found. Using deprecated rocThrust fallback implementation.  Please switch to using libhipcxx.")
  endif()
endif()

# Test dependencies
if(BUILD_TEST OR BUILD_HIPSTDPAR_TEST)
  include(dependencies/googletest)

  if(NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
    find_package(TBB QUIET)
  endif()
  if (NOT TARGET TBB::tbb AND NOT TARGET tbb AND BUILD_HIPSTDPAR_TEST_WITH_TBB)
    message(STATUS "TBB not found or force download TBB on. Downloading and building TBB.")
    set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/tbb CACHE PATH "" FORCE)

    FetchContent_Declare(
      TBB
      GIT_REPOSITORY      https://github.com/oneapi-src/oneTBB.git
      GIT_TAG             1c4c93fc5398c4a1acb3492c02db4699f3048dea # v2021.13.0
      INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/deps/tbb
      CMAKE_ARGS          -DCMAKE_CXX_COMPILER=g++ -DTBB_TEST=OFF -DTBB_BUILD=ON -DTBB_INSTALL=ON -DTBBMALLOC_PROXY_BUILD=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
    )
    FetchContent_MakeAvailable(TBB)
    if(NOT TARGET TBB::tbb)
      add_library(TBB::tbb)
    endif()
  endif()

  # SQlite (for run-to-run bitwise-reproducibility tests)
  # Note: SQLite 3.51.3 to address CVE https://github.com/advisories/GHSA-p36r-6g67-869c
  set(SQLITE_MIN_VERSION "3.51.3")
  string(REPLACE "." "_" SQLITE_VER_UNDERSCORE ${SQLITE_MIN_VERSION})

  if(SQLITE_USE_SYSTEM_PACKAGE)
    find_package(SQLite3 ${SQLITE_MIN_VERSION} REQUIRED)
    list(APPEND static_depends PACKAGE SQLite3)
    set(ROCTHRUST_SQLITE_LIB SQLite::SQLite3)
  else()
    message(STATUS "Force download local copy of SQLite on. Downloading and building SQLite.")
    if(DEFINED ENV{SQLITE_${SQLITE_VER_UNDERSCORE}_SRC_URL})
      set(SQLITE_${SQLITE_VER_UNDERSCORE}_SRC_URL_INIT $ENV{SQLITE_${SQLITE_VER_UNDERSCORE}_SRC_URL})
    else()
      set(SQLITE_${SQLITE_VER_UNDERSCORE}_SRC_URL_INIT https://sqlite.org/2026/sqlite-amalgamation-3510300.zip)
    endif()
    set(SQLITE_${SQLITE_VER_UNDERSCORE}_SRC_URL ${SQLITE_${SQLITE_VER_UNDERSCORE}_SRC_URL_INIT} CACHE STRING "Location of SQLite source code")
    set(SQLITE_SRC_${SQLITE_VER_UNDERSCORE}_SHA3_256 ced02ff9738970f338c9c8e269897b554bcda73f6cf1029d49459e1324dbeaea CACHE STRING "SHA3-256 hash of SQLite source code")

    # embed SQLite
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      # use extract timestamp for fetched files instead of timestamps in the archive
      cmake_policy(SET CMP0135 NEW)
    endif()

    FetchContent_Declare(sqlite_local
      URL ${SQLITE_${SQLITE_VER_UNDERSCORE}_SRC_URL}
      URL_HASH SHA3_256=${SQLITE_SRC_${SQLITE_VER_UNDERSCORE}_SHA3_256}
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
    set(ROCTHRUST_SQLITE_LIB sqlite3)
  endif()
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  fetch_monorepo_dep(PACKAGE rocrand)
endif()
