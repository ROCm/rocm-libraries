# ########################################################################
# Copyright 2019-2026 Advanced Micro Devices, Inc.
# ########################################################################

include(primlibs/dependencies/monorepo)

# Declared here so it appears in the cache and the configure summary even
# when the sqlite3 module below is never included. option() is idempotent.
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
  include(primlibs/dependencies/googletest)

  if(NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
    find_package(TBB QUIET)
  endif()
  if (NOT TARGET TBB::tbb AND NOT TARGET tbb AND BUILD_HIPSTDPAR_TEST_WITH_TBB)
    message(STATUS "TBB not found or force download TBB on. Downloading and building TBB.")
    set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/tbb CACHE PATH "" FORCE)

    FetchContent_Declare(
      TBB
      GIT_REPOSITORY      https://github.com/oneapi-src/oneTBB.git
      GIT_TAG             v2023.0.0
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
  #
  # NOTE(hipccl3): the logic lives in the shared
  # primlibs/dependencies/sqlite3.cmake module. Its sqlite3/ subdirectory
  # removes leaked device flags for the embedded plain C target. The module
  # provides the SQLite::SQLite3 target on both paths, so other hipccl3
  # components can reuse it as-is.
  include(primlibs/dependencies/sqlite3)
  if(SQLITE_USE_SYSTEM_PACKAGE)
    list(APPEND static_depends PACKAGE SQLite3)
  endif()
  set(ROCTHRUST_SQLITE_LIB SQLite::SQLite3)
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  fetch_monorepo_dep(PACKAGE rocrand)
endif()
