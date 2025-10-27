# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(LAPACK QUIET)

if(NOT LAPACK_FOUND AND HIPSPARSELT_ENABLE_FETCH)
  include(FetchContent)
  FetchContent_Declare(
    lapack
    GIT_REPOSITORY https://github.com/Reference-LAPACK/lapack-release
    GIT_TAG lapack-3.7.1
    GIT_SHALLOW TRUE
  )
  enable_language(Fortran)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
  set(CBLAS ON CACHE BOOL "")
  set(LAPACKE OFF CACHE BOOL "")
  set(BUILD_TESTING OFF CACHE BOOL "")
  FetchContent_MakeAvailable(lapack)
  message(STATUS "Fetched LAPACK at: ${lapack_SOURCE_DIR}")
elseif(LAPACK_FOUND)
  message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
else()
  message(FATAL_ERROR 
    "LAPACK not found. Install with your package manager (recommended) or "
    "opt-in to fetch with `-DHIPSPARSELT_ENABLE_FETCH=ON`."
  )
endif()
