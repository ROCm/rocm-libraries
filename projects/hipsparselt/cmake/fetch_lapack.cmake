# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT LAPACK_FOUND AND HIPSPARSELT_ENABLE_FETCH)
    include(FetchContent)
    fetchcontent_declare(
        lapack GIT_REPOSITORY https://github.com/Reference-LAPACK/lapack-release
        GIT_TAG 9f7abc2cc93fac3a5a8905307447c70163abdef2 # lapack-3.7.1@2026-08-28
    )
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
    set(CBLAS ON CACHE BOOL "")
    set(LAPACKE OFF CACHE BOOL "")
    set(BUILD_TESTING OFF CACHE BOOL "")
    fetchcontent_makeavailable(lapack)
    message(STATUS "Fetched LAPACK at: ${lapack_SOURCE_DIR}")
elseif(NOT LAPACK_FOUND)
    message(FATAL_ERROR "LAPACK not found. Install with your package manager (recommended) or "
                        "opt-in to fetch with `-DHIPSPARSELT_ENABLE_FETCH=ON`."
    )
endif()
