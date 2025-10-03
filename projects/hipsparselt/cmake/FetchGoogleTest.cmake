# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest QUIET CONFIG)

if(NOT GTest_FOUND AND HIPSPARSELT_ENABLE_FETCH)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.0
        GIT_SHALLOW TRUE
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    message(STATUS "Fetched GTest and installed to: ${gtest_SOURCE_DIR}")
elseif(GTest_FOUND)
    message(STATUS "Found GTest: ${gtest_SOURCE_DIR}")
else()
  message(FATAL_ERROR 
    "GTest not found. Install with your package manager (recommended) or "
    "opt-in to fetch with `-DHIPSPARSELT_ENABLE_FETCH=ON`."
  )
endif()
