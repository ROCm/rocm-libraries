# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest QUIET CONFIG)

if(NOT GTest_FOUND AND HIPSPARSELT_ENABLE_FETCH)
    include(FetchContent)
    fetchcontent_declare(
        googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG release-1.12.0
        GIT_SHALLOW TRUE
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    fetchcontent_makeavailable(googletest)
    message(STATUS "Fetched GTest and installed to: ${GTest_DIR}")
elseif(GTest_FOUND)
    message(STATUS "Found GTest: ${GTest_DIR}")
else()
    message(FATAL_ERROR "GTest not found. Install with your package manager (recommended) or "
                        "opt-in to fetch with `-DHIPSPARSELT_ENABLE_FETCH=ON`."
    )
endif()
