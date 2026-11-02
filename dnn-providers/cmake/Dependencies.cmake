# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# ALLOW_FETCH_DEPS is shared with projects/hipdnn; option() is a no-op when the
# hipDNN tree already declared it in the same cache. Providers also configure
# standalone, where nothing else declares it.
option(ALLOW_FETCH_DEPS
       "Allow fetching third-party dependencies the build environment does not provide"
       OFF
)

# Finds GTest for the calling provider, fetching it only when permitted.
# Call this from your CMakeLists.txt after deciding tests should be built.
function(fetch_gtest_dependency)
    find_package(GTest CONFIG QUIET)

    # The provider tests link GoogleMock, which a GTest package exports only
    # when it was built with BUILD_GMOCK=ON and INSTALL_GTEST=ON. A package
    # without it is unusable here, so it is treated as not found.
    if(GTest_FOUND AND NOT TARGET GTest::gmock)
        message(STATUS
            "Ignoring the GTest package at ${GTest_DIR}: it does not provide "
            "GoogleMock (GTest::gmock), which the ${PROJECT_NAME} tests link."
        )
        set(GTest_FOUND FALSE)
    endif()

    if(GTest_FOUND)
        message(STATUS "Found system GTest")
        return()
    endif()

    # An explicitly supplied source tree needs no acquisition.
    if(NOT ALLOW_FETCH_DEPS AND
       (NOT FETCHCONTENT_SOURCE_DIR_GOOGLETEST OR
        NOT IS_DIRECTORY "${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}"))
        message(FATAL_ERROR
            "GTest with GoogleMock (GTest::gmock) was not found, and "
            "${PROJECT_NAME} does not fetch "
            "third-party dependencies. They are provided by the build "
            "environment: TheRock's third-party tree, or an install prefix on "
            "CMAKE_PREFIX_PATH. Provide GTest, disable this project's tests, "
            "or configure with -DALLOW_FETCH_DEPS=ON to fetch it. See "
            "https://github.com/ROCm/TheRock/blob/main/docs/development/dependencies.md"
        )
    endif()

    include(FetchContent)

    message(STATUS "Fetching GTest for standalone ${PROJECT_NAME} build")

    fetchcontent_declare(
        googletest URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
                       DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    set(INSTALL_GTEST OFF CACHE INTERNAL "")
    set(BUILD_GMOCK ON CACHE INTERNAL "")

    fetchcontent_makeavailable(googletest)
endfunction()
