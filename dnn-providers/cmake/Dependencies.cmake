# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# ALLOW_FETCH_DEPS is shared with projects/hipdnn; option() is a no-op when the
# hipDNN tree already declared it in the same cache. The deprecated
# HIPDNN_NO_DOWNLOAD shim lives only there, because that variable never reached
# this file and so never governed a provider build.
option(ALLOW_FETCH_DEPS
       "Allow fetching third-party dependencies the build environment does not provide"
       OFF
)

# Finds GTest for the calling provider, fetching it only when permitted.
# Call this from your CMakeLists.txt after deciding tests should be built.
function(fetch_gtest_dependency)
    find_package(GTest CONFIG QUIET)

    if(GTest_FOUND)
        message(STATUS "Found system GTest")
        return()
    endif()

    if(NOT ALLOW_FETCH_DEPS)
        message(FATAL_ERROR
            "GTest was not found, and ${PROJECT_NAME} does not fetch "
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
