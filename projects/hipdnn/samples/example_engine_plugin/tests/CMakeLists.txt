# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

option(ALLOW_FETCH_DEPS
       "Allow fetching third-party dependencies the build environment does not provide"
       OFF
)

set(EXAMPLE_PROVIDER_GTEST_VERSION "1.17.0" CACHE STRING
    "GoogleTest version fetched when no GTest package is available"
)

# The build environment is expected to provide GTest. Fetching is an opt-in
# fallback when no package is available.
find_package(GTest CONFIG QUIET)

# These tests link GoogleMock, which a GTest package exports only when it was
# built with BUILD_GMOCK=ON and INSTALL_GTEST=ON. A package without it is
# unusable here, so it is treated as not found.
if(GTest_FOUND AND NOT TARGET GTest::gmock)
    message(STATUS
        "Ignoring the GTest package at ${GTest_DIR}: it does not provide "
        "GoogleMock (GTest::gmock), which these tests link."
    )
    set(GTest_FOUND FALSE)
endif()

if(NOT GTest_FOUND)
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

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v${EXAMPLE_PROVIDER_GTEST_VERSION}
        SYSTEM
    )
    # Prevent GoogleTest from overriding our compiler/linker options on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Prevent GoogleTest from being installed with cmake --install
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Disable clang-tidy on GoogleTest targets (clang-tidy reports naming/style issues
# in third-party code when building the gtest/gmock libraries themselves)
set(_gtest_targets gtest gtest_main gmock gmock_main)
foreach(_target ${_gtest_targets})
    if(TARGET ${_target})
        set_target_properties(${_target} PROPERTIES CXX_CLANG_TIDY "")
    endif()
endforeach()

add_executable(
    hipdnn_example_provider_unit_tests
    TestExampleProviderContainer.cpp
    TestReluPlanBuilder.cpp
    TestReluPlan.cpp
    TestConvFwdPlanBuilder.cpp
    TestConvFwdPlan.cpp
    TestExampleProviderEngine.cpp
)

target_link_libraries(
    hipdnn_example_provider_unit_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        example_provider_private
        example_provider_compile_options
)

target_include_directories(
    hipdnn_example_provider_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

include(GoogleTest)
if(TEST_ENVIRONMENT)
    gtest_discover_tests(hipdnn_example_provider_unit_tests
        DISCOVERY_MODE PRE_TEST
        PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}"
    )
else()
    gtest_discover_tests(hipdnn_example_provider_unit_tests
        DISCOVERY_MODE PRE_TEST)
endif()
