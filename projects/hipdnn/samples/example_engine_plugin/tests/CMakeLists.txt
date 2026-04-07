# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# Prevent GoogleTest from overriding our compiler/linker options on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
    example_provider_tests
    TestExampleProviderContainer.cpp
    TestReluPlanBuilder.cpp
    TestReluPlan.cpp
    TestConvFwdPlanBuilder.cpp
    TestConvFwdPlan.cpp
)

target_link_libraries(
    example_provider_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        example_provider_private
        example_provider_compile_options
)

target_include_directories(
    example_provider_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

include(GoogleTest)
gtest_discover_tests(example_provider_tests)
