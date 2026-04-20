# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# Prevent GoogleTest from overriding our compiler/linker options on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Prevent GoogleTest from being installed with cmake --install
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Suppress warnings in GoogleTest targets (GoogleTest uses #if instead of #ifdef for
# GTEST_LINKED_AS_SHARED_LIBRARY which triggers -Wundef warnings)
set(_gtest_targets gtest gtest_main gmock gmock_main)
foreach(_target ${_gtest_targets})
    if(TARGET ${_target})
        target_compile_options(${_target} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-undef>)
    endif()
endforeach()

add_executable(
    example_provider_tests
    TestExampleProviderContainer.cpp
    TestReluPlanBuilder.cpp
    TestReluPlan.cpp
    TestConvFwdPlanBuilder.cpp
    TestConvFwdPlan.cpp
    TestExampleProviderEngine.cpp
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
gtest_discover_tests(example_provider_tests DISCOVERY_MODE PRE_TEST)
