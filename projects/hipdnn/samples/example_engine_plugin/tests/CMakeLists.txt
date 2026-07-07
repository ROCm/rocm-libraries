# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    SYSTEM
)
# Prevent GoogleTest from overriding our compiler/linker options on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Prevent GoogleTest from being installed with cmake --install
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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
