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

# Remove strict warning flags from GTest targets to prevent compilation errors
# GTest is an external dependency and should not inherit parent project's -Werror flags
set(GTEST_TARGETS gtest gtest_main gmock gmock_main)
foreach(target ${GTEST_TARGETS})
    if(TARGET ${target})
        get_target_property(target_options ${target} COMPILE_OPTIONS)
        if(target_options)
            # Remove -Werror and other strict warning flags
            list(FILTER target_options EXCLUDE REGEX "^-Werror")
            list(FILTER target_options EXCLUDE REGEX "^/W[1-4X]")
            set_target_properties(${target} PROPERTIES COMPILE_OPTIONS "${target_options}")
        endif()

        get_target_property(interface_options ${target} INTERFACE_COMPILE_OPTIONS)
        if(interface_options)
            list(FILTER interface_options EXCLUDE REGEX "^-Werror")
            list(FILTER interface_options EXCLUDE REGEX "^/W[1-4X]")
            set_target_properties(${target} PROPERTIES INTERFACE_COMPILE_OPTIONS "${interface_options}")
        endif()

        # Suppress all warnings for GTest to avoid -Wundef errors
        target_compile_options(${target} PRIVATE -w)
    endif()
endforeach()

add_executable(
    example_plugin_tests
    TestExamplePluginContainer.cpp
    TestReluPlanBuilder.cpp
    TestReluPlan.cpp
    TestConvFwdPlanBuilder.cpp
    TestConvFwdPlan.cpp
    TestAdvancedEngineScaffold.cpp
)

target_link_libraries(
    example_plugin_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        example_plugin_private
)

target_include_directories(
    example_plugin_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

include(GoogleTest)
gtest_discover_tests(example_plugin_tests)
