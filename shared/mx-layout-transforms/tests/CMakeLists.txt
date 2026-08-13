# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest 1.12.1 QUIET)
if(NOT GTest_FOUND)
    if(ROCMX_LAYOUT_TRANSFORMS_ENABLE_FETCH)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "ROCMxLayoutTransforms tests require GTest 1.12.1 or newer.")
    endif()
endif()

add_executable(mx-layout-transforms-tests pre_swizzle_test.cpp)
target_link_libraries(
    mx-layout-transforms-tests
    PRIVATE
        GTest::gtest_main
        roc::mx-layout-transforms
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
        mx-layout-transforms-tests
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
    )
endif()

include(GoogleTest)
gtest_discover_tests(mx-layout-transforms-tests DISCOVERY_MODE PRE_TEST)

add_test(
    NAME mx-layout-transforms-component-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCOMPONENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/.."
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_component_boundary.cmake"
)

add_test(
    NAME mx-layout-transforms-package
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCOMPONENT_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DPACKAGE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package"
        "-DPACKAGE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-test"
        "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
        "-DTEST_CONFIG=$<CONFIG>"
        "-DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/run_package_test.cmake"
)
