# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# Integration test dependencies
#
# Integration tests exercise the full hipDNN stack: frontend graph API,
# backend engine selection, plugin loading, and graph execution.  They
# require an installed hipDNN with the frontend library and a GPU with
# ROCm for runtime execution (tests skip gracefully if no GPU is found).
# =============================================================================
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)

find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_test_sdk CONFIG REQUIRED)
find_package(hip REQUIRED)

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
    example_plugin_integration_tests
    TestPluginIntegration.cpp
)

target_link_libraries(
    example_plugin_integration_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        hipdnn_frontend
        hipdnn_test_sdk
        hip::host
)

# The test auto-discovers the plugin from its own executable directory.
# No HIPDNN_PLUGIN_DIR environment variable is needed when the plugin
# shared library is co-located with the test executable.
include(GoogleTest)
gtest_discover_tests(example_plugin_integration_tests)
