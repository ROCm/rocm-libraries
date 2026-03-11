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

# Set HIPDNN_PLUGIN_DIR so the hipDNN backend discovers the example plugin
# shared library from the build output directory at test runtime.
set(PLUGIN_DIR "$<TARGET_FILE_DIR:example_plugin>")

# The test reads HIPDNN_PLUGIN_DIR from the environment at runtime.
# gtest_discover_tests sets the ENVIRONMENT property on each discovered test.
include(GoogleTest)
gtest_discover_tests(
    example_plugin_integration_tests
    PROPERTIES
        ENVIRONMENT "HIPDNN_PLUGIN_DIR=${PLUGIN_DIR}"
)
