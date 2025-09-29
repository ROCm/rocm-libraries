# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(GoogleTest)
find_package(Threads REQUIRED)

# Plugin API tests
add_executable(test_fusilli_plugin_api
    test_fusilli_plugin_api.cpp
    utils.h
)
target_compile_options(test_fusilli_plugin_api PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(test_fusilli_plugin_api PRIVATE
    fusilli_plugin  # Link plugin directly
    fusilli::fusilli
    GTest::gtest_main
    hip::host
    hipdnn_sdk
    Threads::Threads
)
target_compile_definitions(test_fusilli_plugin_api PRIVATE
  FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
  FUSILLI_PLUGIN_ENGINE_ID=${FUSILLI_PLUGIN_ENGINE_ID}
)
gtest_discover_tests(test_fusilli_plugin_api)

# Graph import tests
add_executable(test_graph_import
    test_graph_import.cpp
    utils.h
)
target_compile_options(test_graph_import PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(test_graph_import PRIVATE
    fusilli::fusilli
    GTest::gtest_main
    hipdnn_sdk
)
gtest_discover_tests(test_graph_import)

# Integration tests
add_subdirectory(integration)
