# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Threads REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../build_tools")
include(FusilliPluginTestUtils)

# Plugin API tests
add_fusilli_plugin_test(
  NAME fusilli_plugin_api_test
  SRCS
    test_fusilli_plugin_api.cpp
    utils.h
  DEPS
    hip::host
    fusilli_plugin # Link plugin .so directly
    fusilli::fusilli
    GTest::gtest_main
    hipdnn_plugin_sdk
    hipdnn_data_sdk
    hipdnn_test_sdk
    Threads::Threads
)

# Graph import tests
add_fusilli_plugin_test(
  NAME fusilli_plugin_graph_import_test
  SRCS
    test_graph_import.cpp
    utils.h
  DEPS
    fusilli::fusilli
    GTest::gtest_main
    hipdnn_plugin_sdk
    hipdnn_data_sdk
)

# Integration tests
add_subdirectory(integration)
