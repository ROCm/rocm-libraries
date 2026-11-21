# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Host-only tests of the headers the tools share (hipdnn_tools_common): no device or plugin.
add_executable(hipdnn_tools_common_tests TestOptionalFeatures.cpp)
target_compile_options(hipdnn_tools_common_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
# The cross-check computes features through the plugin SDK's extractor, which exists only with
# the kernel ingestor (as for hipdnn_uhd_features).
target_compile_definitions(hipdnn_tools_common_tests PRIVATE HIPDNN_ENABLE_KERNEL_INGESTOR)
target_link_libraries(hipdnn_tools_common_tests
    GTest::gtest GTest::gtest_main hipdnn_tools_common hipdnn_plugin_sdk)

clang_tidy_check(hipdnn_tools_common_tests)
add_hipdnn_test(hipdnn_tools_common_tests ${CMAKE_CURRENT_BINARY_DIR})
