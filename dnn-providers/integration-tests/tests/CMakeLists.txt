# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(gpu_graph_executor_tests
    TestGpuReferenceGraphExecutor.cpp
)

target_include_directories(gpu_graph_executor_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(gpu_graph_executor_tests
    PRIVATE
        hip::host
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)

target_compile_options(gpu_graph_executor_tests
    PRIVATE
        ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS}
)

add_test(NAME gpu_graph_executor_tests COMMAND gpu_graph_executor_tests)
