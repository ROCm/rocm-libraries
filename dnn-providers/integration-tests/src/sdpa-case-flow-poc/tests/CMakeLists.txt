# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Device-free unit tests of the SDPA case builder: each test builds a case from a row and
# checks the returned graph contents. No test runs a graph.
set(SDPA_CASE_FLOW_CASE_GRAPH_TESTS hipdnn_sdpa_case_flow_case_graph_tests)

add_executable(${SDPA_CASE_FLOW_CASE_GRAPH_TESTS}
    TestCaseGraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../CaseGraph.cpp
)

target_include_directories(${SDPA_CASE_FLOW_CASE_GRAPH_TESTS}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(${SDPA_CASE_FLOW_CASE_GRAPH_TESTS}
    PRIVATE
        hip::host
        hipdnn_test_sdk
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)

target_compile_options(${SDPA_CASE_FLOW_CASE_GRAPH_TESTS}
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

set_target_properties(${SDPA_CASE_FLOW_CASE_GRAPH_TESTS}
    PROPERTIES
        BUILD_RPATH_USE_ORIGIN TRUE
)

add_test(NAME ${SDPA_CASE_FLOW_CASE_GRAPH_TESTS} COMMAND ${SDPA_CASE_FLOW_CASE_GRAPH_TESTS})

set(DNN_PROVIDER_CTEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_case_flow.yaml")
_apply_provider_ctest_category_labels(${SDPA_CASE_FLOW_CASE_GRAPH_TESTS})
