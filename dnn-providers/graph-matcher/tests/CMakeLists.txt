# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# GoogleTest: reuse an in-tree target if the enclosing build already provides
# one, otherwise find an installed package.
if(NOT TARGET GTest::gtest_main)
    find_package(GTest CONFIG REQUIRED)
endif()

# The test-SDK graph builders (FlatbufferGraphTestUtils.hpp) are header-only and
# used to construct real FlatBuffer graphs for GraphView tests.
set(_hipdnn_test_sdk_include
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipdnn/test_sdk/include")

add_executable(graph_matcher_tests
    TestOpSchema.cpp
    TestGraphView.cpp
    TestPatternBuilder.cpp
    TestMatcher.cpp
    TestDagAndSymbols.cpp
    TestConstraints.cpp
    TestPatternCompiler.cpp
    TestPredicates.cpp
    TestPatternCodec.cpp
    TestPatternSet.cpp
    TestObservabilityAndParity.cpp
)

target_link_libraries(graph_matcher_tests PRIVATE
    hipdnn_graph_matcher
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(graph_matcher_tests PRIVATE "${_hipdnn_test_sdk_include}")

include(GoogleTest)
gtest_discover_tests(graph_matcher_tests)
