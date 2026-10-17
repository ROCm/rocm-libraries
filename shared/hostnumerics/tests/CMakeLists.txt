# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(HostNumericsTestDependencies)
include(GoogleTest)
add_executable(hostnumerics-core-tests core_test.cpp)
target_link_libraries(hostnumerics-core-tests PRIVATE roc::hostnumerics-core GTest::gtest_main)
gtest_discover_tests(
    hostnumerics-core-tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-core."
)

# Internal C++ result contracts cannot be exercised through the Python bindings.
add_executable(hostnumerics-result-tests result_test.cpp)
target_link_libraries(hostnumerics-result-tests PRIVATE roc::hostnumerics-core GTest::gtest_main)
gtest_discover_tests(
    hostnumerics-result-tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-result."
)
