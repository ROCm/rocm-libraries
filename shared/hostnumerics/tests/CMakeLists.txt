# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(HostNumericsTestDependencies)

add_executable(hostnumerics-version-test version_test.cpp)
target_link_libraries(hostnumerics-version-test PRIVATE roc::hostnumerics GTest::gtest_main)
add_test(NAME hostnumerics-version COMMAND hostnumerics-version-test)

add_executable(hostnumerics-scalar-type-test scalar_type_test.cpp)
target_link_libraries(hostnumerics-scalar-type-test PRIVATE roc::hostnumerics-core GTest::gtest_main)
add_test(NAME hostnumerics-scalar-type COMMAND hostnumerics-scalar-type-test)

# Internal C++ result contracts cannot be exercised through the Python bindings.
add_executable(hostnumerics-result-tests result_test.cpp)
target_link_libraries(hostnumerics-result-tests PRIVATE roc::hostnumerics-core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(
    hostnumerics-result-tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-result."
)
