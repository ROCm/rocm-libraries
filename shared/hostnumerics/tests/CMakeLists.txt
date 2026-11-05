# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(HostNumericsTestDependencies)
include(GoogleTest)

add_executable(hostnumerics-layout-assignment-test layout_assignment_test.cpp)
target_link_libraries(hostnumerics-layout-assignment-test PRIVATE roc::hostnumerics-core)
add_test(NAME hostnumerics-layout-assignment-test COMMAND hostnumerics-layout-assignment-test)

add_executable(hostnumerics-core-tests core_test.cpp expected_test.cpp
    integer_conversion_test.cpp scalar_storage_test.cpp)
target_link_libraries(hostnumerics-core-tests PRIVATE roc::hostnumerics-core GTest::gtest_main)
gtest_discover_tests(
    hostnumerics-core-tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-core."
)

# Strict C++17 keeps the AMD GPU layout headers usable by C++17 consumers and
# independent of the C++20 core headers.
add_executable(hostnumerics-amd-gpu-layout-tests amd_gpu_mx_layout_test.cpp
    amd_gpu_operand_layout_test.cpp)
target_link_libraries(hostnumerics-amd-gpu-layout-tests
    PRIVATE roc::hostnumerics-amd-gpu-layout GTest::gtest_main)
set_target_properties(hostnumerics-amd-gpu-layout-tests
    PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
gtest_discover_tests(
    hostnumerics-amd-gpu-layout-tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-amd-gpu-layout."
)

include(HostNumericsExpectedParity)
