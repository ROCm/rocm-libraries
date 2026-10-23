# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(HostNumericsTestDependencies)
include(GoogleTest)

add_executable(hostnumerics-layout-assignment-test layout_assignment_test.cpp)
target_link_libraries(hostnumerics-layout-assignment-test PRIVATE roc::hostnumerics-core)
add_test(NAME hostnumerics-layout-assignment-test COMMAND hostnumerics-layout-assignment-test)

add_executable(hostnumerics-core-tests core_test.cpp expected_test.cpp
    scalar_conversion_test.cpp scalar_rounding_test.cpp)
target_link_libraries(hostnumerics-core-tests PRIVATE roc::hostnumerics-core GTest::gtest_main)
gtest_discover_tests(
    hostnumerics-core-tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "hostnumerics-core."
)

include(HostNumericsExpectedParity)

# This integer leaf must compile without the throwing convenience adapters.
add_executable(hostnumerics-integer-conversion-no-exceptions-tests
    integer_conversion_no_exceptions_test.cpp)
target_link_libraries(hostnumerics-integer-conversion-no-exceptions-tests
    PRIVATE roc::hostnumerics-core)
if(MSVC)
    target_compile_options(hostnumerics-integer-conversion-no-exceptions-tests PRIVATE /EHs-c-)
else()
    target_compile_options(hostnumerics-integer-conversion-no-exceptions-tests PRIVATE -fno-exceptions)
endif()
add_test(NAME hostnumerics-integer-conversion-no-exceptions-tests
    COMMAND hostnumerics-integer-conversion-no-exceptions-tests)
