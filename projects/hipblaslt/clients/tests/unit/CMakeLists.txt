# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Host-only client unit tests. Kept as a separate executable from hipblaslt-test
# so it can run in a CI lane without a GPU: these tests only exercise pure
# string<->enum helpers and never initialize a device.
add_executable(hipblaslt-client-unit-tests
    ${CMAKE_CURRENT_SOURCE_DIR}/parser_gtest.cpp
)

target_link_libraries(hipblaslt-client-unit-tests
    PRIVATE
        hipblaslt::hipblaslt-clients-common
        hip::device
        GTest::gtest
        GTest::gtest_main
        OpenMP::OpenMP_CXX
)

target_compile_definitions(hipblaslt-client-unit-tests PRIVATE GOOGLE_TEST)

add_test(NAME hipblaslt-client-unit-tests COMMAND hipblaslt-client-unit-tests)
