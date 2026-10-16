# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# Host-only tests: no device or plugin required.
add_executable(hipdnn_bench_tests
    TestVariantPackBuilder.cpp
    TestCandidatePageJson.cpp TestCliNumbers.cpp TestCsvOutput.cpp TestDeviceIdentity.cpp
    TestInputDomain.cpp
    TestNumericalValidation.cpp)

target_compile_options(hipdnn_bench_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
# uhd_gen's dataset tests read this sample of the CSV; TestCsvOutput pins its header to it.
target_compile_definitions(hipdnn_bench_tests PRIVATE
    HIPDNN_BENCH_CSV_SAMPLE="${CMAKE_CURRENT_SOURCE_DIR}/../../uhd_gen/dataset/tests/fixtures/hipdnn_bench.csv")

target_link_libraries(hipdnn_bench_tests
    GTest::gtest GTest::gtest_main Threads::Threads
    hipdnn_bench_lib hipdnn_backend hipdnn_frontend hipdnn_flatbuffers_sdk hipdnn_plugin_sdk
    hip::host)

clang_tidy_check(hipdnn_bench_tests)
add_hipdnn_test(hipdnn_bench_tests ${CMAKE_CURRENT_BINARY_DIR})
