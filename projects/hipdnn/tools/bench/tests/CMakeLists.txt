# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# Host-only tests: no device or plugin required.
add_executable(hipdnn_bench_tests
    TestVariantPackBuilder.cpp
    TestCandidatePageJson.cpp TestCliNumbers.cpp TestCsvOutput.cpp TestDeviceIdentity.cpp
    TestGraphFormat.cpp TestInputDomain.cpp
    TestNumericalValidation.cpp)

target_compile_options(hipdnn_bench_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
# Samples of what hipdnn_bench writes, which uhd_gen's tests read: the CSV (TestCsvOutput pins
# its header) and one `--sweep --json` result entry (TestCandidatePageJson pins it whole).
# Staged and installed with the test data and found from the test binary, so an installed
# run reads its own copy rather than the source tree.
set(_csv_sample_subdir "hipdnn_bench")
set(_samples_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../uhd_gen/dataset/tests/fixtures")
set(_samples "${_samples_dir}/hipdnn_bench.csv" "${_samples_dir}/hipdnn_bench_sweep_result.json")
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/StageFiles.cmake)
hipdnn_tools_stage_files(hipdnn_bench_test_data
    "${HIPDNN_TEST_DATA_DIR}/${_csv_sample_subdir}"
    FILES ${_samples}
    INSTALL_DESTINATION "${HIPDNN_TEST_DATA_INSTALL_DIR}/${_csv_sample_subdir}")
add_dependencies(hipdnn_bench_tests hipdnn_bench_test_data)
target_compile_definitions(hipdnn_bench_tests PRIVATE
    HIPDNN_TEST_DATA_RELDIR="${HIPDNN_TEST_DATA_RELDIR}"
    HIPDNN_BENCH_CSV_SAMPLE_SUBDIR="${_csv_sample_subdir}")

target_link_libraries(hipdnn_bench_tests
    GTest::gtest GTest::gtest_main Threads::Threads
    hipdnn_bench_lib hipdnn_tools_common hipdnn_backend hipdnn_frontend hipdnn_flatbuffers_sdk
    hipdnn_plugin_sdk hipdnn_data_sdk
    hip::host)

clang_tidy_check(hipdnn_bench_tests)
add_hipdnn_test(hipdnn_bench_tests ${CMAKE_CURRENT_BINARY_DIR})
