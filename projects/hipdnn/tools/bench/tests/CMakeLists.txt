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
# Staged and installed with the test data and found from the test binary, so an installed
# run reads its own copy rather than the source tree.
set(_csv_sample_source "${CMAKE_CURRENT_SOURCE_DIR}/../../uhd_gen/dataset/tests/fixtures/hipdnn_bench.csv")
set(_csv_sample_subdir "hipdnn_bench")
set(_csv_sample_staged "${HIPDNN_TEST_DATA_DIR}/${_csv_sample_subdir}/hipdnn_bench.csv")
add_custom_command(
    OUTPUT "${_csv_sample_staged}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${HIPDNN_TEST_DATA_DIR}/${_csv_sample_subdir}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_csv_sample_source}" "${_csv_sample_staged}"
    DEPENDS "${_csv_sample_source}"
    COMMENT "Staging hipdnn_bench's CSV sample beside the test data"
    VERBATIM
)
add_custom_target(hipdnn_bench_test_data
    DEPENDS "${_csv_sample_staged}"
    COMMENT "Staging hipdnn_bench test data")
add_dependencies(hipdnn_bench_tests hipdnn_bench_test_data)
install(FILES "${_csv_sample_source}"
        DESTINATION "${HIPDNN_TEST_DATA_INSTALL_DIR}/${_csv_sample_subdir}")
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
