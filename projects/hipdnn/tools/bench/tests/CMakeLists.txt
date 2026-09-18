find_package(Threads REQUIRED)

# Tensor sizing, initialization and the §13.2 correctness gate: all decisions over host
# buffers, so none of them requires a device or a plugin.
add_executable(hipdnn_bench_tests
    TestVariantPackBuilder.cpp TestTensorFillers.cpp
    TestCsvOutput.cpp TestNumericalValidation.cpp)

target_compile_options(hipdnn_bench_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(hipdnn_bench_tests
    GTest::gtest GTest::gtest_main Threads::Threads
    hipdnn_bench_lib hipdnn_backend hipdnn_frontend)

clang_tidy_check(hipdnn_bench_tests)
add_hipdnn_test(hipdnn_bench_tests ${CMAKE_CURRENT_BINARY_DIR})
