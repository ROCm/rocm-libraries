find_package(Threads REQUIRED)

# Sizing and enumeration only: no device, no plugins, no engine. Everything asserted here is
# arithmetic over tensors and constraints the test constructs, which is what makes it able to
# catch the failures that produce a plausible number rather than an error.
add_executable(hipdnn_bench_tests
    TestVariantPackBuilder.cpp TestKnobEnumeration.cpp TestTensorFillers.cpp
    TestCsvOutput.cpp)

target_compile_options(hipdnn_bench_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(hipdnn_bench_tests
    GTest::gtest GTest::gtest_main Threads::Threads
    hipdnn_bench_lib hipdnn_backend hipdnn_frontend)

clang_tidy_check(hipdnn_bench_tests)
add_hipdnn_test(hipdnn_bench_tests ${CMAKE_CURRENT_BINARY_DIR})
