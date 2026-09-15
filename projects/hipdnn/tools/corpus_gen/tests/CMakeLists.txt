find_package(Threads REQUIRED)

# Pure logic over an injected predicate, so this target links neither the backend nor the
# test SDK: gtest_main rather than the shared main.cpp, which exists to set up HIP error
# handling and logging that nothing here touches.
add_executable(hipdnn_corpus_gen_tests
    TestFeasibleShapeSet.cpp TestOperationMetadata.cpp
    TestProblemSpace.cpp TestGraphBuilderRegistry.cpp TestWorkloadSampling.cpp TestCorpusRealism.cpp TestGraphSize.cpp TestEveryGraphBuilder.cpp TestCorpusOutput.cpp TestOperationDirectory.cpp)

target_compile_options(hipdnn_corpus_gen_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(hipdnn_corpus_gen_tests
    GTest::gtest GTest::gtest_main Threads::Threads hipdnn_corpus_gen
    hipdnn_plugin_sdk nlohmann_json::nlohmann_json)

# The shipped operation declarations are checked as artifacts, not as inline copies, so the
# files themselves cannot rot unnoticed.
target_compile_definitions(hipdnn_corpus_gen_tests PRIVATE
    HIPDNN_CORPUS_GEN_OPERATIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../operations")

clang_tidy_check(hipdnn_corpus_gen_tests)
add_hipdnn_test(hipdnn_corpus_gen_tests ${CMAKE_CURRENT_BINARY_DIR})
