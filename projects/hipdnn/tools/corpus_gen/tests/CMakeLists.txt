find_package(Threads REQUIRED)

# Pure logic, no backend or HIP: plain gtest_main instead of the shared test main.cpp.
add_executable(hipdnn_corpus_gen_tests
    TestFeasibleShapeSet.cpp TestOperationMetadata.cpp
    TestProblemSpace.cpp TestGraphBuilderRegistry.cpp TestWorkloadSampling.cpp TestCorpusRealism.cpp TestGraphSize.cpp TestEveryGraphBuilder.cpp TestCorpusOutput.cpp TestOperationDirectory.cpp TestPoolAssembly.cpp TestRegimeLabel.cpp TestCorpusManifest.cpp TestGraphIdentity.cpp TestDeclaredOracle.cpp TestKernelCatalogSource.cpp TestPointFilter.cpp TestModelShapeSource.cpp TestEngineCoverage.cpp)

target_compile_options(hipdnn_corpus_gen_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# nlohmann arrives as an include dir via hipdnn_flatbuffers_sdk; its IMPORTED target is
# directory-scoped and does not exist here, so do not link nlohmann_json::nlohmann_json.
target_link_libraries(hipdnn_corpus_gen_tests
    GTest::gtest GTest::gtest_main Threads::Threads hipdnn_corpus_gen
    hipdnn_plugin_sdk)

# Tests read the shipped operation declarations directly.
target_compile_definitions(hipdnn_corpus_gen_tests PRIVATE
    HIPDNN_CORPUS_GEN_OPERATIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../operations")

clang_tidy_check(hipdnn_corpus_gen_tests)
add_hipdnn_test(hipdnn_corpus_gen_tests ${CMAKE_CURRENT_BINARY_DIR})
