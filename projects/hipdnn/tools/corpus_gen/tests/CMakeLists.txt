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

# Tests read the shipped operation declarations, staged and installed with the test data
# and found from the test binary (OperationsDir.hpp), so the installed binary TheRock runs
# does not reach back into the source tree.
set(_operations_subdir "corpus_gen_operations")
file(GLOB _operations CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../operations/*.json")
set(_staged_operations "")
foreach(_source IN LISTS _operations)
    get_filename_component(_name "${_source}" NAME)
    set(_staged "${HIPDNN_TEST_DATA_DIR}/${_operations_subdir}/${_name}")
    add_custom_command(
        OUTPUT "${_staged}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HIPDNN_TEST_DATA_DIR}/${_operations_subdir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_source}" "${_staged}"
        DEPENDS "${_source}"
        COMMENT "Staging corpus_gen operation declaration ${_name}"
        VERBATIM
    )
    list(APPEND _staged_operations "${_staged}")
endforeach()
add_custom_target(
    hipdnn_corpus_gen_test_operations
    DEPENDS ${_staged_operations}
    COMMENT "Staging corpus_gen operation declarations beside the test data"
)
add_dependencies(hipdnn_corpus_gen_tests hipdnn_corpus_gen_test_operations)
install(FILES ${_operations} DESTINATION "${HIPDNN_TEST_DATA_INSTALL_DIR}/${_operations_subdir}")

target_compile_definitions(hipdnn_corpus_gen_tests PRIVATE
    HIPDNN_TEST_DATA_RELDIR="${HIPDNN_TEST_DATA_RELDIR}"
    HIPDNN_CORPUS_GEN_OPERATIONS_SUBDIR="${_operations_subdir}")

clang_tidy_check(hipdnn_corpus_gen_tests)
add_hipdnn_test(hipdnn_corpus_gen_tests ${CMAKE_CURRENT_BINARY_DIR})
