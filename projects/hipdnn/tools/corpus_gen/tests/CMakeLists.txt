# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# Pure logic, no backend or HIP: plain gtest_main instead of the shared test main.cpp.
add_executable(hipdnn_corpus_gen_tests
    TestFeasibleShapeSet.cpp TestOperationMetadata.cpp
    TestProblemSpace.cpp TestGraphBuilderRegistry.cpp TestWorkloadSampling.cpp TestCorpusRealism.cpp TestGraphSize.cpp TestEveryGraphBuilder.cpp TestCorpusOutput.cpp TestOperationDirectory.cpp TestPoolAssembly.cpp TestRegimeLabel.cpp TestCorpusManifest.cpp TestGraphIdentity.cpp TestDeclaredOracle.cpp TestKernelCatalogSource.cpp TestPointFilter.cpp TestModelShapeSource.cpp TestEngineCoverage.cpp TestServedExtent.cpp TestDeterministicRandom.cpp)

target_compile_options(hipdnn_corpus_gen_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# nlohmann arrives as an include dir via hipdnn_flatbuffers_sdk; its IMPORTED target is
# directory-scoped and does not exist here, so do not link nlohmann_json::nlohmann_json.
target_link_libraries(hipdnn_corpus_gen_tests
    GTest::gtest GTest::gtest_main Threads::Threads hipdnn_corpus_gen
    hipdnn_plugin_sdk hipdnn_test_sdk)

# Tests read the shipped operation declarations, staged and installed with the test data
# and found from the test binary (OperationsDir.hpp), so the installed binary TheRock runs
# does not reach back into the source tree.
set(_operations_subdir "corpus_gen_operations")
file(GLOB _operations CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../operations/*.json")
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/StageFiles.cmake)
hipdnn_tools_stage_files(hipdnn_corpus_gen_test_operations
    "${HIPDNN_TEST_DATA_DIR}/${_operations_subdir}"
    FILES ${_operations}
    INSTALL_DESTINATION "${HIPDNN_TEST_DATA_INSTALL_DIR}/${_operations_subdir}")
add_dependencies(hipdnn_corpus_gen_tests hipdnn_corpus_gen_test_operations)

target_compile_definitions(hipdnn_corpus_gen_tests PRIVATE
    HIPDNN_TEST_DATA_RELDIR="${HIPDNN_TEST_DATA_RELDIR}"
    HIPDNN_CORPUS_GEN_OPERATIONS_SUBDIR="${_operations_subdir}")

clang_tidy_check(hipdnn_corpus_gen_tests)
add_hipdnn_test(hipdnn_corpus_gen_tests ${CMAKE_CURRENT_BINARY_DIR})

# Reads the built graphs back through the frontend, so it links the backend the pure-logic
# suite above stays free of.
if(HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    add_executable(hipdnn_corpus_gen_frontend_tests TestShippedGraphsValidate.cpp)
    target_compile_options(hipdnn_corpus_gen_frontend_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    target_link_libraries(hipdnn_corpus_gen_frontend_tests
        GTest::gtest GTest::gtest_main Threads::Threads hipdnn_corpus_gen
        hipdnn_frontend hipdnn_backend hipdnn_data_sdk)
    add_dependencies(hipdnn_corpus_gen_frontend_tests hipdnn_corpus_gen_test_operations)
    target_compile_definitions(hipdnn_corpus_gen_frontend_tests PRIVATE
        HIPDNN_TEST_DATA_RELDIR="${HIPDNN_TEST_DATA_RELDIR}"
        HIPDNN_CORPUS_GEN_OPERATIONS_SUBDIR="${_operations_subdir}")
    clang_tidy_check(hipdnn_corpus_gen_frontend_tests)
    add_hipdnn_test(hipdnn_corpus_gen_frontend_tests ${CMAKE_CURRENT_BINARY_DIR})
endif()
