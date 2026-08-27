# Pure logic over an injected predicate, so this target links neither the backend nor the
# test SDK: gtest_main rather than the shared main.cpp, which exists to set up HIP error
# handling and logging that nothing here touches.
add_executable(hipdnn_corpus_gen_tests TestFeasibleRegionProbe.cpp TestFeasibleShapeSet.cpp)

target_compile_options(hipdnn_corpus_gen_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(hipdnn_corpus_gen_tests
    GTest::gtest GTest::gtest_main Threads::Threads hipdnn_corpus_gen)

clang_tidy_check(hipdnn_corpus_gen_tests)
add_hipdnn_test(hipdnn_corpus_gen_tests ${CMAKE_CURRENT_BINARY_DIR})
