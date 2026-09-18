# Corpus generation for UHD training (RFC 0019.13 §5).
#
# Two targets, split along the seam that makes the search testable.
#
# hipdnn_corpus_gen is the search itself: bounds probing and feasible-set construction over
# an injected predicate. It links nothing, so it can be tested against regions whose shape is
# known in advance rather than against whatever an engine happens to do.
#
# hipdnn_corpus_gen (the executable) supplies the other end -- a live engine behind that
# predicate -- and so links the backend and needs a device to run.
add_library(hipdnn_corpus_gen INTERFACE)

target_include_directories(hipdnn_corpus_gen
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# The §6.2 expression language is shared with UMD criteria and UDD dispatch formulas; one
# parser and interpreter serves all three (RFC 0019 §6.2), so this reuses it rather than
# growing a second arithmetic evaluator.
target_link_libraries(hipdnn_corpus_gen INTERFACE hipdnn_plugin_sdk hipdnn_flatbuffers_sdk)

if(HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    add_executable(hipdnn_corpus_gen_tool CorpusGen.cpp)
    set_target_properties(hipdnn_corpus_gen_tool PROPERTIES OUTPUT_NAME hipdnn_corpus_gen)
    target_compile_options(hipdnn_corpus_gen_tool PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    target_link_libraries(hipdnn_corpus_gen_tool
        PRIVATE hipdnn_corpus_gen hipdnn_backend hipdnn_frontend hipdnn_data_sdk)
    clang_tidy_check(hipdnn_corpus_gen_tool)
    install(TARGETS hipdnn_corpus_gen_tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(NOT HIPDNN_SKIP_TESTS)
    add_subdirectory(tests)
endif()
