# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Corpus generation for UHD training (RFC 0019.13 §5).
#
# The library is the search over an injected predicate and links no backend, so it is testable
# against known regions; the executable puts a live engine behind the predicate.
add_library(hipdnn_corpus_gen INTERFACE)

target_include_directories(hipdnn_corpus_gen
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Reuses the shared §6.2 expression parser/interpreter rather than a second evaluator.
target_link_libraries(hipdnn_corpus_gen INTERFACE hipdnn_plugin_sdk hipdnn_flatbuffers_sdk)

if(HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    add_executable(hipdnn_corpus_gen_tool CorpusGen.cpp)
    set_target_properties(hipdnn_corpus_gen_tool PROPERTIES OUTPUT_NAME hipdnn_corpus_gen)
    target_compile_options(hipdnn_corpus_gen_tool PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    target_link_libraries(hipdnn_corpus_gen_tool
        PRIVATE hipdnn_corpus_gen hipdnn_backend hipdnn_frontend hipdnn_data_sdk)
    clang_tidy_check(hipdnn_corpus_gen_tool)
    install(TARGETS hipdnn_corpus_gen_tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # 0 for --help, 1 for arguments it cannot use.
    if(NOT HIPDNN_SKIP_TESTS)
        include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CliExitTests.cmake)
        hipdnn_add_cli_exit_tests(hipdnn_corpus_gen_cli hipdnn_corpus_gen_tool
            "help|--help|0|Usage:"
            "unknown_flag|--operations,.,--no-such-flag|1|Unknown argument: --no-such-flag"
            "flag_as_value|--operations,.,--output,--count,5|1|--output needs a value"
            "trailing_flag|--without-engine,--operations|1|--operations needs a value"
            "count_not_a_number|--without-engine,--operations,.,--count,abc|1|--count needs a number"
            "share_not_a_number|--without-engine,--operations,.,--sweep-share,x|1|--sweep-share needs a number")
    endif()
endif()

if(NOT HIPDNN_SKIP_TESTS)
    add_subdirectory(tests)
endif()
