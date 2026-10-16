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
            # --help after it: a tool that only warned would print the help and exit 0.
            "unknown_flag|--operations,.,--no-such-flag,--help|1|Unknown argument: --no-such-flag"
            "flag_as_value|--operations,.,--output,--count,5|1|--output needs a value"
            "trailing_flag|--without-engine,--operations|1|--operations needs a value"
            "count_not_a_number|--without-engine,--operations,.,--count,abc|1|--count needs a number"
            "share_not_a_number|--without-engine,--operations,.,--sweep-share,x|1|--sweep-share needs a number"
            "count_empty|--without-engine,--operations,.,--count,{empty}|1|--count needs a number, not ''"
            "ceiling_zero|--without-engine,--operations,.,--ceiling,0|1|--ceiling needs a number of at least 1, not '0'"
            "restarts_zero|--without-engine,--operations,.,--restarts,0|1|--restarts needs a number of at least 1"
            "share_negative|--without-engine,--operations,.,--model-share,-0.5|1|--model-share needs a number of at least 0"
            "engine_id_unsigned|--engine-id,17768770578713086094,--help|0|Usage:"
            "engine_id_past_int64|--engine-id,18446744073709551616,--help|1|--engine-id needs a number")

        # A whole run: what it writes, and exit 2 when it generates nothing.
        set(_operations_dir "${CMAKE_CURRENT_SOURCE_DIR}/operations")
        foreach(_run IN ITEMS
                "without_engine|--without-engine,--operations,${_operations_dir},--operation,matmul,--count,4|0|4"
                "empty|--without-engine,--operations,${_operations_dir},--operation,matmul,--count,4,--keep,q.M=-7|2|0")
            string(REPLACE "|" ";" _fields "${_run}")
            list(GET _fields 0 _name)
            list(GET _fields 1 _args)
            list(GET _fields 2 _exit)
            list(GET _fields 3 _graphs)
            set(_test hipdnn_corpus_gen_cli_run_${_name})
            add_test(NAME ${_test}
                COMMAND ${CMAKE_COMMAND} "-DTOOL=$<TARGET_FILE:hipdnn_corpus_gen_tool>"
                        "-DCLI_ARGS=${_args}" "-DEXPECTED_EXIT=${_exit}"
                        "-DEXPECTED_GRAPHS=${_graphs}"
                        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/${_test}"
                        -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CorpusGenRun.cmake")
            set_tests_properties(${_test} PROPERTIES TIMEOUT 120)
            if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
                set_tests_properties(${_test} PROPERTIES
                    ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
            endif()
            _apply_hipdnn_test_category_labels(${_test})
        endforeach()
    endif()
endif()

if(NOT HIPDNN_SKIP_TESTS)
    add_subdirectory(tests)
endif()
