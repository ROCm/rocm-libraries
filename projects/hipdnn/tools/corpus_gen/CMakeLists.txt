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

    # The exit status is what scripts branch on: 0 for --help, 1 for arguments it cannot use.
    # Each entry is name, comma-separated arguments, expected exit, expected output. Build-tree
    # only: the driver script is not installed.
    if(NOT HIPDNN_SKIP_TESTS)
        set(_cli_cases
            "help|--help|0|Usage:"
            "unknown_flag|--operations,.,--no-such-flag|1|Unknown argument: --no-such-flag"
            "flag_as_value|--operations,.,--output,--count,5|1|--output needs a value"
            "trailing_flag|--without-engine,--operations|1|--operations needs a value")
        foreach(_case IN LISTS _cli_cases)
            string(REPLACE "|" ";" _fields "${_case}")
            list(GET _fields 0 _name)
            list(GET _fields 1 _args)
            list(GET _fields 2 _exit)
            list(GET _fields 3 _output)
            set(_test hipdnn_corpus_gen_cli_${_name})
            add_test(NAME ${_test}
                COMMAND ${CMAKE_COMMAND} "-DCORPUS_GEN_TOOL=$<TARGET_FILE:hipdnn_corpus_gen_tool>"
                        "-DCLI_ARGS=${_args}" "-DEXPECTED_EXIT=${_exit}"
                        "-DEXPECTED_OUTPUT=${_output}"
                        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/CliExitStatus.cmake")
            set_tests_properties(${_test} PROPERTIES TIMEOUT 60)
            if(DEFINED TEST_ENVIRONMENT)
                set_tests_properties(${_test} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
            endif()
            # The tool loads the backend DLL from the build tree's bin directory on Windows.
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
