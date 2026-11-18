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
target_link_libraries(hipdnn_corpus_gen INTERFACE hipdnn_plugin_sdk hipdnn_flatbuffers_sdk hipdnn_tools_common)

if(HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    add_executable(hipdnn_corpus_gen_tool CorpusGen.cpp)
    set_target_properties(hipdnn_corpus_gen_tool PROPERTIES OUTPUT_NAME hipdnn_corpus_gen)
    target_compile_options(hipdnn_corpus_gen_tool PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    target_link_libraries(hipdnn_corpus_gen_tool
        PRIVATE hipdnn_corpus_gen hipdnn_tools_common hipdnn_backend hipdnn_frontend hipdnn_data_sdk)
    clang_tidy_check(hipdnn_corpus_gen_tool)
    install(TARGETS hipdnn_corpus_gen_tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # The shipped declarations travel with the tool, so an installed hipdnn_corpus_gen runs
    # without a source checkout: --operations defaults to them, found from the binary. The
    # build tree keeps the same offset.
    set(_operations_install_dir "${CMAKE_INSTALL_DATADIR}/hipdnn/corpus_gen/operations")
    file(RELATIVE_PATH _operations_reldir "/${CMAKE_INSTALL_BINDIR}" "/${_operations_install_dir}")
    target_compile_definitions(hipdnn_corpus_gen_tool
        PRIVATE HIPDNN_CORPUS_GEN_DEFAULT_OPERATIONS_RELDIR="${_operations_reldir}")
    file(GLOB _shipped_operations CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/operations/*.json")
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/StageFiles.cmake)
    hipdnn_tools_stage_files(hipdnn_corpus_gen_operations
        "${CMAKE_BINARY_DIR}/${_operations_install_dir}"
        FILES ${_shipped_operations}
        INSTALL_DESTINATION "${_operations_install_dir}")
    add_dependencies(hipdnn_corpus_gen_tool hipdnn_corpus_gen_operations)

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
            "seed_negative|--without-engine,--operations,.,--seed,-1|1|--seed needs a number of at least 0"
            "kdp_root_missing|--without-engine,--operations,${CMAKE_CURRENT_SOURCE_DIR}/operations,--operation,matmul,--kdp-root,no-such-dir|1|does not exist"
            "model_shapes_missing|--without-engine,--operations,${CMAKE_CURRENT_SOURCE_DIR}/operations,--operation,matmul,--model-shapes,no-such.csv|1|cannot be read"
            "engine_id_unsigned|--engine-id,17768770578713086094,--help|0|Usage:"
            "engine_id_past_int64|--engine-id,18446744073709551616,--help|1|--engine-id needs a number")

        # A whole run: what it writes, and exit 2 when it generates nothing.
        set(_operations_dir "${CMAKE_CURRENT_SOURCE_DIR}/operations")
        set(_model_shapes "${CMAKE_CURRENT_BINARY_DIR}/model_shapes.csv")
        file(WRITE "${_model_shapes}"
             "name,q.batch,q.heads,q.heads_kv,q.seqlen_q,q.seqlen_k,q.head_dim,q.is_causal,"
             "q.alignment,q.generate_stats,q.dtype\n"
             "llama3-70b prefill,1,64,8,4096,4096,128,true,top_left,false,bf16\n")
        foreach(_run IN ITEMS
                "without_engine|--without-engine,--operations,${_operations_dir},--operation,matmul,--count,4|0|4"
                # No --operations: the declarations staged beside the tool.
                "default_operations|--without-engine,--operation,matmul,--count,4|0|4"
                "empty|--without-engine,--operations,${_operations_dir},--operation,matmul,--count,4,--keep,q.M=-7|2|0"
                # With quotas and no --count, the corpus is the quotas alone.
                "regime_quota|--without-engine,--operations,${_operations_dir},--operation,sdpa_fwd,--regime-quota,decode_short_mha=3|0|3"
                # A sweep with no share still delivers the regimes the quotas name.
                "regime_quota_without_sweep|--without-engine,--operations,${_operations_dir},--operation,sdpa_fwd,--regime-quota,decode_short_mha=3,--sweep-share,0|0|3"
                "model_shapes|--without-engine,--operations,${_operations_dir},--operation,sdpa_fwd,--model-shapes,${_model_shapes},--count,1,--model-share,1,--sweep-share,0|0|1"
                # The engine path, host-only: every problem is offered to the test plugin's
                # engine, which serves matmul without a device.
                "engine|--plugin-dir,${HIPDNN_TEST_PLUGIN_DIR}/default,--engine-name,TEST_GOOD_DEFAULT_ENGINE,--operations,${_operations_dir},--operation,matmul,--count,3|0|3")
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

        # The engine run needs no device: the engine's applicability is asked of the plugin, not
        # of a device-ranked list, so it must pass where none is visible.
        set_property(TEST hipdnn_corpus_gen_cli_run_engine APPEND PROPERTY
            ENVIRONMENT_MODIFICATION "HIP_VISIBLE_DEVICES=set:-1")

        # Corpora compared by content key: an excluded corpus shares no graph with the one
        # excluding it, and a seed reproduces its corpus.
        foreach(_mode IN ITEMS exclude seed)
            set(_test hipdnn_corpus_gen_cli_${_mode})
            add_test(NAME ${_test}
                COMMAND ${CMAKE_COMMAND} "-DTOOL=$<TARGET_FILE:hipdnn_corpus_gen_tool>"
                        "-DCLI_ARGS=--without-engine,--operations,${_operations_dir},--operation,matmul,--count,6"
                        "-DMODE=${_mode}"
                        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/${_test}"
                        -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CorpusGenPair.cmake")
            set_tests_properties(${_test} PROPERTIES TIMEOUT 120)
            if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
                set_tests_properties(${_test} PROPERTIES
                    ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
            endif()
            _apply_hipdnn_test_category_labels(${_test})
        endforeach()

        # These tests run CMake driver scripts, not a gtest executable, so add_hipdnn_test()
        # does not track their run-time prerequisites: the tool and, for the engine run, the
        # test plugin whose engine it is offered. Build-tree only, as hipdnn_list_engines_smoke.
        set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} hipdnn_corpus_gen_tool
            ${TEST_GOOD_DEFAULT_PLUGIN_NAME}
            CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE
        )
    endif()
endif()

if(NOT HIPDNN_SKIP_TESTS)
    add_subdirectory(tests)
endif()
