# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Corpus harvest tool (RFC 0019.13 §5.3). It uses the hipdnn_frontend::detail engine-query
# helpers, not the consumer Graph API.
add_library(hipdnn_bench_lib INTERFACE)

target_include_directories(hipdnn_bench_lib
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(hipdnn_bench_lib INTERFACE hipdnn_tools_common)

if(HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    find_package(hip REQUIRED)

    add_executable(hipdnn_bench_tool HipdnnBench.cpp)
    set_target_properties(hipdnn_bench_tool PROPERTIES OUTPUT_NAME hipdnn_bench)
    target_compile_options(hipdnn_bench_tool PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    # detail/EngineQueries.hpp decodes FlatBuffers pages; the frontend does not link the SDK.
    target_link_libraries(hipdnn_bench_tool
        PRIVATE hipdnn_bench_lib hipdnn_tools_common hipdnn_backend hipdnn_frontend hipdnn_flatbuffers_sdk
                hipdnn_data_sdk hipdnn_plugin_sdk hip::host)
    clang_tidy_check(hipdnn_bench_tool)
    install(TARGETS hipdnn_bench_tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # 0 for --help, 1 for arguments it cannot use, decided before any graph or GPU is touched.
    if(NOT HIPDNN_SKIP_TESTS)
        include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CliExitTests.cmake)
        hipdnn_add_cli_exit_tests(hipdnn_bench_cli hipdnn_bench_tool
            "help|--help|0|Usage:"
            # --help after it: a tool that only warned would print the help and exit 0.
            "unknown_flag|--graph,g.fb,--no-such-flag,--help|1|Unknown argument: --no-such-flag"
            "flag_as_value|--problem-id,--no-header,--graph,g.fb|1|--problem-id needs a value"
            "trailing_flag|--graph|1|--graph needs a value"
            "warmup_not_a_number|--graph,g.fb,--warmup,x|1|--warmup needs a number"
            "stability_not_a_number|--graph,g.fb,--stability,5x|1|--stability needs a number"
            "engine_id_not_a_number|--engine-id,ten|1|--engine-id needs a number"
            "error_before_help|--warmup,x,--help|1|--warmup needs a number"
            "warmup_empty|--graph,g.fb,--warmup,{empty},--help|1|--warmup needs a number, not ''"
            "warmup_negative|--warmup,-1,--help|1|--warmup needs a number of at least 0"
            "iterations_past_int|--max-iterations,4294967296,--help|1|--max-iterations needs a number of at most 2147483647"
            "stability_negative|--stability,-0.1,--help|1|--stability needs a number of at least 0"
            "engine_id_unsigned|--engine-id,18446744073709551615,--help|0|Usage:"
            # A workspace limit is a query constraint, which engine-level queries do not take.
            "workspace_limit_collect|--graph,g.fb,--collect-immediate,--workspace-limit,64|1|--workspace-limit"
            "workspace_limit_predict|--graph,g.fb,--predict-engine,--workspace-limit,64|1|--workspace-limit"
            "workspace_limit_describe|--graph,g.fb,--describe-engine-prediction,--knob,global.workspace_size_limit=64|1|--workspace-limit")
        # The tests run a CMake driver script, so their one run-time prerequisite (the tool
        # binary) is appended to CHECK_DEPENDS_GLOBAL by hand, as for hipdnn_list_engines_smoke.
        # Build-tree only.
        set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} hipdnn_bench_tool
            CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE
        )
    endif()
endif()

# The tests link hipdnn_frontend, so they need the same guard as the tool.
if(NOT HIPDNN_SKIP_TESTS AND HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    add_subdirectory(tests)
endif()
