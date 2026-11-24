# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Cross-engine heuristic evaluation harness: selection latency and the regret of the
# heuristic's pick against autotune, for graphs on which several engines compete. It reuses
# the hipdnn_bench helpers (hipdnn_bench_lib, declared by ../bench) and the
# hipdnn_frontend::detail engine-query helpers, not only the consumer Graph API.
if(HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    find_package(hip REQUIRED)

    add_executable(hipdnn_xengine_eval_tool XengineEval.cpp)
    set_target_properties(hipdnn_xengine_eval_tool PROPERTIES OUTPUT_NAME hipdnn_xengine_eval)
    target_compile_options(hipdnn_xengine_eval_tool PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    # detail/EngineQueries.hpp decodes FlatBuffers pages; the frontend does not link the SDK.
    target_link_libraries(hipdnn_xengine_eval_tool
        PRIVATE hipdnn_bench_lib hipdnn_tools_common hipdnn_backend hipdnn_frontend hipdnn_flatbuffers_sdk
                hipdnn_data_sdk hipdnn_plugin_sdk hip::host)
    clang_tidy_check(hipdnn_xengine_eval_tool)
    install(TARGETS hipdnn_xengine_eval_tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
