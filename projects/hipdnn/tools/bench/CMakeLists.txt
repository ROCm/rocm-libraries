# Corpus harvest tool (RFC 0019.13 §5.3). It uses the hipdnn_frontend::detail engine-query
# helpers, not the consumer Graph API.
add_library(hipdnn_bench_lib INTERFACE)

target_include_directories(hipdnn_bench_lib
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    find_package(hip REQUIRED)

    add_executable(hipdnn_bench_tool HipdnnBench.cpp)
    set_target_properties(hipdnn_bench_tool PROPERTIES OUTPUT_NAME hipdnn_bench)
    target_compile_options(hipdnn_bench_tool PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    # detail/EngineQueries.hpp decodes FlatBuffers pages; the frontend does not link the SDK.
    target_link_libraries(hipdnn_bench_tool
        PRIVATE hipdnn_bench_lib hipdnn_backend hipdnn_frontend hipdnn_flatbuffers_sdk
                hipdnn_data_sdk hip::host)
    clang_tidy_check(hipdnn_bench_tool)
    install(TARGETS hipdnn_bench_tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# The tests link hipdnn_frontend, so they need the same guard as the tool.
if(NOT HIPDNN_SKIP_TESTS AND HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    add_subdirectory(tests)
endif()
