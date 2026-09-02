# The harvest end of corpus generation (RFC 0019.13 §5.3).
#
# Two targets on the same seam corpus_gen uses. hipdnn_bench_lib is header-only sizing and
# enumeration logic that needs no device, so it can be tested against tensors and constraints
# the test constructs. The executable adds the device and the engine.
add_library(hipdnn_bench_lib INTERFACE)

target_include_directories(hipdnn_bench_lib
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    find_package(hip REQUIRED)

    add_executable(hipdnn_bench_tool HipdnnBench.cpp)
    set_target_properties(hipdnn_bench_tool PROPERTIES OUTPUT_NAME hipdnn_bench)
    target_compile_options(hipdnn_bench_tool PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
    target_link_libraries(hipdnn_bench_tool
        PRIVATE hipdnn_bench_lib hipdnn_backend hipdnn_frontend hipdnn_data_sdk hip::host)
    clang_tidy_check(hipdnn_bench_tool)
    install(TARGETS hipdnn_bench_tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# The same guard the tool itself carries above. The tests link hipdnn_frontend, so with
# HIPDNN_BUILD_FRONTEND=OFF this resolved a link against a target that was never created and
# failed at configure time rather than being skipped.
if(NOT HIPDNN_SKIP_TESTS AND HIPDNN_BUILD_FRONTEND AND HIPDNN_BUILD_BACKEND)
    add_subdirectory(tests)
endif()
