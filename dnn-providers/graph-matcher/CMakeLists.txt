# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# graph_matcher is a host-only C++17 library (no HIP): a declarative op-graph
# matcher over the FlatBuffers Data SDK. It is linkable by any plugin, so its
# public surface stays C++17 and it depends only on hipdnn_flatbuffers_sdk.
project(hipdnn_graph_matcher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GRAPH_MATCHER_ENABLE_TESTS "Build graph_matcher unit tests" ON)

# SDK dependency: use the in-tree target under the superbuild, else find the
# installed package for a true standalone build.
if(NOT TARGET hipdnn_flatbuffers_sdk)
    find_package(hipdnn_flatbuffers_sdk CONFIG REQUIRED)
endif()

add_library(hipdnn_graph_matcher STATIC
    src/OpSchema.cpp
    src/GraphView.cpp
    src/CompiledPattern.cpp
    src/Matcher.cpp
    src/PatternCompiler.cpp
    src/Predicate.cpp
    src/PatternCodec.cpp
    src/PatternSet.cpp
)

target_include_directories(hipdnn_graph_matcher PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hipdnn_graph_matcher PUBLIC hipdnn_flatbuffers_sdk)

set_target_properties(hipdnn_graph_matcher PROPERTIES
    CXX_VISIBILITY_PRESET hidden
)

add_library(hipdnn::graph_matcher ALIAS hipdnn_graph_matcher)

# Build-time (AOT) criteria compiler: JSON -> serialized CompiledPattern bytes
# emitted as an embeddable C++ array. A host tool; enabled by default so the
# superbuild can invoke it in a custom command to embed in-tree criteria.
option(GRAPH_MATCHER_ENABLE_TOOLS "Build the AOT criteria compiler tool" ON)
if(GRAPH_MATCHER_ENABLE_TOOLS)
    add_executable(graph_matcher_compile_criteria tools/compile_criteria.cpp)
    target_link_libraries(graph_matcher_compile_criteria PRIVATE hipdnn_graph_matcher)
endif()

if(GRAPH_MATCHER_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Fuzz targets for the untrusted parsers/matcher. Off by default; require a
# libFuzzer-capable clang (Linux/CI). Enable with -DGRAPH_MATCHER_ENABLE_FUZZ=ON.
option(GRAPH_MATCHER_ENABLE_FUZZ "Build the libFuzzer targets" OFF)
if(GRAPH_MATCHER_ENABLE_FUZZ)
    foreach(_fuzz fromjson match)
        add_executable(graph_matcher_fuzz_${_fuzz} fuzz/fuzz_${_fuzz}.cpp)
        target_link_libraries(graph_matcher_fuzz_${_fuzz} PRIVATE hipdnn_graph_matcher)
        target_compile_options(graph_matcher_fuzz_${_fuzz}
            PRIVATE -g -O1 -fsanitize=fuzzer,address,undefined)
        target_link_options(graph_matcher_fuzz_${_fuzz}
            PRIVATE -fsanitize=fuzzer,address,undefined)
    endforeach()
endif()

install(TARGETS hipdnn_graph_matcher EXPORT hipdnn_graph_matcher_targets)
install(DIRECTORY include/ DESTINATION include)
