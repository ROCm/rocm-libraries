# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(tilewright_selection_time LANGUAGES CXX)

set(TILEWRIGHT_BUILD_TESTING OFF)
set(TILEWRIGHT_ENABLE_PYTHON OFF)
set(TILEWRIGHT_ENABLE_INSTALL OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../.." tilewright EXCLUDE_FROM_ALL)

add_executable(bench_selection_time bench_selection_time.cpp)
target_link_libraries(bench_selection_time PRIVATE roc::tilewright)
target_compile_features(bench_selection_time PRIVATE cxx_std_17)

add_executable(bench_shipped_rank bench_shipped_rank.cpp)
target_link_libraries(bench_shipped_rank PRIVATE roc::tilewright)
target_compile_features(bench_shipped_rank PRIVATE cxx_std_17)

# Rankings a re-rank memo (TILEWRIGHT_LEVER_MEMO) may answer, hashed for
# comparing builds.
foreach(_tw_check rerank_check memo_check)
    add_executable(${_tw_check} ${_tw_check}.cpp)
    target_link_libraries(${_tw_check} PRIVATE roc::tilewright)
    target_compile_features(${_tw_check} PRIVATE cxx_std_17)
endforeach()
