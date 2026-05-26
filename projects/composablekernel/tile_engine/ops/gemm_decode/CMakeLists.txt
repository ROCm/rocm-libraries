# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Tile engine entry point for the gemm_decode op family. Mirrors the layout
# used by tile_engine/ops/gemm/gemm_universal/ but at P0 is intentionally
# minimal: one universal/ subdirectory with one hand-written tile config.
# The MX and blockscale sub-families (P1/P2) and the Python instance builder
# / JSON sweep harness (P1) land later phases.

set(GEMM_DECODE_DATATYPE "bf16;fp16;fp8" CACHE STRING
    "List of datatypes for gemm_decode benchmarks (semicolon-separated). fp8 is only built when CK_USE_OCP_FP8 is ON.")
set(GEMM_DECODE_CONFIG_FILE "" CACHE STRING
    "Custom config file name (must be in configs/ folder)")
option(ENABLE_CCACHE_GEMM_DECODE "Enable ccache for gemm_decode ops compilation" OFF)

set(GEMM_DECODE_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Filter to architectures with verified support (SmallM warp-per-scalar
# exercises gfx9 64-wide wavefronts; gfx11/12 wave32 paths land in P3+).
set(GEMM_DECODE_GPU_TARGETS_INDIVIDUAL "")
set(_DECODE_DESIRED_TARGETS "gfx942;gfx950")
foreach(_target IN LISTS SUPPORTED_GPU_TARGETS)
    if(_target IN_LIST _DECODE_DESIRED_TARGETS)
        list(APPEND GEMM_DECODE_GPU_TARGETS_INDIVIDUAL ${_target})
    endif()
endforeach()

if(NOT GEMM_DECODE_GPU_TARGETS_INDIVIDUAL)
    message(STATUS "Skipping gemm_decode tile_engine: no gfx942/gfx950 in SUPPORTED_GPU_TARGETS=${SUPPORTED_GPU_TARGETS}")
    return()
endif()

if(ENABLE_CCACHE_GEMM_DECODE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    endif()
endif()

# Master collection targets. The naming mirrors gemm_universal so existing
# `ninja benchmark_gemm_decode_universal_all` style invocations work as soon
# as the codegen is wired in P1+.
add_custom_target(benchmark_gemm_decode_all)
add_custom_target(benchmark_gemm_decode_universal_all)
add_custom_target(benchmark_gemm_decode_blockscale_all)

# Datatype + orientation collection targets. The full sweep over
# (dtype, layout, orientation, traits, tile) lands with the codegen in P1+.
foreach(_dt IN LISTS GEMM_DECODE_DATATYPE)
    add_custom_target(benchmark_gemm_decode_universal_${_dt})
    add_custom_target(benchmark_gemm_decode_blockscale_${_dt})
endforeach()
add_custom_target(benchmark_gemm_decode_universal_smallm)
add_custom_target(benchmark_gemm_decode_universal_smalln)
add_custom_target(benchmark_gemm_decode_blockscale_smallm)

add_dependencies(benchmark_gemm_decode_all benchmark_gemm_decode_universal_all)
add_dependencies(benchmark_gemm_decode_all benchmark_gemm_decode_blockscale_all)

add_subdirectory(universal EXCLUDE_FROM_ALL)
add_subdirectory(blockscale EXCLUDE_FROM_ALL)
