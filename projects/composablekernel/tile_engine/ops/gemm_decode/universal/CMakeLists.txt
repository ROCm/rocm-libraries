# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# P0 wires up the universal family with a single hand-written instance header
# (`gemm_decode_universal_single_default.hpp`). The Python instance builder
# and JSON-driven sweep land in P1+; the directory layout / collection
# targets / per-instance executable model is preserved so the future codegen
# can drop into this exact CMake structure without re-architecting it.

if(NOT GEMM_DECODE_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping gemm_decode universal: no supported GPU targets")
    return()
endif()

set(GEMM_DECODE_UNIVERSAL_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Single tile config target (BF16 SmallM, kVector=8, kMPerWarp=kNPerWarp=1).
set(target_name benchmark_gemm_decode_universal_bf16_smallm_default)
set(instance_header
    ${GEMM_DECODE_UNIVERSAL_SOURCE_DIR}/gemm_decode_universal_single_default.hpp)

add_executable(${target_name} EXCLUDE_FROM_ALL
    ${GEMM_DECODE_UNIVERSAL_SOURCE_DIR}/gemm_decode_universal_benchmark_single.cpp
    ${instance_header})

set_property(TARGET ${target_name}
             PROPERTY HIP_ARCHITECTURES ${GEMM_DECODE_GPU_TARGETS_INDIVIDUAL})

target_include_directories(${target_name} PRIVATE
    ${GEMM_DECODE_SOURCE_DIR}
    ${GEMM_DECODE_UNIVERSAL_SOURCE_DIR})

target_compile_options(${target_name} PRIVATE
    -Wno-undefined-func-template
    -Wno-float-equal
    --offload-compress
    -include ${instance_header})

if(CK_USE_OCP_FP8)
    target_compile_options(${target_name} PRIVATE -DCK_TILE_USE_OCP_FP8)
endif()

add_dependencies(benchmark_gemm_decode_universal_all ${target_name})
add_dependencies(benchmark_gemm_decode_universal_bf16 ${target_name})
add_dependencies(benchmark_gemm_decode_universal_smallm ${target_name})
