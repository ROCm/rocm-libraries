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

# Helper that wires a single-instance benchmark executable. `dtype_tag` is
# one of the dtypes in GEMM_DECODE_DATATYPE so the per-dtype collection
# target picks it up.
function(_gemm_decode_add_universal_instance target_name dtype_tag instance_header_name)
    set(instance_header ${GEMM_DECODE_UNIVERSAL_SOURCE_DIR}/${instance_header_name})

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
    if(TARGET benchmark_gemm_decode_universal_${dtype_tag})
        add_dependencies(benchmark_gemm_decode_universal_${dtype_tag} ${target_name})
    endif()
    add_dependencies(benchmark_gemm_decode_universal_smallm ${target_name})
endfunction()

# BF16 SmallM, kVector=8, kMPerWarp=kNPerWarp=1, unscaled.
_gemm_decode_add_universal_instance(
    benchmark_gemm_decode_universal_bf16_smallm_default
    bf16
    gemm_decode_universal_single_default.hpp)

# FP8 SmallM, kVector=16, kMPerWarp=kNPerWarp=1, PerTensor x PerTensor.
# Only built when CK_USE_OCP_FP8 is on; without it the instance header falls
# back to BF16 (tracked by the GEMM_DECODE_DATATYPE list including fp8 only
# when the host build supports OCP FP8). We still emit the target here so
# `ninja benchmark_gemm_decode_universal_fp8` is a valid alias.
if(CK_USE_OCP_FP8)
    _gemm_decode_add_universal_instance(
        benchmark_gemm_decode_universal_fp8_smallm_pertensor
        fp8
        gemm_decode_universal_fp8_smallm_pertensor.hpp)
endif()
