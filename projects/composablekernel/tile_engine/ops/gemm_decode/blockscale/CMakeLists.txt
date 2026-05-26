# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# P1 wires the blockscale family with a single hand-written instance header
# (`gemm_decode_blockscale_single_dsv3.hpp`) targeting the DeepSeek-V3 a8w8
# blockscale convention (X = Block2D<1, 128>, W = Block2D<128, 128>). The
# Python instance builder + JSON sweep land in P1+; the directory layout
# mirrors `tile_engine/ops/gemm_decode/universal/` so the codegen can drop
# in here without re-architecting the build glue.

if(NOT GEMM_DECODE_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping gemm_decode blockscale: no supported GPU targets")
    return()
endif()

set(GEMM_DECODE_BLOCKSCALE_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

function(_gemm_decode_add_blockscale_instance target_name dtype_tag instance_header_name)
    set(instance_header ${GEMM_DECODE_BLOCKSCALE_SOURCE_DIR}/${instance_header_name})

    add_executable(${target_name} EXCLUDE_FROM_ALL
        ${GEMM_DECODE_BLOCKSCALE_SOURCE_DIR}/gemm_decode_blockscale_benchmark_single.cpp
        ${instance_header})

    set_property(TARGET ${target_name}
                 PROPERTY HIP_ARCHITECTURES ${GEMM_DECODE_GPU_TARGETS_INDIVIDUAL})

    target_include_directories(${target_name} PRIVATE
        ${GEMM_DECODE_SOURCE_DIR}
        ${GEMM_DECODE_BLOCKSCALE_SOURCE_DIR})

    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header})

    if(CK_USE_OCP_FP8)
        target_compile_options(${target_name} PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()

    add_dependencies(benchmark_gemm_decode_all ${target_name})
    add_dependencies(benchmark_gemm_decode_blockscale_all ${target_name})
    if(TARGET benchmark_gemm_decode_blockscale_${dtype_tag})
        add_dependencies(benchmark_gemm_decode_blockscale_${dtype_tag} ${target_name})
    endif()
    add_dependencies(benchmark_gemm_decode_blockscale_smallm ${target_name})
endfunction()

# DSV3 single-config target: FP8 SmallM, kVector=16, kMPerWarp=kNPerWarp=1,
# X = Block2D<1,128>, W = Block2D<128,128>.
if(CK_USE_OCP_FP8)
    _gemm_decode_add_blockscale_instance(
        benchmark_gemm_decode_blockscale_fp8_smallm_dsv3
        fp8
        gemm_decode_blockscale_single_dsv3.hpp)
endif()
