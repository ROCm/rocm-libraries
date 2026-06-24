// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Hand-written instance header for the P0b FP8 PerTensor SmallM tile config:
// FP8/FP8 inputs, BF16 output, FP32 PerTensor scales for X and W, kVector=16,
// kUseDot2=true, kMPerWarp=kNPerWarp=1, kHasBias=false.
//
// Mirrors gemm_decode_universal_single_default.hpp; once the codegen lands in
// P1+ this header (and the bias variant) is what the per-config emitter will
// produce for the (FP8 PerTensor SmallM) row of the sweep matrix.

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

#define GEMM_DECODE_UNIVERSAL_KERNEL_DEFINED 1

// Sweep knobs the codegen / benchmark driver overrides per config via -D.
// kVector is fixed by the dot2 contract on this path (16 for the FP8 dot2
// K-loop, 8 for the BF16 fallback), so it is not a -D knob here.
#ifndef GEMM_DECODE_M_PER_WARP
#define GEMM_DECODE_M_PER_WARP 1
#endif
#ifndef GEMM_DECODE_N_PER_WARP
#define GEMM_DECODE_N_PER_WARP 1
#endif
#ifndef GEMM_DECODE_CHIPLET_SWIZZLE
#define GEMM_DECODE_CHIPLET_SWIZZLE false
#endif
#ifndef GEMM_DECODE_CHIPLET_NUM_XCDS
#define GEMM_DECODE_CHIPLET_NUM_XCDS 8
#endif
#ifndef GEMM_DECODE_CHIPLET_CHUNK
#define GEMM_DECODE_CHIPLET_CHUNK 8
#endif
// P0 wvSplitKQ-recipe knobs (default off). The builder overrides these for the
// multi-warp / fat-WG sweep rows.
#ifndef GEMM_DECODE_WARPS_PER_BLOCK
#define GEMM_DECODE_WARPS_PER_BLOCK 1
#endif
#ifndef GEMM_DECODE_STAGE_A_IN_LDS
#define GEMM_DECODE_STAGE_A_IN_LDS false
#endif
#ifndef GEMM_DECODE_STREAM_B
#define GEMM_DECODE_STREAM_B false
#endif
#ifndef GEMM_DECODE_PERSISTENT
#define GEMM_DECODE_PERSISTENT false
#endif

#ifdef CK_TILE_USE_OCP_FP8
using SelectedADataType = ck_tile::fp8_t;
using SelectedBDataType = ck_tile::fp8_t;
using SelectedCDataType = ck_tile::bf16_t;
#else
// Build is configured without OCP FP8; fall back to BF16 so the executable
// still links. Running with -DCK_USE_OCP_FP8=ON re-routes to the FP8 path.
using SelectedADataType = ck_tile::bf16_t;
using SelectedBDataType = ck_tile::bf16_t;
using SelectedCDataType = ck_tile::bf16_t;
#endif

using SelectedGemmDecodeProblem =
    ck_tile::GemmDecodeProblem<SelectedADataType,
                               SelectedBDataType,
                               /*ComputeDataType=*/float,
                               SelectedCDataType,
                               /*XScaleDataType=*/float,
                               /*WScaleDataType=*/float,
#ifdef CK_TILE_USE_OCP_FP8
                               ck_tile::GemmDecodeScaleLayout::PerTensor,
                               ck_tile::GemmDecodeScaleLayout::PerTensor,
                               /*kVector=*/16,
                               /*kUseDot2=*/true,
#else
                               /*XScaleLayout=*/void,
                               /*WScaleLayout=*/void,
                               /*kVector=*/8,
                               /*kUseDot2=*/false,
#endif
                               /*kUsePackedFp32=*/false,
                               /*kMPerWarp=*/GEMM_DECODE_M_PER_WARP,
                               /*kNPerWarp=*/GEMM_DECODE_N_PER_WARP,
                               ck_tile::GemmDecodeOutputAxis::SmallM,
                               /*kHasBias=*/false,
                               /*kWarpsPerBlock=*/GEMM_DECODE_WARPS_PER_BLOCK,
                               /*kBPreshuffle=*/false,
                               /*kChipletSwizzle=*/GEMM_DECODE_CHIPLET_SWIZZLE,
                               /*kChipletNumXcds=*/GEMM_DECODE_CHIPLET_NUM_XCDS,
                               /*kChipletChunkSize=*/GEMM_DECODE_CHIPLET_CHUNK,
                               /*kBias2D=*/false,
                               /*kStageAInLds=*/GEMM_DECODE_STAGE_A_IN_LDS,
                               /*kStreamB=*/GEMM_DECODE_STREAM_B,
                               /*kPersistent=*/GEMM_DECODE_PERSISTENT>;

using SelectedGemmDecodeUniversalKernel =
    ck_tile::GemmDecodeUniversalKernel<SelectedGemmDecodeProblem, ck_tile::GemmDecodePolicy>;
