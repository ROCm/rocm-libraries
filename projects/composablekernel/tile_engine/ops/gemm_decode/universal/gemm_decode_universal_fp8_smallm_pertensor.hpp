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
                               /*kMPerWarp=*/1,
                               /*kNPerWarp=*/1,
                               ck_tile::GemmDecodeOutputAxis::SmallM,
                               /*kHasBias=*/false,
                               /*kWarpsPerBlock=*/1>;

using SelectedGemmDecodeUniversalKernel =
    ck_tile::GemmDecodeUniversalKernel<SelectedGemmDecodeProblem, ck_tile::GemmDecodePolicy>;
