// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Hand-written instance header for the P1 DSV3 blockscale tile config:
// FP8/FP8 inputs, BF16 output, Block2D<1, 128> X scales (per-token on M,
// blocked on K), Block2D<128, 128> W scales, kVector=16, kUseDot2=true,
// kMPerWarp=kNPerWarp=1, kHasBias=false. Mirrors the (M_Tile=64,
// QuantGroupSize=128) AITER blockscale path at small M.

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

#define GEMM_DECODE_BLOCKSCALE_KERNEL_DEFINED 1

#ifdef CK_TILE_USE_OCP_FP8
using SelectedADataType = ck_tile::fp8_t;
using SelectedBDataType = ck_tile::fp8_t;
using SelectedCDataType = ck_tile::bf16_t;

using SelectedXScaleLayout = ck_tile::GemmDecodeScaleLayout::Block2D<1, 128>;
using SelectedWScaleLayout = ck_tile::GemmDecodeScaleLayout::Block2D<128, 128>;
#else
// Without OCP FP8 the build still links by falling back to BF16 unscaled.
// The Block2D scale layouts are still expressed but the kernel is
// effectively dead; this matches the universal-instance fallback policy.
using SelectedADataType = ck_tile::bf16_t;
using SelectedBDataType = ck_tile::bf16_t;
using SelectedCDataType = ck_tile::bf16_t;

using SelectedXScaleLayout = ck_tile::GemmDecodeScaleLayout::Block2D<1, 128>;
using SelectedWScaleLayout = ck_tile::GemmDecodeScaleLayout::Block2D<128, 128>;
#endif

using SelectedGemmDecodeProblem =
    ck_tile::GemmDecodeProblem<SelectedADataType,
                               SelectedBDataType,
                               /*ComputeDataType=*/float,
                               SelectedCDataType,
                               /*XScaleDataType=*/float,
                               /*WScaleDataType=*/float,
                               SelectedXScaleLayout,
                               SelectedWScaleLayout,
                               /*kVector=*/16,
                               /*kUseDot2=*/true,
                               /*kUsePackedFp32=*/false,
                               /*kMPerWarp=*/1,
                               /*kNPerWarp=*/1,
                               ck_tile::GemmDecodeOutputAxis::SmallM,
                               /*kHasBias=*/false,
                               /*kWarpsPerBlock=*/1>;

using SelectedGemmDecodeBlockscaleKernel =
    ck_tile::GemmDecodeBlockscaleKernel<SelectedGemmDecodeProblem, ck_tile::GemmDecodePolicy>;
