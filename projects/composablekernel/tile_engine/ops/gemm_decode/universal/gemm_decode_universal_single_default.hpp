// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// One-shot instantiation header for the P0 default tile config:
// BF16/BF16 unscaled SmallM with kVector=8, kMPerWarp=kNPerWarp=1.
// In P1+ this header is generated per-config by
// `gemm_decode_instance_builder.py` driven from `configs/default_config.json`;
// during P0 the sweep matrix has a single entry so we keep the file as a
// hand-written reference of what the codegen will emit.

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

#define GEMM_DECODE_UNIVERSAL_KERNEL_DEFINED 1

using SelectedADataType = ck_tile::bf16_t;
using SelectedBDataType = ck_tile::bf16_t;
using SelectedCDataType = ck_tile::bf16_t;

using SelectedGemmDecodeProblem =
    ck_tile::GemmDecodeProblem<SelectedADataType,
                               SelectedBDataType,
                               /*ComputeDataType=*/float,
                               SelectedCDataType,
                               /*XScaleDataType=*/float,
                               /*WScaleDataType=*/float,
                               /*XScaleLayout=*/void,
                               /*WScaleLayout=*/void,
                               /*kVector=*/8,
                               /*kUseDot2=*/false,
                               /*kUsePackedFp32=*/false,
                               /*kMPerWarp=*/1,
                               /*kNPerWarp=*/1,
                               ck_tile::GemmDecodeOutputAxis::SmallM,
                               /*kHasBias=*/false,
                               /*kWarpsPerBlock=*/1>;

using SelectedGemmDecodeUniversalKernel =
    ck_tile::GemmDecodeUniversalKernel<SelectedGemmDecodeProblem, ck_tile::GemmDecodePolicy>;
