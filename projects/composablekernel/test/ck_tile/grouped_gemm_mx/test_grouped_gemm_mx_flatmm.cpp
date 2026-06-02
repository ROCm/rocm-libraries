// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_gemm_mx_flatmm_util.hpp"

using F8  = ck_tile::fp8_t;
using BF8 = ck_tile::bf8_t;
using F16 = ck_tile::half_t;
using F32 = float;
using PF4 = ck_tile::pk_fp4_t;
using PF6 = ck_tile::pk_fp6x16_t;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ADataType, BDataType, CDataType, ArchTraits
#if defined(CK_USE_GFX950)
    // GFX950: MXFlatmmPipelineAGmemBGmemCRegV1
    std::tuple<F8,        F8,        F16,       MXFlatmm_GFX950_FP8FP8_Traits>,
    std::tuple<PF4,       PF4,       F16,       MXFlatmm_GFX950_FP4FP4_Traits>,
    std::tuple<F8,        PF4,       F16,       MXFlatmm_GFX950_FP8FP4_Traits>,
    std::tuple<PF4,       F8,        F16,       MXFlatmm_GFX950_FP4FP8_Traits>,
#endif
#if defined(CK_USE_GFX1250)
    // GFX1250: MXFlatmmPipelineAGmemBGmemCRegV1 (non-TDM)
    std::tuple<F8,        F8,        F16,       MXFlatmm_GFX1250_FP8FP8_Traits>,
    std::tuple<PF4,       PF4,       F16,       MXFlatmm_GFX1250_FP4FP4_Traits>,
    std::tuple<F8,        PF4,       F16,       MXFlatmm_GFX1250_FP8FP4_Traits>,
    std::tuple<PF4,       F8,        F16,       MXFlatmm_GFX1250_FP4FP8_Traits>,
    // GFX1250: MXFlatmmTDM (WeightPreshufflePipelineAGmemBGmemCRegTDM)
    std::tuple<F8,        F8,        F16,       MXFlatmmTDM_GFX1250_FP8FP8_Traits>,
    std::tuple<PF4,       PF4,       F16,       MXFlatmmTDM_GFX1250_FP4FP4_Traits>,
    std::tuple<F8,        PF4,       F16,       MXFlatmmTDM_GFX1250_FP8FP4_Traits>,
    std::tuple<PF4,       F8,        F16,       MXFlatmmTDM_GFX1250_FP4FP8_Traits>
#endif
>;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemmMXFlatmm, KernelTypes);

#include "test_grouped_gemm_mx_flatmm_ut_cases.inc"
