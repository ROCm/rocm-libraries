// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// MXFlatmmPipelineAGmemBGmemCRegV1 grouped-GEMM tests on gfx950 and gfx1250.

#include "test_grouped_gemm_mx_flatmm_common.hpp"

// clang-format off
using KernelTypes = ::testing::Types<
    //         ADataType, BDataType, CDataType, ArchTraits
#if defined(CK_USE_GFX950)
    std::tuple<F8,        F8,        F16,       MXFlatmm_GFX950_FP8FP8_Traits>,
    std::tuple<F4,        F4,        F16,       MXFlatmm_GFX950_FP4FP4_Traits>,
    std::tuple<F6,        F6,        F16,       MXFlatmm_GFX950_FP6FP6_Traits>,
    std::tuple<F8,        F4,        F16,       MXFlatmm_GFX950_FP8FP4_Traits>,
    std::tuple<F4,        F8,        F16,       MXFlatmm_GFX950_FP4FP8_Traits>,
#endif
#if defined(CK_USE_GFX1250)
    std::tuple<F8,        F8,        F16,       MXFlatmm_GFX1250_FP8FP8_Traits>,
    std::tuple<F4,        F4,        F16,       MXFlatmm_GFX1250_FP4FP4_Traits>,
    std::tuple<F6,        F6,        F16,       MXFlatmm_GFX1250_FP6FP6_Traits>,
    std::tuple<F8,        F4,        F16,       MXFlatmm_GFX1250_FP8FP4_Traits>,
    std::tuple<F4,        F8,        F16,       MXFlatmm_GFX1250_FP4FP8_Traits>
#endif
>;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemmMXFlatmm, KernelTypes);

#include "test_grouped_gemm_mx_flatmm_ut_cases.inc"
