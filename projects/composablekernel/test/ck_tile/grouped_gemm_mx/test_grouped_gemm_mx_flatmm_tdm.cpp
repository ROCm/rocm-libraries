// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TDM MX FLATMM grouped-GEMM tests.
// WeightPreshufflePipelineAGmemBGmemCRegTDM on gfx1250

#include "test_grouped_gemm_mx_flatmm_common.hpp"

// clang-format off
// gfx1250-only: built solely under `GPU_TARGETS MATCHES gfx1250` (see
// CMakeLists.txt). No gfx950 TDM traits exist, so no arch selection is needed.
using KernelTypes = ::testing::Types<
    //         ADataType, BDataType, CDataType, ArchTraits
    std::tuple<F8,        F8,        F16,       MXFlatmmTDM_GFX1250_FP8FP8_Traits>,
    std::tuple<F4,        F4,        F16,       MXFlatmmTDM_GFX1250_FP4FP4_Traits>,
    std::tuple<F8,        F4,        F16,       MXFlatmmTDM_GFX1250_FP8FP4_Traits>,
    std::tuple<F4,        F8,        F16,       MXFlatmmTDM_GFX1250_FP4FP8_Traits>
>;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemmMXFlatmm, KernelTypes);

#include "test_grouped_gemm_mx_flatmm_ut_cases.inc"
