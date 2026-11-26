// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The FP8/BF8 16x16x128 dense WMMA layout checks, built at -O0 (see CMakeLists.txt). On
// gfx1250-strict these ops use a fallback whose result depends on how the compiler passes the
// scale operand, which differs between -O0 and optimized builds (ROCM-35020).

#include "test_amdgcn_mma_layout.inc"

// clang-format off
using Gfx1250F8x128Intrinsics = ::testing::Types<
    amdgcn_mma<F8,   F8,   F16,  16u, 16u, 128u, Target1250, MmaOpFamily::DENSE>,
    amdgcn_mma<F8,   BF8,  F16,  16u, 16u, 128u, Target1250, MmaOpFamily::DENSE>,
    amdgcn_mma<BF8,  F8,   F16,  16u, 16u, 128u, Target1250, MmaOpFamily::DENSE>,
    amdgcn_mma<BF8,  BF8,  F16,  16u, 16u, 128u, Target1250, MmaOpFamily::DENSE>,
    amdgcn_mma<F8,   F8,   F32,  16u, 16u, 128u, Target1250, MmaOpFamily::DENSE>,
    amdgcn_mma<F8,   BF8,  F32,  16u, 16u, 128u, Target1250, MmaOpFamily::DENSE>,
    amdgcn_mma<BF8,  F8,   F32,  16u, 16u, 128u, Target1250, MmaOpFamily::DENSE>,
    amdgcn_mma<BF8,  BF8,  F32,  16u, 16u, 128u, Target1250, MmaOpFamily::DENSE>>;
// clang-format on

TYPED_TEST_SUITE(TestMmaLayout, Gfx1250F8x128Intrinsics);
TYPED_TEST(TestMmaLayout, Gfx1250F8x128IntrinsicsO0) { run_mma_layout_test<TypeParam>(); }
