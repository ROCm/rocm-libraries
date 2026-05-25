// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_harness.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

constexpr auto v3 = ck_tile::direct_conv::Version::v3;

// =============================================================================
// v3 tests — cross-wave LDS reduction.
//
// v3 KernelConfigurations layout:
//   Configs  0- 7: SwizzleType::None
//   Configs  8-15: SwizzleType::CyclicShift
//   Configs 16-23: SwizzleType::XOR
//
// Within each swizzle group (offset +0..+7):
//   +0: 16x16x32 Dgrad 4-wave  +1: 16x16x32 Dgrad 2-wave
//   +2: 16x16x32 Fprop 4-wave  +3: 16x16x32 Fprop 2-wave
//   +4: 32x32x16 Dgrad 4-wave  +5: 32x32x16 Dgrad 2-wave
//   +6: 32x32x16 Fprop 4-wave  +7: 32x32x16 Fprop 2-wave
//
// v3 requires C_in == block_c (= waves_per_wg * channels_per_group).
// K_out must be a multiple of block_k_size.
// =============================================================================

struct TileConv32cDenseKernelTraitsV3
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward32CDenseKernel<ConfigIdx, v3>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData32CDenseKernel<ConfigIdx, v3>;
};

// --- Fprop, v3 cross-wave LDS reduction ---

class DirectConvNonGrouped32cFp16V3FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 3: 2 waves — C=64, K=64
TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config3_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config3_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config3_C64_K64_Pad2)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 64, 3, 3, 2, 2)));
}

// Config 3: C != K
TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config3_C64_K128)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config3_C64_K16)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 16, 3, 3, 1, 1)));
}

// Config 3: larger spatial
TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config3_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<3>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config3_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<3>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

// Config 2: 4 waves — C=128, K=128
TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config2_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<2>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config2_C128_K128_NoPad)
{
    ASSERT_TRUE((RunFprop<2>(1, 8, 8, 1, 128, 128, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config2_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<2>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// Config 2: C != K
TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config2_C128_K64)
{
    ASSERT_TRUE((RunFprop<2>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config2_C128_K16)
{
    ASSERT_TRUE((RunFprop<2>(1, 8, 8, 1, 128, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3FpropTest, Fprop_Config2_C128_K128_Ho100)
{
    ASSERT_TRUE((RunFprop<2>(1, 100, 100, 1, 128, 128, 3, 3, 1, 1)));
}

// --- Dgrad, v3 cross-wave LDS reduction ---

class DirectConvNonGrouped32cFp16V3DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 1: 2 waves — C=64, K=64
TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config1_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config1_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config1_C64_K64_Pad2)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 64, 3, 3, 2, 2)));
}

// Config 1: C != K (Dgrad C_in=K=64, C_out=C, so K must be 64)
TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config1_C128_K64)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config1_C48_K64)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 48, 64, 3, 3, 1, 1)));
}

// Config 1: larger spatial
TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config1_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<1>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config1_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<1>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

// Config 0: 4 waves — C=128, K=128
TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config0_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config0_C128_K128_NoPad)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 1, 128, 128, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config0_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<0>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// Config 0: C != K (Dgrad C_in=K=128, C_out=C, so K must be 128)
TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config0_C64_K128)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3DgradTest, Dgrad_Config0_C128_K128_Ho100)
{
    ASSERT_TRUE((RunDgrad<0>(1, 100, 100, 1, 128, 128, 3, 3, 1, 1)));
}

// =============================================================================
// v3 tests — 32x32x16 MFMA shape
//
// 32x32x16: channels_per_group=16, block_k_size=32, block_q=32
//   Config 6: Fprop, 4 waves (C=64, K multiple of 32)
//   Config 7: Fprop, 2 waves (C=32, K multiple of 32)
//   Config 4: Dgrad, 4 waves (C_in=K=64, K_out=C multiple of 32)
//   Config 5: Dgrad, 2 waves (C_in=K=32, K_out=C multiple of 32)
// =============================================================================

// --- Fprop, v3 32x32x16 ---

class DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 7: 2 waves — C=32, K=32
TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config7_C32_K32_Pad1)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config7_C32_K32_NoPad)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config7_C32_K32_Pad2)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 32, 32, 3, 3, 2, 2)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config7_C32_K64)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 32, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config7_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<7>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config7_C32_K32_Ho100)
{
    ASSERT_TRUE((RunFprop<7>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

// Config 6: 4 waves — C=64, K=64
TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config6_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<6>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config6_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<6>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config6_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<6>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config6_C64_K32)
{
    ASSERT_TRUE((RunFprop<6>(1, 8, 8, 1, 64, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config6_C64_K128)
{
    ASSERT_TRUE((RunFprop<6>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32FpropTest, Fprop_Config6_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<6>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

// --- Dgrad, v3 32x32x16 ---

class DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 5: 2 waves — C_in=K=32, K_out=C multiple of 32
TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config5_C32_K32_Pad1)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config5_C32_K32_NoPad)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config5_C32_K32_Pad2)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 32, 32, 3, 3, 2, 2)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config5_C64_K32)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 64, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config5_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<5>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config5_C32_K32_Ho100)
{
    ASSERT_TRUE((RunDgrad<5>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

// Config 4: 4 waves — C_in=K=64, K_out=C multiple of 32
TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config4_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config4_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config4_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config4_C32_K64)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 1, 32, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Mfma32x32DgradTest, Dgrad_Config4_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<4>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

// =============================================================================
// v3 tests — CyclicShift swizzle (configs 8-15)
//
// Same MFMA shape / wave / direction mapping as configs 0-7, with
// SwizzleType::CyclicShift applied to input LDS.
//
//   Config  8: 16x16x32 Dgrad 4-wave  Config  9: 16x16x32 Dgrad 2-wave
//   Config 10: 16x16x32 Fprop 4-wave  Config 11: 16x16x32 Fprop 2-wave
//   Config 12: 32x32x16 Dgrad 4-wave  Config 13: 32x32x16 Dgrad 2-wave
//   Config 14: 32x32x16 Fprop 4-wave  Config 15: 32x32x16 Fprop 2-wave
// =============================================================================

// --- CyclicShift Fprop 16x16x32 ---

class DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K128)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<11>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<11>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config10_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<10>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config10_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<10>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- CyclicShift Dgrad 16x16x32 ---

class DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<9>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<9>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C128_K64)
{
    ASSERT_TRUE((RunDgrad<9>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<9>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<9>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config8_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config8_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<8>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- CyclicShift Fprop 32x32x16 ---

class DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32FpropTest, Fprop_Config15_C32_K32_Pad1)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32FpropTest, Fprop_Config15_C32_K32_NoPad)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32FpropTest, Fprop_Config15_C32_K64)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32FpropTest, Fprop_Config15_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<15>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32FpropTest, Fprop_Config15_C32_K32_Ho100)
{
    ASSERT_TRUE((RunFprop<15>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32FpropTest, Fprop_Config14_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<14>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32FpropTest, Fprop_Config14_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<14>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// --- CyclicShift Dgrad 32x32x16 ---

class DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32DgradTest, Dgrad_Config13_C32_K32_Pad1)
{
    ASSERT_TRUE((RunDgrad<13>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32DgradTest, Dgrad_Config13_C32_K32_NoPad)
{
    ASSERT_TRUE((RunDgrad<13>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32DgradTest, Dgrad_Config13_C64_K32)
{
    ASSERT_TRUE((RunDgrad<13>(1, 8, 8, 1, 64, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32DgradTest, Dgrad_Config13_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<13>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32DgradTest, Dgrad_Config13_C32_K32_Ho100)
{
    ASSERT_TRUE((RunDgrad<13>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32DgradTest, Dgrad_Config12_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<12>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftMfma32x32DgradTest, Dgrad_Config12_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<12>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// =============================================================================
// v3 tests — XOR swizzle (configs 16-25)
//
//   Config 16: 16x16x32 Dgrad 4-wave  Config 17: 16x16x32 Dgrad 2-wave
//   Config 18: 16x16x32 Fprop 4-wave  Config 19: 16x16x32 Fprop 2-wave
//   Config 20: 32x32x16 Dgrad 4-wave  Config 21: 32x32x16 Dgrad 2-wave
//   Config 22: 32x32x16 Fprop 4-wave  Config 23: 32x32x16 Fprop 2-wave
//   Config 24: 32x32x16 Dgrad 8-wave  Config 25: 32x32x16 Fprop 8-wave
// =============================================================================

// --- XOR Fprop 16x16x32 ---

class DirectConvNonGrouped32cFp16V3XorFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K128)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<19>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<19>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config18_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<18>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config18_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<18>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- XOR Dgrad 16x16x32 ---

class DirectConvNonGrouped32cFp16V3XorDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<17>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<17>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C128_K64)
{
    ASSERT_TRUE((RunDgrad<17>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<17>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<17>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config16_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<16>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config16_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<16>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- XOR Fprop 32x32x16 ---

class DirectConvNonGrouped32cFp16V3XorMfma32x32FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32FpropTest, Fprop_Config23_C32_K32_Pad1)
{
    ASSERT_TRUE((RunFprop<23>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32FpropTest, Fprop_Config23_C32_K32_NoPad)
{
    ASSERT_TRUE((RunFprop<23>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32FpropTest, Fprop_Config23_C32_K64)
{
    ASSERT_TRUE((RunFprop<23>(1, 8, 8, 1, 32, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32FpropTest, Fprop_Config23_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<23>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32FpropTest, Fprop_Config23_C32_K32_Ho100)
{
    ASSERT_TRUE((RunFprop<23>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32FpropTest, Fprop_Config22_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<22>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32FpropTest, Fprop_Config22_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<22>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// --- XOR Dgrad 32x32x16 ---

class DirectConvNonGrouped32cFp16V3XorMfma32x32DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32DgradTest, Dgrad_Config21_C32_K32_Pad1)
{
    ASSERT_TRUE((RunDgrad<21>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32DgradTest, Dgrad_Config21_C32_K32_NoPad)
{
    ASSERT_TRUE((RunDgrad<21>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32DgradTest, Dgrad_Config21_C64_K32)
{
    ASSERT_TRUE((RunDgrad<21>(1, 8, 8, 1, 64, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32DgradTest, Dgrad_Config21_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<21>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32DgradTest, Dgrad_Config21_C32_K32_Ho100)
{
    ASSERT_TRUE((RunDgrad<21>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32DgradTest, Dgrad_Config20_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<20>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorMfma32x32DgradTest, Dgrad_Config20_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<20>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// =============================================================================
// v3 tests — LDS-staged epilogue (16B DRAM writes)
//
// CyclicShift + LDS epilogue (configs 26-33):
//   Config 26: 16x16x32 Dgrad 4-wave  Config 27: 16x16x32 Dgrad 2-wave
//   Config 28: 16x16x32 Fprop 4-wave  Config 29: 16x16x32 Fprop 2-wave
//   Config 30: 32x32x16 Dgrad 4-wave  Config 31: 32x32x16 Dgrad 2-wave
//   Config 32: 32x32x16 Fprop 4-wave  Config 33: 32x32x16 Fprop 2-wave
//
// XOR + LDS epilogue (configs 34-43):
//   Config 34: 16x16x32 Dgrad 4-wave  Config 35: 16x16x32 Dgrad 2-wave
//   Config 36: 16x16x32 Fprop 4-wave  Config 37: 16x16x32 Fprop 2-wave
//   Config 38: 32x32x16 Dgrad 4-wave  Config 39: 32x32x16 Dgrad 2-wave
//   Config 40: 32x32x16 Fprop 4-wave  Config 41: 32x32x16 Fprop 2-wave
//   Config 42: 32x32x16 Dgrad 8-wave  Config 43: 32x32x16 Fprop 8-wave
// =============================================================================

// --- CyclicShift + LDS epilogue, Fprop 16x16x32 ---

class DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<29>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<29>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K128)
{
    ASSERT_TRUE((RunFprop<29>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<29>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<29>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config28_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<28>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config28_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<28>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- CyclicShift + LDS epilogue, Dgrad 16x16x32 ---

class DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<27>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<27>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C128_K64)
{
    ASSERT_TRUE((RunDgrad<27>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<27>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<27>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config26_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<26>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config26_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<26>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- CyclicShift + LDS epilogue, Fprop 32x32x16 ---

class DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32FpropTest, Fprop_Config33_C32_K32_Pad1)
{
    ASSERT_TRUE((RunFprop<33>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32FpropTest, Fprop_Config33_C32_K32_NoPad)
{
    ASSERT_TRUE((RunFprop<33>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32FpropTest, Fprop_Config33_C32_K64)
{
    ASSERT_TRUE((RunFprop<33>(1, 8, 8, 1, 32, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32FpropTest, Fprop_Config33_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<33>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32FpropTest, Fprop_Config33_C32_K32_Ho100)
{
    ASSERT_TRUE((RunFprop<33>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32FpropTest, Fprop_Config32_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<32>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32FpropTest, Fprop_Config32_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<32>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// --- CyclicShift + LDS epilogue, Dgrad 32x32x16 ---

class DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32DgradTest, Dgrad_Config31_C32_K32_Pad1)
{
    ASSERT_TRUE((RunDgrad<31>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32DgradTest, Dgrad_Config31_C32_K32_NoPad)
{
    ASSERT_TRUE((RunDgrad<31>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32DgradTest, Dgrad_Config31_C64_K32)
{
    ASSERT_TRUE((RunDgrad<31>(1, 8, 8, 1, 64, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32DgradTest, Dgrad_Config31_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<31>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32DgradTest, Dgrad_Config31_C32_K32_Ho100)
{
    ASSERT_TRUE((RunDgrad<31>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32DgradTest, Dgrad_Config30_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<30>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftMfma32x32DgradTest, Dgrad_Config30_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<30>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// --- XOR + LDS epilogue, Fprop 16x16x32 ---

class DirectConvNonGrouped32cFp16V3LdsXorFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<37>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<37>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K128)
{
    ASSERT_TRUE((RunFprop<37>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<37>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<37>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config36_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<36>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config36_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<36>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- XOR + LDS epilogue, Dgrad 16x16x32 ---

class DirectConvNonGrouped32cFp16V3LdsXorDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<35>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<35>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C128_K64)
{
    ASSERT_TRUE((RunDgrad<35>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<35>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<35>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config34_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<34>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config34_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<34>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- XOR + LDS epilogue, Fprop 32x32x16 ---

class DirectConvNonGrouped32cFp16V3LdsXorMfma32x32FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32FpropTest, Fprop_Config41_C32_K32_Pad1)
{
    ASSERT_TRUE((RunFprop<41>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32FpropTest, Fprop_Config41_C32_K32_NoPad)
{
    ASSERT_TRUE((RunFprop<41>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32FpropTest, Fprop_Config41_C32_K64)
{
    ASSERT_TRUE((RunFprop<41>(1, 8, 8, 1, 32, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32FpropTest, Fprop_Config41_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<41>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32FpropTest, Fprop_Config41_C32_K32_Ho100)
{
    ASSERT_TRUE((RunFprop<41>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32FpropTest, Fprop_Config40_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<40>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32FpropTest, Fprop_Config40_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<40>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// --- XOR + LDS epilogue, Dgrad 32x32x16 ---

class DirectConvNonGrouped32cFp16V3LdsXorMfma32x32DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32DgradTest, Dgrad_Config39_C32_K32_Pad1)
{
    ASSERT_TRUE((RunDgrad<39>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32DgradTest, Dgrad_Config39_C32_K32_NoPad)
{
    ASSERT_TRUE((RunDgrad<39>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32DgradTest, Dgrad_Config39_C64_K32)
{
    ASSERT_TRUE((RunDgrad<39>(1, 8, 8, 1, 64, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32DgradTest, Dgrad_Config39_C32_K32_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<39>(2, 16, 16, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32DgradTest, Dgrad_Config39_C32_K32_Ho100)
{
    ASSERT_TRUE((RunDgrad<39>(1, 100, 100, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32DgradTest, Dgrad_Config38_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<38>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorMfma32x32DgradTest, Dgrad_Config38_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<38>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

