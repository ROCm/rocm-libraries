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
// v3 tests — cross-wave LDS reduction, 16x16x32 MFMA only.
//
// v3 KernelConfigurations tested:
//   Configs  0- 3: SwizzleType::None
//   Configs  4- 7: SwizzleType::CyclicShift
//   Configs  8-11: SwizzleType::XOR
//   Configs 12-15: CyclicShift + LDS epilogue
//   Configs 16-19: XOR + LDS epilogue
//   Configs 20-23: CyclicShift 8-wave 3x3
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
// v3 tests — CyclicShift swizzle (configs 4-7)
//
//   Config 4: 16x16x32 Dgrad 4-wave  Config 5: 16x16x32 Dgrad 2-wave
//   Config 6: 16x16x32 Fprop 4-wave  Config 7: 16x16x32 Fprop 2-wave
// =============================================================================

// --- CyclicShift Fprop 16x16x32 ---

class DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K128)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<7>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config11_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<7>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config10_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<6>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftFpropTest, Fprop_Config10_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<6>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- CyclicShift Dgrad 16x16x32 ---

class DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C128_K64)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<5>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config9_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<5>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config8_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShiftDgradTest, Dgrad_Config8_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// =============================================================================
// v3 tests — XOR swizzle (configs 8-11)
//
//   Config  8: 16x16x32 Dgrad 4-wave  Config  9: 16x16x32 Dgrad 2-wave
//   Config 10: 16x16x32 Fprop 4-wave  Config 11: 16x16x32 Fprop 2-wave
// =============================================================================

// --- XOR Fprop 16x16x32 ---

class DirectConvNonGrouped32cFp16V3XorFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K128)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<11>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config19_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<11>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config18_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<10>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorFpropTest, Fprop_Config18_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<10>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- XOR Dgrad 16x16x32 ---

class DirectConvNonGrouped32cFp16V3XorDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<9>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<9>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C128_K64)
{
    ASSERT_TRUE((RunDgrad<9>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<9>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config17_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<9>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config16_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3XorDgradTest, Dgrad_Config16_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<8>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// =============================================================================
// v3 tests — LDS-staged epilogue (16B DRAM writes)
//
// CyclicShift + LDS epilogue (configs 12-15):
//   Config 12: 16x16x32 Dgrad 4-wave  Config 13: 16x16x32 Dgrad 2-wave
//   Config 14: 16x16x32 Fprop 4-wave  Config 15: 16x16x32 Fprop 2-wave
//
// XOR + LDS epilogue (configs 16-19):
//   Config 16: 16x16x32 Dgrad 4-wave  Config 17: 16x16x32 Dgrad 2-wave
//   Config 18: 16x16x32 Fprop 4-wave  Config 19: 16x16x32 Fprop 2-wave
// =============================================================================

// --- CyclicShift + LDS epilogue, Fprop 16x16x32 ---

class DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K128)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<15>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config29_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<15>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config28_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<14>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftFpropTest, Fprop_Config28_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<14>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- CyclicShift + LDS epilogue, Dgrad 16x16x32 ---

class DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<13>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<13>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C128_K64)
{
    ASSERT_TRUE((RunDgrad<13>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<13>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config27_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<13>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config26_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<12>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsCyclicShiftDgradTest, Dgrad_Config26_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<12>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- XOR + LDS epilogue, Fprop 16x16x32 ---

class DirectConvNonGrouped32cFp16V3LdsXorFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K128)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<19>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config37_C64_K64_Ho100)
{
    ASSERT_TRUE((RunFprop<19>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config36_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<18>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorFpropTest, Fprop_Config36_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<18>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// --- XOR + LDS epilogue, Dgrad 16x16x32 ---

class DirectConvNonGrouped32cFp16V3LdsXorDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<17>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<17>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C128_K64)
{
    ASSERT_TRUE((RunDgrad<17>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<17>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config35_C64_K64_Ho100)
{
    ASSERT_TRUE((RunDgrad<17>(1, 100, 100, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config34_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<16>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3LdsXorDgradTest, Dgrad_Config34_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<16>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// =============================================================================
// CyclicShift 8-wave 3x3 configs (20-23)
//   Config 20: Dgrad, DRAM epilogue (block_c=256)
//   Config 21: Fprop, DRAM epilogue (block_c=256)
//   Config 22: Dgrad, LDS epilogue  (block_c=256)
//   Config 23: Fprop, LDS epilogue  (block_c=256)
// =============================================================================

// --- CyclicShift 8-wave 3x3 Fprop ---

class DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3FpropTest, Fprop_Config45_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<21>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3FpropTest, Fprop_Config45_C256_K256_NoPad)
{
    ASSERT_TRUE((RunFprop<21>(1, 8, 8, 1, 256, 256, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3FpropTest, Fprop_Config45_C256_K128)
{
    ASSERT_TRUE((RunFprop<21>(1, 8, 8, 1, 256, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3FpropTest, Fprop_Config45_C256_K256_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<21>(2, 16, 16, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3FpropTest, Fprop_Config47_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<23>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3FpropTest, Fprop_Config47_C256_K128)
{
    ASSERT_TRUE((RunFprop<23>(1, 8, 8, 1, 256, 128, 3, 3, 1, 1)));
}

// --- CyclicShift 8-wave 3x3 Dgrad ---

class DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3DgradTest, Dgrad_Config44_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<20>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3DgradTest, Dgrad_Config44_C256_K256_NoPad)
{
    ASSERT_TRUE((RunDgrad<20>(1, 8, 8, 1, 256, 256, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3DgradTest, Dgrad_Config44_C128_K256)
{
    ASSERT_TRUE((RunDgrad<20>(1, 8, 8, 1, 128, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3DgradTest, Dgrad_Config44_C256_K256_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<20>(2, 16, 16, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3DgradTest, Dgrad_Config46_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<22>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3DgradTest, Dgrad_Config46_C128_K256)
{
    ASSERT_TRUE((RunDgrad<22>(1, 8, 8, 1, 128, 256, 3, 3, 1, 1)));
}
