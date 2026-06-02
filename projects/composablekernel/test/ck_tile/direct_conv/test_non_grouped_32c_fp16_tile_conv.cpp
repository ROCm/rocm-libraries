// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_harness.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_32c_dense.hpp"
#include "ck_tile/ops/direct_convolution/configs/direct_conv_32c_dense_configs.hpp"
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
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward32CDenseKernel<
        ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<
            ck_tile::direct_conv::DataType::fp16>::configs_map.get(ConfigIdx),
        v3>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData32CDenseKernel<
        ck_tile::direct_conv::conv_32c_tile::v3::KernelConfigurations<
            ck_tile::direct_conv::DataType::fp16>::configs_map.get(ConfigIdx),
        v3>;
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

// =============================================================================
// v3 tests — wave counts 3,5,6,7 (CyclicShift only) + XOR 8-wave (40-43)
//
// For Fprop: C == waves_per_wg * 32, K must be multiple of 16.
// For Dgrad: K == waves_per_wg * 32, C must be multiple of 16.
//
// XOR swizzle requires power-of-2 waves_per_wg; odd wave counts use CyclicShift only.
//
// waves=3: block_c=96   (Fprop CyclicShift DRAM=25, LDS=27 / Dgrad DRAM=24, LDS=26)
// waves=5: block_c=160  (Fprop CyclicShift DRAM=29, LDS=31 / Dgrad DRAM=28, LDS=30)
// waves=6: block_c=192  (Fprop CyclicShift DRAM=33, LDS=35 / Dgrad DRAM=32, LDS=34)
// waves=7: block_c=224  (Fprop CyclicShift DRAM=37, LDS=39 / Dgrad DRAM=36, LDS=38)
// waves=8 XOR:          (Fprop XOR DRAM=41, LDS=43         / Dgrad DRAM=40, LDS=42)
//
// For waves=3,5,6,7: BLOCK_C8 does not divide 64, so the tile distribution
// maps fewer than 64 lanes. Excess lanes are masked by the load_active guard
// in the base InputLoader.
// =============================================================================

// --- waves=3 Fprop (CyclicShift DRAM=25, CyclicShift+LDS=27) ---

class DirectConvNonGrouped32cFp16V3Waves3FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves3FpropTest, Fprop_Cfg25_CyclicShift_DRAM_C96_K64)
{
    ASSERT_TRUE((RunFprop<25>(1, 8, 8, 1, 96, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves3FpropTest, Fprop_Cfg25_CyclicShift_DRAM_C96_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<25>(1, 8, 8, 1, 96, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves3FpropTest, Fprop_Cfg25_CyclicShift_DRAM_C96_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<25>(2, 16, 16, 1, 96, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves3FpropTest, Fprop_Cfg27_CyclicShift_LDS_C96_K64)
{
    ASSERT_TRUE((RunFprop<27>(1, 8, 8, 1, 96, 64, 3, 3, 1, 1)));
}

// --- waves=3 Dgrad (CyclicShift DRAM=24, CyclicShift+LDS=26) ---

class DirectConvNonGrouped32cFp16V3Waves3DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves3DgradTest, Dgrad_Cfg24_CyclicShift_DRAM_C64_K96)
{
    ASSERT_TRUE((RunDgrad<24>(1, 8, 8, 1, 64, 96, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves3DgradTest, Dgrad_Cfg24_CyclicShift_DRAM_C64_K96_NoPad)
{
    ASSERT_TRUE((RunDgrad<24>(1, 8, 8, 1, 64, 96, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves3DgradTest, Dgrad_Cfg24_CyclicShift_DRAM_C64_K96_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<24>(2, 16, 16, 1, 64, 96, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves3DgradTest, Dgrad_Cfg26_CyclicShift_LDS_C64_K96)
{
    ASSERT_TRUE((RunDgrad<26>(1, 8, 8, 1, 64, 96, 3, 3, 1, 1)));
}

// --- waves=5 Fprop (CyclicShift DRAM=29, CyclicShift+LDS=31) ---

class DirectConvNonGrouped32cFp16V3Waves5FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves5FpropTest, Fprop_Cfg29_CyclicShift_DRAM_C160_K64)
{
    ASSERT_TRUE((RunFprop<29>(1, 8, 8, 1, 160, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves5FpropTest, Fprop_Cfg29_CyclicShift_DRAM_C160_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<29>(1, 8, 8, 1, 160, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves5FpropTest, Fprop_Cfg29_CyclicShift_DRAM_C160_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<29>(2, 16, 16, 1, 160, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves5FpropTest, Fprop_Cfg31_CyclicShift_LDS_C160_K64)
{
    ASSERT_TRUE((RunFprop<31>(1, 8, 8, 1, 160, 64, 3, 3, 1, 1)));
}

// --- waves=5 Dgrad (CyclicShift DRAM=28, CyclicShift+LDS=30) ---

class DirectConvNonGrouped32cFp16V3Waves5DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves5DgradTest, Dgrad_Cfg28_CyclicShift_DRAM_C64_K160)
{
    ASSERT_TRUE((RunDgrad<28>(1, 8, 8, 1, 64, 160, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves5DgradTest, Dgrad_Cfg28_CyclicShift_DRAM_C64_K160_NoPad)
{
    ASSERT_TRUE((RunDgrad<28>(1, 8, 8, 1, 64, 160, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves5DgradTest, Dgrad_Cfg28_CyclicShift_DRAM_C64_K160_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<28>(2, 16, 16, 1, 64, 160, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves5DgradTest, Dgrad_Cfg30_CyclicShift_LDS_C64_K160)
{
    ASSERT_TRUE((RunDgrad<30>(1, 8, 8, 1, 64, 160, 3, 3, 1, 1)));
}

// --- waves=6 Fprop (CyclicShift DRAM=33, CyclicShift+LDS=35) ---

class DirectConvNonGrouped32cFp16V3Waves6FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves6FpropTest, Fprop_Cfg33_CyclicShift_DRAM_C192_K64)
{
    ASSERT_TRUE((RunFprop<33>(1, 8, 8, 1, 192, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves6FpropTest, Fprop_Cfg33_CyclicShift_DRAM_C192_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<33>(1, 8, 8, 1, 192, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves6FpropTest, Fprop_Cfg33_CyclicShift_DRAM_C192_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<33>(2, 16, 16, 1, 192, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves6FpropTest, Fprop_Cfg35_CyclicShift_LDS_C192_K64)
{
    ASSERT_TRUE((RunFprop<35>(1, 8, 8, 1, 192, 64, 3, 3, 1, 1)));
}

// --- waves=6 Dgrad (CyclicShift DRAM=32, CyclicShift+LDS=34) ---

class DirectConvNonGrouped32cFp16V3Waves6DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves6DgradTest, Dgrad_Cfg32_CyclicShift_DRAM_C64_K192)
{
    ASSERT_TRUE((RunDgrad<32>(1, 8, 8, 1, 64, 192, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves6DgradTest, Dgrad_Cfg32_CyclicShift_DRAM_C64_K192_NoPad)
{
    ASSERT_TRUE((RunDgrad<32>(1, 8, 8, 1, 64, 192, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves6DgradTest, Dgrad_Cfg32_CyclicShift_DRAM_C64_K192_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<32>(2, 16, 16, 1, 64, 192, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves6DgradTest, Dgrad_Cfg34_CyclicShift_LDS_C64_K192)
{
    ASSERT_TRUE((RunDgrad<34>(1, 8, 8, 1, 64, 192, 3, 3, 1, 1)));
}

// --- waves=7 Fprop (CyclicShift DRAM=37, CyclicShift+LDS=39) ---

class DirectConvNonGrouped32cFp16V3Waves7FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves7FpropTest, Fprop_Cfg37_CyclicShift_DRAM_C224_K64)
{
    ASSERT_TRUE((RunFprop<37>(1, 8, 8, 1, 224, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves7FpropTest, Fprop_Cfg37_CyclicShift_DRAM_C224_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<37>(1, 8, 8, 1, 224, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves7FpropTest, Fprop_Cfg37_CyclicShift_DRAM_C224_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<37>(2, 16, 16, 1, 224, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves7FpropTest, Fprop_Cfg39_CyclicShift_LDS_C224_K64)
{
    ASSERT_TRUE((RunFprop<39>(1, 8, 8, 1, 224, 64, 3, 3, 1, 1)));
}

// --- waves=7 Dgrad (CyclicShift DRAM=36, CyclicShift+LDS=38) ---

class DirectConvNonGrouped32cFp16V3Waves7DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves7DgradTest, Dgrad_Cfg36_CyclicShift_DRAM_C64_K224)
{
    ASSERT_TRUE((RunDgrad<36>(1, 8, 8, 1, 64, 224, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves7DgradTest, Dgrad_Cfg36_CyclicShift_DRAM_C64_K224_NoPad)
{
    ASSERT_TRUE((RunDgrad<36>(1, 8, 8, 1, 64, 224, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves7DgradTest, Dgrad_Cfg36_CyclicShift_DRAM_C64_K224_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<36>(2, 16, 16, 1, 64, 224, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves7DgradTest, Dgrad_Cfg38_CyclicShift_LDS_C64_K224)
{
    ASSERT_TRUE((RunDgrad<38>(1, 8, 8, 1, 64, 224, 3, 3, 1, 1)));
}

// --- Integration tests: odd-wave configs with larger K ---
// These test the failing profiler case (C=192, K=48) and similar shapes
// where K is not a power of 2 or requires multiple K-blocks.

class DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// waves=3 (C=96): moderate spatial, multiple block_q positions
TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Fprop_Cfg25_Waves3_C96_K48)
{
    ASSERT_TRUE((RunFprop<25>(2, 14, 14, 1, 96, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Dgrad_Cfg24_Waves3_C48_K96)
{
    ASSERT_TRUE((RunDgrad<24>(2, 14, 14, 1, 48, 96, 3, 3, 1, 1)));
}

// waves=5 (C=160): non-square spatial, wide W exercises many block_q values
TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Fprop_Cfg29_Waves5_C160_K48)
{
    ASSERT_TRUE((RunFprop<29>(1, 8, 56, 1, 160, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Dgrad_Cfg28_Waves5_C48_K160)
{
    ASSERT_TRUE((RunDgrad<28>(1, 8, 56, 1, 48, 160, 3, 3, 1, 1)));
}

// waves=6 (C=192): small spatial baseline
TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Fprop_Cfg33_Waves6_C192_K48)
{
    ASSERT_TRUE((RunFprop<33>(1, 8, 8, 1, 192, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Fprop_Cfg35_Waves6_C192_K48_LDS)
{
    ASSERT_TRUE((RunFprop<35>(1, 8, 8, 1, 192, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Dgrad_Cfg32_Waves6_C48_K192)
{
    ASSERT_TRUE((RunDgrad<32>(1, 8, 8, 1, 48, 192, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Dgrad_Cfg34_Waves6_C48_K192_LDS)
{
    ASSERT_TRUE((RunDgrad<34>(1, 8, 8, 1, 48, 192, 3, 3, 1, 1)));
}

// waves=6 (C=192): large spatial — matches the failing profiler test case
TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Fprop_Cfg33_Waves6_C192_K48_LargeSpatial)
{
    ASSERT_TRUE((RunFprop<33>(8, 64, 64, 1, 192, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Dgrad_Cfg32_Waves6_C48_K192_LargeSpatial)
{
    ASSERT_TRUE((RunDgrad<32>(8, 64, 64, 1, 48, 192, 3, 3, 1, 1)));
}

// waves=7 (C=224): W=17 just over block_q=16, tests block_q boundary
TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Fprop_Cfg37_Waves7_C224_K48)
{
    ASSERT_TRUE((RunFprop<37>(1, 17, 17, 1, 224, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest, Dgrad_Cfg36_Waves7_C48_K224)
{
    ASSERT_TRUE((RunDgrad<36>(1, 17, 17, 1, 48, 224, 3, 3, 1, 1)));
}

// --- waves=8 XOR Fprop (XOR DRAM=41, XOR+LDS=43) ---

class DirectConvNonGrouped32cFp16V3Waves8XorFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves8XorFpropTest, Fprop_Cfg41_XOR_DRAM_C256_K64)
{
    ASSERT_TRUE((RunFprop<41>(1, 8, 8, 1, 256, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves8XorFpropTest, Fprop_Cfg41_XOR_DRAM_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<41>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves8XorFpropTest, Fprop_Cfg43_XOR_LDS_C256_K64)
{
    ASSERT_TRUE((RunFprop<43>(1, 8, 8, 1, 256, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves8XorFpropTest, Fprop_Cfg43_XOR_LDS_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<43>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

// --- waves=8 XOR Dgrad (XOR DRAM=40, XOR+LDS=42) ---

class DirectConvNonGrouped32cFp16V3Waves8XorDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves8XorDgradTest, Dgrad_Cfg40_XOR_DRAM_C64_K256)
{
    ASSERT_TRUE((RunDgrad<40>(1, 8, 8, 1, 64, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves8XorDgradTest, Dgrad_Cfg40_XOR_DRAM_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<40>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves8XorDgradTest, Dgrad_Cfg42_XOR_LDS_C64_K256)
{
    ASSERT_TRUE((RunDgrad<42>(1, 8, 8, 1, 64, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves8XorDgradTest, Dgrad_Cfg42_XOR_LDS_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<42>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

// =============================================================================
// v3 c_slices_per_wave > 1 — 16x16x32 only.
//
// Each wave streams N C-chunks of 32 channels through the same fixed-size
// LDS buffers. total_block_c = waves_per_wg * N * 32. LDS footprint matches
// (waves_per_wg, N=1); the wave just does N× more MFMA per row.
//
//   Config 49: Fprop, waves=2, N=2 (total_block_c=128) — same C-coverage as
//              Config 2 (waves=4, N=1) with half the reduction LDS.
//   Config 50: Dgrad, waves=2, N=2 (total_block_c=128)
//   Config 51: Fprop, waves=2, N=4 (total_block_c=256)
//   Config 52: Dgrad, waves=2, N=4 (total_block_c=256)
//   Config 53/54: Fprop/Dgrad N=2 with CyclicShift swizzle
//   Config 55/56: Fprop/Dgrad N=2 with XOR swizzle
// =============================================================================

class DirectConvNonGrouped32cFp16V3CspwFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 49: Fprop, waves=2, N=2 — C=128, K=128, no swizzle
TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config49_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<44>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config49_C128_K128_NoPad)
{
    ASSERT_TRUE((RunFprop<44>(1, 8, 8, 1, 128, 128, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config49_C128_K64)
{
    ASSERT_TRUE((RunFprop<44>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config49_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<44>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config49_C128_K128_Ho100)
{
    ASSERT_TRUE((RunFprop<44>(1, 100, 100, 1, 128, 128, 3, 3, 1, 1)));
}

// Config 51: Fprop, waves=2, N=4 — C=256, K=128
TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config51_C256_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<46>(1, 8, 8, 1, 256, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config51_C256_K256)
{
    ASSERT_TRUE((RunFprop<46>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config51_C256_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<46>(2, 16, 16, 1, 256, 128, 3, 3, 1, 1)));
}

// Config 53: Fprop, waves=2, N=2, CyclicShift
TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config53_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<48>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

// Config 55: Fprop, waves=2, N=2, XOR
TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config55_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<50>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

class DirectConvNonGrouped32cFp16V3CspwDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 50: Dgrad, waves=2, N=2 — C_out=128, K=C_in=128
TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config50_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<45>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config50_C128_K128_NoPad)
{
    ASSERT_TRUE((RunDgrad<45>(1, 8, 8, 1, 128, 128, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config50_C64_K128)
{
    ASSERT_TRUE((RunDgrad<45>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config50_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<45>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// Config 52: Dgrad, waves=2, N=4 — K=256
TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config52_C128_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<47>(1, 8, 8, 1, 128, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config52_C256_K256)
{
    ASSERT_TRUE((RunDgrad<47>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

// Config 54: Dgrad, waves=2, N=2, CyclicShift
TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config54_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<49>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

// Config 56: Dgrad, waves=2, N=2, XOR
TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config56_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<51>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

// =============================================================================
// v3 c_slices_per_wave > 1, waves_per_wg = 4 — 16x16x32.
//
// Same chunked-streaming schedule as configs 49-56 but with 4 waves per WG.
// total_block_c = 4 * N * 32.
//
//   Config 57: Fprop, waves=4, N=2 (total_block_c=256)
//   Config 58: Dgrad, waves=4, N=2 (total_block_c=256)
//   Config 59: Fprop, waves=4, N=4 (total_block_c=512)
//   Config 60: Dgrad, waves=4, N=4 (total_block_c=512)
//   Config 61/62: Fprop/Dgrad N=2, CyclicShift
//   Config 63/64: Fprop/Dgrad N=2, XOR
// =============================================================================

// Config 57: Fprop, waves=4, N=2 — C=256
TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config57_C256_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<52>(1, 8, 8, 1, 256, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config57_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<52>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config57_C256_K256_NoPad)
{
    ASSERT_TRUE((RunFprop<52>(1, 8, 8, 1, 256, 256, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config57_C256_K256_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<52>(2, 16, 16, 1, 256, 256, 3, 3, 1, 1)));
}

// Config 59: Fprop, waves=4, N=4 — C=512
TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config59_C512_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<54>(1, 8, 8, 1, 512, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config59_C512_K256)
{
    ASSERT_TRUE((RunFprop<54>(1, 8, 8, 1, 512, 256, 3, 3, 1, 1)));
}

// Config 61: Fprop, waves=4, N=2, CyclicShift
TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config61_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<56>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

// Config 63: Fprop, waves=4, N=2, XOR
TEST_F(DirectConvNonGrouped32cFp16V3CspwFpropTest, Fprop_Config63_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<58>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

// Config 58: Dgrad, waves=4, N=2 — K=256
TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config58_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<53>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config58_C128_K256)
{
    ASSERT_TRUE((RunDgrad<53>(1, 8, 8, 1, 128, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config58_C256_K256_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<53>(2, 16, 16, 1, 256, 256, 3, 3, 1, 1)));
}

// Config 60: Dgrad, waves=4, N=4 — K=512
TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config60_C256_K512_Pad1)
{
    ASSERT_TRUE((RunDgrad<55>(1, 8, 8, 1, 256, 512, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config60_C128_K512)
{
    ASSERT_TRUE((RunDgrad<55>(1, 8, 8, 1, 128, 512, 3, 3, 1, 1)));
}

// Config 62: Dgrad, waves=4, N=2, CyclicShift
TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config62_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<57>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

// Config 64: Dgrad, waves=4, N=2, XOR
TEST_F(DirectConvNonGrouped32cFp16V3CspwDgradTest, Dgrad_Config64_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<59>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}
