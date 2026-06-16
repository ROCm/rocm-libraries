// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_harness.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_32c_dense.hpp"
#include "configs/direct_conv_32c_dense_configs.hpp"
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

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3FpropTest,
       Fprop_Config45_C256_K256_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3CyclicShift8wave3x3DgradTest,
       Dgrad_Config44_C256_K256_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3Waves3FpropTest,
       Fprop_Cfg25_CyclicShift_DRAM_C96_K64_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3Waves3DgradTest,
       Dgrad_Cfg24_CyclicShift_DRAM_C64_K96_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3Waves5FpropTest,
       Fprop_Cfg29_CyclicShift_DRAM_C160_K64_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3Waves5DgradTest,
       Dgrad_Cfg28_CyclicShift_DRAM_C64_K160_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3Waves6FpropTest,
       Fprop_Cfg33_CyclicShift_DRAM_C192_K64_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3Waves6DgradTest,
       Dgrad_Cfg32_CyclicShift_DRAM_C64_K192_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3Waves7FpropTest,
       Fprop_Cfg37_CyclicShift_DRAM_C224_K64_LargerSpatial)
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

TEST_F(DirectConvNonGrouped32cFp16V3Waves7DgradTest,
       Dgrad_Cfg36_CyclicShift_DRAM_C64_K224_LargerSpatial)
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
TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest,
       Fprop_Cfg33_Waves6_C192_K48_LargeSpatial)
{
    ASSERT_TRUE((RunFprop<33>(8, 64, 64, 1, 192, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OddWavesIntegrationTest,
       Dgrad_Cfg32_Waves6_C48_K192_LargeSpatial)
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

// =============================================================================
// Channel padding (Phase 1) — 8-channel-granularity reduction + output masking.
//
// Exercises shapes where the reduction-channel count (C_in) is NOT an exact
// multiple of the config's total_block_c(), and/or the output-channel count
// (K_out) is NOT a multiple of block_k_size (16). Both must still be multiples
// of 8. Channel padding zero-fills the partial reduction tiles and masks the
// partial output tiles.
//
// Only the SwizzleType::None DRAM-epilogue configs (0-3) are eligible:
//   Config 0: Dgrad 4-wave (total_block_c=128)  Config 1: Dgrad 2-wave (=64)
//   Config 2: Fprop 4-wave (total_block_c=128)  Config 3: Fprop 2-wave (=64)
//
// is_valid_config requires C_in to land in
// (total_block_c - 32, total_block_c], so the minimal covering config is used.
// =============================================================================

// --- Fprop channel padding ---

class DirectConvNonGrouped32cFp16V3ChannelPadFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 3 (2-wave, total_block_c=64): partial reduction channels.
TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg3_C48_K64)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 48, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg3_C40_K64)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 40, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg3_C56_K64)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 56, 64, 3, 3, 1, 1)));
}

// Config 3: partial output channels (K not a multiple of block_k_size=16).
TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg3_C64_K24)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg3_C64_K8)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 8, 3, 3, 1, 1)));
}

// Config 3: both reduction and output padding.
TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg3_C48_K24)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 48, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg3_C48_K24_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<3>(2, 16, 16, 1, 48, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg3_C48_K24_Ho100)
{
    ASSERT_TRUE((RunFprop<3>(1, 100, 100, 1, 48, 24, 3, 3, 1, 1)));
}

// Config 2 (4-wave, total_block_c=128): partial reduction + output channels.
TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg2_C112_K64)
{
    ASSERT_TRUE((RunFprop<2>(1, 8, 8, 1, 112, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg2_C120_K72)
{
    ASSERT_TRUE((RunFprop<2>(1, 8, 8, 1, 120, 72, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadFpropTest, Fprop_Cfg2_C104_K40_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<2>(2, 16, 16, 1, 104, 40, 3, 3, 1, 1)));
}

// --- Dgrad channel padding ---
// For Dgrad: reduction dim C_in = k_tot, output dim K_out = c_tot.

class DirectConvNonGrouped32cFp16V3ChannelPadDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 1 (2-wave, total_block_c=64): partial reduction channels (K=k_tot).
TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg1_C64_K48)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg1_C64_K40)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 40, 3, 3, 1, 1)));
}

// Config 1: partial output channels (C=c_tot not a multiple of 16).
// Note: the output dim (c_tot for Dgrad) must stay > 32 — is_non_grouped()
// routes c_tot <= 32 to the grouped kernels, so output padding below 33 is
// unreachable through the dense dispatch.
TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg1_C40_K64)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 40, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg1_C56_K64)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 56, 64, 3, 3, 1, 1)));
}

// Config 1: both reduction and output padding.
TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg1_C40_K48)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 40, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg1_C40_K48_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<1>(2, 16, 16, 1, 40, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg1_C40_K48_Ho100)
{
    ASSERT_TRUE((RunDgrad<1>(1, 100, 100, 1, 40, 48, 3, 3, 1, 1)));
}

// Config 0 (4-wave, total_block_c=128): partial reduction + output channels.
TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg0_C128_K112)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 1, 128, 112, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg0_C72_K120)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 1, 72, 120, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ChannelPadDgradTest, Dgrad_Cfg0_C40_K104_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<0>(2, 16, 16, 1, 40, 104, 3, 3, 1, 1)));
}

// =============================================================================
// v3 Phase 2 — sub-8 channel padding tests.
//
// Channel counts that are NOT multiples of 8. The straddling channel tile is
// loaded and zeroed at element granularity:
//   - reduction (C_in): the weight loader element-wise zeros lanes >= C_real,
//     so the MFMA contribution of the partial tile's invalid lanes is exactly 0
//     (the input partial tile is loaded as-is — its garbage lanes x 0 = 0).
//   - output (K_out): the output writer masks per-element, so partial 4-K / 8-K
//     write groups straddling K_real only emit the valid channels.
//
// Same config eligibility as Phase 1: only the SwizzleType::None DRAM-epilogue
// configs (0-3) are padding-eligible.
//   Config 0: Dgrad 4-wave (total_block_c=128)  Config 1: Dgrad 2-wave (=64)
//   Config 2: Fprop 4-wave (total_block_c=128)  Config 3: Fprop 2-wave (=64)
// =============================================================================

// --- Fprop sub-8 channel padding ---

class DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 3 (2-wave, total_block_c=64): sub-8 reduction channels.
// C_in must land in (32, 64].
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C44_K64)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 44, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C60_K64)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 60, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C33_K64)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 33, 64, 3, 3, 1, 1)));
}

// Config 3: sub-8 output channels (k_tot not a multiple of 8).
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C64_K20)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 20, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C64_K12)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 12, 3, 3, 1, 1)));
}

// Sub-4 output: K=6 exercises per-element masking within a single 4-K group.
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C64_K6)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 6, 3, 3, 1, 1)));
}

// Sub-8 output spanning multiple K-blocks (k_tot = 66 > block_k_size=16).
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C64_K66)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 64, 66, 3, 3, 1, 1)));
}

// Config 3: both reduction and output sub-8.
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C44_K20)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 44, 20, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C44_K20_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<3>(2, 16, 16, 1, 44, 20, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg3_C33_K6_Ho50)
{
    ASSERT_TRUE((RunFprop<3>(1, 50, 50, 1, 33, 6, 3, 3, 1, 1)));
}

// Config 2 (4-wave, total_block_c=128): C_in in (96, 128], sub-8 both dims.
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg2_C100_K20)
{
    ASSERT_TRUE((RunFprop<2>(1, 8, 8, 1, 100, 20, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadFpropTest, Fprop_Cfg2_C108_K70_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<2>(2, 16, 16, 1, 108, 70, 3, 3, 1, 1)));
}

// --- Dgrad sub-8 channel padding ---
// For Dgrad: reduction dim C_in = k_tot, output dim K_out = c_tot (c_tot > 32).

class DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 1 (2-wave, total_block_c=64): sub-8 reduction channels (k_tot in (32,64]).
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C64_K44)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 44, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C64_K60)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 60, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C64_K33)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 33, 3, 3, 1, 1)));
}

// Config 1: sub-8 output channels (c_tot not a multiple of 8, must stay > 32).
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C44_K64)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 44, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C36_K64)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 36, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C52_K64)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 52, 64, 3, 3, 1, 1)));
}

// Config 1: both reduction and output sub-8.
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C44_K44)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 44, 44, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C44_K44_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<1>(2, 16, 16, 1, 44, 44, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg1_C36_K33_Ho50)
{
    ASSERT_TRUE((RunDgrad<1>(1, 50, 50, 1, 36, 33, 3, 3, 1, 1)));
}

// Config 0 (4-wave, total_block_c=128): k_tot in (96, 128], sub-8 both dims.
TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg0_C100_K100)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 1, 100, 100, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3SubChannelPadDgradTest, Dgrad_Cfg0_C36_K108_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<0>(2, 16, 16, 1, 36, 108, 3, 3, 1, 1)));
}

// =============================================================================
// Output-only channel padding with swizzled / multi-slice configs.
//
// When only the OUTPUT channel count is padded (the reduction C_in exactly
// fills total_block_c()), is_valid_config imposes NO swizzle / c_slices_per_wave
// restriction: output padding is handled solely by the output writer's
// element-precise K-mask and the weight loader's K-row zeroing, both of which
// are independent of the swizzle and the number of c-slices. These tests cover
// that path on waves_per_wg=4, c_slices_per_wave=2 (total_block_c=256) configs
// with None / CyclicShift / XOR swizzle.
//   Fprop configs: 52=None, 56=CyclicShift, 58=XOR.
//   Dgrad configs: 53=None, 57=CyclicShift, 59=XOR.
// =============================================================================

class DirectConvNonGrouped32cFp16V3OutputPadSwizzleFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Reduction C=256 fills total_block_c; output K=36 is padded (36 % 16 != 0).
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleFpropTest, Fprop_Cfg52_None_C256_K36)
{
    ASSERT_TRUE((RunFprop<52>(1, 8, 8, 1, 256, 36, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleFpropTest, Fprop_Cfg56_CyclicShift_C256_K36)
{
    ASSERT_TRUE((RunFprop<56>(1, 8, 8, 1, 256, 36, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleFpropTest, Fprop_Cfg58_XOR_C256_K36)
{
    ASSERT_TRUE((RunFprop<58>(1, 8, 8, 1, 256, 36, 3, 3, 1, 1)));
}

// The requested large output shape (256, 2376): K=2376 is padded (2376 % 16 != 0).
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleFpropTest, Fprop_Cfg56_CyclicShift_C256_K2376)
{
    ASSERT_TRUE((RunFprop<56>(1, 8, 8, 1, 256, 2376, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleFpropTest, Fprop_Cfg58_XOR_C256_K2376)
{
    ASSERT_TRUE((RunFprop<58>(1, 8, 8, 1, 256, 2376, 3, 3, 1, 1)));
}

// Unpadded sanity: K=256 is a multiple of 16, so no padding is engaged.
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleFpropTest, Fprop_Cfg56_CyclicShift_C256_K256)
{
    ASSERT_TRUE((RunFprop<56>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleFpropTest, Fprop_Cfg56_CyclicShift_C256_K36_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<56>(2, 16, 16, 1, 256, 36, 3, 3, 1, 1)));
}

class DirectConvNonGrouped32cFp16V3OutputPadSwizzleDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// For Dgrad: reduction C_in = k_tot, output K_out = c_tot (must be > 32).
// Reduction K=256 fills total_block_c; output C=36 is padded (36 % 16 != 0).
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleDgradTest, Dgrad_Cfg53_None_C36_K256)
{
    ASSERT_TRUE((RunDgrad<53>(1, 8, 8, 1, 36, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleDgradTest, Dgrad_Cfg57_CyclicShift_C36_K256)
{
    ASSERT_TRUE((RunDgrad<57>(1, 8, 8, 1, 36, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleDgradTest, Dgrad_Cfg59_XOR_C36_K256)
{
    ASSERT_TRUE((RunDgrad<59>(1, 8, 8, 1, 36, 256, 3, 3, 1, 1)));
}

// Output C=72 padded (72 % 16 != 0), reduction K=256 clean.
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleDgradTest, Dgrad_Cfg57_CyclicShift_C72_K256)
{
    ASSERT_TRUE((RunDgrad<57>(1, 8, 8, 1, 72, 256, 3, 3, 1, 1)));
}

// =============================================================================
// v3 Phase 2 — REDUCTION-channel padding combined with SWIZZLE.
//
// The descriptor-based input path makes the reduction-pad validity check
// swizzle-aware: a thread is masked iff the SWIZZLED physical channel block it
// loads, swizzled_c8(global_spatial, logical_c8), lands at or beyond
// ceil(C_in/8). Because that check is derived from the same forward-swizzle
// descriptor as the load, pad and swizzle compose, lifting the old restriction
// that reduction padding required swizzle_type == None.
//
// Eligible: DRAM-epilogue CyclicShift/XOR configs with c_slices_per_wave == 1:
//   Config 4: CyclicShift Dgrad 4-wave   Config 5: CyclicShift Dgrad 2-wave
//   Config 6: CyclicShift Fprop 4-wave   Config 7: CyclicShift Fprop 2-wave
//   Config 8: XOR Dgrad 4-wave           Config 9: XOR Dgrad 2-wave
//   Config 10: XOR Fprop 4-wave          Config 11: XOR Fprop 2-wave
// =============================================================================

// --- Fprop reduction padding + swizzle (reduction dim C_in = c_tot) ---

class DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 7 (CyclicShift, 2-wave, total_block_c=64): partial reduction channels.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg7_CS_C48_K64)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 48, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg7_CS_C40_K64)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 40, 64, 3, 3, 1, 1)));
}

// Sub-8 reduction with swizzle.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg7_CS_C44_K64)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 44, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg7_CS_C48_K24_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<7>(2, 16, 16, 1, 48, 24, 3, 3, 1, 1)));
}

// Config 11 (XOR, 2-wave, total_block_c=64).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg11_XOR_C48_K64)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 48, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg11_XOR_C44_K64)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 44, 64, 3, 3, 1, 1)));
}

// Config 6 (CyclicShift, 4-wave, total_block_c=128).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg6_CS_C112_K64)
{
    ASSERT_TRUE((RunFprop<6>(1, 8, 8, 1, 112, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg6_CS_C100_K40_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<6>(2, 16, 16, 1, 100, 40, 3, 3, 1, 1)));
}

// Config 10 (XOR, 4-wave, total_block_c=128).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg10_XOR_C112_K64)
{
    ASSERT_TRUE((RunFprop<10>(1, 8, 8, 1, 112, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleFpropTest, Fprop_Cfg10_XOR_C120_K72)
{
    ASSERT_TRUE((RunFprop<10>(1, 8, 8, 1, 120, 72, 3, 3, 1, 1)));
}

// --- Dgrad reduction padding + swizzle (reduction dim C_in = k_tot) ---
// Output dim K_out = c_tot must stay > 32 (is_non_grouped).

class DirectConvNonGrouped32cFp16V3ReductionPadSwizzleDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 5 (CyclicShift, Dgrad, 2-wave, total_block_c=64): partial reduction K.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleDgradTest, Dgrad_Cfg5_CS_C64_K48)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 64, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleDgradTest, Dgrad_Cfg5_CS_C64_K44)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 64, 44, 3, 3, 1, 1)));
}

// Config 9 (XOR, Dgrad, 2-wave).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleDgradTest, Dgrad_Cfg9_XOR_C64_K48)
{
    ASSERT_TRUE((RunDgrad<9>(1, 8, 8, 1, 64, 48, 3, 3, 1, 1)));
}

// Config 4 (CyclicShift, Dgrad, 4-wave, total_block_c=128).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleDgradTest, Dgrad_Cfg4_CS_C128_K112)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 1, 128, 112, 3, 3, 1, 1)));
}

// Config 8 (XOR, Dgrad, 4-wave).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleDgradTest, Dgrad_Cfg8_XOR_C128_K112_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<8>(2, 16, 16, 1, 128, 112, 3, 3, 1, 1)));
}

// --- Output-channel padding + swizzle on cspw=1 configs (4-11) ---
// Phase 3: exercise the descriptor-based output-channel pad
// (DenseSharedDescriptors<TC>::Output::MakeChannelPadDescriptor) on the
// swizzled cspw=1 configs. Reduction dim is kept clean (== total_block_c)
// so only the output channel count is padded (K_out for Fprop, C_out for
// Dgrad), isolating the new output-writer pad path under swizzle.

class DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 7 (CyclicShift, 2-wave): C=64 clean, K padded (24 / 8 / sub-8 6).
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1FpropTest, Fprop_Cfg7_CS_C64_K24)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 64, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1FpropTest, Fprop_Cfg7_CS_C64_K8)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 64, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1FpropTest, Fprop_Cfg7_CS_C64_K6)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 1, 64, 6, 3, 3, 1, 1)));
}

// Config 11 (XOR, 2-wave).
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1FpropTest, Fprop_Cfg11_XOR_C64_K24)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1FpropTest, Fprop_Cfg11_XOR_C64_K72_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<11>(2, 16, 16, 1, 64, 72, 3, 3, 1, 1)));
}

// Config 6 (CyclicShift, 4-wave): C=128 clean, K padded.
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1FpropTest, Fprop_Cfg6_CS_C128_K40)
{
    ASSERT_TRUE((RunFprop<6>(1, 8, 8, 1, 128, 40, 3, 3, 1, 1)));
}

// Config 10 (XOR, 4-wave).
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1FpropTest, Fprop_Cfg10_XOR_C128_K20)
{
    ASSERT_TRUE((RunFprop<10>(1, 8, 8, 1, 128, 20, 3, 3, 1, 1)));
}

// --- Dgrad output (C_out) padding + swizzle (C_out must stay > 32) ---

class DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config 5 (CyclicShift, Dgrad, 2-wave): K=64 clean (reduction), C padded.
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1DgradTest, Dgrad_Cfg5_CS_C48_K64)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 48, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1DgradTest, Dgrad_Cfg5_CS_C36_K64)
{
    ASSERT_TRUE((RunDgrad<5>(1, 8, 8, 1, 36, 64, 3, 3, 1, 1)));
}

// Config 9 (XOR, Dgrad, 2-wave).
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1DgradTest, Dgrad_Cfg9_XOR_C40_K64)
{
    ASSERT_TRUE((RunDgrad<9>(1, 8, 8, 1, 40, 64, 3, 3, 1, 1)));
}

// Config 4 (CyclicShift, Dgrad, 4-wave): K=128 clean, C padded.
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1DgradTest, Dgrad_Cfg4_CS_C72_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(2, 16, 16, 1, 72, 128, 3, 3, 1, 1)));
}

// Config 8 (XOR, Dgrad, 4-wave).
TEST_F(DirectConvNonGrouped32cFp16V3OutputPadSwizzleCspw1DgradTest, Dgrad_Cfg8_XOR_C100_K128)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 100, 128, 3, 3, 1, 1)));
}

// =============================================================================
// v3 — REDUCTION-channel padding combined with SWIZZLE and c_slices_per_wave > 1.
//
// Lifts the last reduction-pad restriction (c_slices_per_wave == 1). Each wave
// streams CS = 0..cspw-1 chunks of BLOCK_C8 channel-8 blocks; the global
// channel-8 index of a thread's load in chunk CS is CS * BLOCK_C8 +
// swizzled_c8(global_spatial, logical_c8). The input loader now evaluates the
// reduction-pad mask per chunk in prefetch_tile_to_lds<CS>, so pad + swizzle +
// chunking compose. Correctness ultimately comes from the weight loader zeroing
// every invalid reduction channel (garbage_input * 0_weight = 0 in the MFMA),
// which is already chunk-aware via the global channel index.
//
// cspw=2 swizzled configs (total_block_c = waves * 2 * 32):
//   waves=2 (total_block_c=128): 48=CS Fprop, 49=CS Dgrad, 50=XOR Fprop, 51=XOR Dgrad
//   waves=4 (total_block_c=256): 56=CS Fprop, 57=CS Dgrad, 58=XOR Fprop, 59=XOR Dgrad
// =============================================================================

// --- Fprop reduction padding + swizzle + cspw>1 (reduction dim C_in = c_tot) ---

class DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config key 48 (CyclicShift, 2-wave, cspw=2, total_block_c=128): C_in in (96,128].
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg48_CS_C112_K128)
{
    ASSERT_TRUE((RunFprop<48>(1, 8, 8, 1, 112, 128, 3, 3, 1, 1)));
}

// Sub-8 reduction with swizzle + cspw>1.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg48_CS_C100_K128)
{
    ASSERT_TRUE((RunFprop<48>(1, 8, 8, 1, 100, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg48_CS_C112_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<48>(2, 16, 16, 1, 112, 64, 3, 3, 1, 1)));
}

// Config key 50 (XOR, 2-wave, cspw=2).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg50_XOR_C112_K128)
{
    ASSERT_TRUE((RunFprop<50>(1, 8, 8, 1, 112, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg50_XOR_C100_K128)
{
    ASSERT_TRUE((RunFprop<50>(1, 8, 8, 1, 100, 128, 3, 3, 1, 1)));
}

// Config key 56 (CyclicShift, 4-wave, cspw=2, total_block_c=256): C_in in (224,256].
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg56_CS_C240_K128)
{
    ASSERT_TRUE((RunFprop<56>(1, 8, 8, 1, 240, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg56_CS_C232_K128)
{
    ASSERT_TRUE((RunFprop<56>(1, 8, 8, 1, 232, 128, 3, 3, 1, 1)));
}

// Config key 58 (XOR, 4-wave, cspw=2).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg58_XOR_C240_K128)
{
    ASSERT_TRUE((RunFprop<58>(1, 8, 8, 1, 240, 128, 3, 3, 1, 1)));
}

// Combined reduction + output padding under swizzle + cspw>1.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwFpropTest, Fprop_Cfg58_XOR_C240_K36)
{
    ASSERT_TRUE((RunFprop<58>(1, 8, 8, 1, 240, 36, 3, 3, 1, 1)));
}

// --- Dgrad reduction padding + swizzle + cspw>1 (reduction dim C_in = k_tot) ---
// Output dim K_out = c_tot must stay > 32 (is_non_grouped).

class DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config key 49 (CyclicShift, Dgrad, 2-wave, cspw=2): K_in in (96,128], C_out clean.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwDgradTest, Dgrad_Cfg49_CS_C128_K112)
{
    ASSERT_TRUE((RunDgrad<49>(1, 8, 8, 1, 128, 112, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwDgradTest, Dgrad_Cfg49_CS_C128_K100)
{
    ASSERT_TRUE((RunDgrad<49>(1, 8, 8, 1, 128, 100, 3, 3, 1, 1)));
}

// Config key 51 (XOR, Dgrad, 2-wave, cspw=2).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwDgradTest, Dgrad_Cfg51_XOR_C128_K112)
{
    ASSERT_TRUE((RunDgrad<51>(1, 8, 8, 1, 128, 112, 3, 3, 1, 1)));
}

// Config key 57 (CyclicShift, Dgrad, 4-wave, cspw=2): K_in in (224,256].
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwDgradTest, Dgrad_Cfg57_CS_C256_K240)
{
    ASSERT_TRUE((RunDgrad<57>(1, 8, 8, 1, 256, 240, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwDgradTest, Dgrad_Cfg57_CS_C256_K232_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<57>(2, 16, 16, 1, 256, 232, 3, 3, 1, 1)));
}

// Config key 59 (XOR, Dgrad, 4-wave, cspw=2).
TEST_F(DirectConvNonGrouped32cFp16V3ReductionPadSwizzleCspwDgradTest, Dgrad_Cfg59_XOR_C256_K240)
{
    ASSERT_TRUE((RunDgrad<59>(1, 8, 8, 1, 256, 240, 3, 3, 1, 1)));
}

// =============================================================================
// v3 — COVERING-WINDOW GAP FILL (W = waves * c_slices_per_wave = 9 and 10).
//
// The dispatcher previously instantiated W in {2,3,4,5,6,7,8,12,16,24,32,48,64}
// only, leaving reductions needing W=9 (total_block_c=288) or W=10
// (total_block_c=320) uncovered. New configs:
//   key 64/65 = waves=3, cspw=3 (W=9, total_block_c=288): Dgrad/Fprop CyclicShift
//   key 66/67 = waves=5, cspw=2 (W=10, total_block_c=320): Dgrad/Fprop CyclicShift
// Non-power-of-2 waves (3, 5) require CyclicShift (XOR static-asserts pow2).
// The covering rule places the exact reduction at the top of the window, so
// these cases are NOT reduction-padded (C_in == total_block_c).
// =============================================================================

// --- Fprop covering gap (reduction dim C_in = c_tot) ---

class DirectConvNonGrouped32cFp16V3CoveringGapFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config key 65 (W=9, total_block_c=288): C=288, output K=144 (>32).
TEST_F(DirectConvNonGrouped32cFp16V3CoveringGapFpropTest, Fprop_Cfg65_W9_C288_K144)
{
    ASSERT_TRUE((RunFprop<65>(1, 8, 8, 1, 288, 144, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CoveringGapFpropTest, Fprop_Cfg65_W9_C288_K144_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<65>(2, 16, 16, 1, 288, 144, 3, 3, 1, 1)));
}

// Config key 67 (W=10, total_block_c=320): C=320, output K=160/K=80 (>32).
TEST_F(DirectConvNonGrouped32cFp16V3CoveringGapFpropTest, Fprop_Cfg67_W10_C320_K160)
{
    ASSERT_TRUE((RunFprop<67>(1, 8, 8, 1, 320, 160, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CoveringGapFpropTest, Fprop_Cfg67_W10_C320_K80)
{
    ASSERT_TRUE((RunFprop<67>(1, 8, 8, 1, 320, 80, 3, 3, 1, 1)));
}

// --- Dgrad covering gap (reduction dim C_in = k_tot, output = c_tot > 32) ---

class DirectConvNonGrouped32cFp16V3CoveringGapDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config key 64 (W=9, total_block_c=288): K=288, output C=144 (>32).
TEST_F(DirectConvNonGrouped32cFp16V3CoveringGapDgradTest, Dgrad_Cfg64_W9_C144_K288)
{
    ASSERT_TRUE((RunDgrad<64>(1, 8, 8, 1, 144, 288, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CoveringGapDgradTest, Dgrad_Cfg64_W9_C144_K288_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<64>(2, 16, 16, 1, 144, 288, 3, 3, 1, 1)));
}

// Config key 66 (W=10, total_block_c=320): K=320, output C=160/C=80 (>32).
TEST_F(DirectConvNonGrouped32cFp16V3CoveringGapDgradTest, Dgrad_Cfg66_W10_C160_K320)
{
    ASSERT_TRUE((RunDgrad<66>(1, 8, 8, 1, 160, 320, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3CoveringGapDgradTest, Dgrad_Cfg66_W10_C80_K320)
{
    ASSERT_TRUE((RunDgrad<66>(1, 8, 8, 1, 80, 320, 3, 3, 1, 1)));
}

// =============================================================================
// v3 — REDUCTION <= 32 (waves_per_wg=1, total_block_c=32).
//
// The smallest covering config used to be waves=2 (window (32,64]), so any
// reduction <= 32 had no instance, and the dense applicability gate also
// rejected groups==1 shapes with channels <= 32. The gate is now groups != 1,
// and waves=1 configs cover reduction in (0,32]:
//   key 60/61 = waves=1 None        (Dgrad/Fprop) — robust for sub-32 padding
//   key 62/63 = waves=1 CyclicShift (Dgrad/Fprop) — exact C==32 path
// Output channel kept > 32 so the shape is genuinely non-grouped.
// =============================================================================

// --- Fprop reduction <= 32 (reduction dim C_in = c_tot) ---

class DirectConvNonGrouped32cFp16V3ReductionLE32FpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config key 61 (None, waves=1): padded reductions C in {3,16,24}, output K=64.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32FpropTest, Fprop_Cfg61_None_C3_K64)
{
    ASSERT_TRUE((RunFprop<61>(1, 8, 8, 1, 3, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32FpropTest, Fprop_Cfg61_None_C16_K64)
{
    ASSERT_TRUE((RunFprop<61>(1, 8, 8, 1, 16, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32FpropTest, Fprop_Cfg61_None_C24_K64)
{
    ASSERT_TRUE((RunFprop<61>(1, 8, 8, 1, 24, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32FpropTest, Fprop_Cfg61_None_C32_K64)
{
    ASSERT_TRUE((RunFprop<61>(1, 8, 8, 1, 32, 64, 3, 3, 1, 1)));
}

// Config key 63 (CyclicShift, waves=1): exact C==32 path, output K=64.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32FpropTest, Fprop_Cfg63_CS_C32_K64)
{
    ASSERT_TRUE((RunFprop<63>(1, 8, 8, 1, 32, 64, 3, 3, 1, 1)));
}

// --- Dgrad reduction <= 32 (reduction dim C_in = k_tot, output = c_tot > 32) ---

class DirectConvNonGrouped32cFp16V3ReductionLE32DgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// Config key 60 (None, waves=1): padded reductions K in {3,16,24}, output C=64.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32DgradTest, Dgrad_Cfg60_None_C64_K3)
{
    ASSERT_TRUE((RunDgrad<60>(1, 8, 8, 1, 64, 3, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32DgradTest, Dgrad_Cfg60_None_C64_K16)
{
    ASSERT_TRUE((RunDgrad<60>(1, 8, 8, 1, 64, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32DgradTest, Dgrad_Cfg60_None_C64_K24)
{
    ASSERT_TRUE((RunDgrad<60>(1, 8, 8, 1, 64, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32DgradTest, Dgrad_Cfg60_None_C64_K32)
{
    ASSERT_TRUE((RunDgrad<60>(1, 8, 8, 1, 64, 32, 3, 3, 1, 1)));
}

// Config key 62 (CyclicShift, waves=1): exact K==32 path, output C=64.
TEST_F(DirectConvNonGrouped32cFp16V3ReductionLE32DgradTest, Dgrad_Cfg62_CS_C64_K32)
{
    ASSERT_TRUE((RunDgrad<62>(1, 8, 8, 1, 64, 32, 3, 3, 1, 1)));
}

// =============================================================================
// v3 — waves_per_wg=1, c_slices_per_wave > 1 (config keys 68-83).
//
// Single-wave configs that stream the full C-reduction as cspw chunks of 32
// channels through ONE wavefront, eliminating the cross-wave LDS reduction.
// total_block_c = cspw * 32, covering reduction in (32*(cspw-1), 32*cspw]:
//   cspw=2 -> (32,64]   cspw=3 -> (64,96]   cspw=4 -> (96,128]   cspw=6 -> (160,192]
// CyclicShift swizzle; both DRAM and LDS-staged epilogues; Fprop and Dgrad.
//
//   Fprop DRAM: 68(cspw2) 72(cspw3) 76(cspw4) 80(cspw6)
//   Fprop LDS : 70(cspw2) 74(cspw3) 78(cspw4) 82(cspw6)
//   Dgrad DRAM: 69(cspw2) 73(cspw3) 77(cspw4) 81(cspw6)
//   Dgrad LDS : 71(cspw2) 75(cspw3) 79(cspw4) 83(cspw6)
// For Fprop: reduction dim C_in = c_tot. For Dgrad: reduction dim C_in = k_tot,
// output dim K_out = c_tot (must stay > 32 to be genuinely non-grouped).
// =============================================================================

// --- Fprop, DRAM epilogue ---

class DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// cspw=2 (total_block_c=64): exact C=64 and padded C=48.
TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg68_cspw2_C64_K64)
{
    ASSERT_TRUE((RunFprop<68>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg68_cspw2_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<68>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg68_cspw2_C48_K64)
{
    ASSERT_TRUE((RunFprop<68>(1, 8, 8, 1, 48, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg68_cspw2_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<68>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// cspw=3 (total_block_c=96): exact C=96 and padded C=80.
TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg72_cspw3_C96_K64)
{
    ASSERT_TRUE((RunFprop<72>(1, 8, 8, 1, 96, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg72_cspw3_C80_K64)
{
    ASSERT_TRUE((RunFprop<72>(1, 8, 8, 1, 80, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg72_cspw3_C96_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<72>(2, 16, 16, 1, 96, 64, 3, 3, 1, 1)));
}

// cspw=4 (total_block_c=128): exact C=128 and padded C=112.
TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg76_cspw4_C128_K64)
{
    ASSERT_TRUE((RunFprop<76>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg76_cspw4_C112_K64)
{
    ASSERT_TRUE((RunFprop<76>(1, 8, 8, 1, 112, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg76_cspw4_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<76>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// cspw=6 (total_block_c=192): exact C=192 and padded C=176.
TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg80_cspw6_C192_K64)
{
    ASSERT_TRUE((RunFprop<80>(1, 8, 8, 1, 192, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg80_cspw6_C176_K64)
{
    ASSERT_TRUE((RunFprop<80>(1, 8, 8, 1, 176, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwFpropTest, Fprop_Cfg80_cspw6_C192_K48_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<80>(2, 16, 16, 1, 192, 48, 3, 3, 1, 1)));
}

// --- Fprop, LDS-staged epilogue ---

class DirectConvNonGrouped32cFp16V3Waves1CspwLdsFpropTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsFpropTest, Fprop_Cfg70_cspw2_C64_K64)
{
    ASSERT_TRUE((RunFprop<70>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsFpropTest, Fprop_Cfg70_cspw2_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<70>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsFpropTest, Fprop_Cfg74_cspw3_C96_K64)
{
    ASSERT_TRUE((RunFprop<74>(1, 8, 8, 1, 96, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsFpropTest, Fprop_Cfg78_cspw4_C128_K64)
{
    ASSERT_TRUE((RunFprop<78>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsFpropTest, Fprop_Cfg82_cspw6_C192_K64)
{
    ASSERT_TRUE((RunFprop<82>(1, 8, 8, 1, 192, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsFpropTest, Fprop_Cfg82_cspw6_C192_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<82>(2, 16, 16, 1, 192, 128, 3, 3, 1, 1)));
}

// --- Dgrad, DRAM epilogue (reduction dim C_in = k_tot; output c_tot > 32) ---

class DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

// cspw=2 (total_block_c=64): exact K=64 and padded K=48.
TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg69_cspw2_C64_K64)
{
    ASSERT_TRUE((RunDgrad<69>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg69_cspw2_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<69>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg69_cspw2_C64_K48)
{
    ASSERT_TRUE((RunDgrad<69>(1, 8, 8, 1, 64, 48, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg69_cspw2_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<69>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// cspw=3 (total_block_c=96): exact K=96.
TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg73_cspw3_C64_K96)
{
    ASSERT_TRUE((RunDgrad<73>(1, 8, 8, 1, 64, 96, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg73_cspw3_C64_K80)
{
    ASSERT_TRUE((RunDgrad<73>(1, 8, 8, 1, 64, 80, 3, 3, 1, 1)));
}

// cspw=4 (total_block_c=128): exact K=128.
TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg77_cspw4_C64_K128)
{
    ASSERT_TRUE((RunDgrad<77>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg77_cspw4_C64_K112)
{
    ASSERT_TRUE((RunDgrad<77>(1, 8, 8, 1, 64, 112, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg77_cspw4_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<77>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// cspw=6 (total_block_c=192): exact K=192.
TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg81_cspw6_C64_K192)
{
    ASSERT_TRUE((RunDgrad<81>(1, 8, 8, 1, 64, 192, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwDgradTest, Dgrad_Cfg81_cspw6_C64_K176)
{
    ASSERT_TRUE((RunDgrad<81>(1, 8, 8, 1, 64, 176, 3, 3, 1, 1)));
}

// --- Dgrad, LDS-staged epilogue ---

class DirectConvNonGrouped32cFp16V3Waves1CspwLdsDgradTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraitsV3>
{
};

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsDgradTest, Dgrad_Cfg71_cspw2_C64_K64)
{
    ASSERT_TRUE((RunDgrad<71>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsDgradTest, Dgrad_Cfg71_cspw2_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<71>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsDgradTest, Dgrad_Cfg75_cspw3_C64_K96)
{
    ASSERT_TRUE((RunDgrad<75>(1, 8, 8, 1, 64, 96, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsDgradTest, Dgrad_Cfg79_cspw4_C64_K128)
{
    ASSERT_TRUE((RunDgrad<79>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsDgradTest, Dgrad_Cfg83_cspw6_C64_K192)
{
    ASSERT_TRUE((RunDgrad<83>(1, 8, 8, 1, 64, 192, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16V3Waves1CspwLdsDgradTest, Dgrad_Cfg83_cspw6_C128_K192_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<83>(2, 16, 16, 1, 128, 192, 3, 3, 1, 1)));
}
