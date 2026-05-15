// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_harness.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

constexpr auto v2 = ck_tile::direct_conv::Version::v2;

struct TileConv32cKernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward32CKernel<ConfigIdx, v2>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData32CKernel<ConfigIdx, v2>;
};

// =============================================================================
// V2 Group 0 tests: No swizzle, direct DRAM epilogue
// Dgrad indices 0-7, Fprop indices 8-15
// waves_per_wg: 16,14,12,10,8,6,4,2 → block_groups: 8,7,6,5,4,3,2,1
// =============================================================================

class DirectConvGrouped32cFp16TileConvV2Test
    : public DirectConvGroupedTestHarness<TileConv32cKernelTraits>
{
};

// --- V2 Fprop ---

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config15_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config15_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config15_Groups1_Pad2)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 32, 3, 3, 2, 2)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config15_Groups1_Pad3)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 32, 3, 3, 3, 3)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config15_Groups8)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config14_Groups2)
{
    // waves_per_wg=4, block_groups=2 → needs groups%2==0
    ASSERT_TRUE((RunFprop<14>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config13_Groups3_LargerSpatial)
{
    // waves_per_wg=6, block_groups=3 → needs groups%3==0
    ASSERT_TRUE((RunFprop<13>(4, 16, 16, 3, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config12_Groups8_LargerSpatial)
{
    // waves_per_wg=8, block_groups=4 → needs groups%4==0
    ASSERT_TRUE((RunFprop<12>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config11_Groups5_LargerSpatial)
{
    // waves_per_wg=10, block_groups=5 → needs groups%5==0
    ASSERT_TRUE((RunFprop<11>(4, 16, 16, 5, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Fprop_Config10_Groups6_LargerSpatial)
{
    // waves_per_wg=12, block_groups=6 → needs groups%6==0
    ASSERT_TRUE((RunFprop<10>(4, 16, 16, 6, 32, 32, 3, 3, 1, 1)));
}

// --- V2 Dgrad ---

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config7_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config7_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config7_Groups1_Pad2)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 1, 32, 32, 3, 3, 2, 2)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config7_Groups1_Pad3)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 1, 32, 32, 3, 3, 3, 3)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config7_Groups8)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config6_Groups2)
{
    ASSERT_TRUE((RunDgrad<6>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config5_Groups3_LargerSpatial)
{
    // waves_per_wg=6, block_groups=3 → needs groups%3==0
    ASSERT_TRUE((RunDgrad<5>(4, 16, 16, 3, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config4_Groups8_LargerSpatial)
{
    // waves_per_wg=8, block_groups=4 → needs groups%4==0
    ASSERT_TRUE((RunDgrad<4>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config3_Groups5_LargerSpatial)
{
    // waves_per_wg=10, block_groups=5 → needs groups%5==0
    ASSERT_TRUE((RunDgrad<3>(4, 16, 16, 5, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2Test, Dgrad_Config2_Groups6_LargerSpatial)
{
    // waves_per_wg=12, block_groups=6 → needs groups%6==0
    ASSERT_TRUE((RunDgrad<2>(4, 16, 16, 6, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 1 tests: No swizzle, LDS-staged epilogue
// Dgrad indices 16-23, Fprop indices 24-31
// =============================================================================

class DirectConvGrouped32cFp16TileConvV2LdsTest
    : public DirectConvGroupedTestHarness<TileConv32cKernelTraits>
{
};

// --- Group 1 Fprop (indices 24-31) ---

TEST_F(DirectConvGrouped32cFp16TileConvV2LdsTest, Fprop_Config31_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<31>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2LdsTest, Fprop_Config31_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<31>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2LdsTest, Fprop_Config30_Groups2)
{
    ASSERT_TRUE((RunFprop<30>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2LdsTest, Fprop_Config28_Groups8_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<28>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

// --- Group 1 Dgrad (indices 16-23) ---

TEST_F(DirectConvGrouped32cFp16TileConvV2LdsTest, Dgrad_Config23_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<23>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2LdsTest, Dgrad_Config23_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<23>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2LdsTest, Dgrad_Config22_Groups2)
{
    ASSERT_TRUE((RunDgrad<22>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2LdsTest, Dgrad_Config20_Groups8_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<20>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 2 tests: XOR swizzle, direct DRAM epilogue
// Dgrad indices 32-39, Fprop indices 40-47
// XOR constraint: waves_per_wg must divide 8 for multi-tile spatial.
// =============================================================================

class DirectConvGrouped32cFp16TileConvV2XorTest
    : public DirectConvGroupedTestHarness<TileConv32cKernelTraits>
{
};

// --- Group 2 Fprop (indices 40-47): use configs with block_groups ∈ {1,2,4} ---

TEST_F(DirectConvGrouped32cFp16TileConvV2XorTest, Fprop_Config47_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<47>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorTest, Fprop_Config47_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<47>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorTest, Fprop_Config46_Groups2)
{
    ASSERT_TRUE((RunFprop<46>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorTest, Fprop_Config44_Groups8_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<44>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

// --- Group 2 Dgrad (indices 32-39) ---

TEST_F(DirectConvGrouped32cFp16TileConvV2XorTest, Dgrad_Config39_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<39>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorTest, Dgrad_Config39_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<39>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorTest, Dgrad_Config38_Groups2)
{
    ASSERT_TRUE((RunDgrad<38>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorTest, Dgrad_Config36_Groups8_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<36>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 3 tests: XOR swizzle, LDS-staged epilogue
// Dgrad indices 48-55, Fprop indices 56-63
// =============================================================================

class DirectConvGrouped32cFp16TileConvV2XorLdsTest
    : public DirectConvGroupedTestHarness<TileConv32cKernelTraits>
{
};

// --- Group 3 Fprop (indices 56-63) ---

TEST_F(DirectConvGrouped32cFp16TileConvV2XorLdsTest, Fprop_Config63_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<63>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorLdsTest, Fprop_Config63_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<63>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorLdsTest, Fprop_Config62_Groups2)
{
    ASSERT_TRUE((RunFprop<62>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorLdsTest, Fprop_Config60_Groups8_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<60>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

// --- Group 3 Dgrad (indices 48-55) ---

TEST_F(DirectConvGrouped32cFp16TileConvV2XorLdsTest, Dgrad_Config55_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<55>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorLdsTest, Dgrad_Config55_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<55>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorLdsTest, Dgrad_Config54_Groups2)
{
    ASSERT_TRUE((RunDgrad<54>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2XorLdsTest, Dgrad_Config52_Groups8_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<52>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// Cyclic-shift swizzle instances (indices 64-67)
// =============================================================================

class DirectConvGrouped32cFp16TileConvV2CyclicShiftSwizzleTest
    : public DirectConvGroupedTestHarness<TileConv32cKernelTraits>
{
};

TEST_F(DirectConvGrouped32cFp16TileConvV2CyclicShiftSwizzleTest, Fprop_Config64_Groups8_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunFprop<64>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2CyclicShiftSwizzleTest, Fprop_Config65_Groups8_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunFprop<65>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2CyclicShiftSwizzleTest, Dgrad_Config66_Groups8_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunDgrad<66>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2CyclicShiftSwizzleTest, Dgrad_Config67_Groups8_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunDgrad<67>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// Padded channel tests: c_per_group and/or k_per_group < 32 (but > 16 for at least one)
// Uses CyclicShift configs with small vector_size:
//   Dgrad indices 68-72 (vector_size = 16, 8, 4, 2, 1)
//   Fprop indices 73-77 (vector_size = 16, 8, 4, 2, 1)
// and None fallback configs: Dgrad 78, Fprop 79 (vector_size = 1)
// =============================================================================

class DirectConvGrouped32cFp16TileConvV2PaddedTest
    : public DirectConvGroupedTestHarness<TileConv32cKernelTraits>
{
};

// --- Fprop padded: C == K ---

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C24_K24)
{
    // vector_size=8 config (index 74): 24 % 8 == 0
    ASSERT_TRUE((RunFprop<74>(1, 8, 8, 4, 24, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C20_K20)
{
    // vector_size=4 config (index 75): 20 % 4 == 0
    ASSERT_TRUE((RunFprop<75>(1, 8, 8, 4, 20, 20, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C18_K18)
{
    // vector_size=2 config (index 76): 18 % 2 == 0
    ASSERT_TRUE((RunFprop<76>(1, 8, 8, 4, 18, 18, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C17_K17)
{
    // vector_size=1 config (index 77): any channel count
    ASSERT_TRUE((RunFprop<77>(1, 8, 8, 4, 17, 17, 3, 3, 1, 1)));
}

// --- Fprop padded: C != K ---

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C17_K32)
{
    // vector_size=1 config (index 77): 17 not divisible by 2/4/8/16
    ASSERT_TRUE((RunFprop<77>(1, 8, 8, 4, 17, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C32_K17)
{
    ASSERT_TRUE((RunFprop<77>(1, 8, 8, 4, 32, 17, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C24_K16)
{
    // vector_size=8 config (index 74): 24 % 8 == 0, 16 % 8 == 0
    ASSERT_TRUE((RunFprop<74>(1, 8, 8, 4, 24, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C20_K28)
{
    // vector_size=4 config (index 75): 20 % 4 == 0, 28 % 4 == 0
    ASSERT_TRUE((RunFprop<75>(1, 8, 8, 4, 20, 28, 3, 3, 1, 1)));
}

// --- Dgrad padded: C == K ---

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C24_K24)
{
    // vector_size=8 config (index 69): 24 % 8 == 0
    ASSERT_TRUE((RunDgrad<69>(1, 8, 8, 4, 24, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C20_K20)
{
    // vector_size=4 config (index 70): 20 % 4 == 0
    ASSERT_TRUE((RunDgrad<70>(1, 8, 8, 4, 20, 20, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C18_K18)
{
    // vector_size=2 config (index 71): 18 % 2 == 0
    ASSERT_TRUE((RunDgrad<71>(1, 8, 8, 4, 18, 18, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C17_K17)
{
    // vector_size=1 config (index 72)
    ASSERT_TRUE((RunDgrad<72>(1, 8, 8, 4, 17, 17, 3, 3, 1, 1)));
}

// --- Dgrad padded: C != K ---

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C17_K32)
{
    ASSERT_TRUE((RunDgrad<72>(1, 8, 8, 4, 17, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C32_K17)
{
    ASSERT_TRUE((RunDgrad<72>(1, 8, 8, 4, 32, 17, 3, 3, 1, 1)));
}

// --- No-swizzle fallback for padding ---

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_NoSwizzle_C17_K17)
{
    // Config 79: Fprop, None swizzle, vector_size=1
    ASSERT_TRUE((RunFprop<79>(1, 8, 8, 4, 17, 17, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_NoSwizzle_C17_K17)
{
    // Config 78: Dgrad, None swizzle, vector_size=1
    ASSERT_TRUE((RunDgrad<78>(1, 8, 8, 4, 17, 17, 3, 3, 1, 1)));
}

// --- Larger spatial with padding ---

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C24_K24_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<74>(4, 16, 16, 4, 24, 24, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C24_K24_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<69>(4, 16, 16, 4, 24, 24, 3, 3, 1, 1)));
}

// --- Padded-channel spatial padding tests (pad=2, pad=3) ---

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C24_K24_Pad2)
{
    ASSERT_TRUE((RunFprop<74>(1, 8, 8, 4, 24, 24, 3, 3, 2, 2)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Fprop_C24_K24_Pad3)
{
    ASSERT_TRUE((RunFprop<74>(1, 8, 8, 4, 24, 24, 3, 3, 3, 3)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C24_K24_Pad2)
{
    ASSERT_TRUE((RunDgrad<69>(1, 8, 8, 4, 24, 24, 3, 3, 2, 2)));
}

TEST_F(DirectConvGrouped32cFp16TileConvV2PaddedTest, Dgrad_C24_K24_Pad3)
{
    ASSERT_TRUE((RunDgrad<69>(1, 8, 8, 4, 24, 24, 3, 3, 3, 3)));
}
