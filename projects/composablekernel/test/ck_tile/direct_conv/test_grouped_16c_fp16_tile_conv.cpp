// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_grouped_4c_fp16_harness.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_16c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_16c_fp16_tile_conv_kernel.hpp"
#pragma clang diagnostic pop

constexpr auto v2 = ck_tile::direct_conv::Version::v2;

struct TileConv16cKernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward16CFp16Kernel<ConfigIdx, v2>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData16CFp16Kernel<ConfigIdx, v2>;
};

// =============================================================================
// V2 tests (CK Tile tile distribution implementation)
// =============================================================================

class DirectConvGrouped16cFp16TileConvV2Test
    : public DirectConvGrouped4cFp16TestHarness<TileConv16cKernelTraits>
{
};

// --- V2 Fprop ---

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config17_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config17_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config17_Groups1_Pad2)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 16, 16, 3, 3, 2, 2)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config17_Groups1_Pad3)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 16, 16, 3, 3, 3, 3)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config17_Groups16)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config16_Groups16)
{
    ASSERT_TRUE((RunFprop<16>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config15_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<15>(4, 16, 16, 6, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config14_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<14>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config13_Groups10_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<13>(4, 16, 16, 10, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Fprop_Config12_Groups12_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<12>(4, 16, 16, 12, 16, 16, 3, 3, 1, 1)));
}

// --- V2 Dgrad ---

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config8_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config8_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config8_Groups1_Pad2)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 16, 16, 3, 3, 2, 2)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config8_Groups1_Pad3)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 16, 16, 3, 3, 3, 3)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config8_Groups16)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config7_Groups16)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config6_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<6>(4, 16, 16, 6, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config5_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<5>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config4_Groups10_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(4, 16, 16, 10, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2Test, Dgrad_Config3_Groups12_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<3>(4, 16, 16, 12, 16, 16, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 1 tests: No swizzle, LDS-staged epilogue
// Config index offset: +18 from Group 0
// =============================================================================

class DirectConvGrouped16cFp16TileConvV2LdsTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv16cKernelTraits>
{
};

// --- Group 1 Fprop (indices 27-35) ---

TEST_F(DirectConvGrouped16cFp16TileConvV2LdsTest, Fprop_Config35_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<35>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2LdsTest, Fprop_Config35_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<35>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2LdsTest, Fprop_Config34_Groups16)
{
    ASSERT_TRUE((RunFprop<34>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2LdsTest, Fprop_Config32_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<32>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

// --- Group 1 Dgrad (indices 18-26) ---

TEST_F(DirectConvGrouped16cFp16TileConvV2LdsTest, Dgrad_Config26_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<26>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2LdsTest, Dgrad_Config26_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<26>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2LdsTest, Dgrad_Config25_Groups16)
{
    ASSERT_TRUE((RunDgrad<25>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2LdsTest, Dgrad_Config23_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<23>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 2 tests: XOR swizzle, direct DRAM epilogue
// Config index offset: +36 from Group 0
// XOR constraint: waves_per_wg must divide 8 for multi-tile spatial.
// =============================================================================

class DirectConvGrouped16cFp16TileConvV2XorTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv16cKernelTraits>
{
};

// --- Group 2 Fprop (indices 45-53): use waves_per_wg ∈ {1,2,4,8} ---

TEST_F(DirectConvGrouped16cFp16TileConvV2XorTest, Fprop_Config53_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<53>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorTest, Fprop_Config53_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<53>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorTest, Fprop_Config52_Groups16)
{
    ASSERT_TRUE((RunFprop<52>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorTest, Fprop_Config50_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<50>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

// --- Group 2 Dgrad (indices 36-44) ---

TEST_F(DirectConvGrouped16cFp16TileConvV2XorTest, Dgrad_Config44_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<44>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorTest, Dgrad_Config44_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<44>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorTest, Dgrad_Config43_Groups16)
{
    ASSERT_TRUE((RunDgrad<43>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorTest, Dgrad_Config41_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<41>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 3 tests: XOR swizzle, LDS-staged epilogue
// Config index offset: +54 from Group 0
// =============================================================================

class DirectConvGrouped16cFp16TileConvV2XorLdsTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv16cKernelTraits>
{
};

// --- Group 3 Fprop (indices 63-71) ---

TEST_F(DirectConvGrouped16cFp16TileConvV2XorLdsTest, Fprop_Config71_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<71>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorLdsTest, Fprop_Config71_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<71>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorLdsTest, Fprop_Config70_Groups16)
{
    ASSERT_TRUE((RunFprop<70>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorLdsTest, Fprop_Config68_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<68>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

// --- Group 3 Dgrad (indices 54-62) ---

TEST_F(DirectConvGrouped16cFp16TileConvV2XorLdsTest, Dgrad_Config62_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<62>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorLdsTest, Dgrad_Config62_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<62>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorLdsTest, Dgrad_Config61_Groups16)
{
    ASSERT_TRUE((RunDgrad<61>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2XorLdsTest, Dgrad_Config59_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<59>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

// --- Test cyclic-shift swizzle instances (indices 72-75)

class DirectConvGrouped16cFp16TileConvV2CyclicShiftSwizzleTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv16cKernelTraits>
{
};

TEST_F(DirectConvGrouped16cFp16TileConvV2CyclicShiftSwizzleTest, Fprop_Config72_Groups16_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunFprop<72>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2CyclicShiftSwizzleTest, Fprop_Config73_Groups16_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunFprop<73>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2CyclicShiftSwizzleTest, Dgrad_Config74_Groups16_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunDgrad<74>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2CyclicShiftSwizzleTest, Dgrad_Config75_Groups16_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunDgrad<75>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

// =============================================================================
// Padded channel tests: c_per_group and/or k_per_group < 16 (but > 8 for at least one)
// Uses CyclicShift configs with small vector_size (indices 76-83)
// and None fallback configs (indices 84-85).
// =============================================================================

class DirectConvGrouped16cFp16TileConvV2PaddedTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv16cKernelTraits>
{
};

// --- Fprop padded: C == K ---

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C12_K12)
{
    // vector_size=4 config (index 81): 12 % 4 == 0
    ASSERT_TRUE((RunFprop<81>(1, 8, 8, 8, 12, 12, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C10_K10)
{
    // vector_size=2 config (index 82): 10 % 2 == 0
    ASSERT_TRUE((RunFprop<82>(1, 8, 8, 8, 10, 10, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C9_K9)
{
    // vector_size=1 config (index 83): any channel count
    ASSERT_TRUE((RunFprop<83>(1, 8, 8, 8, 9, 9, 3, 3, 1, 1)));
}

// --- Fprop padded: C != K ---

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C9_K16)
{
    // vector_size=1 config (index 83): 9 is not divisible by 2/4/8
    ASSERT_TRUE((RunFprop<83>(1, 8, 8, 8, 9, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C16_K9)
{
    ASSERT_TRUE((RunFprop<83>(1, 8, 8, 8, 16, 9, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C12_K8)
{
    // vector_size=4 config (index 81): 12 % 4 == 0, 8 % 4 == 0
    ASSERT_TRUE((RunFprop<81>(1, 8, 8, 8, 12, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C10_K14)
{
    // vector_size=2 config (index 82): 10 % 2 == 0, 14 % 2 == 0
    ASSERT_TRUE((RunFprop<82>(1, 8, 8, 8, 10, 14, 3, 3, 1, 1)));
}

// --- Dgrad padded: C == K ---

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_C12_K12)
{
    // vector_size=4 config (index 77): 12 % 4 == 0
    ASSERT_TRUE((RunDgrad<77>(1, 8, 8, 8, 12, 12, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_C10_K10)
{
    // vector_size=2 config (index 78): 10 % 2 == 0
    ASSERT_TRUE((RunDgrad<78>(1, 8, 8, 8, 10, 10, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_C9_K9)
{
    // vector_size=1 config (index 79)
    ASSERT_TRUE((RunDgrad<79>(1, 8, 8, 8, 9, 9, 3, 3, 1, 1)));
}

// --- Dgrad padded: C != K ---

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_C9_K16)
{
    ASSERT_TRUE((RunDgrad<79>(1, 8, 8, 8, 9, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_C16_K9)
{
    ASSERT_TRUE((RunDgrad<79>(1, 8, 8, 8, 16, 9, 3, 3, 1, 1)));
}

// --- No-swizzle fallback for padding ---

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_NoSwizzle_C9_K9)
{
    // Config 85: Fprop, None swizzle, vector_size=1
    ASSERT_TRUE((RunFprop<85>(1, 8, 8, 8, 9, 9, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_NoSwizzle_C9_K9)
{
    // Config 84: Dgrad, None swizzle, vector_size=1
    ASSERT_TRUE((RunDgrad<84>(1, 8, 8, 8, 9, 9, 3, 3, 1, 1)));
}

// --- Larger spatial with padding ---

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C12_K12_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<81>(4, 16, 16, 8, 12, 12, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_C12_K12_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<77>(4, 16, 16, 8, 12, 12, 3, 3, 1, 1)));
}

// --- Padded-channel spatial padding tests (pad=2, pad=3) ---

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C12_K12_Pad2)
{
    ASSERT_TRUE((RunFprop<81>(1, 8, 8, 8, 12, 12, 3, 3, 2, 2)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Fprop_C12_K12_Pad3)
{
    ASSERT_TRUE((RunFprop<81>(1, 8, 8, 8, 12, 12, 3, 3, 3, 3)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_C12_K12_Pad2)
{
    ASSERT_TRUE((RunDgrad<77>(1, 8, 8, 8, 12, 12, 3, 3, 2, 2)));
}

TEST_F(DirectConvGrouped16cFp16TileConvV2PaddedTest, Dgrad_C12_K12_Pad3)
{
    ASSERT_TRUE((RunDgrad<77>(1, 8, 8, 8, 12, 12, 3, 3, 3, 3)));
}

