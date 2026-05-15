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

struct TileConv16cBf16KernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward16CKernel<
        ConfigIdx, v2, ck_tile::direct_conv::DataType::bf16>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData16CKernel<
        ConfigIdx, v2, ck_tile::direct_conv::DataType::bf16>;
};

// =============================================================================
// V2 Group 0: No swizzle, direct DRAM epilogue
// =============================================================================

class DirectConvGrouped16cBf16TileConvV2Test
    : public DirectConvGroupedTestHarness<TileConv16cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

// --- V2 Fprop ---

TEST_F(DirectConvGrouped16cBf16TileConvV2Test, Fprop_Config17_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2Test, Fprop_Config17_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2Test, Fprop_Config17_Groups16)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2Test, Fprop_Config15_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<15>(4, 16, 16, 6, 16, 16, 3, 3, 1, 1)));
}

// --- V2 Dgrad ---

TEST_F(DirectConvGrouped16cBf16TileConvV2Test, Dgrad_Config8_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2Test, Dgrad_Config8_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 16, 16, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2Test, Dgrad_Config8_Groups16)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2Test, Dgrad_Config6_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<6>(4, 16, 16, 6, 16, 16, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 1: No swizzle, LDS-staged epilogue
// =============================================================================

class DirectConvGrouped16cBf16TileConvV2LdsTest
    : public DirectConvGroupedTestHarness<TileConv16cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

TEST_F(DirectConvGrouped16cBf16TileConvV2LdsTest, Fprop_Config35_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<35>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2LdsTest, Fprop_Config34_Groups16)
{
    ASSERT_TRUE((RunFprop<34>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2LdsTest, Dgrad_Config26_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<26>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2LdsTest, Dgrad_Config25_Groups16)
{
    ASSERT_TRUE((RunDgrad<25>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 2: XOR swizzle, direct DRAM epilogue
// =============================================================================

class DirectConvGrouped16cBf16TileConvV2XorTest
    : public DirectConvGroupedTestHarness<TileConv16cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

TEST_F(DirectConvGrouped16cBf16TileConvV2XorTest, Fprop_Config53_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<53>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2XorTest, Fprop_Config52_Groups16)
{
    ASSERT_TRUE((RunFprop<52>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2XorTest, Dgrad_Config44_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<44>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2XorTest, Dgrad_Config43_Groups16)
{
    ASSERT_TRUE((RunDgrad<43>(1, 8, 8, 16, 16, 16, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 3: XOR swizzle, LDS-staged epilogue
// =============================================================================

class DirectConvGrouped16cBf16TileConvV2XorLdsTest
    : public DirectConvGroupedTestHarness<TileConv16cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

TEST_F(DirectConvGrouped16cBf16TileConvV2XorLdsTest, Fprop_Config71_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<71>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2XorLdsTest, Dgrad_Config62_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<62>(1, 8, 8, 1, 16, 16, 3, 3, 1, 1)));
}

// =============================================================================
// Cyclic-shift swizzle instances (indices 72-75)
// =============================================================================

class DirectConvGrouped16cBf16TileConvV2CyclicShiftSwizzleTest
    : public DirectConvGroupedTestHarness<TileConv16cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

TEST_F(DirectConvGrouped16cBf16TileConvV2CyclicShiftSwizzleTest, Fprop_Config72_Groups16_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunFprop<72>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped16cBf16TileConvV2CyclicShiftSwizzleTest, Dgrad_Config74_Groups16_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunDgrad<74>(4, 16, 16, 16, 16, 16, 3, 3, 1, 1)));
}
