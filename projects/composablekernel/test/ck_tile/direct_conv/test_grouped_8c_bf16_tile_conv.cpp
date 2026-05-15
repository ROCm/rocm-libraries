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

struct TileConv8cBf16KernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward8CKernel<
        ConfigIdx, v2, ck_tile::direct_conv::DataType::bf16>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData8CKernel<
        ConfigIdx, v2, ck_tile::direct_conv::DataType::bf16>;
};

class DirectConvGrouped8cBf16TileConvV2Test
    : public DirectConvGroupedTestHarness<TileConv8cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

// =============================================================================
// V2 Group 0: No swizzle, direct DRAM epilogue
// =============================================================================

// Fprop
TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Fprop_Config17_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Fprop_Config17_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 8, 8, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Fprop_Config17_Groups16)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Fprop_Config15_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<15>(4, 16, 16, 6, 8, 8, 3, 3, 1, 1)));
}

// Dgrad
TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Dgrad_Config8_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Dgrad_Config8_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 8, 8, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Dgrad_Config8_Groups16)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Dgrad_Config6_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<6>(4, 16, 16, 6, 8, 8, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 1: No swizzle, LDS-staged epilogue
// =============================================================================

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Fprop_LDS_Config35_Groups1)
{
    ASSERT_TRUE((RunFprop<35>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Dgrad_LDS_Config26_Groups1)
{
    ASSERT_TRUE((RunDgrad<26>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 2: XOR swizzle, direct DRAM epilogue
// =============================================================================

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Fprop_XOR_Config53_Groups1)
{
    ASSERT_TRUE((RunFprop<53>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Dgrad_XOR_Config44_Groups1)
{
    ASSERT_TRUE((RunDgrad<44>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 3: XOR swizzle, LDS-staged epilogue
// =============================================================================

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Fprop_XORLDS_Config71_Groups1)
{
    ASSERT_TRUE((RunFprop<71>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Dgrad_XORLDS_Config62_Groups1)
{
    ASSERT_TRUE((RunDgrad<62>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Cyclic-shift swizzle
// =============================================================================

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Fprop_CyclicShift_LDS_Config72_Groups8)
{
    ASSERT_TRUE((RunFprop<72>(1, 8, 8, 8, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cBf16TileConvV2Test, Dgrad_CyclicShift_LDS_Config74_Groups8)
{
    ASSERT_TRUE((RunDgrad<74>(1, 8, 8, 8, 8, 8, 3, 3, 1, 1)));
}
