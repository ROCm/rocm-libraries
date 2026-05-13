// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_grouped_4c_fp16_harness.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

constexpr auto v2 = ck_tile::direct_conv::Version::v2;

struct TileConv32cBf16KernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward32CKernel<
        ConfigIdx, v2, ck_tile::direct_conv::DataType::bf16>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData32CKernel<
        ConfigIdx, v2, ck_tile::direct_conv::DataType::bf16>;
};

// =============================================================================
// V2 Group 0: No swizzle, direct DRAM epilogue
// =============================================================================

class DirectConvGrouped32cBf16TileConvV2Test
    : public DirectConvGrouped4cFp16TestHarness<TileConv32cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

// --- V2 Fprop ---

TEST_F(DirectConvGrouped32cBf16TileConvV2Test, Fprop_Config15_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2Test, Fprop_Config15_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2Test, Fprop_Config15_Groups8)
{
    ASSERT_TRUE((RunFprop<15>(1, 8, 8, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2Test, Fprop_Config13_Groups3_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<13>(4, 16, 16, 3, 32, 32, 3, 3, 1, 1)));
}

// --- V2 Dgrad ---

TEST_F(DirectConvGrouped32cBf16TileConvV2Test, Dgrad_Config7_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2Test, Dgrad_Config7_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2Test, Dgrad_Config7_Groups8)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2Test, Dgrad_Config5_Groups3_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<5>(4, 16, 16, 3, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 1: No swizzle, LDS-staged epilogue
// =============================================================================

class DirectConvGrouped32cBf16TileConvV2LdsTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv32cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

TEST_F(DirectConvGrouped32cBf16TileConvV2LdsTest, Fprop_Config31_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<31>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2LdsTest, Fprop_Config30_Groups2)
{
    ASSERT_TRUE((RunFprop<30>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2LdsTest, Dgrad_Config23_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<23>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2LdsTest, Dgrad_Config22_Groups2)
{
    ASSERT_TRUE((RunDgrad<22>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 2: XOR swizzle, direct DRAM epilogue
// =============================================================================

class DirectConvGrouped32cBf16TileConvV2XorTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv32cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

TEST_F(DirectConvGrouped32cBf16TileConvV2XorTest, Fprop_Config47_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<47>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2XorTest, Fprop_Config46_Groups2)
{
    ASSERT_TRUE((RunFprop<46>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2XorTest, Dgrad_Config39_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<39>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2XorTest, Dgrad_Config38_Groups2)
{
    ASSERT_TRUE((RunDgrad<38>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// V2 Group 3: XOR swizzle, LDS-staged epilogue
// =============================================================================

class DirectConvGrouped32cBf16TileConvV2XorLdsTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv32cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

TEST_F(DirectConvGrouped32cBf16TileConvV2XorLdsTest, Fprop_Config63_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<63>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2XorLdsTest, Dgrad_Config55_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<55>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

// =============================================================================
// Cyclic-shift swizzle instances (indices 64-67)
// =============================================================================

class DirectConvGrouped32cBf16TileConvV2CyclicShiftSwizzleTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv32cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

TEST_F(DirectConvGrouped32cBf16TileConvV2CyclicShiftSwizzleTest, Fprop_Config64_Groups8_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunFprop<64>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cBf16TileConvV2CyclicShiftSwizzleTest, Dgrad_Config66_Groups8_LargerSpatial_CyclicShift)
{
    ASSERT_TRUE((RunDgrad<66>(4, 16, 16, 8, 32, 32, 3, 3, 1, 1)));
}
