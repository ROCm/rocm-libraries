// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_grouped_4c_fp16_harness.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

constexpr auto v3 = ck_tile::direct_conv::Version::v3;

struct TileConv4cBf16KernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward4CKernel<
        ConfigIdx, v3, ck_tile::direct_conv::DataType::bf16>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData4CKernel<
        ConfigIdx, v3, ck_tile::direct_conv::DataType::bf16>;
};

class DirectConvGrouped4cBf16TileConvTestV3
    : public DirectConvGrouped4cFp16TestHarness<TileConv4cBf16KernelTraits, ck_tile::bfloat16_t>
{
};

// =============================================================================
// V3 Group 0: No swizzle, direct DRAM epilogue
// =============================================================================

// --- Fprop: groups=16, only config 9 (waves_c64=1) is compatible ---

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_Config9_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_Config9_Groups16_NoPad)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Fprop: groups=32, configs 7-9 compatible (8x8 input, out_w=8) ---

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_Config7_Groups32)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_Config9_Groups32)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop: groups=32, 16x16 input (out_w=16), configs 6-9 compatible ---

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_Config6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<6>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop: groups=32, 32x32 input (out_w=32), config 5 (wq4=8) compatible ---

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_Config5_Groups32_Pad1)
{
    ASSERT_TRUE((RunFprop<5>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad: groups=16, only config 4 (waves_c64=1) is compatible ---

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_Config4_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_Config4_Groups16_NoPad)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad: groups=32, configs 2-4 compatible (8x8 input, out_w=8) ---

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_Config2_Groups32)
{
    ASSERT_TRUE((RunDgrad<2>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_Config4_Groups32)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad: groups=32, 16x16 input, configs 1-4 compatible ---

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_Config1_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<1>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad: groups=32, 32x32 input (out_w=32), config 0 (wq4=8) compatible ---

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_Config0_Groups32_Pad1)
{
    ASSERT_TRUE((RunDgrad<0>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

// =============================================================================
// V3 Group 1: No swizzle, LDS-staged epilogue (Fprop 15-19, Dgrad 10-14)
// =============================================================================

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_LdsEpilogue_Config19_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_LdsEpilogue_Config17_Groups32)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config14_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<14>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config12_Groups32)
{
    ASSERT_TRUE((RunDgrad<12>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// =============================================================================
// V3 Group 2: XOR swizzle, direct DRAM epilogue (Fprop 25-29, Dgrad 20-24)
// =============================================================================

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_XOR_Config29_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<29>(1, 4, 4, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_XOR_Config27_Groups32)
{
    ASSERT_TRUE((RunFprop<27>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_XOR_Config24_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<24>(1, 4, 4, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_XOR_Config22_Groups32)
{
    ASSERT_TRUE((RunDgrad<22>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// =============================================================================
// V3 Group 3: XOR swizzle, LDS-staged epilogue (Fprop 35-39, Dgrad 30-34)
// =============================================================================

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config39_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<39>(1, 4, 4, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config37_Groups32)
{
    ASSERT_TRUE((RunFprop<37>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config34_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<34>(1, 4, 4, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config32_Groups32)
{
    ASSERT_TRUE((RunDgrad<32>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// =============================================================================
// Cyclic-shift swizzle tests (instances 40-43)
// =============================================================================

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, FProp_V3_CyclicShift_LdsEpilogue_Groups32_NoPad)
{
    ASSERT_TRUE((RunFprop<40>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped4cBf16TileConvTestV3, Dgrad_V3_CyclicShift_LdsEpilogue_Groups32_NoPad)
{
    ASSERT_TRUE((RunDgrad<42>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}
