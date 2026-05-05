// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_grouped_4c_fp16_harness.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_4c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_4c_fp16_tile_conv_kernel.hpp"
#pragma clang diagnostic pop

constexpr auto v3 = ck_tile::direct_conv::Version::v3;

struct TileConvKernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward4CFp16Kernel<ConfigIdx, v3>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData4CFp16Kernel<ConfigIdx, v3>;
};

class DirectConvGrouped4cFp16TileConvTestV3
    : public DirectConvGrouped4cFp16TestHarness<TileConvKernelTraits>
{
};

// =============================================================================
// V3 Config index reference (40 configs total, 4 groups × 10):
//
// Group 0 (No swizzle, DRAM epilogue):
//   Dgrad: 0 (wc64=2,wq4=8), 1 (2,4), 2 (2,2), 3 (2,1), 4 (1,1)
//   Fprop: 5 (wc64=2,wq4=8), 6 (2,4), 7 (2,2), 8 (2,1), 9 (1,1)
//
// Group 1 (No swizzle, LDS epilogue):
//   Dgrad: 10-14,  Fprop: 15-19  (same wave pattern)
//
// Group 2 (XOR swizzle, DRAM epilogue):
//   Dgrad: 20-24,  Fprop: 25-29
//
// Group 3 (XOR swizzle, LDS epilogue):
//   Dgrad: 30-34,  Fprop: 35-39
//
// Compatibility:
//   waves_c64=2 requires groups % 32 == 0
//   waves_c64=1 requires groups % 16 == 0
//   waves_q4>1  requires output_width >= waves_q4 * 4
// =============================================================================

// =============================================================================
// V3 Group 0: No swizzle, direct DRAM epilogue
// =============================================================================

// --- Fprop: groups=16, only config 9 (waves_c64=1) is compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config9_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config9_Groups16_NoPad)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Fprop: groups=32, configs 7-9 compatible (8x8 input, out_w=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config7_Groups32)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config8_Groups32)
{
    ASSERT_TRUE((RunFprop<8>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config9_Groups32)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop: groups=32, 16x16 input (out_w=16), configs 6-9 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<6>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config7_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<7>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config8_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<8>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config9_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<9>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop: groups=64, configs 7-9 compatible (8x8 input) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config7_ManyGroups)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config8_ManyGroups)
{
    ASSERT_TRUE((RunFprop<8>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config9_ManyGroups)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop: groups=32, 32x32 input (out_w=32), config 5 (wq4=8) compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config5_Groups32_Pad1)
{
    ASSERT_TRUE((RunFprop<5>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Config5_Groups32_NoPad)
{
    ASSERT_TRUE((RunFprop<5>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad: groups=16, only config 4 (waves_c64=1) is compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config4_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config4_Groups16_NoPad)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad: groups=32, configs 2-4 compatible (8x8 input, out_w=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config2_Groups32)
{
    ASSERT_TRUE((RunDgrad<2>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config3_Groups32)
{
    ASSERT_TRUE((RunDgrad<3>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config4_Groups32)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad: groups=32, 16x16 input, configs 1-4 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config1_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<1>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config2_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<2>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config3_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<3>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config4_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad: groups=32, 32x32 input (out_w=32), config 0 (wq4=8) compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config0_Groups32_Pad1)
{
    ASSERT_TRUE((RunDgrad<0>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_Config0_Groups32_NoPad)
{
    ASSERT_TRUE((RunDgrad<0>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// =============================================================================
// V3 Group 1: No swizzle, LDS-staged epilogue (Fprop 15-19, Dgrad 10-14)
// =============================================================================

// --- Fprop LDS: groups=16, only config 19 (waves_c64=1) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config19_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config19_Groups16_NoPad)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Fprop LDS: groups=32, configs 17-19 compatible (8x8 input, out_w=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config17_Groups32)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config18_Groups32)
{
    ASSERT_TRUE((RunFprop<18>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config19_Groups32)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop LDS: groups=32, 16x16 input, configs 16-19 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config16_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<16>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config17_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<17>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config18_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<18>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config19_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<19>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop LDS: groups=64, configs 17-19 compatible (8x8 input) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config17_ManyGroups)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config18_ManyGroups)
{
    ASSERT_TRUE((RunFprop<18>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config19_ManyGroups)
{
    ASSERT_TRUE((RunFprop<19>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop LDS: groups=32, 32x32 input, config 15 (wq4=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config15_Groups32_Pad1)
{
    ASSERT_TRUE((RunFprop<15>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_LdsEpilogue_Config15_Groups32_NoPad)
{
    ASSERT_TRUE((RunFprop<15>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad LDS: groups=16, only config 14 (waves_c64=1) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config14_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<14>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config14_Groups16_NoPad)
{
    ASSERT_TRUE((RunDgrad<14>(1, 8, 8, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad LDS: groups=32, configs 12-14 compatible (8x8 input, out_w=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config12_Groups32)
{
    ASSERT_TRUE((RunDgrad<12>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config13_Groups32)
{
    ASSERT_TRUE((RunDgrad<13>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config14_Groups32)
{
    ASSERT_TRUE((RunDgrad<14>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad LDS: groups=32, 16x16 input, configs 11-14 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config11_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<11>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config12_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<12>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config13_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<13>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config14_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<14>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad LDS: groups=32, 32x32 input, config 10 (wq4=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config10_Groups32_Pad1)
{
    ASSERT_TRUE((RunDgrad<10>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_LdsEpilogue_Config10_Groups32_NoPad)
{
    ASSERT_TRUE((RunDgrad<10>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// =============================================================================
// V3 Group 2: XOR swizzle, direct DRAM epilogue (Fprop 25-29, Dgrad 20-24)
//
// XOR swizzle constraint: block_q must be aligned to BLOCK_C8 (= block_c/8).
// For waves_c64=2: BLOCK_C8=16, so block_q must be multiple of 16.
//   Config 25/20 (wq4=8, block_q=32): aligned ✓  multi-tile OK
//   Config 26/21 (wq4=4, block_q=16): aligned ✓  multi-tile OK
//   Config 27/22 (wq4=2, block_q=8):  NOT aligned, single-tile only (out_w<=8)
//   Config 28/23 (wq4=1, block_q=4):  NOT aligned, single-tile only (out_w<=4)
// For waves_c64=1: BLOCK_C8=8, so block_q must be multiple of 8.
//   Config 29/24 (wq4=1, block_q=4):  NOT aligned, single-tile only (out_w<=4)
// =============================================================================

// --- Fprop XOR: groups=16, config 29 (c64=1), 4x4 input (single tile) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config29_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<29>(1, 4, 4, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config29_Groups16_NoPad)
{
    ASSERT_TRUE((RunFprop<29>(1, 4, 4, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Fprop XOR: groups=32, 8x8 input (out_w=8), configs 27 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config27_Groups32)
{
    ASSERT_TRUE((RunFprop<27>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop XOR: groups=32, 4x4 input (out_w=4, single tile), configs 28-29 ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config28_Groups32)
{
    ASSERT_TRUE((RunFprop<28>(1, 4, 4, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config29_Groups32)
{
    ASSERT_TRUE((RunFprop<29>(1, 4, 4, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop XOR: groups=32, 16x16 input (out_w=16), configs 25-26 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config25_LargerSpatial)
{
    // Config 25: waves_q4=8, block_q=32 — need out_q >= 32
    ASSERT_TRUE((RunFprop<25>(4, 34, 34, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config26_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<26>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop XOR: groups=64, 8x8 input, config 27 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config27_ManyGroups)
{
    ASSERT_TRUE((RunFprop<27>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop XOR: groups=32, 32x32 input, config 25 (wq4=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config25_Groups32_Pad1)
{
    ASSERT_TRUE((RunFprop<25>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_Config25_Groups32_NoPad)
{
    ASSERT_TRUE((RunFprop<25>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad XOR: groups=16, config 24 (c64=1), 4x4 input (single tile) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config24_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<24>(1, 4, 4, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config24_Groups16_NoPad)
{
    ASSERT_TRUE((RunDgrad<24>(1, 4, 4, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad XOR: groups=32, 8x8 input, config 22 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config22_Groups32)
{
    ASSERT_TRUE((RunDgrad<22>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad XOR: groups=32, 4x4 input (single tile), configs 23-24 ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config23_Groups32)
{
    ASSERT_TRUE((RunDgrad<23>(1, 4, 4, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config24_Groups32)
{
    ASSERT_TRUE((RunDgrad<24>(1, 4, 4, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad XOR: groups=32, 16x16 input, configs 20-21 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config20_LargerSpatial)
{
    // Config 20: waves_q4=8, block_q=32 — need out_w >= 32
    ASSERT_TRUE((RunDgrad<20>(4, 34, 34, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config21_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<21>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad XOR: groups=32, 32x32 input, config 20 (wq4=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config20_Groups32_Pad1)
{
    ASSERT_TRUE((RunDgrad<20>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_Config20_Groups32_NoPad)
{
    ASSERT_TRUE((RunDgrad<20>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// =============================================================================
// V3 Group 3: XOR swizzle, LDS-staged epilogue (Fprop 35-39, Dgrad 30-34)
// Same XOR alignment constraints as Group 2.
// =============================================================================

// --- Fprop XOR LDS: groups=16, config 39 (c64=1), 4x4 input (single tile) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config39_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<39>(1, 4, 4, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config39_Groups16_NoPad)
{
    ASSERT_TRUE((RunFprop<39>(1, 4, 4, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Fprop XOR LDS: groups=32, 8x8 input, config 37 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config37_Groups32)
{
    ASSERT_TRUE((RunFprop<37>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop XOR LDS: groups=32, 4x4 input (single tile), configs 38-39 ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config38_Groups32)
{
    ASSERT_TRUE((RunFprop<38>(1, 4, 4, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config39_Groups32)
{
    ASSERT_TRUE((RunFprop<39>(1, 4, 4, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop XOR LDS: groups=32, 16x16 input, configs 35-36 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config35_LargerSpatial)
{
    // Config 35: waves_q4=8, block_q=32 — need out_q >= 32
    ASSERT_TRUE((RunFprop<35>(4, 34, 34, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config36_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<36>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop XOR LDS: groups=64, 8x8 input, config 37 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config37_ManyGroups)
{
    ASSERT_TRUE((RunFprop<37>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop XOR LDS: groups=32, 32x32 input, config 35 (wq4=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config35_Groups32_Pad1)
{
    ASSERT_TRUE((RunFprop<35>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_XOR_LdsEpilogue_Config35_Groups32_NoPad)
{
    ASSERT_TRUE((RunFprop<35>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad XOR LDS: groups=16, config 34 (c64=1), 4x4 input (single tile) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config34_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<34>(1, 4, 4, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config34_Groups16_NoPad)
{
    ASSERT_TRUE((RunDgrad<34>(1, 4, 4, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad XOR LDS: groups=32, 8x8 input, config 32 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config32_Groups32)
{
    ASSERT_TRUE((RunDgrad<32>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad XOR LDS: groups=32, 4x4 input (single tile), configs 33-34 ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config33_Groups32)
{
    ASSERT_TRUE((RunDgrad<33>(1, 4, 4, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config34_Groups32)
{
    ASSERT_TRUE((RunDgrad<34>(1, 4, 4, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad XOR LDS: groups=32, 16x16 input, configs 30-31 compatible ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config30_LargerSpatial)
{
    // Config 30: waves_q4=8, block_q=32 — need out_w >= 32
    ASSERT_TRUE((RunDgrad<30>(4, 34, 34, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config31_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<31>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad XOR LDS: groups=32, 32x32 input, config 30 (wq4=8) ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config30_Groups32_Pad1)
{
    ASSERT_TRUE((RunDgrad<30>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Dgrad_V3_XOR_LdsEpilogue_Config30_Groups32_NoPad)
{
    ASSERT_TRUE((RunDgrad<30>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// Cyclic-shift swizzle tests (instances 40-43)
class DirectConvGrouped4cFp16TileConvV3CyclicShiftSwizzleTest
    : public DirectConvGrouped4cFp16TestHarness<TileConvKernelTraits>
{
};

TEST_F(DirectConvGrouped4cFp16TileConvV3CyclicShiftSwizzleTest, FProp_V3_CyclicShift_LdsEpilogue_Groups32_NoPad)
{
    ASSERT_TRUE((RunFprop<40>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3CyclicShiftSwizzleTest, FProp_V3_CyclicShift_SkipdLds_Groups32_NoPad)
{
    ASSERT_TRUE((RunFprop<41>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3CyclicShiftSwizzleTest, Dgrad_V3_CyclicShift_LdsEpilogue_Groups32_NoPad)
{
    ASSERT_TRUE((RunDgrad<42>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3CyclicShiftSwizzleTest, Dgrad_V3_CyclicShift_SkipdLds_Groups32_NoPad)
{
    ASSERT_TRUE((RunDgrad<43>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}

// =============================================================================
// V3 C != K padded convolution tests
//
// Configs 44-46: Dgrad CyclicShift, vector_size = {4, 2, 1}
// Configs 47-49: Fprop CyclicShift, vector_size = {4, 2, 1}
//
// vector_size=1 (configs 46/49) supports any c_per_group/k_per_group in {1,2,3,4}
// vector_size=2 (configs 45/48) requires c_per_group % 2 == 0 and k_per_group % 2 == 0
// =============================================================================

class DirectConvGrouped4cFp16TileConvV3PaddedTest
    : public DirectConvGrouped4cFp16TestHarness<TileConvKernelTraits>
{
};

// --- Fprop: C != K, vector_size=1 (config 49) ---
// Config 49: waves_c64=2 (groups%32==0), waves_q4=8 (out_q>=32 needed)
// With 34x34 input, pad=1, 3x3 filter: out_q = 34

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Fprop_C3_K2)
{
    ASSERT_TRUE((RunFprop<49>(1, 34, 34, 32, 3, 2, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Fprop_C1_K4)
{
    ASSERT_TRUE((RunFprop<49>(1, 34, 34, 32, 1, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Fprop_C4_K1)
{
    ASSERT_TRUE((RunFprop<49>(1, 34, 34, 32, 4, 1, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Fprop_C1_K3)
{
    ASSERT_TRUE((RunFprop<49>(1, 34, 34, 32, 1, 3, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Fprop_C2_K3)
{
    ASSERT_TRUE((RunFprop<49>(1, 34, 34, 32, 2, 3, 3, 3, 1, 1)));
}

// --- Fprop: C != K, vector_size=2 (config 48) ---

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Fprop_Vec2_C2_K4)
{
    ASSERT_TRUE((RunFprop<48>(1, 34, 34, 32, 2, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Fprop_Vec2_C4_K2)
{
    ASSERT_TRUE((RunFprop<48>(1, 34, 34, 32, 4, 2, 3, 3, 1, 1)));
}

// --- Dgrad: C != K, vector_size=1 (config 46) ---
// For Dgrad, out_q = par.w (input width). waves_q4=8 needs w >= 32.

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Dgrad_C3_K2)
{
    ASSERT_TRUE((RunDgrad<46>(1, 34, 34, 32, 3, 2, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Dgrad_C1_K4)
{
    ASSERT_TRUE((RunDgrad<46>(1, 34, 34, 32, 1, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Dgrad_C4_K1)
{
    ASSERT_TRUE((RunDgrad<46>(1, 34, 34, 32, 4, 1, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Dgrad_C1_K3)
{
    ASSERT_TRUE((RunDgrad<46>(1, 34, 34, 32, 1, 3, 3, 3, 1, 1)));
}

// --- Dgrad: C != K, vector_size=2 (config 45) ---

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Dgrad_Vec2_C2_K4)
{
    ASSERT_TRUE((RunDgrad<45>(1, 34, 34, 32, 2, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvV3PaddedTest, Dgrad_Vec2_C4_K2)
{
    ASSERT_TRUE((RunDgrad<45>(1, 34, 34, 32, 4, 2, 3, 3, 1, 1)));
}

