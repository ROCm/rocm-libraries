// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_grouped_4c_fp16_harness.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_8c_fp16_tile_conv_kernel.hpp"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_bwd_data_8c_fp16_tile_conv_kernel.hpp"
#pragma clang diagnostic pop

constexpr auto v1 = ck_tile::direct_conv::Version::v1;
constexpr auto v2 = ck_tile::direct_conv::Version::v2;

struct TileConv8cKernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward8CFp16Kernel<ConfigIdx, v1>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData8CFp16Kernel<ConfigIdx, v1>;
};

// =============================================================================
// V1 tests (CK Tile implementation of 8c Toeplitz kernel)
// =============================================================================

class DirectConvGrouped8cFp16TileConvV1Test
    : public DirectConvGrouped4cFp16TestHarness<TileConv8cKernelTraits>
{
};

// Config index reference (8c kernel):
//   Dgrad: 0-8 (waves_per_wg = 16,8,7,6,5,4,3,2,1)
//   Fprop: 9-17 (waves_per_wg = 16,8,7,6,5,4,3,2,1)
//
// Compatibility: groups % waves_per_wg == 0

// --- V1 Fprop ---

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Fprop_Config17_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Fprop_Config17_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 8, 8, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Fprop_Config17_Groups16)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Fprop_Config16_Groups16)
{
    ASSERT_TRUE((RunFprop<16>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Fprop_Config15_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<15>(4, 16, 16, 6, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Fprop_Config14_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<14>(4, 16, 16, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Fprop_Config13_Groups10_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<13>(4, 16, 16, 10, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Fprop_Config12_Groups12_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<12>(4, 16, 16, 12, 8, 8, 3, 3, 1, 1)));
}

// --- V1 Dgrad ---

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Dgrad_Config8_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Dgrad_Config8_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 8, 8, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Dgrad_Config8_Groups16)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Dgrad_Config7_Groups16)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Dgrad_Config6_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<6>(4, 16, 16, 6, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Dgrad_Config5_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<5>(4, 16, 16, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Dgrad_Config4_Groups10_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(4, 16, 16, 10, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV1Test, Dgrad_Config3_Groups12_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<3>(4, 16, 16, 12, 8, 8, 3, 3, 1, 1)));
}

// =============================================================================
// V2 tests (CK Tile abstractions for data movement, same Toeplitz compute loop)
// =============================================================================

struct TileConv8cV2KernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward8CFp16Kernel<ConfigIdx, v2>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData8CFp16Kernel<ConfigIdx, v2>;
};

class DirectConvGrouped8cFp16TileConvV2Test
    : public DirectConvGrouped4cFp16TestHarness<TileConv8cV2KernelTraits>
{
};

// V2 config index reference (8c kernel):
//   Group 0 (no swizzle, direct DRAM):  Dgrad 0-8, Fprop 9-17
//   Group 1 (no swizzle, LDS epilogue): Dgrad 18-26, Fprop 27-35
//   Group 2 (XOR swizzle, direct DRAM): Dgrad 36-44, Fprop 45-53
//   Group 3 (XOR swizzle, LDS epilogue): Dgrad 54-62, Fprop 63-71
//   Cyclic-shift: 72 (Fprop LDS), 73 (Fprop direct), 74 (Dgrad LDS), 75 (Dgrad direct)
//
// Compatibility: groups % waves_per_wg == 0

// --- V2 Group 0: No swizzle, direct DRAM epilogue ---

// Fprop
TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_Config17_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_Config17_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 8, 8, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_Config17_Groups16)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_Config16_Groups16)
{
    ASSERT_TRUE((RunFprop<16>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_Config15_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<15>(4, 16, 16, 6, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_Config14_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<14>(4, 16, 16, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_Config13_Groups10_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<13>(4, 16, 16, 10, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_Config12_Groups12_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<12>(4, 16, 16, 12, 8, 8, 3, 3, 1, 1)));
}

// Dgrad
TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_Config8_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_Config8_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 8, 8, 3, 3, 0, 0)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_Config8_Groups16)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_Config7_Groups16)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_Config6_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<6>(4, 16, 16, 6, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_Config5_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<5>(4, 16, 16, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_Config4_Groups10_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(4, 16, 16, 10, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_Config3_Groups12_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<3>(4, 16, 16, 12, 8, 8, 3, 3, 1, 1)));
}

// --- V2 Group 1: No swizzle, LDS-staged epilogue ---

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_LDS_Config35_Groups1)
{
    ASSERT_TRUE((RunFprop<35>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_LDS_Config28_Groups16)
{
    ASSERT_TRUE((RunFprop<28>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_LDS_Config26_Groups1)
{
    ASSERT_TRUE((RunDgrad<26>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_LDS_Config19_Groups16)
{
    ASSERT_TRUE((RunDgrad<19>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

// --- V2 Group 2: XOR swizzle, direct DRAM epilogue ---

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_XOR_Config53_Groups1)
{
    ASSERT_TRUE((RunFprop<53>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_XOR_Config46_Groups16)
{
    ASSERT_TRUE((RunFprop<46>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_XOR_Config44_Groups1)
{
    ASSERT_TRUE((RunDgrad<44>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_XOR_Config37_Groups16)
{
    ASSERT_TRUE((RunDgrad<37>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

// --- V2 Group 3: XOR swizzle, LDS-staged epilogue ---

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_XORLDS_Config71_Groups1)
{
    ASSERT_TRUE((RunFprop<71>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_XORLDS_Config64_Groups16)
{
    ASSERT_TRUE((RunFprop<64>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_XORLDS_Config62_Groups1)
{
    ASSERT_TRUE((RunDgrad<62>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_XORLDS_Config55_Groups16)
{
    ASSERT_TRUE((RunDgrad<55>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

// --- V2 Cyclic-shift swizzle ---

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_CyclicShift_LDS_Config72_Groups8)
{
    ASSERT_TRUE((RunFprop<72>(1, 8, 8, 8, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Fprop_CyclicShift_Direct_Config73_Groups8)
{
    ASSERT_TRUE((RunFprop<73>(1, 8, 8, 8, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_CyclicShift_LDS_Config74_Groups8)
{
    ASSERT_TRUE((RunDgrad<74>(1, 8, 8, 8, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2Test, Dgrad_CyclicShift_Direct_Config75_Groups8)
{
    ASSERT_TRUE((RunDgrad<75>(1, 8, 8, 8, 8, 8, 3, 3, 1, 1)));
}

// =============================================================================
// Padded channel tests: c_per_group and/or k_per_group < 8 (but > 4 for at least one)
// Uses CyclicShift configs with small vector_size (indices 76-81)
// and None fallback configs (indices 82-83).
// =============================================================================

class DirectConvGrouped8cFp16TileConvV2PaddedTest
    : public DirectConvGrouped4cFp16TestHarness<TileConv8cV2KernelTraits>
{
};

// --- Fprop padded: C == K ---

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Fprop_C6_K6)
{
    // vector_size=2 config (index 80): 6 % 2 == 0
    ASSERT_TRUE((RunFprop<80>(1, 8, 8, 8, 6, 6, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Fprop_C5_K5)
{
    // vector_size=1 config (index 81): any channel count
    ASSERT_TRUE((RunFprop<81>(1, 8, 8, 8, 5, 5, 3, 3, 1, 1)));
}

// --- Fprop padded: C != K ---

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Fprop_C5_K8)
{
    // vector_size=1 config (index 81)
    ASSERT_TRUE((RunFprop<81>(1, 8, 8, 8, 5, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Fprop_C8_K5)
{
    ASSERT_TRUE((RunFprop<81>(1, 8, 8, 8, 8, 5, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Fprop_C6_K4)
{
    // vector_size=2 config (index 80): 6 % 2 == 0, 4 % 2 == 0; c > 4 so 8c kernel applies
    ASSERT_TRUE((RunFprop<80>(1, 8, 8, 8, 6, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Fprop_C5_K7)
{
    // vector_size=1 config (index 81)
    ASSERT_TRUE((RunFprop<81>(1, 8, 8, 8, 5, 7, 3, 3, 1, 1)));
}

// --- Dgrad padded: C == K ---

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Dgrad_C6_K6)
{
    // vector_size=2 config (index 77): 6 % 2 == 0
    ASSERT_TRUE((RunDgrad<77>(1, 8, 8, 8, 6, 6, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Dgrad_C5_K5)
{
    // vector_size=1 config (index 78)
    ASSERT_TRUE((RunDgrad<78>(1, 8, 8, 8, 5, 5, 3, 3, 1, 1)));
}

// --- Dgrad padded: C != K ---

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Dgrad_C5_K8)
{
    ASSERT_TRUE((RunDgrad<78>(1, 8, 8, 8, 5, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Dgrad_C8_K5)
{
    ASSERT_TRUE((RunDgrad<78>(1, 8, 8, 8, 8, 5, 3, 3, 1, 1)));
}

// --- No-swizzle fallback for padding ---

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Fprop_NoSwizzle_C5_K5)
{
    // Config 83: Fprop, None swizzle, vector_size=1
    ASSERT_TRUE((RunFprop<83>(1, 8, 8, 8, 5, 5, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Dgrad_NoSwizzle_C5_K5)
{
    // Config 82: Dgrad, None swizzle, vector_size=1
    ASSERT_TRUE((RunDgrad<82>(1, 8, 8, 8, 5, 5, 3, 3, 1, 1)));
}

// --- Larger spatial with padding ---

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Fprop_C6_K6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<80>(4, 16, 16, 8, 6, 6, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16TileConvV2PaddedTest, Dgrad_C6_K6_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<77>(4, 16, 16, 8, 6, 6, 3, 3, 1, 1)));
}
