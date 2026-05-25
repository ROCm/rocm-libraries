// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_harness.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

constexpr auto v3 = ck_tile::direct_conv::Version::v3;

// =============================================================================
// v3 tests — cross-wave LDS reduction.
//
// v3 KernelConfigurations layout:
//   Config 0: Dgrad, 4 waves (C=128, K multiple of 16)
//   Config 1: Dgrad, 2 waves (C=64, K multiple of 16)
//   Config 2: Fprop, 4 waves (C=128, K multiple of 16)
//   Config 3: Fprop, 2 waves (C=64, K multiple of 16)
//
// v3 requires C == block_c (= waves_per_wg * 32), i.e. num_c_blocks = 1.
// K must be a multiple of 16.
// =============================================================================

struct TileConv32cDenseKernelTraitsV3
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward32CDenseKernel<ConfigIdx, v3>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData32CDenseKernel<ConfigIdx, v3>;
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
