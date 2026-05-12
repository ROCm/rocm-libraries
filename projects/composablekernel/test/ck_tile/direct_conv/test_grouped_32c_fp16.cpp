// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_grouped_4c_fp16_harness.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

struct HipConv32cKernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectHipConvForward32CFp16Kernel<ConfigIdx>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectHipConvBwdData32CFp16Kernel<ConfigIdx>;
};

class DirectConvGrouped32cFp16HipTest
    : public DirectConvGrouped4cFp16TestHarness<HipConv32cKernelTraits>
{
};

// Config index reference (32c HIP kernel):
//   Dgrad: 0 (waves_per_wg=4, block_groups=2), 1 (waves_per_wg=2, block_groups=1)
//   Fprop: 2 (waves_per_wg=4, block_groups=2), 3 (waves_per_wg=2, block_groups=1)
//
// Compatibility: groups % block_groups == 0

// --- Fprop: groups=1, only config 3 (block_groups=1) is compatible ---

TEST_F(DirectConvGrouped32cFp16HipTest, Fprop_Config3_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16HipTest, Fprop_Config3_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

// --- Fprop: groups=2, both configs compatible ---

TEST_F(DirectConvGrouped32cFp16HipTest, Fprop_Config3_Groups2)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16HipTest, Fprop_Config2_Groups2)
{
    ASSERT_TRUE((RunFprop<2>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

// --- Fprop: larger spatial ---

TEST_F(DirectConvGrouped32cFp16HipTest, Fprop_Config3_Groups4_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<3>(4, 16, 16, 4, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16HipTest, Fprop_Config2_Groups4_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<2>(4, 16, 16, 4, 32, 32, 3, 3, 1, 1)));
}

// --- Dgrad: groups=1, only config 1 (block_groups=1) is compatible ---

TEST_F(DirectConvGrouped32cFp16HipTest, Dgrad_Config1_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16HipTest, Dgrad_Config1_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 32, 32, 3, 3, 0, 0)));
}

// --- Dgrad: groups=2, both configs compatible ---

TEST_F(DirectConvGrouped32cFp16HipTest, Dgrad_Config1_Groups2)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16HipTest, Dgrad_Config0_Groups2)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 2, 32, 32, 3, 3, 1, 1)));
}

// --- Dgrad: larger spatial ---

TEST_F(DirectConvGrouped32cFp16HipTest, Dgrad_Config1_Groups4_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<1>(4, 16, 16, 4, 32, 32, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped32cFp16HipTest, Dgrad_Config0_Groups4_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<0>(4, 16, 16, 4, 32, 32, 3, 3, 1, 1)));
}
