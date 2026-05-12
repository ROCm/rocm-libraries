// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_grouped_4c_fp16_harness.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

struct HipConv8cKernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectHipConvForward8CFp16Kernel<ConfigIdx>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectHipConvBwdData8CFp16Kernel<ConfigIdx>;
};

class DirectConvGrouped8cFp16HipTest
    : public DirectConvGrouped4cFp16TestHarness<HipConv8cKernelTraits>
{
};

// Config index reference (8c kernel):
//   Dgrad: 0-8 (waves_per_wg = 16,8,7,6,5,4,3,2,1)
//   Fprop: 9-17 (waves_per_wg = 16,8,7,6,5,4,3,2,1)
//
// Compatibility: groups % waves_per_wg == 0

// --- Fprop: groups=1, only config 17 (waves_per_wg=1) is compatible ---

TEST_F(DirectConvGrouped8cFp16HipTest, Fprop_Config17_Groups1_Pad1)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Fprop_Config17_Groups1_NoPad)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 1, 8, 8, 3, 3, 0, 0)));
}

// --- Fprop: groups=16, configs 14-17 compatible ---

TEST_F(DirectConvGrouped8cFp16HipTest, Fprop_Config17_Groups16)
{
    ASSERT_TRUE((RunFprop<17>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Fprop_Config16_Groups16)
{
    ASSERT_TRUE((RunFprop<16>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

// Config 15: waves_per_wg=3, needs groups%3==0
TEST_F(DirectConvGrouped8cFp16HipTest, Fprop_Config15_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<15>(4, 16, 16, 6, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Fprop_Config14_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<14>(4, 16, 16, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Fprop_Config13_Groups10_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<13>(4, 16, 16, 10, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Fprop_Config12_Groups12_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<12>(4, 16, 16, 12, 8, 8, 3, 3, 1, 1)));
}

// --- Dgrad: groups=1, only config 8 (waves_per_wg=1) is compatible ---

TEST_F(DirectConvGrouped8cFp16HipTest, Dgrad_Config8_Groups1_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Dgrad_Config8_Groups1_NoPad)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 8, 8, 3, 3, 0, 0)));
}

// --- Dgrad: groups=16, configs 5-8 compatible ---

TEST_F(DirectConvGrouped8cFp16HipTest, Dgrad_Config8_Groups16)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Dgrad_Config7_Groups16)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 16, 8, 8, 3, 3, 1, 1)));
}

// Config 6: waves_per_wg=3, needs groups%3==0
TEST_F(DirectConvGrouped8cFp16HipTest, Dgrad_Config6_Groups6_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<6>(4, 16, 16, 6, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Dgrad_Config5_Groups16_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<5>(4, 16, 16, 16, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Dgrad_Config4_Groups10_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(4, 16, 16, 10, 8, 8, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped8cFp16HipTest, Dgrad_Config3_Groups12_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<3>(4, 16, 16, 12, 8, 8, 3, 3, 1, 1)));
}
