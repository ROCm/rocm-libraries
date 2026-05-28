// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_harness.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

struct HipConvKernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectHipConvForward4CFp16Kernel<ConfigIdx>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectHipConvBwdData4CFp16Kernel<ConfigIdx>;
};

class DirectConvGrouped4cFp16HipTest
    : public DirectConvGroupedTestHarness<HipConvKernelTraits>
{
};

// Config index reference:
//   Fprop: 5 (wc64=2,wq4=8), 6 (2,4), 7 (2,2), 8 (2,1), 9 (1,1)
//   Dgrad: 0 (wc64=2,wq4=8), 1 (2,4), 2 (2,2), 3 (2,1), 4 (1,1)
//
// Compatibility:
//   waves_c64=2 requires groups % 32 == 0
//   waves_c64=1 requires groups % 16 == 0
//   waves_q4>1  requires output_width >= waves_q4 * 4

// --- Fprop: groups=16, only config 9 (waves_c64=1) is compatible ---

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config9_Groups16_Pad1)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config9_Groups16_NoPad)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Fprop: groups=32, configs 7-9 compatible (8x8 input, out_w=8) ---
//     Config 5 needs out_w>=32, config 6 needs out_w>=16 — both fail for 8x8.

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config7_Groups32)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config8_Groups32)
{
    ASSERT_TRUE((RunFprop<8>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config9_Groups32)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop: groups=32, 16x16 input (out_w=16), configs 6-9 compatible ---

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config6_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<6>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config7_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<7>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config8_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<8>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config9_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<9>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Fprop: groups=64, configs 7-9 compatible (8x8 input) ---

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config7_ManyGroups)
{
    ASSERT_TRUE((RunFprop<7>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config8_ManyGroups)
{
    ASSERT_TRUE((RunFprop<8>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Fprop_Config9_ManyGroups)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 64, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad: groups=16, only config 4 (waves_c64=1) is compatible ---

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config4_Groups16_Pad1)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 16, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config4_Groups16_NoPad)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 16, 4, 4, 3, 3, 0, 0)));
}

// --- Dgrad: groups=32, configs 2-4 compatible (8x8 input, out_w=8) ---

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config2_Groups32)
{
    ASSERT_TRUE((RunDgrad<2>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config3_Groups32)
{
    ASSERT_TRUE((RunDgrad<3>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config4_Groups32)
{
    ASSERT_TRUE((RunDgrad<4>(1, 8, 8, 32, 4, 4, 3, 3, 1, 1)));
}

// --- Dgrad: groups=32, 16x16 input, configs 1-4 compatible ---

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config1_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<1>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config2_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<2>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config3_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<3>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16HipTest, Dgrad_Config4_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<4>(4, 16, 16, 32, 4, 4, 3, 3, 1, 1)));
}
