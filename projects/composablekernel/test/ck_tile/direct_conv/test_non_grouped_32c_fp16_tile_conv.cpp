// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_harness.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_kernels.hpp"
#pragma clang diagnostic pop

constexpr auto v1 = ck_tile::direct_conv::Version::v1;

struct TileConv32cDenseKernelTraits
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward32CDenseKernel<ConfigIdx, v1>;
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvBwdData32CDenseKernel<ConfigIdx, v1>;
};

// =============================================================================
// Non-grouped (dense) convolution — 32c MFMA with C-reduction.
//
// KernelConfigurations layout:
//   Configs 0-2:  Dgrad, direct DRAM epilogue  (waves 8, 4, 2)
//   Configs 3-5:  Fprop, direct DRAM epilogue  (waves 8, 4, 2)
//   Configs 6-8:  Dgrad, LDS-staged epilogue   (waves 8, 4, 2)
//   Configs 9-11: Fprop, LDS-staged epilogue   (waves 8, 4, 2)
//
// Alignment requirements:
//   Config 2/5/8/11 (2 waves): C % 32 == 0, K % 32 == 0
//   Config 1/4/7/10 (4 waves): C % 64 == 0, K % 64 == 0
//   Config 0/3/6/9  (8 waves): C % 128 == 0, K % 128 == 0
// =============================================================================

// --- Fprop, direct DRAM epilogue ---

class DirectConvNonGrouped32cFp16DramTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraits>
{
};

// Config 5: 2 waves — smallest config, C=K=64
TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config5_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<5>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config5_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<5>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config5_C64_K64_Pad2)
{
    ASSERT_TRUE((RunFprop<5>(1, 8, 8, 1, 64, 64, 3, 3, 2, 2)));
}

// Config 5: C != K
TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config5_C64_K128)
{
    ASSERT_TRUE((RunFprop<5>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config5_C128_K64)
{
    ASSERT_TRUE((RunFprop<5>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

// Config 5: larger spatial
TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config5_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<5>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// Config 4: 4 waves — C=K=128
TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config4_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<4>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config4_C128_K128_NoPad)
{
    ASSERT_TRUE((RunFprop<4>(1, 8, 8, 1, 128, 128, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config4_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<4>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// Config 4: C != K
TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config4_C64_K256)
{
    ASSERT_TRUE((RunFprop<4>(1, 8, 8, 1, 64, 256, 3, 3, 1, 1)));
}

// Config 3: 8 waves — C=K=256
TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config3_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DramTest, Fprop_Config3_C128_K128)
{
    ASSERT_TRUE((RunFprop<3>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

// --- Fprop, LDS-staged epilogue ---

class DirectConvNonGrouped32cFp16LdsTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraits>
{
};

// Config 11: 2 waves — C=K=64
TEST_F(DirectConvNonGrouped32cFp16LdsTest, Fprop_Config11_C64_K64_Pad1)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16LdsTest, Fprop_Config11_C64_K64_NoPad)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16LdsTest, Fprop_Config11_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunFprop<11>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// Config 11: C != K
TEST_F(DirectConvNonGrouped32cFp16LdsTest, Fprop_Config11_C64_K128)
{
    ASSERT_TRUE((RunFprop<11>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

// Config 10: 4 waves — C=K=128
TEST_F(DirectConvNonGrouped32cFp16LdsTest, Fprop_Config10_C128_K128_Pad1)
{
    ASSERT_TRUE((RunFprop<10>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16LdsTest, Fprop_Config10_C128_K128_NoPad)
{
    ASSERT_TRUE((RunFprop<10>(1, 8, 8, 1, 128, 128, 3, 3, 0, 0)));
}

// Config 9: 8 waves — C=K=256
TEST_F(DirectConvNonGrouped32cFp16LdsTest, Fprop_Config9_C256_K256_Pad1)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16LdsTest, Fprop_Config9_C128_K128)
{
    ASSERT_TRUE((RunFprop<9>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

// =============================================================================
// Dgrad tests
// =============================================================================

// --- Dgrad, direct DRAM epilogue ---

class DirectConvNonGrouped32cFp16DgradDramTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraits>
{
};

// Config 2: 2 waves — C=K=64
TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config2_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<2>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config2_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<2>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config2_C64_K64_Pad2)
{
    ASSERT_TRUE((RunDgrad<2>(1, 8, 8, 1, 64, 64, 3, 3, 2, 2)));
}

// Config 2: C != K
TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config2_C64_K128)
{
    ASSERT_TRUE((RunDgrad<2>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config2_C128_K64)
{
    ASSERT_TRUE((RunDgrad<2>(1, 8, 8, 1, 128, 64, 3, 3, 1, 1)));
}

// Config 2: larger spatial
TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config2_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<2>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// Config 1: 4 waves — C=K=128
TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config1_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config1_C128_K128_NoPad)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 128, 128, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config1_C128_K128_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<1>(2, 16, 16, 1, 128, 128, 3, 3, 1, 1)));
}

// Config 1: C != K
TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config1_C64_K256)
{
    ASSERT_TRUE((RunDgrad<1>(1, 8, 8, 1, 64, 256, 3, 3, 1, 1)));
}

// Config 0: 8 waves — C=K=256
TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config0_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradDramTest, Dgrad_Config0_C128_K128)
{
    ASSERT_TRUE((RunDgrad<0>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

// --- Dgrad, LDS-staged epilogue ---

class DirectConvNonGrouped32cFp16DgradLdsTest
    : public DirectConvGroupedTestHarness<TileConv32cDenseKernelTraits>
{
};

// Config 8: 2 waves — C=K=64
TEST_F(DirectConvNonGrouped32cFp16DgradLdsTest, Dgrad_Config8_C64_K64_Pad1)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 64, 64, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradLdsTest, Dgrad_Config8_C64_K64_NoPad)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 64, 64, 3, 3, 0, 0)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradLdsTest, Dgrad_Config8_C64_K64_LargerSpatial)
{
    ASSERT_TRUE((RunDgrad<8>(2, 16, 16, 1, 64, 64, 3, 3, 1, 1)));
}

// Config 8: C != K
TEST_F(DirectConvNonGrouped32cFp16DgradLdsTest, Dgrad_Config8_C64_K128)
{
    ASSERT_TRUE((RunDgrad<8>(1, 8, 8, 1, 64, 128, 3, 3, 1, 1)));
}

// Config 7: 4 waves — C=K=128
TEST_F(DirectConvNonGrouped32cFp16DgradLdsTest, Dgrad_Config7_C128_K128_Pad1)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradLdsTest, Dgrad_Config7_C128_K128_NoPad)
{
    ASSERT_TRUE((RunDgrad<7>(1, 8, 8, 1, 128, 128, 3, 3, 0, 0)));
}

// Config 6: 8 waves — C=K=256
TEST_F(DirectConvNonGrouped32cFp16DgradLdsTest, Dgrad_Config6_C256_K256_Pad1)
{
    ASSERT_TRUE((RunDgrad<6>(1, 8, 8, 1, 256, 256, 3, 3, 1, 1)));
}

TEST_F(DirectConvNonGrouped32cFp16DgradLdsTest, Dgrad_Config6_C128_K128)
{
    ASSERT_TRUE((RunDgrad<6>(1, 8, 8, 1, 128, 128, 3, 3, 1, 1)));
}
