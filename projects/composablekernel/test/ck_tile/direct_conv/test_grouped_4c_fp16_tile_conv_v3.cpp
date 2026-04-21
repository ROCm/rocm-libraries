// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_grouped_4c_fp16_harness.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/direct_conv_fwd_4c_fp16_tile_conv_kernel.hpp"
#pragma clang diagnostic pop

template <ck_tile::direct_conv::Version Ver>
struct TileConvKernelTraitsV3
{
    template <int ConfigIdx>
    using FwdKernel = ck_tile::direct_conv::DirectTileConvForward4CFp16Kernel<ConfigIdx, Ver>;
    // BwdData not yet implemented for v3.
    template <int ConfigIdx>
    using BwdDataKernel = ck_tile::direct_conv::DirectTileConvForward4CFp16Kernel<ConfigIdx, Ver>;
};

constexpr auto v3 = ck_tile::direct_conv::Version::v3;

class DirectConvGrouped4cFp16TileConvTestV3
    : public DirectConvGrouped4cFp16TestHarness<TileConvKernelTraitsV3<v3>>
{
};

// =============================================================================
// V3 async_load_tile — Fprop config 1
// =============================================================================

// --- Fprop V3: groups=32 ---

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Groups32_Pad1)
{
    ASSERT_TRUE((RunFprop<1>(1, 32, 32, 32, 4, 4, 3, 3, 1, 1)));
}

TEST_F(DirectConvGrouped4cFp16TileConvTestV3, Fprop_V3_Groups32_NoPad)
{
    ASSERT_TRUE((RunFprop<1>(2, 64, 64, 32, 4, 4, 3, 3, 0, 0)));
}
