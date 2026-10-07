// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core/arch/arch.hpp"

#include <hip/hip_runtime.h>

using namespace ck_tile;

namespace {

template <index_t BlockSize, index_t MinBlockPerCu, index_t WarpSize, typename Arch>
constexpr index_t budget =
    detail::get_vgpr_budget_per_wave<BlockSize, MinBlockPerCu, WarpSize>(Arch{});

// gfx1250: wave32, 1024-entry VGPR file per SIMD, granule 16, 1024 addressable per wave.
static_assert(budget<64, 1, 32, gfx125_t> == 1024);
static_assert(budget<128, 1, 32, gfx125_t> == 1024);
static_assert(budget<128, 2, 32, gfx125_t> == 512);
static_assert(budget<256, 1, 32, gfx125_t> == 512);
static_assert(budget<256, 2, 32, gfx125_t> == 512);
static_assert(budget<512, 1, 32, gfx125_t> == 256);
static_assert(budget<512, 2, 32, gfx125_t> == 256);
static_assert(budget<1024, 1, 32, gfx125_t> == 128);
static_assert(budget<1024, 2, 32, gfx125_t> == 128);
// granule rounding: 1024 / 3 = 341 -> 336
static_assert(budget<128, 3, 32, gfx125_t> == 336);

// gfx9 (modelled as gfx90a/gfx942) / gfx950: wave64, 512-entry unified file, granule 8,
// 512 addressable per wave.
static_assert(budget<256, 1, 64, gfx9_t> == 512);
static_assert(budget<256, 2, 64, gfx9_t> == 256);
static_assert(budget<512, 1, 64, gfx9_t> == 256);
static_assert(budget<1024, 1, 64, gfx9_t> == 128);
static_assert(budget<64, 3, 64, gfx9_t> == 168);
static_assert(budget<256, 1, 64, gfx950_t> == 512);
static_assert(budget<256, 2, 64, gfx950_t> == 256);

// RDNA (gfx10.3 / gfx11 / gfx12): wave32, per-wave maximum of 256 dominates small blocks.
static_assert(budget<256, 2, 32, gfx11_t> == 256);
static_assert(budget<512, 1, 32, gfx11_t> == 256);
static_assert(budget<1024, 1, 32, gfx11_t> == 128);
static_assert(budget<256, 2, 32, gfx120_t> == 256);
static_assert(budget<1024, 2, 32, gfx120_t> == 128);
static_assert(budget<256, 1, 32, gfx103_t> == 256);
static_assert(budget<256, 1, 64, gfx11_t> == 256);
static_assert(budget<1024, 1, 64, gfx11_t> == 128);

static_assert(budget<256, 2, 32, gfx_invalid_t> == 0);

} // namespace

// The device-side wrapper must agree with the arch-explicit form for the arch being compiled.
__global__ void check_vgpr_budget_matches_device_arch()
{
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(get_vgpr_budget_per_wave<256, 2>() ==
                  detail::get_vgpr_budget_per_wave<256, 2, get_warp_size()>(get_device_arch()));
#if defined(__gfx125__)
    static_assert(get_vgpr_budget_per_wave<128, 1>() == 1024);
    static_assert(get_vgpr_budget_per_wave<256, 2>() == 512);
    static_assert(get_vgpr_budget_per_wave<512, 2>() == 256);
    static_assert(get_vgpr_budget_per_wave<1024, 2>() == 128);
#elif defined(__gfx9__)
    static_assert(get_vgpr_budget_per_wave<256, 2>() == 256);
#endif
#endif
}

TEST(TestVgprBudget, Gfx1250Table)
{
    EXPECT_EQ((budget<128, 1, 32, gfx125_t>), 1024);
    EXPECT_EQ((budget<256, 2, 32, gfx125_t>), 512);
    EXPECT_EQ((budget<512, 2, 32, gfx125_t>), 256);
    EXPECT_EQ((budget<1024, 2, 32, gfx125_t>), 128);
}

TEST(TestVgprBudget, Gfx9Table)
{
    EXPECT_EQ((budget<256, 1, 64, gfx9_t>), 512);
    EXPECT_EQ((budget<256, 2, 64, gfx9_t>), 256);
    EXPECT_EQ((budget<256, 2, 64, gfx950_t>), 256);
}
