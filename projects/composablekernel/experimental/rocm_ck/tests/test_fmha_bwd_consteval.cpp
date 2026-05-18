// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tests for the consteval tile config dispatch table (getTileConfig).
// Verifies that each (hdim_q, hdim_v, dtype, arch) combination returns
// the correct tile geometry from fmha_bwd.py, and that makeSpec()
// integrates the tile config correctly.

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::DataType;
using ::rocm_ck::FmhaBiasType;
using ::rocm_ck::FmhaBwdDQDKDVConfig;
using ::rocm_ck::FmhaBwdDQDKDVTileConfig;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::getTileConfig;
using ::rocm_ck::GpuTarget;
using ::rocm_ck::makeSpec;
using ::rocm_ck::wavefrontSize;

// ============================================================================
// getTileConfig: GFX9 fp16/bf16 tile configs
// ============================================================================

TEST(TileConfig, GFX9_FP16_D32)
{
    constexpr auto t = getTileConfig(32, 32, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(t.hdim_q, 32);
    EXPECT_EQ(t.hdim_v, 32);
    EXPECT_EQ(t.bm0, 32);
    EXPECT_EQ(t.bn0, 128);
    EXPECT_EQ(t.bk0, 32);
    EXPECT_EQ(t.bk1, 32);
    EXPECT_EQ(t.bk2, 32);
    EXPECT_EQ(t.bk3, 32);
    EXPECT_EQ(t.bk4, 64);
    EXPECT_EQ(t.rm0, 1);
    EXPECT_EQ(t.rn0, 4);
    EXPECT_EQ(t.rk0, 1);
    EXPECT_EQ(t.wm0, 16);
    EXPECT_EQ(t.wn0, 16);
    EXPECT_EQ(t.wk0, 32);
    EXPECT_EQ(t.rm1, 4);
    EXPECT_EQ(t.rn1, 1);
    EXPECT_EQ(t.rk1, 1);
    EXPECT_EQ(t.wm1, 16);
    EXPECT_EQ(t.wn1, 16);
    EXPECT_EQ(t.wk1, 16);
    EXPECT_EQ(t.rm2, 2);
    EXPECT_EQ(t.rn2, 2);
    EXPECT_EQ(t.rk2, 1);
    EXPECT_EQ(t.occupancy, 1);
    EXPECT_EQ(t.max_seq_q, 0);
    EXPECT_EQ(t.num_warps(), 4);
    EXPECT_EQ(t.block_size(GpuTarget::gfx942), 256);
}

TEST(TileConfig, GFX9_FP16_D64)
{
    constexpr auto t = getTileConfig(64, 64, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(t.hdim_q, 64);
    EXPECT_EQ(t.hdim_v, 64);
    EXPECT_EQ(t.bm0, 32);
    EXPECT_EQ(t.bn0, 128);
    EXPECT_EQ(t.bk0, 64);
    EXPECT_EQ(t.bk1, 32);
    EXPECT_EQ(t.bk2, 64);
    EXPECT_EQ(t.bk3, 32);
    EXPECT_EQ(t.bk4, 32);
    EXPECT_EQ(t.rm2, 1);
    EXPECT_EQ(t.rn2, 4);
    EXPECT_EQ(t.rk2, 1);
    EXPECT_EQ(t.occupancy, 1);
    EXPECT_EQ(t.max_seq_q, 0);
    EXPECT_EQ(t.block_size(GpuTarget::gfx942), 256);
}

TEST(TileConfig, GFX9_FP16_D96)
{
    constexpr auto t = getTileConfig(96, 96, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(t.hdim_q, 96);
    EXPECT_EQ(t.hdim_v, 96);
    EXPECT_EQ(t.bm0, 32);
    EXPECT_EQ(t.bn0, 128);
    EXPECT_EQ(t.bk0, 96);
    EXPECT_EQ(t.bk1, 32);
    EXPECT_EQ(t.bk2, 96);
    EXPECT_EQ(t.bk3, 32);
    EXPECT_EQ(t.bk4, 32);
    EXPECT_EQ(t.rm2, 2);
    EXPECT_EQ(t.rn2, 2);
    EXPECT_EQ(t.rk2, 1);
    EXPECT_EQ(t.occupancy, 1);
    EXPECT_EQ(t.max_seq_q, 0);
    EXPECT_EQ(t.block_size(GpuTarget::gfx942), 256);
}

TEST(TileConfig, GFX9_FP16_D128)
{
    constexpr auto t = getTileConfig(128, 128, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(t.hdim_q, 128);
    EXPECT_EQ(t.hdim_v, 128);
    EXPECT_EQ(t.bm0, 16);
    EXPECT_EQ(t.bn0, 128);
    EXPECT_EQ(t.bk0, 128);
    EXPECT_EQ(t.bk1, 16);
    EXPECT_EQ(t.bk2, 128);
    EXPECT_EQ(t.bk3, 16);
    EXPECT_EQ(t.bk4, 32);
    EXPECT_EQ(t.rm0, 1);
    EXPECT_EQ(t.rn0, 4);
    EXPECT_EQ(t.rk0, 1);
    EXPECT_EQ(t.wm0, 16);
    EXPECT_EQ(t.wn0, 16);
    EXPECT_EQ(t.wk0, 32);
    EXPECT_EQ(t.rm1, 4);
    EXPECT_EQ(t.rn1, 1);
    EXPECT_EQ(t.rk1, 1);
    EXPECT_EQ(t.wm1, 16);
    EXPECT_EQ(t.wn1, 16);
    EXPECT_EQ(t.wk1, 16);
    EXPECT_EQ(t.rm2, 1);
    EXPECT_EQ(t.rn2, 4);
    EXPECT_EQ(t.rk2, 1);
    EXPECT_EQ(t.occupancy, 1);
    EXPECT_EQ(t.max_seq_q, 0);
    EXPECT_EQ(t.num_warps(), 4);
    EXPECT_EQ(t.block_size(GpuTarget::gfx942), 256);
}

TEST(TileConfig, GFX9_FP16_D256)
{
    constexpr auto t = getTileConfig(256, 256, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(t.hdim_q, 256);
    EXPECT_EQ(t.hdim_v, 256);
    EXPECT_EQ(t.bm0, 16);
    EXPECT_EQ(t.bn0, 64);
    EXPECT_EQ(t.bk0, 256);
    EXPECT_EQ(t.bk1, 16);
    EXPECT_EQ(t.bk2, 256);
    EXPECT_EQ(t.bk3, 16);
    EXPECT_EQ(t.bk4, 32);
    EXPECT_EQ(t.occupancy, 1);
    EXPECT_EQ(t.max_seq_q, 0);
    EXPECT_EQ(t.block_size(GpuTarget::gfx942), 256);
}

// BF16 should produce the same tile configs as FP16
TEST(TileConfig, GFX9_BF16_D128_MatchesFP16)
{
    constexpr auto t_fp16 = getTileConfig(128, 128, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t_bf16 = getTileConfig(128, 128, DataType::BF16, GpuTarget::gfx942);

    EXPECT_EQ(t_fp16.bm0, t_bf16.bm0);
    EXPECT_EQ(t_fp16.bn0, t_bf16.bn0);
    EXPECT_EQ(t_fp16.bk0, t_bf16.bk0);
    EXPECT_EQ(t_fp16.occupancy, t_bf16.occupancy);
}

// BF16 compile-time checks for all head dims
static_assert(getTileConfig(32, 32, DataType::BF16, GpuTarget::gfx942).bn0 ==
              getTileConfig(32, 32, DataType::FP16, GpuTarget::gfx942).bn0);
static_assert(getTileConfig(64, 64, DataType::BF16, GpuTarget::gfx942).bn0 ==
              getTileConfig(64, 64, DataType::FP16, GpuTarget::gfx942).bn0);
static_assert(getTileConfig(96, 96, DataType::BF16, GpuTarget::gfx942).bn0 ==
              getTileConfig(96, 96, DataType::FP16, GpuTarget::gfx942).bn0);
static_assert(getTileConfig(256, 256, DataType::BF16, GpuTarget::gfx942).bn0 ==
              getTileConfig(256, 256, DataType::FP16, GpuTarget::gfx942).bn0);

// ============================================================================
// wavefrontSize helper (from arch_properties.hpp)
// ============================================================================

TEST(TileConfig, WavefrontSize_GFX9)
{
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx942), 64);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx90a), 64);
}

TEST(TileConfig, WavefrontSize_GFX950) { EXPECT_EQ(wavefrontSize(GpuTarget::gfx950), 64); }

TEST(TileConfig, WavefrontSize_RDNA)
{
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx1100), 32);
    EXPECT_EQ(wavefrontSize(GpuTarget::gfx1101), 32);
}

// ============================================================================
// getTileConfig: table count
// ============================================================================

TEST(TileConfig, GFX9_FP16_TableCount) { EXPECT_EQ(rocm_ck::GFX9_FP16_DQDKDV_TILES_COUNT, 5); }

// ============================================================================
// makeSpec integration: tile config flows into spec
// ============================================================================

TEST(TileConfig, MakeSpec_D32_BlockSize)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 32, .hdim_v = 32, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
    EXPECT_EQ(k.block_per_cu, 1);
}

TEST(TileConfig, MakeSpec_D64_BlockSize)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64, .hdim_v = 64, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(TileConfig, MakeSpec_D96_BlockSize)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 96, .hdim_v = 96, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(TileConfig, MakeSpec_D128_BlockSize)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(TileConfig, MakeSpec_D256_BlockSize)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 256,
                                                   .hdim_v = 256,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 64);
}

// Verify that block_per_cu auto-resolution (-1) uses tile occupancy
TEST(TileConfig, MakeSpec_BlockPerCu_AutoResolvesToOccupancy)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    // The d128 tile config has occupancy=1, so block_per_cu should be 1
    EXPECT_EQ(k.block_per_cu, 1);
}

// Verify that explicit block_per_cu overrides tile occupancy
TEST(TileConfig, MakeSpec_BlockPerCu_ExplicitOverride)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8, .block_per_cu = 2}});

    EXPECT_EQ(k.block_per_cu, 2);
}

// ============================================================================
// Consteval compile-time validation
// ============================================================================

// These are compile-time checks: getTileConfig is consteval, so if these
// compile, the lookup succeeded.
static_assert(getTileConfig(32, 32, DataType::FP16, GpuTarget::gfx942).bn0 == 128);
static_assert(getTileConfig(64, 64, DataType::FP16, GpuTarget::gfx942).bn0 == 128);
static_assert(getTileConfig(96, 96, DataType::FP16, GpuTarget::gfx942).bn0 == 128);
static_assert(getTileConfig(128, 128, DataType::FP16, GpuTarget::gfx942).bn0 == 128);
static_assert(getTileConfig(256, 256, DataType::FP16, GpuTarget::gfx942).bn0 == 64);

// Verify block_size computation
static_assert(getTileConfig(128, 128, DataType::FP16, GpuTarget::gfx942)
                  .block_size(GpuTarget::gfx942) == 256);
static_assert(getTileConfig(128, 128, DataType::BF16, GpuTarget::gfx942)
                  .block_size(GpuTarget::gfx942) == 256);

// Verify num_warps (all gfx9 fp16/bf16 configs have 4 warps)
static_assert(getTileConfig(32, 32, DataType::FP16, GpuTarget::gfx942).num_warps() == 4);
static_assert(getTileConfig(256, 256, DataType::FP16, GpuTarget::gfx942).num_warps() == 4);
