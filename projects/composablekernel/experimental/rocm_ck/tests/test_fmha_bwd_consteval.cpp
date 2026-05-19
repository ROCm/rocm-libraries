// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tests for the consteval tile config generation (getTileConfig,
// generateTileConfig, computeGemm4Warps). Verifies that each
// (hdim_q, hdim_v, dtype, arch) combination returns the correct tile
// geometry from fmha_bwd.py, and that makeSpec() integrates the tile
// config correctly. Covers symmetric AND asymmetric head dimensions.

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::computeGemm4Warps;
using ::rocm_ck::DataType;
using ::rocm_ck::FmhaBiasType;
using ::rocm_ck::FmhaBwdBaseTile;
using ::rocm_ck::FmhaBwdDQDKDVConfig;
using ::rocm_ck::FmhaBwdDQDKDVTileConfig;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::generateTileConfig;
using ::rocm_ck::getBaseTile;
using ::rocm_ck::getTileConfig;
using ::rocm_ck::GpuTarget;
using ::rocm_ck::makeSpec;

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

// BF16 should produce the same tile configs as FP16
TEST(TileConfig, GFX9_BF16_D128_MatchesFP16)
{
    constexpr auto t_fp16 = getTileConfig(128, 128, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t_bf16 = getTileConfig(128, 128, DataType::BF16, GpuTarget::gfx942);

    EXPECT_EQ(t_fp16.bm0, t_bf16.bm0);
    EXPECT_EQ(t_fp16.bn0, t_bf16.bn0);
    EXPECT_EQ(t_fp16.bk0, t_bf16.bk0);
    EXPECT_EQ(t_fp16.bk1, t_bf16.bk1);
    EXPECT_EQ(t_fp16.bk2, t_bf16.bk2);
    EXPECT_EQ(t_fp16.bk3, t_bf16.bk3);
    EXPECT_EQ(t_fp16.bk4, t_bf16.bk4);
    EXPECT_EQ(t_fp16.rm2, t_bf16.rm2);
    EXPECT_EQ(t_fp16.rn2, t_bf16.rn2);
    EXPECT_EQ(t_fp16.occupancy, t_bf16.occupancy);
}

// BF16 compile-time checks for all head dims
TEST(TileConfig, GFX9_BF16_MatchesFP16_AllHeadDims)
{
    constexpr auto bf16_32  = getTileConfig(32, 32, DataType::BF16, GpuTarget::gfx942);
    constexpr auto fp16_32  = getTileConfig(32, 32, DataType::FP16, GpuTarget::gfx942);
    constexpr auto bf16_64  = getTileConfig(64, 64, DataType::BF16, GpuTarget::gfx942);
    constexpr auto fp16_64  = getTileConfig(64, 64, DataType::FP16, GpuTarget::gfx942);
    constexpr auto bf16_96  = getTileConfig(96, 96, DataType::BF16, GpuTarget::gfx942);
    constexpr auto fp16_96  = getTileConfig(96, 96, DataType::FP16, GpuTarget::gfx942);
    constexpr auto bf16_256 = getTileConfig(256, 256, DataType::BF16, GpuTarget::gfx942);
    constexpr auto fp16_256 = getTileConfig(256, 256, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(bf16_32.bn0, fp16_32.bn0);
    EXPECT_EQ(bf16_64.bn0, fp16_64.bn0);
    EXPECT_EQ(bf16_96.bn0, fp16_96.bn0);
    EXPECT_EQ(bf16_256.bn0, fp16_256.bn0);
}

// ============================================================================
// Base tile table count
// ============================================================================

TEST(TileConfig, GFX9_FP16_BaseTileCount)
{
    EXPECT_EQ(rocm_ck::GFX9_FP16_DQDKDV_BASE_TILES_COUNT, 5);
}

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
TEST(TileConfig, ConstevalSymmetricBn0)
{
    constexpr auto t32  = getTileConfig(32, 32, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t64  = getTileConfig(64, 64, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t96  = getTileConfig(96, 96, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t128 = getTileConfig(128, 128, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t256 = getTileConfig(256, 256, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(t32.bn0, 128);
    EXPECT_EQ(t64.bn0, 128);
    EXPECT_EQ(t96.bn0, 128);
    EXPECT_EQ(t128.bn0, 128);
    EXPECT_EQ(t256.bn0, 64);
}

// Verify block_size computation and num_warps
TEST(TileConfig, ConstevalBlockSizeAndWarps)
{
    constexpr auto t128_fp16 = getTileConfig(128, 128, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t128_bf16 = getTileConfig(128, 128, DataType::BF16, GpuTarget::gfx942);
    constexpr auto t32       = getTileConfig(32, 32, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t256      = getTileConfig(256, 256, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(t128_fp16.block_size(GpuTarget::gfx942), 256);
    EXPECT_EQ(t128_bf16.block_size(GpuTarget::gfx942), 256);
    EXPECT_EQ(t32.num_warps(), 4);
    EXPECT_EQ(t256.num_warps(), 4);
}

// ============================================================================
// computeGemm4Warps: GEMM4 block warp derivation
// ============================================================================

// (1,4) preferred when hdim_q % 64 == 0
// (2,2) when hdim_q % 64 != 0 but hdim_q % 32 == 0 and bm0 >= 32
TEST(TileConfig, ComputeGemm4Warps_WarpDistributions)
{
    // (1,4) cases
    constexpr auto w_32_64  = computeGemm4Warps(32, 64);
    constexpr auto w_16_128 = computeGemm4Warps(16, 128);
    constexpr auto w_16_256 = computeGemm4Warps(16, 256);
    EXPECT_EQ(w_32_64.rm2, 1);
    EXPECT_EQ(w_32_64.rn2, 4);
    EXPECT_EQ(w_16_128.rm2, 1);
    EXPECT_EQ(w_16_128.rn2, 4);
    EXPECT_EQ(w_16_256.rm2, 1);
    EXPECT_EQ(w_16_256.rn2, 4);

    // (2,2) cases
    constexpr auto w_32_32 = computeGemm4Warps(32, 32);
    constexpr auto w_32_96 = computeGemm4Warps(32, 96);
    EXPECT_EQ(w_32_32.rm2, 2);
    EXPECT_EQ(w_32_32.rn2, 2);
    EXPECT_EQ(w_32_96.rm2, 2);
    EXPECT_EQ(w_32_96.rn2, 2);
}

TEST(TileConfig, ComputeGemm4Warps_MatchesSymmetricEntries)
{
    // Verify derivation matches all hand-tuned symmetric entries
    constexpr auto w32  = computeGemm4Warps(32, 32);
    constexpr auto w64  = computeGemm4Warps(32, 64);
    constexpr auto w96  = computeGemm4Warps(32, 96);
    constexpr auto w128 = computeGemm4Warps(16, 128);
    constexpr auto w256 = computeGemm4Warps(16, 256);

    EXPECT_EQ(w32.rm2, 2);
    EXPECT_EQ(w32.rn2, 2); // hdim=32
    EXPECT_EQ(w64.rm2, 1);
    EXPECT_EQ(w64.rn2, 4); // hdim=64
    EXPECT_EQ(w96.rm2, 2);
    EXPECT_EQ(w96.rn2, 2); // hdim=96
    EXPECT_EQ(w128.rm2, 1);
    EXPECT_EQ(w128.rn2, 4); // hdim=128
    EXPECT_EQ(w256.rm2, 1);
    EXPECT_EQ(w256.rn2, 4); // hdim=256
}

// ============================================================================
// generateTileConfig: invariant derivation rules
// ============================================================================

TEST(TileConfig, GenerateTileConfig_InvariantRules)
{
    constexpr FmhaBwdBaseTile base{32, 128, 32, 1, 0};
    constexpr auto t = generateTileConfig(64, 96, base);

    // Invariant: bk0 = hdim_q
    EXPECT_EQ(t.bk0, 64);
    // Invariant: bk1 = bm0
    EXPECT_EQ(t.bk1, 32);
    // Invariant: bk2 = hdim_v
    EXPECT_EQ(t.bk2, 96);
    // Invariant: bk3 = bm0
    EXPECT_EQ(t.bk3, 32);
    // Base tile passthrough
    EXPECT_EQ(t.bm0, 32);
    EXPECT_EQ(t.bn0, 128);
    EXPECT_EQ(t.bk4, 32);
    // GFX9 fp16/bf16 constants
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
}

TEST(TileConfig, GenerateTileConfig_MatchesAllSymmetricEntries)
{
    // Verify the generation function produces identical output to the
    // original hand-tuned table for all 5 symmetric entries.
    constexpr auto b32  = getBaseTile(32, DataType::FP16, GpuTarget::gfx942);
    constexpr auto b64  = getBaseTile(64, DataType::FP16, GpuTarget::gfx942);
    constexpr auto b96  = getBaseTile(96, DataType::FP16, GpuTarget::gfx942);
    constexpr auto b128 = getBaseTile(128, DataType::FP16, GpuTarget::gfx942);
    constexpr auto b256 = getBaseTile(256, DataType::FP16, GpuTarget::gfx942);

    // D32 symmetric
    constexpr auto t32 = generateTileConfig(32, 32, b32);
    EXPECT_EQ(t32.bm0, 32);
    EXPECT_EQ(t32.bn0, 128);
    EXPECT_EQ(t32.bk0, 32);
    EXPECT_EQ(t32.bk1, 32);
    EXPECT_EQ(t32.bk2, 32);
    EXPECT_EQ(t32.bk3, 32);
    EXPECT_EQ(t32.bk4, 64);
    EXPECT_EQ(t32.rm2, 2);
    EXPECT_EQ(t32.rn2, 2);

    // D64 symmetric
    constexpr auto t64 = generateTileConfig(64, 64, b64);
    EXPECT_EQ(t64.bm0, 32);
    EXPECT_EQ(t64.bn0, 128);
    EXPECT_EQ(t64.bk0, 64);
    EXPECT_EQ(t64.bk4, 32);
    EXPECT_EQ(t64.rm2, 1);
    EXPECT_EQ(t64.rn2, 4);

    // D96 symmetric
    constexpr auto t96 = generateTileConfig(96, 96, b96);
    EXPECT_EQ(t96.bm0, 32);
    EXPECT_EQ(t96.bn0, 128);
    EXPECT_EQ(t96.bk0, 96);
    EXPECT_EQ(t96.bk2, 96);
    EXPECT_EQ(t96.rm2, 2);
    EXPECT_EQ(t96.rn2, 2);

    // D128 symmetric
    constexpr auto t128 = generateTileConfig(128, 128, b128);
    EXPECT_EQ(t128.bm0, 16);
    EXPECT_EQ(t128.bn0, 128);
    EXPECT_EQ(t128.bk0, 128);
    EXPECT_EQ(t128.bk1, 16);
    EXPECT_EQ(t128.rm2, 1);
    EXPECT_EQ(t128.rn2, 4);

    // D256 symmetric
    constexpr auto t256 = generateTileConfig(256, 256, b256);
    EXPECT_EQ(t256.bm0, 16);
    EXPECT_EQ(t256.bn0, 64);
    EXPECT_EQ(t256.bk0, 256);
    EXPECT_EQ(t256.bk2, 256);
    EXPECT_EQ(t256.rm2, 1);
    EXPECT_EQ(t256.rn2, 4);
}

// ============================================================================
// getTileConfig: asymmetric head dimensions
// ============================================================================

// --- Same-tier asymmetric (Tier 1: bm0=32, hdim ∈ {32,64,96}) ---

TEST(TileConfig, Asymmetric_D64_D32)
{
    constexpr auto t = getTileConfig(64, 32, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.hdim_q, 64);
    EXPECT_EQ(t.hdim_v, 32);
    EXPECT_EQ(t.bm0, 32); // from max(64,32)=64 base tile
    EXPECT_EQ(t.bn0, 128);
    EXPECT_EQ(t.bk0, 64); // = hdim_q
    EXPECT_EQ(t.bk2, 32); // = hdim_v
    EXPECT_EQ(t.bk1, 32); // = bm0
    EXPECT_EQ(t.bk3, 32); // = bm0
    EXPECT_EQ(t.rm2, 1);  // hdim_q=64 → (1,4)
    EXPECT_EQ(t.rn2, 4);
    EXPECT_EQ(t.num_warps(), 4);
}

TEST(TileConfig, Asymmetric_D32_D64)
{
    constexpr auto t = getTileConfig(32, 64, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.hdim_q, 32);
    EXPECT_EQ(t.hdim_v, 64);
    EXPECT_EQ(t.bm0, 32); // from max(32,64)=64 base tile
    EXPECT_EQ(t.bk0, 32); // = hdim_q
    EXPECT_EQ(t.bk2, 64); // = hdim_v
    EXPECT_EQ(t.rm2, 2);  // hdim_q=32 → (2,2)
    EXPECT_EQ(t.rn2, 2);
}

TEST(TileConfig, Asymmetric_D96_D32)
{
    constexpr auto t = getTileConfig(96, 32, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 32);
    EXPECT_EQ(t.bk0, 96);
    EXPECT_EQ(t.bk2, 32);
    EXPECT_EQ(t.rm2, 2); // hdim_q=96 → (2,2)
    EXPECT_EQ(t.rn2, 2);
}

TEST(TileConfig, Asymmetric_D32_D96)
{
    constexpr auto t = getTileConfig(32, 96, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 32);
    EXPECT_EQ(t.bk0, 32);
    EXPECT_EQ(t.bk2, 96);
    EXPECT_EQ(t.rm2, 2); // hdim_q=32 → (2,2)
    EXPECT_EQ(t.rn2, 2);
}

TEST(TileConfig, Asymmetric_D64_D96)
{
    constexpr auto t = getTileConfig(64, 96, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 32); // from max=96 base tile
    EXPECT_EQ(t.bk0, 64);
    EXPECT_EQ(t.bk2, 96);
    EXPECT_EQ(t.rm2, 1); // hdim_q=64 → (1,4)
    EXPECT_EQ(t.rn2, 4);
}

TEST(TileConfig, Asymmetric_D96_D64)
{
    constexpr auto t = getTileConfig(96, 64, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 32);
    EXPECT_EQ(t.bk0, 96);
    EXPECT_EQ(t.bk2, 64);
    EXPECT_EQ(t.rm2, 2); // hdim_q=96 → (2,2)
    EXPECT_EQ(t.rn2, 2);
}

// --- Same-tier asymmetric (Tier 2: bm0=16, hdim ∈ {128,256}) ---

TEST(TileConfig, Asymmetric_D128_D256)
{
    constexpr auto t = getTileConfig(128, 256, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16); // from max=256 base tile
    EXPECT_EQ(t.bn0, 64);
    EXPECT_EQ(t.bk0, 128);
    EXPECT_EQ(t.bk2, 256);
    EXPECT_EQ(t.rm2, 1);
    EXPECT_EQ(t.rn2, 4);
}

TEST(TileConfig, Asymmetric_D256_D128)
{
    constexpr auto t = getTileConfig(256, 128, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16);
    EXPECT_EQ(t.bn0, 64);
    EXPECT_EQ(t.bk0, 256);
    EXPECT_EQ(t.bk2, 128);
    EXPECT_EQ(t.rm2, 1);
    EXPECT_EQ(t.rn2, 4);
}

// --- Cross-tier asymmetric (valid: hdim_q in higher tier) ---

TEST(TileConfig, Asymmetric_D128_D32)
{
    constexpr auto t = getTileConfig(128, 32, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16); // from max=128 base tile
    EXPECT_EQ(t.bn0, 128);
    EXPECT_EQ(t.bk0, 128);
    EXPECT_EQ(t.bk2, 32);
    EXPECT_EQ(t.rm2, 1);
    EXPECT_EQ(t.rn2, 4);
}

TEST(TileConfig, Asymmetric_D128_D64)
{
    constexpr auto t = getTileConfig(128, 64, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16);
    EXPECT_EQ(t.bk0, 128);
    EXPECT_EQ(t.bk2, 64);
    EXPECT_EQ(t.rm2, 1);
    EXPECT_EQ(t.rn2, 4);
}

TEST(TileConfig, Asymmetric_D128_D96)
{
    constexpr auto t = getTileConfig(128, 96, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16);
    EXPECT_EQ(t.bk0, 128);
    EXPECT_EQ(t.bk2, 96);
    EXPECT_EQ(t.rm2, 1);
    EXPECT_EQ(t.rn2, 4);
}

TEST(TileConfig, Asymmetric_D64_D128)
{
    constexpr auto t = getTileConfig(64, 128, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16); // from max=128 base tile
    EXPECT_EQ(t.bk0, 64);
    EXPECT_EQ(t.bk2, 128);
    EXPECT_EQ(t.rm2, 1); // hdim_q=64 → (1,4)
    EXPECT_EQ(t.rn2, 4);
}

TEST(TileConfig, Asymmetric_D256_D32)
{
    constexpr auto t = getTileConfig(256, 32, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16); // from max=256 base tile
    EXPECT_EQ(t.bn0, 64);
    EXPECT_EQ(t.bk0, 256);
    EXPECT_EQ(t.bk2, 32);
}

TEST(TileConfig, Asymmetric_D256_D64)
{
    constexpr auto t = getTileConfig(256, 64, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16); // from max=256 base tile
    EXPECT_EQ(t.bn0, 64);
    EXPECT_EQ(t.bk0, 256); // = hdim_q
    EXPECT_EQ(t.bk2, 64);  // = hdim_v
    EXPECT_EQ(t.bk1, 16);  // = bm0
    EXPECT_EQ(t.bk3, 16);  // = bm0
    EXPECT_EQ(t.rm2, 1);   // hdim_q=256 → (1,4)
    EXPECT_EQ(t.rn2, 4);
}

TEST(TileConfig, Asymmetric_D256_D96)
{
    constexpr auto t = getTileConfig(256, 96, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16); // from max=256 base tile
    EXPECT_EQ(t.bn0, 64);
    EXPECT_EQ(t.bk0, 256); // = hdim_q
    EXPECT_EQ(t.bk2, 96);  // = hdim_v
    EXPECT_EQ(t.bk1, 16);  // = bm0
    EXPECT_EQ(t.bk3, 16);  // = bm0
    EXPECT_EQ(t.rm2, 1);   // hdim_q=256 → (1,4)
    EXPECT_EQ(t.rn2, 4);
}

TEST(TileConfig, Asymmetric_D64_D256)
{
    constexpr auto t = getTileConfig(64, 256, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t.bm0, 16); // from max=256 base tile
    EXPECT_EQ(t.bn0, 64);
    EXPECT_EQ(t.bk0, 64);
    EXPECT_EQ(t.bk2, 256);
    EXPECT_EQ(t.rm2, 1); // hdim_q=64 → (1,4)
    EXPECT_EQ(t.rn2, 4);
}

// --- Cross-tier: BF16 asymmetric works identically ---

TEST(TileConfig, Asymmetric_BF16_D128_D64)
{
    constexpr auto fp16 = getTileConfig(128, 64, DataType::FP16, GpuTarget::gfx942);
    constexpr auto bf16 = getTileConfig(128, 64, DataType::BF16, GpuTarget::gfx942);
    EXPECT_EQ(fp16.bm0, bf16.bm0);
    EXPECT_EQ(fp16.bn0, bf16.bn0);
    EXPECT_EQ(fp16.bk0, bf16.bk0);
    EXPECT_EQ(fp16.bk2, bf16.bk2);
    EXPECT_EQ(fp16.rm2, bf16.rm2);
    EXPECT_EQ(fp16.rn2, bf16.rn2);
}

// --- Asymmetric configs that are NOT supported ---
// These fail at compile time because no valid GEMM4 warp distribution exists.
// When bm0=16 (forced by large hdim_v) and hdim_q ∈ {32,96} (not divisible
// by 64, and bm0 too small for (2,2)), no (rm2,rn2) satisfies the constraints.
//
// Unsupported: (32,128), (96,128), (32,256), (96,256)
//
// These are NOT tested here because they are compile errors (consteval throw).
// See compile_fail/ for negative tests if needed.

// --- Compile-time validation of asymmetric configs ---

// Asymmetric configs compile and have correct invariant relationships
TEST(TileConfig, Asymmetric_CompileTimeInvariants)
{
    constexpr auto t64_32  = getTileConfig(64, 32, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t128_64 = getTileConfig(128, 64, DataType::FP16, GpuTarget::gfx942);
    constexpr auto t256_32 = getTileConfig(256, 32, DataType::FP16, GpuTarget::gfx942);

    EXPECT_EQ(t64_32.bk0, 64);
    EXPECT_EQ(t64_32.bk2, 32);
    EXPECT_EQ(t128_64.bk0, 128);
    EXPECT_EQ(t128_64.bk2, 64);
    EXPECT_EQ(t256_32.bk0, 256);
    EXPECT_EQ(t256_32.bk2, 32);

    // All asymmetric configs have 4 warps
    EXPECT_EQ(t64_32.num_warps(), 4);
    EXPECT_EQ(t128_64.num_warps(), 4);

    constexpr auto t64_256 = getTileConfig(64, 256, DataType::FP16, GpuTarget::gfx942);
    EXPECT_EQ(t64_256.num_warps(), 4);
}

// ============================================================================
// makeSpec integration: asymmetric configs
// ============================================================================

TEST(TileConfig, MakeSpec_Asymmetric_D128_D64)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 64,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 64);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
    EXPECT_EQ(k.block_per_cu, 1);
}

TEST(TileConfig, MakeSpec_Asymmetric_D64_D32)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64, .hdim_v = 32, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.hdim_q, 64);
    EXPECT_EQ(k.hdim_v, 32);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}
