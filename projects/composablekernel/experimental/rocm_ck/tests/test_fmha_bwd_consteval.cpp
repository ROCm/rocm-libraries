// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Consteval tests for the FMHA BWD dQ/dK/dV kernel family. Two groups:
//
//   TileConfig        -- compile-time tile-geometry dispatch (getTileConfig,
//                        generateTileConfig, computeGemm4Warps). Verifies each
//                        (hdim, dtype, arch) returns the correct geometry from
//                        fmha_bwd.py and that makeSpec() integrates it. Only
//                        symmetric head dims (hdim_q == hdim_v) are supported;
//                        asymmetric rejection is covered by compile_fail/.
//
//   FmhaBwdConsteval  -- sliding-window attention (SWA) and bottom-right causal
//                        mask configs. Verifies compile-time guarantees for
//                        window_size_left/right and mask_type slot layouts, and
//                        that SWA/CMaskBR variant specs derive correctly from
//                        the underlying cmask config. The SWA/CMaskBR/CMask
//                        variants share one compiled kernel binary; mask_type
//                        and window sizes are runtime-parametrized via scalar
//                        slots, so these tests pin the slot infrastructure.

#include "rocm_fmha_bwd_registry.hpp"

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::computeGemm4Warps;
using ::rocm_ck::DataType;
using ::rocm_ck::FmhaBiasType;
using ::rocm_ck::FmhaBwdBaseTile;
using ::rocm_ck::FmhaBwdDQDKDVConfig;
using ::rocm_ck::FmhaBwdDQDKDVSpec;
using ::rocm_ck::FmhaBwdDQDKDVTileConfig;
using ::rocm_ck::FmhaMaskType;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::generateTileConfig;
using ::rocm_ck::getBaseTile;
using ::rocm_ck::getTileConfig;
using ::rocm_ck::GpuTarget;
using ::rocm_ck::hasMask;
using ::rocm_ck::makeSpec;
using ::rocm_ck::usesBatchSizeSlot;
namespace S = ::rocm_ck::fmha_bwd_dqdkdv_slots;

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

// (1,4) preferred when hdim % 64 == 0
// (2,2) when hdim % 64 != 0 but hdim % 32 == 0 and bm0 >= 32
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
    constexpr auto t = generateTileConfig(64, base);

    // Invariant: bk0 = hdim
    EXPECT_EQ(t.bk0, 64);
    // Invariant: bk1 = bm0
    EXPECT_EQ(t.bk1, 32);
    // Invariant: bk2 = hdim (== bk0 for symmetric head dims)
    EXPECT_EQ(t.bk2, 64);
    // Invariant: bk3 = bm0
    EXPECT_EQ(t.bk3, 32);
    // Symmetric: both head-dim fields take the single hdim
    EXPECT_EQ(t.hdim_q, 64);
    EXPECT_EQ(t.hdim_v, 64);
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
    constexpr auto t32 = generateTileConfig(32, b32);
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
    constexpr auto t64 = generateTileConfig(64, b64);
    EXPECT_EQ(t64.bm0, 32);
    EXPECT_EQ(t64.bn0, 128);
    EXPECT_EQ(t64.bk0, 64);
    EXPECT_EQ(t64.bk4, 32);
    EXPECT_EQ(t64.rm2, 1);
    EXPECT_EQ(t64.rn2, 4);

    // D96 symmetric
    constexpr auto t96 = generateTileConfig(96, b96);
    EXPECT_EQ(t96.bm0, 32);
    EXPECT_EQ(t96.bn0, 128);
    EXPECT_EQ(t96.bk0, 96);
    EXPECT_EQ(t96.bk2, 96);
    EXPECT_EQ(t96.rm2, 2);
    EXPECT_EQ(t96.rn2, 2);

    // D128 symmetric
    constexpr auto t128 = generateTileConfig(128, b128);
    EXPECT_EQ(t128.bm0, 16);
    EXPECT_EQ(t128.bn0, 128);
    EXPECT_EQ(t128.bk0, 128);
    EXPECT_EQ(t128.bk1, 16);
    EXPECT_EQ(t128.rm2, 1);
    EXPECT_EQ(t128.rn2, 4);

    // D256 symmetric
    constexpr auto t256 = generateTileConfig(256, b256);
    EXPECT_EQ(t256.bm0, 16);
    EXPECT_EQ(t256.bn0, 64);
    EXPECT_EQ(t256.bk0, 256);
    EXPECT_EQ(t256.bk2, 256);
    EXPECT_EQ(t256.rm2, 1);
    EXPECT_EQ(t256.rn2, 4);
}

// ============================================================================
// Asymmetric head dimensions (hdim_q != hdim_v): UNSUPPORTED
// ============================================================================
// Asymmetric configs are rejected at compile time by getTileConfig() because
// CK Tile's fmha_bwd.py defines tuned tile configs only for symmetric head
// dims. Because the rejection is a consteval throw, it cannot be exercised
// from a normal (must-compile) gtest TU; the negative coverage lives in:
//   compile_fail/dqdkdv_asymmetric_unsupported.cpp
//   compile_fail/dqdkdv_asymmetric_unsupported_reversed.cpp
// each of which must FAIL to compile (WILL_FAIL ctest property).

// Pin the scalar-slot layout for the SWA / mask infrastructure. The runtime
// argument-packing code in the .hip variants writes window sizes and mask_type
// into these exact indices; if anyone renumbers the slots, the build must fail
// here rather than silently corrupting kernel inputs.
static_assert(S::WINDOW_SIZE_LEFT == 8, "WINDOW_SIZE_LEFT slot index drifted");
static_assert(S::WINDOW_SIZE_RIGHT == 9, "WINDOW_SIZE_RIGHT slot index drifted");
static_assert(S::MASK_TYPE == 10, "MASK_TYPE slot index drifted");

// ============================================================================
// Spec equivalence: SWA / CMaskBR / CMask share compiled spec (all fields)
// ============================================================================

TEST(FmhaBwdConsteval, SWA_AllFieldsMatchCMask)
{
    // SWA, CMaskBR, and CMask produce identical compiled specs.
    // The difference is purely runtime (mask_type + window sizes).
    constexpr auto k_cmask =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask");
    constexpr auto k_swa =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_swa");

    // Every field must match -- these share a compiled kernel binary.
    EXPECT_EQ(k_swa.dtype, k_cmask.dtype);
    EXPECT_EQ(k_swa.hdim_q, k_cmask.hdim_q);
    EXPECT_EQ(k_swa.hdim_v, k_cmask.hdim_v);
    EXPECT_EQ(k_swa.mode, k_cmask.mode);
    EXPECT_EQ(k_swa.bias_type, k_cmask.bias_type);
    EXPECT_EQ(k_swa.has_bias_grad, k_cmask.has_bias_grad);
    EXPECT_EQ(hasMask(k_swa), hasMask(k_cmask));
    // mask_type itself is the runtime discriminator and intentionally differs.
    EXPECT_NE(k_swa.mask_type, k_cmask.mask_type);
    EXPECT_EQ(k_swa.has_dropout, k_cmask.has_dropout);
    EXPECT_EQ(k_swa.is_deterministic, k_cmask.is_deterministic);
    EXPECT_EQ(k_swa.pad_hdim_q, k_cmask.pad_hdim_q);
    EXPECT_EQ(k_swa.pad_hdim_v, k_cmask.pad_hdim_v);
    EXPECT_EQ(k_swa.block_per_cu, k_cmask.block_per_cu);
    EXPECT_EQ(k_swa.block_size, k_cmask.block_size);
    EXPECT_EQ(k_swa.block_n0, k_cmask.block_n0);
}

TEST(FmhaBwdConsteval, CMaskBR_AllFieldsMatchCMask)
{
    constexpr auto k_cmask =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask");
    constexpr auto k_br =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask_br");

    EXPECT_EQ(k_br.dtype, k_cmask.dtype);
    EXPECT_EQ(k_br.hdim_q, k_cmask.hdim_q);
    EXPECT_EQ(k_br.hdim_v, k_cmask.hdim_v);
    EXPECT_EQ(k_br.mode, k_cmask.mode);
    EXPECT_EQ(k_br.bias_type, k_cmask.bias_type);
    EXPECT_EQ(k_br.has_bias_grad, k_cmask.has_bias_grad);
    EXPECT_EQ(hasMask(k_br), hasMask(k_cmask));
    // mask_type itself is the runtime discriminator and intentionally differs.
    EXPECT_NE(k_br.mask_type, k_cmask.mask_type);
    EXPECT_EQ(k_br.has_dropout, k_cmask.has_dropout);
    EXPECT_EQ(k_br.is_deterministic, k_cmask.is_deterministic);
    EXPECT_EQ(k_br.pad_hdim_q, k_cmask.pad_hdim_q);
    EXPECT_EQ(k_br.pad_hdim_v, k_cmask.pad_hdim_v);
    EXPECT_EQ(k_br.block_per_cu, k_cmask.block_per_cu);
    EXPECT_EQ(k_br.block_size, k_cmask.block_size);
    EXPECT_EQ(k_br.block_n0, k_cmask.block_n0);
}

// ============================================================================
// Spec generation with mask: scalar slot requirements
// (These test the underlying makeSpec() system. The SWA/CMaskBR/CMask variants
//  all use this same spec infrastructure.)
// ============================================================================

TEST(FmhaBwdConsteval, MaskedSpec_RequiredScalars_DeterministicBatch)
{
    // Deterministic batch mode adds BATCH_SIZE slot (index 11) -> 12 scalars.
    // This dominates the mask slots (index 10) -> 11 scalars.
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_EQ(S::requiredScalars(k), 12);
}

TEST(FmhaBwdConsteval, MaskedSpec_RequiredScalars_DeterministicGroup)
{
    // Group mode does not use the BATCH_SIZE slot (kernel derives batch count
    // from seqstart pointers), so mask slots dominate -> 11 scalars.
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_EQ(S::requiredScalars(k), 11);
}

// ============================================================================
// usesBatchSizeSlot with mask combinations
// ============================================================================

TEST(FmhaBwdConsteval, MaskedSpec_UsesBatchSizeSlot_WhenDeterministicBatch)
{
    // Deterministic + batch mode -> uses BATCH_SIZE slot.
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_TRUE(usesBatchSizeSlot(k));
}

TEST(FmhaBwdConsteval, MaskedSpec_DoesNotUseBatchSizeSlot_WhenDeterministicGroup)
{
    // Deterministic + group mode -> does NOT use BATCH_SIZE slot.
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_FALSE(usesBatchSizeSlot(k));
}

TEST(FmhaBwdConsteval, MaskedSpec_DoesNotUseBatchSizeSlot_WhenNonDeterministic)
{
    // Mask without deterministic flag: batch slot is NOT required.
    // Only mask+deterministic+batch uses the batch slot.
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_FALSE(usesBatchSizeSlot(k));
}

// ============================================================================
// Tensor slot invariance: mask does not add tensor slots
// ============================================================================

TEST(FmhaBwdConsteval, MaskedSpec_RequiredTensors_UnchangedForGroup)
{
    // Group mode always requires 16 tensor slots, regardless of mask.
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(S::requiredTensors(k), 16);
}

// ============================================================================
// findVariant: mask_type disambiguates the causal family
// ============================================================================

TEST(FmhaBwdConsteval, VariantRegistry_FindReturnsBaseCMaskForMaskedQuery)
{
    // Post AE-1, findVariant() matches on mask_type, so a TOP_LEFT_CAUSAL query
    // resolves to _cmask. GENERIC / BOTTOM_RIGHT_CAUSAL resolve to _swa /
    // _cmask_br instead of aliasing onto _cmask -- see the compat suite's
    // Registry_DqDkDv_DisambiguatesMaskType.
    const auto* v = ::rocm_ck::findVariant(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_dqdkdv_fp16_d128_batch_cmask");
}

// ============================================================================
// BF16 mask spec: SWA configurations work with both dtypes
// ============================================================================

TEST(FmhaBwdConsteval, MaskedSpec_BF16_WithMask)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::BF16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.mask_type, FmhaMaskType::TOP_LEFT_CAUSAL);
    EXPECT_EQ(S::requiredScalars(k), 11);
}

TEST(FmhaBwdConsteval, MaskedSpec_BF16_WithMaskAndDeterministic)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::BF16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.mask_type, FmhaMaskType::TOP_LEFT_CAUSAL);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_EQ(S::requiredScalars(k), 12);
}
