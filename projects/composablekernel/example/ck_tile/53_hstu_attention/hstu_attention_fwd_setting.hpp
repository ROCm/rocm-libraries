// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_tile_setting_define.hpp"
#include "hstu_attention_util.hpp"

using WarpTile_16x16x16 = ck_tile::sequence<16, 16, 16>;
using WarpTile_16x16x32 = ck_tile::sequence<16, 16, 32>;
using WarpTile_32x32x16 = ck_tile::sequence<32, 32, 16>;

#if !defined(BUILD_HSTU_FOR_GFX95_ONLY)
template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionNoSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N0Sub N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
// bespoke_idx5_round (B=120,H=4,N=16384,D=64 inference): the d=64 no-softmax
// kernels are VGPR-bound at 1 wave/SIMD (VGPR 167-192 > 128 of 256/SIMD, AGPR=0;
// ISA metadata) and the op is compute/reuse-bound, NOT HBM- or occupancy-bound
// (the grid oversubscribes CUs ~95x; MemUnitBusy~0%). Shrinking the M-tile to
// raise occupancy (kM0=64 -> 2 waves/SIMD) was MEASURED +27% SLOWER: it halves
// K/V operand reuse and doubles streaming/loop overhead. The win is the opposite
// lever: a LARGER kM0=192 (MTile==64 slot) -> 1.5x more M-rows reuse each K/V
// load -> fewer M-blocks -> less LDS streaming. VGPR lands at 255 (private_seg=0,
// no spill), still 1 wave/SIMD; kM0=256 would spill. Measured 24.04 ms vs 26.4-
// 27.5 for kM0=128 (-10/-12%), bit-exact to the kM0=128 reference. MTile==128
// keeps the original kM0=128 geometry byte-identical.
template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdBlockTile<64, MTile>
{
    // MTile==64 slot repurposed as the bespoke kM0=192 max-reuse tile.
    using type        = std::conditional_t<MTile == 64,
                                     ck_tile::sequence<192, 64, 32, 64, 32, 64>,
                                     ck_tile::sequence<128, 64, 32, 64, 32, 64>>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdBlockTile<96, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 128, 32, 96>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<128, 64>
{
    using type        = ck_tile::sequence<64, 32, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<128, 128>
{
    using type        = ck_tile::sequence<128, 32, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdBlockTile<256, MTile>
{
    using type        = ck_tile::sequence<128, 32, 16, 256, 16, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionWithSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N0Sub N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdBlockTile<64, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdBlockTile<96, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 128, 32, 96>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<128, 64>
{
    using type        = ck_tile::sequence<64, 64, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<128, 128>
{
    using type        = ck_tile::sequence<128, 64, 16, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdBlockTile<256, MTile>
{
    using type        = ck_tile::sequence<128, 32, 16, 256, 16, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionNoSoftmaxFwdTileSetting;

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdTileSetting<64, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::gemm1_warps,
        WarpTile_16x16x16>;
};

template struct HstuAttentionNoSoftmaxFwdTileSetting<64, 64>;
template struct HstuAttentionNoSoftmaxFwdTileSetting<64, 128>;

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdTileSetting<96, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::gemm1_warps,
        WarpTile_16x16x16>;
};

template struct HstuAttentionNoSoftmaxFwdTileSetting<96, 64>;
template struct HstuAttentionNoSoftmaxFwdTileSetting<96, 128>;

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<128, 64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<128, 128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdTileSetting<256, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm1_warps,
        WarpTile_16x16x16>;
};

template struct HstuAttentionNoSoftmaxFwdTileSetting<256, 64>;
template struct HstuAttentionNoSoftmaxFwdTileSetting<256, 128>;

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionWithSoftmaxFwdTileSetting;

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdTileSetting<64, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::gemm0_warps,
        WarpTile_32x32x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::gemm1_warps,
        WarpTile_32x32x16>;
};

template struct HstuAttentionWithSoftmaxFwdTileSetting<64, 64>;
template struct HstuAttentionWithSoftmaxFwdTileSetting<64, 128>;

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdTileSetting<96, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::gemm0_warps,
        WarpTile_32x32x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::gemm1_warps,
        WarpTile_32x32x16>;
};

template struct HstuAttentionWithSoftmaxFwdTileSetting<96, 64>;
template struct HstuAttentionWithSoftmaxFwdTileSetting<96, 128>;

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<128, 64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<128, 128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::gemm1_warps,
        WarpTile_16x16x16>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdTileSetting<256, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm0_warps,
        WarpTile_16x16x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm1_warps,
        WarpTile_16x16x16>;
};

template struct HstuAttentionWithSoftmaxFwdTileSetting<256, 64>;
template struct HstuAttentionWithSoftmaxFwdTileSetting<256, 128>;
#endif

#if defined(BUILD_HSTU_FOR_GFX95_ONLY)
template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionNoSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N0Sub N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdBlockTile<64, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdBlockTile<96, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 128, 32, 96>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<128, 64>
{
    using type        = ck_tile::sequence<64, 32, 32, 128, 32, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionNoSoftmaxFwdBlockTile<128, 128>
{
    using type        = ck_tile::sequence<128, 32, 32, 128, 32, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdBlockTile<256, MTile>
{
    using type        = ck_tile::sequence<128, 32, 32, 256, 32, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK, ck_tile::index_t = 0>
struct HstuAttentionWithSoftmaxFwdBlockTile;

// Tile-sizes: M N0 N0Sub N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdBlockTile<64, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdBlockTile<96, MTile>
{
    using type        = ck_tile::sequence<128, 64, 32, 128, 32, 96>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<128, 64>
{
    using type        = ck_tile::sequence<64, 64, 32, 128, 32, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct HstuAttentionWithSoftmaxFwdBlockTile<128, 128>
{
    using type        = ck_tile::sequence<128, 64, 32, 128, 16, 128>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdBlockTile<256, MTile>
{
    using type        = ck_tile::sequence<128, 64, 16, 256, 32, 256>;
    using gemm0_warps = ck_tile::sequence<4, 1, 1>;
    using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionNoSoftmaxFwdTileSetting;

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdTileSetting<64, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::gemm0_warps,
        WarpTile_16x16x32,
        typename HstuAttentionNoSoftmaxFwdBlockTile<64>::gemm1_warps,
        WarpTile_16x16x32>;
};

template struct HstuAttentionNoSoftmaxFwdTileSetting<64, 64>;
template struct HstuAttentionNoSoftmaxFwdTileSetting<64, 128>;

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdTileSetting<96, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::gemm0_warps,
        WarpTile_16x16x32,
        typename HstuAttentionNoSoftmaxFwdBlockTile<96>::gemm1_warps,
        WarpTile_16x16x32>;
};

template struct HstuAttentionNoSoftmaxFwdTileSetting<96, 64>;
template struct HstuAttentionNoSoftmaxFwdTileSetting<96, 128>;

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<128, 64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::gemm0_warps,
        WarpTile_16x16x32,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 64>::gemm1_warps,
        WarpTile_16x16x32>;
};

template <>
struct HstuAttentionNoSoftmaxFwdTileSetting<128, 128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::gemm0_warps,
        WarpTile_16x16x32,
        typename HstuAttentionNoSoftmaxFwdBlockTile<128, 128>::gemm1_warps,
        WarpTile_16x16x32>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionNoSoftmaxFwdTileSetting<256, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm0_warps,
        WarpTile_16x16x32,
        typename HstuAttentionNoSoftmaxFwdBlockTile<256>::gemm1_warps,
        WarpTile_16x16x32>;
};

template struct HstuAttentionNoSoftmaxFwdTileSetting<256, 64>;
template struct HstuAttentionNoSoftmaxFwdTileSetting<256, 128>;

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct HstuAttentionWithSoftmaxFwdTileSetting;

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdTileSetting<64, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::gemm0_warps,
        WarpTile_32x32x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<64>::gemm1_warps,
        WarpTile_32x32x16>;
};

template struct HstuAttentionWithSoftmaxFwdTileSetting<64, 64>;
template struct HstuAttentionWithSoftmaxFwdTileSetting<64, 128>;

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdTileSetting<96, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::gemm0_warps,
        WarpTile_32x32x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<96>::gemm1_warps,
        WarpTile_32x32x16>;
};

template struct HstuAttentionWithSoftmaxFwdTileSetting<96, 64>;
template struct HstuAttentionWithSoftmaxFwdTileSetting<96, 128>;

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<128, 64>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::gemm0_warps,
        WarpTile_16x16x32,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 64>::gemm1_warps,
        WarpTile_16x16x32>;
};

template <>
struct HstuAttentionWithSoftmaxFwdTileSetting<128, 128>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::gemm0_warps,
        WarpTile_32x32x16,
        typename HstuAttentionWithSoftmaxFwdBlockTile<128, 128>::gemm1_warps,
        WarpTile_32x32x16>;
};

template <ck_tile::index_t MTile>
struct HstuAttentionWithSoftmaxFwdTileSetting<256, MTile>
{
    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::type,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm0_warps,
        WarpTile_16x16x32,
        typename HstuAttentionWithSoftmaxFwdBlockTile<256>::gemm1_warps,
        WarpTile_16x16x32>;
};

template struct HstuAttentionWithSoftmaxFwdTileSetting<256, 64>;
template struct HstuAttentionWithSoftmaxFwdTileSetting<256, 128>;

#endif

// Parametric no-softmax tile setting that lets the dispatcher select the
// warp-K dim of the 16x16x{K} bf16 MFMA family at JIT time. WarpK=16 maps
// to v_mfma_f32_16x16x16_bf16 (the original default); WarpK=32 maps to
// v_mfma_f32_16x16x32_bf16 (~1.6x more useful ops/cycle on bf16 for
// gfx94x/gfx95x).  Only safe when BlockTile::kK1 >= WarpK; the deployment
// shape (max_k=64) has kK1=32, so the static_assert below is satisfied.
template <ck_tile::index_t WarpK>
struct HstuChooseWarpTile_16x16
{
    static_assert(WarpK == 16 || WarpK == 32, "WarpK must be 16 or 32");
    using type = std::conditional_t<WarpK == 32, WarpTile_16x16x32, WarpTile_16x16x16>;
};

template <ck_tile::index_t MaxK, ck_tile::index_t MTile, ck_tile::index_t WarpK>
struct HstuAttentionNoSoftmaxFwdTileSettingW
{
    using BlockTileT = HstuAttentionNoSoftmaxFwdBlockTile<MaxK, MTile>;
    static_assert(BlockTileT::type::at(ck_tile::number<4>{}) >= WarpK,
                  "BlockTile::kK1 must be >= WarpK for the warp tile to fit one MFMA");

    using Type = ck_tile::HstuAttentionFwdTileSettingClass<
        typename BlockTileT::type,
        typename BlockTileT::gemm0_warps,
        typename HstuChooseWarpTile_16x16<WarpK>::type,
        typename BlockTileT::gemm1_warps,
        typename HstuChooseWarpTile_16x16<WarpK>::type>;
};

static int get_hstu_attention_fwd_mtile(int num_batches, int num_heads, int max_seqlen_q)
{
    int num_CUs  = get_number_of_cu();
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    if(max_seqlen_q <= 64)
        return 64;

    int nbatch_nhead_mblocks = num_batches * num_heads * ceildiv(max_seqlen_q, 128);

    // assuming each CU is assigned two work-groups
    if(nbatch_nhead_mblocks >= static_cast<int>(0.85f * num_CUs * 2.0f))
        return 128;

    // currently, only hdim-128 actually uses mtile-64, for other hdim, the settings for
    // mtile-64 can be added through tuning/verification
    return 64;
};

static float get_estimated_cu_coverage_ratio(int num_batches, int num_heads, int max_seqlen_q)
{
    int num_CUs  = get_number_of_cu();
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    int nbatch_nhead_mblocks = num_batches * num_heads * ceildiv(max_seqlen_q, 64);

    // assume each CU can run two work-groups, common cases for hdim128
    return static_cast<float>(nbatch_nhead_mblocks) / (2.0f * num_CUs);
};

static bool shall_use_splitkv(int num_batches, int num_heads, int max_seqlen_q)
{
    // Please tune the threshold here
    const float threshold = 0.8f;

    if(get_estimated_cu_coverage_ratio(num_batches, num_heads, max_seqlen_q) < threshold)
        return true;
    return false;
};

static int get_suggested_num_splits(int num_batches, int num_heads, int max_seqlen_q)
{
    int i = 2;

    // Please tune the threshold here
    const float threshold = 1.5f;
    while(get_estimated_cu_coverage_ratio(num_batches, num_heads, max_seqlen_q) * i < threshold)
        i++;

    // the num_splits shall not be bigger than 64
    return ck_tile::min(i, 64);
};
