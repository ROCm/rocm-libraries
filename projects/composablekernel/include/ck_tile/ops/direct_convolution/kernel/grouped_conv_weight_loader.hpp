// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"

namespace ck_tile {
namespace direct_conv {

// -----------------------------------------------------------------------
// WeightAccessor — register-resident filter weight buffer with 2D
// (R, S) coordinate access via CK Tile tensor descriptor.
//
// Template parameters:
//   KH, KW — filter height and width (compile-time constants).
//
// The underlying storage is a flat fp16x4_t[KH*KW] array. Each element
// holds one fp16x4_t (4 fp16 values) per filter position — the MFMA
// B-operand for a single (R, S) coordinate.
//
// Access methods use make_tensor_coordinate with a packed 2D [KH, KW]
// tensor descriptor to compute the flat index from (R, S) coordinates.
//
// WeightAccessor serves as the base class for variant-specific
// WeightLoader structs, which populate the weights[] array from LDS
// and inherit the get<R,S>() / get_transposed<R,S>() accessors.
// -----------------------------------------------------------------------
template <int KH, int KW>
struct WeightAccessor
{
    fp16x4_t weights[KH * KW];

    static constexpr auto desc_ = ck_tile::make_naive_tensor_descriptor_packed(
        ck_tile::make_tuple(ck_tile::number<KH>{}, ck_tile::number<KW>{}));

    template <int R, int S>
    __device__ __forceinline__ fp16x4_t get() const
    {
        constexpr auto coord = ck_tile::make_tensor_coordinate(
            desc_, ck_tile::make_tuple(ck_tile::number<R>{}, ck_tile::number<S>{}));
        return weights[coord.get_offset()];
    }

    template <int R, int S>
    __device__ __forceinline__ fp16x4_t get_transposed() const
    {
        return get<KH - 1 - R, KW - 1 - S>();
    }
};

// -----------------------------------------------------------------------
// WeightAccessor8 — register-resident filter weight buffer with fp16x8_t
// (8 fp16 values) per filter position. Used by 32c kernel where the MFMA
// B-operand is fp16x8_t (mfma_f32_16x16x32_f16).
// -----------------------------------------------------------------------
template <int KH, int KW>
struct WeightAccessor8
{
    fp16x8_t weights[KH * KW];

    static constexpr auto desc_ = ck_tile::make_naive_tensor_descriptor_packed(
        ck_tile::make_tuple(ck_tile::number<KH>{}, ck_tile::number<KW>{}));

    template <int R, int S>
    __device__ __forceinline__ fp16x8_t get() const
    {
        constexpr auto coord = ck_tile::make_tensor_coordinate(
            desc_, ck_tile::make_tuple(ck_tile::number<R>{}, ck_tile::number<S>{}));
        return weights[coord.get_offset()];
    }

    template <int R, int S>
    __device__ __forceinline__ fp16x8_t get_transposed() const
    {
        return get<KH - 1 - R, KW - 1 - S>();
    }
};

// Shared weight load function for grouped convolution kernels.
//
// Loads weight data from global memory into LDS using async tile operations.
// Supports multi-pass loading when the weight buffer exceeds one block_size.
//
// TC must provide:
//   TC::Weight::MakeDramReadDescriptor()
//   TC::Weight::MakeDramReadTileDistribution()
//   TC::Weight::MakeLdsWriteDescriptor()
//   TC::Weight::NUM_WEIGHT_PASSES
//   TC::GROUP_SIZE
//
// cfg must provide:
//   cfg.kh, cfg.kw, cfg.block_size()
template <typename TC, auto cfg, typename BlockCoords_>
__device__ void weight_load_to_lds(const BlockCoords_& bc,
                                   uint4* weight_lds,
                                   const _Float16* __restrict__ wei,
                                   const int c_per_group,
                                   const int k_per_group)
{
    constexpr auto weight_dram_desc = TC::Weight::MakeDramReadDescriptor();
    constexpr auto weight_dram_dist = TC::Weight::MakeDramReadTileDistribution();

    if (TC::GROUP_SIZE != c_per_group || TC::GROUP_SIZE != k_per_group)
    {
        // Padded path: c_per_group < GROUP_SIZE or k_per_group < GROUP_SIZE.
        //
        // MakeDramReadDescriptorPadded returns a 2D [WEIGHT_LDS_PADDED_UINT4, 8]
        // descriptor (same shape as MakeDramReadDescriptor) built by:
        //   1. 4D raw DRAM view: [BLOCK_GROUPS, k_per_group, KH_KW, c_per_group]
        //   2. Pad K → GROUP_SIZE, pad C → GROUP_SIZE (OOB reads as zero)
        //   3. Merge all 4 dims → flat 1D
        //   4. Unmerge → [WEIGHT_LDS_SIZE_UINT4, 8]
        //   5. Pad rows → [WEIGHT_LDS_PADDED_UINT4, 8]
        //
        // The buffer base is offset to bc.block_group so the BLOCK_GROUPS
        // dimension covers exactly this block's groups.
        const auto weight_padded_dram_desc =
            TC::Weight::template MakeDramReadDescriptorPadded<cfg.vector_size>(k_per_group, c_per_group);

        // Offset wei to bc.block_group * k_per_group * KH_KW * c_per_group.
        const _Float16* wei_block =
            wei + static_cast<size_t>(bc.block_group) * k_per_group * cfg.kh * cfg.kw * c_per_group;

        auto weight_padded_dram_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            wei_block, weight_padded_dram_desc);

        auto weight_padded_dram_window = ck_tile::make_tile_window(
            weight_padded_dram_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_size()>{}, ck_tile::number<8>{}),
            {0, 0},
            weight_dram_dist);

        constexpr auto weight_lds_desc = TC::Weight::MakeLdsWriteDescriptor();
        auto weight_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(weight_lds), weight_lds_desc);

        // LDS window needs the same distribution as the DRAM window so that
        // store_tile knows where each thread writes its register tile elements.
        auto weight_lds_window = ck_tile::make_tile_window(
            weight_lds_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_size()>{}, ck_tile::number<8>{}),
            {0, 0},
            weight_dram_dist);

        static_for<TC::Weight::NUM_WEIGHT_PASSES>(
            [&]<int Pass>()
            {
                // load_tile applies per-element OOB checking via the pad transforms,
                // correctly zeroing padded K and C positions. This is correct for any
                // c_per_group, unlike async_load_tile which issues vector loads that
                // bypass per-element OOB checks.
                auto weight_reg = ck_tile::load_tile(weight_padded_dram_window);
                ck_tile::store_tile(weight_lds_window, weight_reg);
                if constexpr(Pass < TC::Weight::NUM_WEIGHT_PASSES - 1)
                {
                    ck_tile::move_tile_window(weight_padded_dram_window, {cfg.block_size(), 0});
                    ck_tile::move_tile_window(weight_lds_window, {cfg.block_size(), 0});
                }
            });
    }
    else
    {
        // Unpadded path: c_per_group == GROUP_SIZE && k_per_group == GROUP_SIZE.
        // Use efficient async_load_tile (buffer_load_lds) with the flat 2D descriptor.
        auto weight_dram_buf = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(
            wei + static_cast<size_t>(bc.block_k) * cfg.kh * cfg.kw * TC::GROUP_SIZE,
            static_cast<ck_tile::index_t>(weight_dram_desc.get_element_space_size()));
        auto weight_dram_view =
            ck_tile::tensor_view<remove_cvref_t<decltype(weight_dram_buf)>,
                                remove_cvref_t<decltype(weight_dram_desc)>>{
                weight_dram_buf, weight_dram_desc};

        auto weight_dram_window = ck_tile::make_tile_window(
            weight_dram_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_size()>{}, ck_tile::number<8>{}),
            {0, 0},
            weight_dram_dist);

        constexpr auto weight_lds_desc = TC::Weight::MakeLdsWriteDescriptor();
        auto weight_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(weight_lds), weight_lds_desc);
        auto weight_lds_window = ck_tile::make_tile_window(
            weight_lds_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_size()>{}, ck_tile::number<8>{}),
            {0, 0});

        // Multi-pass weight loading: when the weight data is larger than
        // block_size, we need multiple async loads with advancing offsets.
        // The pad transform on the DRAM descriptor suppresses OOB reads
        // in the final pass.
        static_for<TC::Weight::NUM_WEIGHT_PASSES>(
            [&]<int Pass>()
            {
                ck_tile::async_load_tile(weight_lds_window, weight_dram_window);
                if constexpr(Pass < TC::Weight::NUM_WEIGHT_PASSES - 1)
                {
                    ck_tile::move_tile_window(weight_dram_window, {cfg.block_size(), 0});
                    ck_tile::move_tile_window(weight_lds_window, {cfg.block_size(), 0});
                }
            });
    }
}

} // namespace direct_conv
} // namespace ck_tile
