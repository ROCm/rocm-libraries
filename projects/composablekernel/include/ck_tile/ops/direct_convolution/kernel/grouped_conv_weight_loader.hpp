// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"

namespace ck_tile {
namespace direct_conv {

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
                                   const _Float16* __restrict__ wei)
{
    constexpr auto weight_dram_desc = TC::Weight::MakeDramReadDescriptor();
    auto weight_dram_buf = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(
        wei + static_cast<size_t>(bc.block_k) * cfg.kh * cfg.kw * TC::GROUP_SIZE,
        static_cast<ck_tile::index_t>(weight_dram_desc.get_element_space_size()));
    auto weight_dram_view =
        ck_tile::tensor_view<remove_cvref_t<decltype(weight_dram_buf)>,
                            remove_cvref_t<decltype(weight_dram_desc)>>{
            weight_dram_buf, weight_dram_desc};

    constexpr auto weight_dram_dist = TC::Weight::MakeDramReadTileDistribution();
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

} // namespace direct_conv
} // namespace ck_tile
