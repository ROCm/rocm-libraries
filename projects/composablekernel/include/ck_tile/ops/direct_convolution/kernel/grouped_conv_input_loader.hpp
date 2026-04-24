// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
namespace direct_conv {

// Shared InputLoader for grouped convolution kernels.
//
// Template parameters:
//   TC  — TileConstants type providing Input/Mfma descriptors and distributions.
//   cfg — Config value providing kw and other kernel parameters.
//
// TC must provide:
//   TC::Input::MakeDramDescriptor(hi, wi, C_total, px)  — px = left padding
//   TC::Input::MakeDramDistribution()
//   TC::Input::MakeLdsStoreDescriptor()
//   TC::Input::MakeLdsReadDescriptor()
//   TC::Mfma::MakeDistribution()
//   TC::TOTAL_SPATIAL, TC::BLOCK_C8, TC::BLOCK_C4, TC::BLOCK_Q
//   TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8, TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16
template <typename TC, auto cfg>
struct InputLoader
{
    // Derive the DRAM window type from the factory functions with dummy args.
    using InputDramWindowType = decltype(ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            static_cast<const _Float16*>(nullptr),
            TC::Input::MakeDramDescriptor(int{}, int{}, int{}, int{})),
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        ck_tile::multi_index<4>{},
        TC::Input::MakeDramDistribution()));

    // LDS window has no distribution — same descriptor, no distribution arg.
    using LdsWindowType = decltype(ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            static_cast<_Float16*>(nullptr),
            TC::Input::MakeLdsStoreDescriptor()),
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        ck_tile::multi_index<4>{}));

    static constexpr auto mfma_desc = TC::Input::MakeLdsReadDescriptor();
    static constexpr auto mfma_dist = TC::Mfma::MakeDistribution();

    using MfmaBuf      = ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>;
    using MfmaViewType = ck_tile::tensor_view<MfmaBuf, ck_tile::remove_cvref_t<decltype(mfma_desc)>>;

    using MfmaWindowType = decltype(ck_tile::make_tile_window(
        MfmaViewType{},
        ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0},
        mfma_dist));

    // Members
    InputDramWindowType input_dram_window;
    LdsWindowType       lds_window_0;
    LdsWindowType       lds_window_1;
    MfmaWindowType      mfma_window_0;
    MfmaWindowType      mfma_window_1;
    uint4*              input_lds_ptr;

    template <typename BlockCoords_>
    __device__ InputLoader(const BlockCoords_& bc,
                           uint4* input_lds,
                           const _Float16* __restrict__ in,
                           int hi,
                           int wi,
                           int px)
                : input_lds_ptr(input_lds)
    {
        // x_start = block_q - px  (global start column of this tile's halo)
        const auto input_dram_desc = TC::Input::MakeDramDescriptor(hi, wi, bc.C, px);
        const auto input_dram_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C + bc.block_k,
            input_dram_desc);

        // DRAM tile window with distribution. Window = [1, TOTAL_SPATIAL, BLOCK_C8, 8].
        constexpr auto input_dram_dist = TC::Input::MakeDramDistribution();
        input_dram_window = ck_tile::make_tile_window(
            input_dram_view,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {0, bc.block_q, 0, 0},
            input_dram_dist);

        // LDS tile windows for double-buffered store (no distribution needed).
        constexpr auto lds_store_desc = TC::Input::MakeLdsStoreDescriptor();
        auto lds_view_0 = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(&input_lds[0]), lds_store_desc);
        auto lds_view_1 = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(&input_lds[TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8]),
            lds_store_desc);
        lds_window_0 = ck_tile::make_tile_window(
            lds_view_0,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {0, 0, 0, 0});
        lds_window_1 = ck_tile::make_tile_window(
            lds_view_1,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {0, 0, 0, 0});

        // Create MFMA windows for the two LDS buffers, to be used for register reads.
        auto mfma_buf_0 = MfmaBuf{
            reinterpret_cast<_Float16*>(input_lds_ptr),
            static_cast<ck_tile::index_t>(TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16)};
        auto mfma_view_0 = MfmaViewType{mfma_buf_0, mfma_desc};
        mfma_window_0 = ck_tile::make_tile_window(
            mfma_view_0,
            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0},
            mfma_dist);

        auto mfma_buf_1 = MfmaBuf{
            reinterpret_cast<_Float16*>(input_lds_ptr) + TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16,
            static_cast<ck_tile::index_t>(TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16)};
        auto mfma_view_1 = MfmaViewType{mfma_buf_1, mfma_desc};
        mfma_window_1 = ck_tile::make_tile_window(
            mfma_view_1,
            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0},
            mfma_dist);
    }

    __device__ void fetch_tile_to_lds(int lds_buffer_index)
    {
        ck_tile::move_tile_window(input_dram_window, {1, 0, 0, 0});
        if(lds_buffer_index == 0)
            ck_tile::async_load_tile(lds_window_0, input_dram_window);
        else
            ck_tile::async_load_tile(lds_window_1, input_dram_window);
    }

    __device__ void prefetch_tile_to_lds(int lds_buffer_index)
    {
        if(lds_buffer_index == 0)
            ck_tile::async_load_tile(lds_window_0, input_dram_window);
        else
            ck_tile::async_load_tile(lds_window_1, input_dram_window);
    }

    // Read a given kw slice for this thread from LDS into registers.
    // Must be called for slice = 0, 1, ..., kw-1 in order.
    // Assumes the relevant tile is already loaded and synced in LDS.
    __device__ void read_from_lds(ck_tile::fp16x4_t& input_reg, int slice, int lds_buffer_index)
    {
        auto& window = (lds_buffer_index == 0) ? mfma_window_0 : mfma_window_1;
        auto tile = ck_tile::load_tile(window);
        __builtin_memcpy(&input_reg, &tile.get_thread_buffer()(ck_tile::number<0>{}), sizeof(ck_tile::fp16x4_t));
        if(slice < cfg.kw - 1)
        {
            ck_tile::move_tile_window(window, {1, 0, 0});
        }
        else
        {
            // Reset to origin so the next row starts at slice 0.
            ck_tile::move_tile_window(window, {-(cfg.kw - 1), 0, 0});
        }
    }
};

} // namespace direct_conv
} // namespace ck_tile
