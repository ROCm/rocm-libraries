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

    if (TC::GROUP_SIZE != c_per_group || TC::GROUP_SIZE != k_per_group)
    {
        // Padded path: c_per_group < GROUP_SIZE or k_per_group < GROUP_SIZE.
        //
        // We cannot use async_load_tile (buffer_load_lds) here because it loads
        // 8 contiguous fp16 elements per thread. For GROUP_SIZE=4, each row of 8
        // fp16 spans two (k, yx) positions and the pad transform can't zero-pad
        // individual elements within a contiguous 16-byte load.
        //
        // Instead, use a cooperative scalar loop: each thread processes a subset
        // of the padded LDS buffer, reading real weight data from DRAM for valid
        // (k, c) positions and writing zero for padded positions.
        //
        // LDS layout (matches the unpadded path): flat contiguous buffer of
        //   BLOCK_GROUPS * GROUP_SIZE * KH_KW * GROUP_SIZE fp16 elements,
        // indexed as [g, k, yx, c] with strides [GROUP_SIZE*KH_KW*GROUP_SIZE,
        //   KH_KW*GROUP_SIZE, GROUP_SIZE, 1].
        //
        // DRAM layout (real tensor): [groups, k_per_group, KH_KW, c_per_group]
        // with strides [k_per_group*KH_KW*c_per_group, KH_KW*c_per_group,
        //   c_per_group, 1].

        constexpr int GROUP_SZ = TC::GROUP_SIZE;
        constexpr int KH_KW = cfg.kh * cfg.kw;
        constexpr int BLOCK_GRP = cfg.block_groups();
        constexpr int TOTAL_PADDED_ELEMS = BLOCK_GRP * GROUP_SZ * KH_KW * GROUP_SZ;

        // Base pointer for this block's weight slice in DRAM.
        // block_group = blockIdx.y * block_groups() selects which group slice.
        // In the real tensor, group g starts at offset g * k_per_group * KH_KW * c_per_group.
        const _Float16* wei_base = wei + static_cast<size_t>(bc.block_group) * k_per_group * KH_KW * c_per_group;

        const int dram_k_stride  = KH_KW * c_per_group;
        const int dram_g_stride  = k_per_group * dram_k_stride;

        _Float16* lds_fp16 = reinterpret_cast<_Float16*>(weight_lds);

        const int tid = static_cast<int>(threadIdx.x);
        for(int i = tid; i < TOTAL_PADDED_ELEMS; i += cfg.block_size())
        {
            // Decompose flat LDS index into 4D coordinates [g, k, yx, c].
            const int c  = i % GROUP_SZ;
            const int yx = (i / GROUP_SZ) % KH_KW;
            const int k  = (i / (GROUP_SZ * KH_KW)) % GROUP_SZ;
            const int g  = i / (GROUP_SZ * KH_KW * GROUP_SZ);

            _Float16 val;
            if(k < k_per_group && c < c_per_group)
            {
                // Valid position: read from DRAM.
                int dram_offset = g * dram_g_stride + k * dram_k_stride + yx * c_per_group + c;
                val = wei_base[dram_offset];
            }
            else
            {
                // Padded position: write zero.
                val = static_cast<_Float16>(0.0f);
            }
            lds_fp16[i] = val;
        }
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
}

} // namespace direct_conv
} // namespace ck_tile
