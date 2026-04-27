// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
namespace direct_conv {

// Shared OutputWriter for grouped convolution kernels.
// Writes MFMA accumulator results directly to global memory.
//
// Uses precomputed scalar state instead of persistent tile_window objects
// to minimize register pressure and spill traffic.
//
// TC must provide:
//   TC::Output::MakeDramDistribution()
//   TC::Output::MakeDramDescriptor(ho, wo, C)
//   TC::BLOCK_Q, TC::BLOCK_C4
template <typename TC>
struct OutputWriter
{
    // Type aliases for temporary tile_window construction.
    using OutputDramDesc =
        ck_tile::remove_cvref_t<decltype(TC::Output::MakeDramDescriptor(int{}, int{}, int{}))>;
    using OutputDramBuf =
        ck_tile::buffer_view<ck_tile::address_space_enum::global, _Float16, ck_tile::index_t, true>;
    using OutputDramView = ck_tile::tensor_view<OutputDramBuf, OutputDramDesc>;
    using OutputDramWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputDramView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<TC::BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        TC::Output::MakeDramDistribution()))>;

    // Persistent members — scalar state only.
    _Float16*         output_base;         // base output pointer for this block
    ck_tile::index_t  output_elem_offset;  // per-thread element offset (within-tile spatial+channel)
    ck_tile::index_t  row_stride_elems;    // elements per output row (wo * C)
    bool              store_valid;         // whether this thread's output position is in bounds

    template <typename BlockCoords_>
    __device__ OutputWriter(const BlockCoords_& bc,
                            uint4*, // Unused, matches OutputWriterLds constructor signature.
                            _Float16* __restrict__ out,
                            int ho,
                            int wo)
    {
        output_base = out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C + bc.block_k;
        row_stride_elems = wo * bc.C;

        // Create temporary DRAM tile_window to extract per-thread offset and validity.
        {
            constexpr auto out_dist = TC::Output::MakeDramDistribution();
            const auto out_desc = TC::Output::MakeDramDescriptor(ho, wo, bc.C);
            auto out_buf = OutputDramBuf{
                output_base,
                static_cast<ck_tile::index_t>(out_desc.get_element_space_size())};
            auto out_view = OutputDramView{out_buf, out_desc};

            auto tmp_window = ck_tile::make_tile_window(
                out_view,
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<TC::BLOCK_Q>{},
                                    ck_tile::number<TC::BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                {0, bc.block_q, 0, 0},
                out_dist);

            // Extract per-thread output element offset (encodes spatial + channel position).
            output_elem_offset = tmp_window.pre_computed_coords_[ck_tile::number<0>{}]
                                                                [ck_tile::number<1>{}].get_offset();

            // Check validity via the pad transform's coordinate-based check.
            // The element_space_size check is insufficient because OOB threads
            // in the spatial dimension can still have offsets within the multi-row
            // buffer range.
            store_valid = ck_tile::coordinate_has_valid_offset_assuming_top_index_is_valid(
                out_desc, tmp_window.pre_computed_coords_[ck_tile::number<0>{}]
                                                        [ck_tile::number<1>{}]);
        } // tmp_window goes out of scope — no persistent register cost
    }

    // Convert fp32x4 accumulator to fp16x4 and write directly to global memory.
    __device__ void flush(fp32x4_t acc_val, int p_out)
    {
        // 1. Convert fp32→fp16.
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

        // 2. Direct store to DRAM: base + row offset + per-thread offset.
        if(store_valid)
        {
            ck_tile::index_t store_offset = output_elem_offset
                + static_cast<ck_tile::index_t>(p_out) * row_stride_elems;
            __builtin_memcpy(output_base + store_offset, &out_reg, sizeof(out_reg));
        }
    }
};

// Shared OutputWriterLds for grouped convolution kernels.
// Stages output through LDS before writing to global memory.
//
// Uses precomputed scalar state instead of persistent tile_window objects
// to minimize register pressure and spill traffic.
//
// TC must provide:
//   TC::Mfma::MakeDistribution()
//   TC::Output::MakeDramDistribution()
//   TC::Output::MakeLdsWriteDescriptor()
//   TC::Output::MakeLdsReadDescriptor()
//   TC::Output::MakeDramDescriptor(ho, wo, C)
//   TC::Output::OUTPUT_LDS_BUFFER_SIZE
//   TC::Weight::WEIGHT_LDS_PADDED_UINT4
//   TC::BLOCK_Q, TC::BLOCK_C4
template <typename TC>
struct OutputWriterLds
{
    // Type aliases for temporary tile_window construction.
    static constexpr auto OutputLdsDist  = TC::Mfma::MakeDistribution();
    static constexpr auto OutputDramDist = TC::Output::MakeDramDistribution();

    using OutputLdsBuf = ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>;

    using OutputLdsWriteDesc   = ck_tile::remove_cvref_t<decltype(TC::Output::MakeLdsWriteDescriptor())>;
    using OutputLdsWriteView   = ck_tile::tensor_view<OutputLdsBuf, OutputLdsWriteDesc>;
    using OutputLdsWriteWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputLdsWriteView{},
        ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0},
        OutputLdsDist))>;

    using OutputLdsReadDesc   = ck_tile::remove_cvref_t<decltype(TC::Output::MakeLdsReadDescriptor())>;
    using OutputLdsReadView   = ck_tile::tensor_view<OutputLdsBuf, OutputLdsReadDesc>;
    using OutputLdsReadWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputLdsReadView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<TC::BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        OutputDramDist))>;

    using OutputDramDesc =
        ck_tile::remove_cvref_t<decltype(TC::Output::MakeDramDescriptor(int{}, int{}, int{}))>;
    using OutputDramBuf =
        ck_tile::buffer_view<ck_tile::address_space_enum::global, _Float16, ck_tile::index_t, true>;
    using OutputDramView = ck_tile::tensor_view<OutputDramBuf, OutputDramDesc>;
    using OutputDramWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputDramView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<TC::BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        OutputDramDist))>;

    // Persistent members — scalar state only.
    _Float16*         output_base;         // base output pointer for this block
    _Float16*         lds_base;            // LDS buffer base pointer
    ck_tile::index_t  lds_write_offset;    // per-thread LDS write element offset (MFMA distribution)
    ck_tile::index_t  lds_read_offset;     // per-thread LDS read element offset (DRAM distribution)
    ck_tile::index_t  output_elem_offset;  // per-thread output DRAM element offset
    ck_tile::index_t  row_stride_elems;    // wo * C elements per output row
    ck_tile::index_t  lds_buf_size;        // LDS buffer size in fp16 elements
    bool              store_valid;         // whether this thread's output position is in bounds

    template <typename BlockCoords_>
    __device__ OutputWriterLds(const BlockCoords_& bc,
                               uint4* output_lds,
                               _Float16* __restrict__ out,
                               int ho,
                               int wo)
    {
        output_base = out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C + bc.block_k;
        lds_base = reinterpret_cast<_Float16*>(output_lds);
        row_stride_elems = wo * bc.C;
        lds_buf_size = static_cast<ck_tile::index_t>(
            ck_tile::max(TC::Weight::WEIGHT_LDS_PADDED_UINT4, TC::Output::OUTPUT_LDS_BUFFER_SIZE) *
            (sizeof(uint4) / sizeof(_Float16)));

        // Create temporary tile_windows to extract per-thread offsets, then discard.
        auto lds_buf = OutputLdsBuf{lds_base, lds_buf_size};

        // LDS write offset (MFMA distribution → swizzled LDS layout).
        {
            constexpr auto lds_write_desc = TC::Output::MakeLdsWriteDescriptor();
            auto lds_write_view = OutputLdsWriteView{lds_buf, lds_write_desc};
            auto tmp_write = ck_tile::make_tile_window(
                lds_write_view,
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                    ck_tile::number<TC::BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                {0, 0, 0},
                TC::Mfma::MakeDistribution());

            lds_write_offset = tmp_write.pre_computed_coords_[ck_tile::number<0>{}]
                                                             [ck_tile::number<1>{}].get_offset();
        }

        // LDS read offset (DRAM distribution → coalesced layout for DRAM store).
        {
            constexpr auto lds_read_desc = TC::Output::MakeLdsReadDescriptor();
            auto lds_read_view = OutputLdsReadView{lds_buf, lds_read_desc};
            auto tmp_read = ck_tile::make_tile_window(
                lds_read_view,
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<TC::BLOCK_Q>{},
                                    ck_tile::number<TC::BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                {0, 0, 0, 0},
                TC::Output::MakeDramDistribution());

            lds_read_offset = tmp_read.pre_computed_coords_[ck_tile::number<0>{}]
                                                           [ck_tile::number<1>{}].get_offset();
        }

        // Output DRAM offset and validity.
        {
            constexpr auto out_dist = TC::Output::MakeDramDistribution();
            const auto out_desc = TC::Output::MakeDramDescriptor(ho, wo, bc.C);
            auto out_buf = OutputDramBuf{
                output_base,
                static_cast<ck_tile::index_t>(out_desc.get_element_space_size())};
            auto out_view = OutputDramView{out_buf, out_desc};

            auto tmp_dram = ck_tile::make_tile_window(
                out_view,
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<TC::BLOCK_Q>{},
                                    ck_tile::number<TC::BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                {0, bc.block_q, 0, 0},
                out_dist);

            output_elem_offset = tmp_dram.pre_computed_coords_[ck_tile::number<0>{}]
                                                              [ck_tile::number<1>{}].get_offset();

            // Check validity via the pad transform's coordinate-based check.
            store_valid = ck_tile::coordinate_has_valid_offset_assuming_top_index_is_valid(
                out_desc, tmp_dram.pre_computed_coords_[ck_tile::number<0>{}]
                                                      [ck_tile::number<1>{}]);
        }
    }

    // Convert fp32x4 accumulator to fp16x4 and write through LDS to global memory.
    __device__ void flush(fp32x4_t acc_val, int p_out)
    {
        // 1. Convert fp32→fp16.
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

        // 2. Store to LDS via precomputed offset (MFMA swizzled layout).
        __builtin_memcpy(lds_base + lds_write_offset, &out_reg, sizeof(out_reg));

        // 3. Wait for LDS writes from all threads.
        ck_tile::s_waitcnt_lgkm<0>();

        // 4. Read from LDS at coalesced offset (DRAM distribution layout).
        fp16x4_t lds_data;
        __builtin_memcpy(&lds_data, lds_base + lds_read_offset, sizeof(lds_data));

        // 5. Store to DRAM: base + row offset + per-thread offset.
        if(store_valid)
        {
            ck_tile::index_t store_offset = output_elem_offset
                + static_cast<ck_tile::index_t>(p_out) * row_stride_elems;
            __builtin_memcpy(output_base + store_offset, &lds_data, sizeof(lds_data));
        }
    }
};

} // namespace direct_conv
} // namespace ck_tile
