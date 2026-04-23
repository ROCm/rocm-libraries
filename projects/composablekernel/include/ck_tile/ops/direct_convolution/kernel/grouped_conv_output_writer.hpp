// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
namespace direct_conv {

// Shared OutputWriter for grouped convolution kernels.
// Writes MFMA accumulator results directly to global memory.
//
// TC must provide:
//   TC::Output::MakeDramDistribution()
//   TC::Output::MakeDramDescriptor(ho, wo, C)
//   TC::BLOCK_Q, TC::BLOCK_C4
template <typename TC>
struct OutputWriter
{
    // Output tile distribution and distributed tensor type for direct DRAM writes.
    static constexpr auto OutputDramDist = TC::Output::MakeDramDistribution();
    using OutputDstrTensor =
        ck_tile::static_distributed_tensor<_Float16, ck_tile::remove_cvref_t<decltype(OutputDramDist)>>;

    // DRAM tile window type — derived from MakeDramDescriptor with dummy args.
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

    OutputDramWindow dram_window;
    int last_p_out; // last row stored (for move_tile_window delta)

    template <typename BlockCoords_>
    __device__ OutputWriter(const BlockCoords_& bc,
                            uint4*, // Unused, matches OutputWriterLds constructor signature.
                            _Float16* __restrict__ out,
                            int ho,
                            int wo)
        : last_p_out(0)
    {
        constexpr auto out_dist = TC::Output::MakeDramDistribution();
        const auto out_desc = TC::Output::MakeDramDescriptor(ho, wo, bc.C);
        auto out_buf = OutputDramBuf{
            out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C + bc.block_k,
            static_cast<ck_tile::index_t>(out_desc.get_element_space_size())};
        auto out_view = OutputDramView{out_buf, out_desc};

        dram_window = ck_tile::make_tile_window(
            out_view,
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<TC::BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, bc.block_q, 0, 0},
            out_dist);
    }

    // Convert fp32x4 accumulator to fp16x4 and write directly to global memory.
    __device__ void flush(fp32x4_t acc_val, int p_out)
    {
        // 1. Convert fp32→fp16 and pack into distributed tensor.
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        const auto* fp16_ptr = reinterpret_cast<const _Float16*>(halves);

        OutputDstrTensor output_tile;
        output_tile.get_thread_buffer()(ck_tile::number<0>{}) = fp16_ptr[0];
        output_tile.get_thread_buffer()(ck_tile::number<1>{}) = fp16_ptr[1];
        output_tile.get_thread_buffer()(ck_tile::number<2>{}) = fp16_ptr[2];
        output_tile.get_thread_buffer()(ck_tile::number<3>{}) = fp16_ptr[3];

        // 2. Move window to current output row.
        ck_tile::move_tile_window(dram_window, {p_out - last_p_out, 0, 0, 0});
        last_p_out = p_out;

        // 3. Direct store to DRAM — pad transform handles OOB.
        ck_tile::store_tile(dram_window, output_tile);
    }
};

// Shared OutputWriterLds for grouped convolution kernels.
// Stages output through LDS before writing to global memory.
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
    // Output tile distribution and distributed tensor type for LDS writes.
    static constexpr auto OutputLdsDist  = TC::Mfma::MakeDistribution();
    static constexpr auto OutputDramDist = TC::Output::MakeDramDistribution();

    using OutputDstrTensor =
        ck_tile::static_distributed_tensor<_Float16, ck_tile::remove_cvref_t<decltype(OutputLdsDist)>>;

    // LDS buffer - same for read and write
    using OutputLdsBuf = ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>;

    // LDS write tile window type.
    using OutputLdsWriteDesc   = ck_tile::remove_cvref_t<decltype(TC::Output::MakeLdsWriteDescriptor())>;
    using OutputLdsWriteView   = ck_tile::tensor_view<OutputLdsBuf, OutputLdsWriteDesc>;
    using OutputLdsWriteWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputLdsWriteView{},
        ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0},
        OutputLdsDist))>;

    // Output tile distribution and distributed tensor for LDS reads before DRAM store.
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

    // DRAM tile window type.
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

    OutputLdsWriteWindow lds_write_window;
    OutputLdsReadWindow  lds_read_window;
    OutputDramWindow     dram_window;
    int last_p_out;

    template <typename BlockCoords_>
    __device__ OutputWriterLds(const BlockCoords_& bc,
                               uint4* output_lds,
                               _Float16* __restrict__ out,
                               int ho,
                               int wo)
        : last_p_out(0)
    {
        // Construct the common LDS buffer for write and read.
        auto lds_buf = OutputLdsBuf{
            reinterpret_cast<_Float16*>(output_lds),
            static_cast<ck_tile::index_t>(
                ck_tile::max(TC::Weight::WEIGHT_LDS_PADDED_UINT4, TC::Output::OUTPUT_LDS_BUFFER_SIZE) *
                (sizeof(uint4) / sizeof(_Float16)))};

        // Construct LDS write window for the MFMA output staging to LDS.
        constexpr auto lds_write_desc = TC::Output::MakeLdsWriteDescriptor();
        constexpr auto lds_write_dist = TC::Mfma::MakeDistribution();
        auto lds_write_view = OutputLdsWriteView{lds_buf, lds_write_desc};
        lds_write_window = ck_tile::make_tile_window(
            lds_write_view,
            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0},
            lds_write_dist);

        // Construct LDS read window for staging before DRAM store.
        constexpr auto lds_read_desc = TC::Output::MakeLdsReadDescriptor();
        constexpr auto lds_read_dist = TC::Output::MakeDramDistribution();
        auto lds_read_view = OutputLdsReadView{lds_buf, lds_read_desc};
        lds_read_window = ck_tile::make_tile_window(
            lds_read_view,
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<TC::BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0, 0},
            lds_read_dist);

        // Construct output DRAM window for global memory writes.
        constexpr auto out_dist = TC::Output::MakeDramDistribution();
        const auto out_desc = TC::Output::MakeDramDescriptor(ho, wo, bc.C);
        auto out_buf = OutputDramBuf{
            out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C + bc.block_k,
            static_cast<ck_tile::index_t>(out_desc.get_element_space_size())};
        auto out_view = OutputDramView{out_buf, out_desc};

        dram_window = ck_tile::make_tile_window(
            out_view,
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<TC::BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, bc.block_q, 0, 0},
            out_dist);
    }

    // Convert fp32x4 accumulator to fp16x4 and write through LDS to global memory.
    __device__ void flush(fp32x4_t acc_val, int p_out)
    {
        // 1. Convert fp32→fp16 and pack into distributed tensor.
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        const auto* fp16_ptr = reinterpret_cast<const _Float16*>(halves);

        OutputDstrTensor output_tile;
        output_tile.get_thread_buffer()(ck_tile::number<0>{}) = fp16_ptr[0];
        output_tile.get_thread_buffer()(ck_tile::number<1>{}) = fp16_ptr[1];
        output_tile.get_thread_buffer()(ck_tile::number<2>{}) = fp16_ptr[2];
        output_tile.get_thread_buffer()(ck_tile::number<3>{}) = fp16_ptr[3];

        // 2. Store to LDS via tile window.
        ck_tile::store_tile(lds_write_window, output_tile);

        // 3. Synchronize such that all threads' LDS writes are visible.
        ck_tile::s_waitcnt_lgkm<0>();

        // 4. LDS back to registers for coalesced store to DRAM.
        const auto lds_tile = ck_tile::load_tile(lds_read_window);

        // 5. Move window to current output row.
        ck_tile::move_tile_window(dram_window, {p_out - last_p_out, 0, 0, 0});
        last_p_out = p_out;

        // 6. Store LDS tile to DRAM — pad transform handles OOB.
        ck_tile::store_tile(dram_window, lds_tile);
    }
};

} // namespace direct_conv
} // namespace ck_tile
