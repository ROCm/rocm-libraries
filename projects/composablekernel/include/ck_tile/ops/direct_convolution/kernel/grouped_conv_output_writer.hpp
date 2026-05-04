// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"

namespace ck_tile {
namespace direct_conv {

// Shared OutputWriter for grouped convolution kernels.
// Writes MFMA accumulator results directly to global memory.
//
// Uses precomputed scalar state instead of persistent tile_window objects
// to minimize register pressure and spill traffic.
//
// TC must provide:
//   TC::Output::MakeDramWriteTileDistributionNarrow()
//   TC::Output::MakeDramWriteDescriptorNarrow(ho, wo, C)
//   TC::BLOCK_Q, TC::BLOCK_C4
template <typename TC>
struct OutputWriter
{
    // Type aliases for temporary tile_window construction.
    using OutputDramDesc =
        ck_tile::remove_cvref_t<decltype(TC::Output::MakeDramWriteDescriptorNarrow(int{}, int{}, int{}))>;
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
        TC::Output::MakeDramWriteTileDistributionNarrow()))>;

    // Persistent members — scalar state only.
    _Float16*         output_base;         // base output pointer for this block
    ck_tile::index_t  output_elem_offset;  // per-thread element offset (within-tile spatial+channel)
    ck_tile::index_t  row_stride_elems;    // elements per output row (wo * C)
    bool              store_valid;         // whether this thread's output position is in bounds

    // Additional state for padded path (k_per_group < GROUP_SIZE).
    int               k_valid_count_;    // how many of thread's 4 output channels are valid (0-4)

    template <typename BlockCoords_>
    __device__ OutputWriter(const BlockCoords_& bc,
                            uint4*, // Unused, matches OutputWriterLds constructor signature.
                            _Float16* __restrict__ out,
                            int ho,
                            int wo,
                            int k_per_group = TC::GROUP_SIZE)
    {
        output_base = out + static_cast<size_t>(bc.block_n) * ho * wo * bc.K + bc.block_k_out;
        row_stride_elems = wo * bc.K;

        // Create temporary DRAM tile_window to extract per-thread offset and validity.
        {
            constexpr auto out_dist = TC::Output::MakeDramWriteTileDistributionNarrow();

            auto extract_offset_and_validity = [&](const auto& out_desc) {
                auto out_buf = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(
                    output_base,
                    static_cast<ck_tile::index_t>(out_desc.get_element_space_size()));
                auto out_view = ck_tile::tensor_view<
                    ck_tile::remove_cvref_t<decltype(out_buf)>,
                    ck_tile::remove_cvref_t<decltype(out_desc)>>{out_buf, out_desc};

                auto tmp_window = ck_tile::make_tile_window(
                    out_view,
                    ck_tile::make_tuple(ck_tile::number<1>{},
                                        ck_tile::number<TC::BLOCK_Q>{},
                                        ck_tile::number<TC::BLOCK_C4>{},
                                        ck_tile::number<4>{}),
                    {0, bc.block_q, 0, 0},
                    out_dist);

                output_elem_offset = tmp_window.pre_computed_coords_[ck_tile::number<0>{}]
                                                                    [ck_tile::number<1>{}].get_offset();
                store_valid = ck_tile::coordinate_has_valid_offset_assuming_top_index_is_valid(
                    out_desc, tmp_window.pre_computed_coords_[ck_tile::number<0>{}]
                                                            [ck_tile::number<1>{}]);
            };

            if constexpr(requires { TC::Output::MakeDramWriteDescriptorNarrowPadded(int{}, int{}, int{}, int{}); })
            {
                if(k_per_group < TC::GROUP_SIZE)
                {
                    // Padded path: use padded descriptor for correct group-strided offsets.
                    extract_offset_and_validity(
                        TC::Output::MakeDramWriteDescriptorNarrowPadded(ho, wo, bc.K, k_per_group));
                    k_valid_count_ = k_per_group;
                }
                else
                {
                    extract_offset_and_validity(
                        TC::Output::MakeDramWriteDescriptorNarrow(ho, wo, bc.K));
                    k_valid_count_ = 4;
                }
            }
            else
            {
                // Variant without padded descriptor support: always use unpadded path.
                extract_offset_and_validity(
                    TC::Output::MakeDramWriteDescriptorNarrow(ho, wo, bc.K));
                k_valid_count_ = 4;
            }
        } // tmp_window goes out of scope
    }

    // Convert fp32x4 accumulator to fp16x4 and write directly to global memory.
    __device__ __forceinline__ void flush(fp32x4_t acc_val, int p_out)
    {
        if(!store_valid)
            return;

        // 1. Convert fp32→fp16.
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

        // 2. Direct store to DRAM: base + row offset + per-thread offset.
        ck_tile::index_t store_offset = output_elem_offset
            + static_cast<ck_tile::index_t>(p_out) * row_stride_elems;

        if(k_valid_count_ == 4)
        {
            // Full 8B write: all 4 channels valid.
            __builtin_memcpy(output_base + store_offset, &out_reg, sizeof(out_reg));
        }
        else
        {
            // Partial write: only k_valid_count_ channels valid.
            const _Float16* src = reinterpret_cast<const _Float16*>(&out_reg);
            for(int i = 0; i < k_valid_count_; i++)
            {
                output_base[store_offset + i] = src[i];
            }
        }
    }
};

// Shared OutputWriterLds for grouped convolution kernels.
// Stages output through LDS before writing to global memory.
//
// Uses precomputed scalar state instead of persistent tile_window objects
// to minimize register pressure and spill traffic.
//
// The LDS write uses the MFMA distribution (all threads write 8B each).
// The LDS read + DRAM store uses wider 16B (uint4) operations with only
// STORE_VECS = BLOCK_Q * BLOCK_C8 threads active, this gives a better 
// memory throughput (ds_read_b128 + global_store_dwordx4).
//
// Thread activity is managed by the store distribution's MaxThreadId parameter,
// which marks threads with tid >= STORE_VECS as inactive. The distribution
// maps all block_size threads to [STORE_Q, BLOCK_C8, 8] tile positions,
// but the swizzle and bounds checking are handled by the CK Tile descriptor
// and coordinate infrastructure.
//
// TC must provide:
//   TC::Mfma::MakeAccTileDistribution()
//   TC::Output::MakeLdsWriteDescriptor()
//   TC::Output::MakeDramWriteTileDistributionWide()
//   TC::Output::MakeLdsReadDescriptorWide()
//   TC::Output::MakeDramWriteDescriptorWide(wo, C)
//   TC::Output::OUTPUT_LDS_BUFFER_SIZE
//   TC::Output::STORE_Q
//   TC::Weight::WEIGHT_LDS_SIZE_UINT4
//   TC::BLOCK_Q, TC::BLOCK_C4, TC::BLOCK_C8
template <typename TC>
struct OutputWriterLds
{
    // Type aliases for the LDS write path (MFMA distribution).
    static constexpr auto OutputLdsDist = TC::Mfma::MakeAccTileDistribution();

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

    // Type aliases for the store path (store distribution with MaxThreadId).
    static constexpr auto StoreDist = TC::Output::MakeDramWriteTileDistributionWide();

    using StoreLdsReadDesc = ck_tile::remove_cvref_t<decltype(TC::Output::MakeLdsReadDescriptorWide())>;
    using StoreLdsReadView = ck_tile::tensor_view<OutputLdsBuf, StoreLdsReadDesc>;

    using StoreDramDesc =
        ck_tile::remove_cvref_t<decltype(TC::Output::MakeDramWriteDescriptorWide(int{}, int{}))>;
    using StoreDramBuf =
        ck_tile::buffer_view<ck_tile::address_space_enum::global, _Float16, ck_tile::index_t, true>;
    using StoreDramView = ck_tile::tensor_view<StoreDramBuf, StoreDramDesc>;

    // Persistent members — scalar state only.
    _Float16*         output_base;           // base output pointer for this block
    _Float16*         lds_base;              // LDS buffer base pointer
    ck_tile::index_t  lds_write_offset;      // per-thread LDS write element offset (MFMA distribution)
    ck_tile::index_t  lds_read_offset;       // precomputed swizzled LDS offset in fp16 elements
    ck_tile::index_t  output_elem_offset;    // per-thread output DRAM element offset (C8-aligned)
    ck_tile::index_t  row_stride_elems;      // wo * C elements per output row
    ck_tile::index_t  lds_buf_size;          // LDS buffer size in fp16 elements
    bool              store_valid;           // whether this thread should store to DRAM

    // k_per_group for padded path (< GROUP_SIZE means partial writes).
    int               k_per_group_;

    template <typename BlockCoords_>
    __device__ OutputWriterLds(const BlockCoords_& bc,
                               uint4* output_lds,
                               _Float16* __restrict__ out,
                               int ho,
                               int wo,
                               int k_per_group = TC::GROUP_SIZE)
    {
        output_base = out + static_cast<size_t>(bc.block_n) * ho * wo * bc.K + bc.block_k_out;
        lds_base = reinterpret_cast<_Float16*>(output_lds);
        row_stride_elems = wo * bc.K;
        lds_buf_size = static_cast<ck_tile::index_t>(
            ck_tile::max(TC::Weight::WEIGHT_LDS_SIZE_UINT4, TC::Output::OUTPUT_LDS_BUFFER_SIZE) *
            (sizeof(uint4) / sizeof(_Float16)));

        // LDS write offset (MFMA distribution → swizzled LDS layout).
        {
            auto lds_buf = OutputLdsBuf{lds_base, lds_buf_size};
            constexpr auto lds_write_desc = TC::Output::MakeLdsWriteDescriptor();
            auto lds_write_view = OutputLdsWriteView{lds_buf, lds_write_desc};
            auto tmp_write = ck_tile::make_tile_window(
                lds_write_view,
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                    ck_tile::number<TC::BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                {0, 0, 0},
                TC::Mfma::MakeAccTileDistribution());

            lds_write_offset = tmp_write.pre_computed_coords_[ck_tile::number<0>{}]
                                                             [ck_tile::number<1>{}].get_offset();
        }

        // LDS read offset (store distribution → swizzled LDS layout).
        // The store distribution maps all threads to [STORE_Q, BLOCK_C8, 8].
        // The descriptor applies the same swizzle as the write descriptor.
        {
            constexpr auto store_dist = TC::Output::MakeDramWriteTileDistributionWide();
            constexpr auto lds_read_desc = TC::Output::MakeLdsReadDescriptorWide();
            auto lds_buf = OutputLdsBuf{lds_base, lds_buf_size};
            auto lds_read_view = StoreLdsReadView{lds_buf, lds_read_desc};
            auto tmp_read = ck_tile::make_tile_window(
                lds_read_view,
                ck_tile::make_tuple(ck_tile::number<TC::Output::STORE_Q>{},
                                    ck_tile::number<TC::BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                {0, 0, 0},
                store_dist);

            lds_read_offset = tmp_read.pre_computed_coords_[ck_tile::number<0>{}]
                                                           [ck_tile::number<1>{}].get_offset();
        }

        // DRAM store offset and validity (store distribution → padded DRAM layout).
        // The pad transform marks threads with Q >= wo as invalid.
        // The is_thread_active() query marks threads with tid >= STORE_VECS as inactive.
        {
            constexpr auto store_dist = TC::Output::MakeDramWriteTileDistributionWide();

            auto extract_store_offset = [&](const auto& out_desc) {
                auto out_buf = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(
                    output_base,
                    static_cast<ck_tile::index_t>(out_desc.get_element_space_size()));
                auto out_view = ck_tile::tensor_view<
                    ck_tile::remove_cvref_t<decltype(out_buf)>,
                    ck_tile::remove_cvref_t<decltype(out_desc)>>{out_buf, out_desc};
                auto tmp_dram = ck_tile::make_tile_window(
                    out_view,
                    ck_tile::make_tuple(ck_tile::number<TC::Output::STORE_Q>{},
                                        ck_tile::number<TC::BLOCK_C8>{},
                                        ck_tile::number<8>{}),
                    {bc.block_q, 0, 0},
                    store_dist);

                output_elem_offset = tmp_dram.pre_computed_coords_[ck_tile::number<0>{}]
                                                                  [ck_tile::number<1>{}].get_offset();
                store_valid = store_dist.is_thread_active()
                    && ck_tile::coordinate_has_valid_offset_assuming_top_index_is_valid(
                           out_desc, tmp_dram.pre_computed_coords_[ck_tile::number<0>{}]
                                                                  [ck_tile::number<1>{}]);
            };

            if constexpr(requires { TC::Output::MakeDramWriteDescriptorWidePadded(int{}, int{}, int{}); })
            {
                if(k_per_group < TC::GROUP_SIZE)
                    extract_store_offset(
                        TC::Output::MakeDramWriteDescriptorWidePadded(wo, bc.K, k_per_group));
                else
                    extract_store_offset(
                        TC::Output::MakeDramWriteDescriptorWide(wo, bc.K));
            }
            else
            {
                extract_store_offset(
                    TC::Output::MakeDramWriteDescriptorWide(wo, bc.K));
            }

            k_per_group_ = k_per_group;
        }
    }

    // Convert fp32x4 accumulator to fp16x4 and write through LDS to global memory.
    __device__ __forceinline__ void flush(fp32x4_t acc_val, int p_out)
    {
        // 1. Convert fp32→fp16.
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

        // 2. Store 8B to LDS via precomputed offset (MFMA swizzled layout).
        // All threads participate in the write.
        __builtin_memcpy(lds_base + lds_write_offset, &out_reg, sizeof(out_reg));

        // 3. Wait for ALL threads' LDS writes to complete.
        // __syncthreads() is required here (not just s_waitcnt lgkmcnt(0)) because
        // each thread reads from a coalesced offset that was written by a DIFFERENT
        // thread. s_waitcnt only guarantees this thread's own writes are visible.
        __syncthreads();

        if(store_valid)
        {
            // 4. Read 16B (uint4) from LDS at distribution-computed swizzled offset.
            const uint4* output_lds_uint4 = reinterpret_cast<const uint4*>(lds_base);
            uint4 lds_data = output_lds_uint4[lds_read_offset / 8];

            // 5. Store to DRAM: base + row offset + per-thread offset.
            ck_tile::index_t store_offset = output_elem_offset
                + static_cast<ck_tile::index_t>(p_out) * row_stride_elems;

            if(k_per_group_ >= TC::GROUP_SIZE)
            {
                // Full 16B write: all 8 channels valid.
                __builtin_memcpy(output_base + store_offset, &lds_data, sizeof(lds_data));
            }
            else
            {
                // Partial write: only k_per_group_ channels within each group.
                // The 8 channels span (8 / GROUP_SIZE) groups, each with GROUP_SIZE channels.
                const _Float16* src = reinterpret_cast<const _Float16*>(&lds_data);
                constexpr int GROUP_SIZE = TC::GROUP_SIZE;
                for(int g = 0; g < 8 / GROUP_SIZE; g++)
                {
                    for(int c = 0; c < k_per_group_; c++)
                    {
                        output_base[store_offset + g * GROUP_SIZE + c] = src[g * GROUP_SIZE + c];
                    }
                }
            }
        }

    }
};

} // namespace direct_conv
} // namespace ck_tile
