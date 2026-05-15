// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
namespace direct_conv {

// Shared InputLoader for grouped convolution kernels.
//
// Uses precomputed scalar state instead of persistent tile_window objects
// to minimize register pressure and spill traffic. A temporary tile_window
// is created at construction to extract per-thread offsets via CK Tile's
// distribution machinery, then immediately discarded.
//
// Template parameters:
//   TC  — TileConstants type providing Input/Mfma descriptors and distributions.
//   cfg — Config value providing kw and other kernel parameters.
//
// TC must provide:
//   TC::Input::MakeDramReadDescriptor(hi, wi, C_total, px, py, dx, dy, sx, sy)
//   TC::Input::MakeDramReadTileDistribution()
//   TC::Input::MakeLdsWriteDescriptor()
//   TC::Input::MakeLdsReadDescriptor()
//   TC::Mfma::MakeAccTileDistribution()
//   TC::TOTAL_SPATIAL, TC::BLOCK_W, TC::BLOCK_C8, TC::BLOCK_C4, TC::BLOCK_Q
//   TC::INPUT_LDS_BUFFER_SIZE_C8, TC::INPUT_LDS_BUFFER_SIZE_FP16
template <typename TC, auto cfg, typename InputType = ck_tile::fp16x4_t, bool Padded = true, typename ElementType = _Float16>
struct InputLoader
{
    // Register type for MFMA input operand (matches read_from_lds parameter type).
    using input_type = InputType;

    // Type aliases needed for temporary tile_window construction and MFMA reads.
    using InputDramWindowType = decltype(ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            static_cast<const ElementType*>(nullptr),
            TC::Input::MakeDramReadDescriptor(int{}, int{}, int{}, int{}, int{}, int{}, int{}, int{}, int{})),
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        ck_tile::multi_index<4>{},
        TC::Input::MakeDramReadTileDistribution()));

    static constexpr auto mfma_desc = TC::Input::MakeLdsReadDescriptor();
    static constexpr auto mfma_dist = TC::Mfma::MakeAccTileDistribution();

    using MfmaBuf      = ck_tile::buffer_view<ck_tile::address_space_enum::lds, ElementType, ck_tile::index_t, true>;
    using MfmaViewType = ck_tile::tensor_view<MfmaBuf, ck_tile::remove_cvref_t<decltype(mfma_desc)>>;

    using MfmaWindowType = decltype(ck_tile::make_tile_window(
        MfmaViewType{},
        ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0},
        mfma_dist));

    // Persistent members — scalar state only, no tile_window objects.
    __amdgpu_buffer_rsrc_t            input_rsrc;          // buffer resource for DRAM async loads
    ck_tile::index_t                  input_voffset;        // per-thread DRAM byte offset (advances per row)
    CK_TILE_LDS_ADDR ElementType*     store_input_lds;      // per-thread LDS write destination (lane-0 address)
    ck_tile::index_t                  row_stride_bytes;     // bytes per input row (for y-advance)
    ck_tile::index_t                  is_valid;             // per-thread pad-transform validity (constant across rows)
    bool                              load_active;          // whether this thread should issue buffer_load_lds
    uint4*                            input_lds_ptr;        // LDS buffer base (for MFMA reads)
    ck_tile::index_t                  mfma_lds_offsets[cfg.kw]; // precomputed element offsets per kw slice

    // Additional state for padded path (c_per_group < GROUP_SIZE).
    // When Padded=false, these are not needed and are eliminated by the compiler.
    struct PaddedState
    {
        int               hi_;                  // input height
        int               wi_;                  // input width
        int               C_in_;                // groups * c_per_group
        int               c_per_group_;         // channels per group (< GROUP_SIZE)
        int               px_;                  // spatial padding in width
        int               py_;                  // spatial padding in height
        int               dx_;                  // dilation in width
        int               dy_;                  // dilation in height
        int               sx_;                  // stride in width
        int               sy_;                  // stride in height
        int               current_row_;         // current input row for padded fetch
        int               block_q_;             // spatial block offset
        const ElementType* input_base_padded_;  // base pointer for padded DRAM reads
    };
    struct EmptyState {};
    [[no_unique_address]] std::conditional_t<Padded, PaddedState, EmptyState> padded_state_;

    template <typename BlockCoords_>
    __device__ InputLoader(const BlockCoords_& bc,
                           uint4* input_lds,
                           const ElementType* __restrict__ in,
                           int hi,
                           int wi,
                           int px,
                           int py,
                           int dx,
                           int dy,
                           int sx,
                           int sy,
                           int c_per_group = TC::GROUP_SIZE,
                           bool init_mfma_offsets = true)
                : input_lds_ptr(input_lds)
    {
        constexpr auto input_dram_dist = TC::Input::MakeDramReadTileDistribution();
        constexpr auto tile_lengths = ck_tile::make_tuple(
            ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{});

        // Extract load_active from a tile_window's warp-level LDS mapping.
        // The thread's LDS write position must fall within the BLOCK_W region;
        // threads beyond BLOCK_W are inactive and must not write to LDS.
        auto compute_load_active = [&](const auto& dram_window) {
            constexpr auto lds_store_desc = TC::Input::MakeLdsWriteDescriptor();
            auto warp_bottom_idx =
                dram_window.pre_computed_warp_coords_[ck_tile::number<0>{}]
                                                     [ck_tile::number<0>{}].get_bottom_index();
            auto lds_offset = ck_tile::make_tensor_coordinate(lds_store_desc, warp_bottom_idx).get_offset();
            const int lane_id = ck_tile::get_lane_id();
            return lds_offset + lane_id * 8 < TC::BLOCK_W * TC::BLOCK_C8 * 8;
        };

        // ---- Unpadded init: extracted as a lambda so it can be called from
        // both the Padded=true (but c_per_group==GROUP_SIZE at runtime) case
        // and the Padded=false case without code duplication. ----
        auto init_unpadded = [&]() {
            const auto input_dram_desc = TC::Input::MakeDramReadDescriptor(hi, wi, bc.C, px, py, dx, dy, sx, sy);
            const ElementType* input_base = in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C + bc.block_k;
            const auto input_dram_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
                input_base, input_dram_desc);

            auto tmp_dram_window = ck_tile::make_tile_window(
                input_dram_view, tile_lengths, {0, bc.block_q, 0, 0}, input_dram_dist);

            load_active = compute_load_active(tmp_dram_window);

            // Extract per-thread DRAM source element offset.
            auto dram_elem_offset = tmp_dram_window.pre_computed_coords_[ck_tile::number<0>{}]
                                                                        [ck_tile::number<1>{}].get_offset();

            // Extract per-thread pad-transform validity flag.
            is_valid = ck_tile::coordinate_has_valid_offset_assuming_top_index_is_valid(
                input_dram_desc, tmp_dram_window.pre_computed_coords_[ck_tile::number<0>{}]
                                                                     [ck_tile::number<1>{}]);

            // Create buffer resource.
            auto elem_space_size = input_dram_desc.get_element_space_size();
            input_rsrc = ck_tile::make_builtin_buffer_resource(
                input_base,
                static_cast<uint32_t>(elem_space_size * sizeof(ElementType)));

            // Per-thread DRAM byte offset.
            input_voffset = static_cast<ck_tile::index_t>(dram_elem_offset * sizeof(ElementType));

            // LDS destination pointer.
            constexpr auto lds_store_desc = TC::Input::MakeLdsWriteDescriptor();
            auto warp_adaptor_bottom_idx =
                tmp_dram_window.pre_computed_warp_coords_[ck_tile::number<0>{}]
                                                         [ck_tile::number<0>{}].get_bottom_index();
            auto warp_lds_offset = ck_tile::make_tensor_coordinate(
                lds_store_desc, warp_adaptor_bottom_idx).get_offset();
            auto* lds_elem_base = reinterpret_cast<CK_TILE_LDS_ADDR ElementType*>(
                reinterpret_cast<uintptr_t>(reinterpret_cast<ElementType*>(&input_lds[0])));
            store_input_lds = lds_elem_base + warp_lds_offset;

            // Row stride in bytes.
            row_stride_bytes = static_cast<ck_tile::index_t>(wi * bc.C * sizeof(ElementType));
        };

        if constexpr(Padded)
        {
            if(c_per_group != TC::GROUP_SIZE)
            {
                // ---- Padded path: c_per_group < GROUP_SIZE ----
                // Store scalar state for creating temporary tile_windows per row.
                padded_state_.hi_               = hi;
                padded_state_.wi_               = wi;
                padded_state_.C_in_             = bc.C_in;
                padded_state_.c_per_group_      = c_per_group;
                padded_state_.px_               = px;
                padded_state_.py_               = py;
                padded_state_.dx_               = dx;
                padded_state_.dy_               = dy;
                padded_state_.sx_               = sx;
                padded_state_.sy_               = sy;
                padded_state_.current_row_      = 0;
                padded_state_.block_q_          = bc.block_q;
                padded_state_.input_base_padded_ = in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C_in + bc.block_k_in;

                // Extract load_active using the padded DRAM descriptor (correct strides).
                {
                    const auto padded_desc =
                        TC::Input::template MakeDramReadDescriptorPadded<cfg.vector_size>(
                            hi, wi, bc.C_in, c_per_group, px, py, dx, dy, sx, sy);
                    auto padded_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
                        padded_state_.input_base_padded_, padded_desc);
                    auto tmp_window = ck_tile::make_tile_window(
                        padded_view, tile_lengths, {0, bc.block_q, 0, 0}, input_dram_dist);
                    load_active = compute_load_active(tmp_window);
                }

                // Mark async path members as unused.
                input_voffset = 0;
                store_input_lds = nullptr;
                row_stride_bytes = 0;
                is_valid = 0;
            }
            else
            {
                init_unpadded();
                // Initialize c_per_group_ so runtime checks in fetch_tile_to_lds
                // and prefetch_tile_to_lds correctly take the unpadded path.
                padded_state_.c_per_group_ = TC::GROUP_SIZE;
            }
        }
        else
        {
            init_unpadded();
        }

        // Precompute per-thread MFMA LDS read offsets for each kw slice.
        // Shared between padded and unpadded paths — LDS layout is identical.
        if (init_mfma_offsets)
        {
            auto mfma_buf_tmp = MfmaBuf{
                reinterpret_cast<ElementType*>(input_lds_ptr),
                static_cast<ck_tile::index_t>(TC::INPUT_LDS_BUFFER_SIZE_FP16)};
            auto mfma_view_tmp = MfmaViewType{mfma_buf_tmp, mfma_desc};
            auto mfma_window_tmp = ck_tile::make_tile_window(
                mfma_view_tmp,
                ck_tile::make_tuple(ck_tile::number<TC::BLOCK_Q>{},
                                    ck_tile::number<TC::BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                {0, 0, 0},
                mfma_dist);

            for(int s = 0; s < cfg.kw; s++)
            {
                mfma_lds_offsets[s] =
                    mfma_window_tmp.pre_computed_coords_[ck_tile::number<0>{}]
                                                       [ck_tile::number<1>{}].get_offset();
                if(s < cfg.kw - 1)
                    ck_tile::move_tile_window(mfma_window_tmp, {1, 0, 0});
            }
        }
    }

    __device__ __forceinline__ void fetch_tile_to_lds(int lds_buffer_index)
    {
        if constexpr(Padded)
        {
            if(padded_state_.c_per_group_ != TC::GROUP_SIZE)
            {
                padded_state_.current_row_++;
            }
            else
            {
                if(load_active)
                {
                    input_voffset += row_stride_bytes;
                }
            }
        }
        else
        {
            if(load_active)
            {
                input_voffset += row_stride_bytes;
            }
        }
        prefetch_tile_to_lds(lds_buffer_index);
    }

    __device__ __forceinline__ void prefetch_tile_to_lds(int lds_buffer_index)
    {
        if(load_active)
        {
            if constexpr(Padded)
            {
                if(padded_state_.c_per_group_ != TC::GROUP_SIZE)
                {
                    prefetch_tile_to_lds_padded(lds_buffer_index);
                }
                else
                {
                    prefetch_tile_to_lds_unpadded(lds_buffer_index);
                }
            }
            else
            {
                prefetch_tile_to_lds_unpadded(lds_buffer_index);
            }
        }
    }

    __device__ __forceinline__ void prefetch_tile_to_lds_unpadded(int lds_buffer_index)
    {
        CK_TILE_LDS_ADDR ElementType* lds_dest =
            store_input_lds + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;

        ck_tile::amd_async_buffer_load<ElementType, 8,
            ck_tile::amd_buffer_coherence_enum::coherence_default, true>(
            lds_dest,
            input_rsrc,
            input_voffset,
            0,
            ck_tile::number<0>{},
            is_valid);
    }

    // Padded path: create temporary tile_windows per row, use load_tile + store_tile.
    // This correctly zero-pads channels beyond c_per_group via the pad transform's
    // per-element OOB checking.
    __device__ __forceinline__ void prefetch_tile_to_lds_padded(int lds_buffer_index)
        requires(Padded)
    {
        constexpr auto input_dram_dist = TC::Input::MakeDramReadTileDistribution();

        // Create padded DRAM descriptor and view.
        const auto padded_dram_desc =
            TC::Input::template MakeDramReadDescriptorPadded<cfg.vector_size>(
                padded_state_.hi_, padded_state_.wi_, padded_state_.C_in_,
                padded_state_.c_per_group_, padded_state_.px_, padded_state_.py_,
                padded_state_.dx_, padded_state_.dy_, padded_state_.sx_, padded_state_.sy_);

        auto padded_dram_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            padded_state_.input_base_padded_, padded_dram_desc);

        auto padded_dram_window = ck_tile::make_tile_window(
            padded_dram_view,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {padded_state_.current_row_, padded_state_.block_q_, 0, 0},
            input_dram_dist);

        // Load from DRAM with per-element OOB checking (pad transform zeros padded channels).
        auto input_reg = ck_tile::load_tile(padded_dram_window);

        // Create LDS write window
        constexpr auto lds_write_desc = TC::Input::MakeLdsWriteDescriptor();
        ElementType* lds_base = reinterpret_cast<ElementType*>(input_lds_ptr)
                             + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;
        auto lds_write_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            lds_base, lds_write_desc);

        auto lds_write_window = ck_tile::make_tile_window(
            lds_write_view,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {0, 0, 0, 0},
            input_dram_dist);

        ck_tile::store_tile(lds_write_window, input_reg);
    }

    // Read a given kw slice for this thread from LDS into registers.
    // Uses precomputed element offsets and direct memcpy load
    // (no buffer_view overhead).
    __device__ __forceinline__ void read_from_lds(InputType& input_reg, int slice, int lds_buffer_index) const
    {
        const ElementType* base = reinterpret_cast<const ElementType*>(input_lds_ptr)
                               + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;
        __builtin_memcpy(&input_reg, base + mfma_lds_offsets[slice], sizeof(input_reg));
    }

    // Read from a specific C-section of the input LDS (for non-grouped conv C-reduction).
    //
    // In non-grouped conv, the input LDS holds BLOCK_C = waves_per_wg/2 * 32 channels.
    // Within each c_block, each wave iterates over all C-sections (c_local = 0..c_local_max),
    // reading from section c_local rather than the wave's own section.
    //
    // c_section_delta_elements: (c_local - wave_group) * 32 — the signed offset in fp16
    // elements from the wave's own C-section to the target c_local section.
    __device__ __forceinline__ void read_from_lds_at_section(
        InputType& input_reg, int slice, int lds_buffer_index,
        int c_section_delta_elements) const
    {
        const ElementType* base = reinterpret_cast<const ElementType*>(input_lds_ptr)
                               + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;
        __builtin_memcpy(&input_reg,
                         base + mfma_lds_offsets[slice] + c_section_delta_elements,
                         sizeof(input_reg));
    }
};

} // namespace direct_conv
} // namespace ck_tile
