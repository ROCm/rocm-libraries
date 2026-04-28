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
//   TC::Input::MakeDramDescriptor(hi, wi, C_total, px)  — px = left padding
//   TC::Input::MakeDramDistribution()
//   TC::Input::MakeLdsStoreDescriptor()
//   TC::Input::MakeLdsReadDescriptor()
//   TC::Mfma::MakeDistribution()
//   TC::TOTAL_SPATIAL, TC::BLOCK_W, TC::BLOCK_C8, TC::BLOCK_C4, TC::BLOCK_Q
//   TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8, TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16
template <typename TC, auto cfg>
struct InputLoader
{
    // Type aliases needed for temporary tile_window construction and MFMA reads.
    using InputDramWindowType = decltype(ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            static_cast<const _Float16*>(nullptr),
            TC::Input::MakeDramDescriptor(int{}, int{}, int{}, int{})),
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        ck_tile::multi_index<4>{},
        TC::Input::MakeDramDistribution()));

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

    // Persistent members — scalar state only, no tile_window objects.
    __amdgpu_buffer_rsrc_t            input_rsrc;          // buffer resource for DRAM async loads
    ck_tile::index_t                  input_voffset;        // per-thread DRAM byte offset (advances per row)
    CK_TILE_LDS_ADDR _Float16*        store_input_lds;      // per-thread LDS write destination (lane-0 address)
    ck_tile::index_t                  row_stride_bytes;     // bytes per input row (for y-advance)
    ck_tile::index_t                  is_valid;             // per-thread pad-transform validity (constant across rows)
    bool                              load_active;          // whether this thread should issue buffer_load_lds
    uint4*                            input_lds_ptr;        // LDS buffer base (for MFMA reads)
    ck_tile::index_t                  mfma_lds_offsets[cfg.kw]; // precomputed element offsets per kw slice

    template <typename BlockCoords_>
    __device__ InputLoader(const BlockCoords_& bc,
                           uint4* input_lds,
                           const _Float16* __restrict__ in,
                           int hi,
                           int wi,
                           int px)
                : input_lds_ptr(input_lds)
    {
        // Create temporary DRAM tile_window to extract per-thread offsets.
        const auto input_dram_desc = TC::Input::MakeDramDescriptor(hi, wi, bc.C, px);
        const _Float16* input_base = in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C + bc.block_k;
        const auto input_dram_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            input_base, input_dram_desc);

        constexpr auto input_dram_dist = TC::Input::MakeDramDistribution();
        {
            auto tmp_dram_window = ck_tile::make_tile_window(
                input_dram_view,
                ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                    ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
                {0, bc.block_q, 0, 0},
                input_dram_dist);

            // Extract per-thread DRAM source element offset.
            auto dram_elem_offset = tmp_dram_window.pre_computed_coords_[ck_tile::number<0>{}]
                                                                        [ck_tile::number<1>{}].get_offset();

            // Extract per-thread pad-transform validity flag.
            // The pad transform is on the spatial (wi) dimension. A thread's spatial
            // position is fixed at construction (doesn't change per row), so validity
            // is constant across all rows.
            is_valid = ck_tile::coordinate_has_valid_offset_assuming_top_index_is_valid(
                input_dram_desc, tmp_dram_window.pre_computed_coords_[ck_tile::number<0>{}]
                                                                     [ck_tile::number<1>{}]);

            // Extract lane-0 LDS destination offset from the DRAM window's adaptor
            // warp coordinate. The adaptor coordinate's bottom index gives tile-local
            // indices (row, spatial, c8, c), which we map through the LDS store
            // descriptor to get the LDS element offset. This correctly decouples the
            // LDS layout from DRAM descriptor transforms (pad, XOR, etc.).
            constexpr auto lds_store_desc = TC::Input::MakeLdsStoreDescriptor();
            auto warp_adaptor_bottom_idx =
                tmp_dram_window.pre_computed_warp_coords_[ck_tile::number<0>{}]
                                                         [ck_tile::number<0>{}].get_bottom_index();
            auto lds_coord = ck_tile::make_tensor_coordinate(lds_store_desc, warp_adaptor_bottom_idx);
            auto warp_lds_offset = lds_coord.get_offset();

            // Determine load_active from the thread's LDS write position.
            // buffer_load_lds writes lane l at warp_lds_offset + l*8 fp16
            // elements. In the LDS layout [TOTAL_SPATIAL, BLOCK_C8, 8],
            // the valid region spans BLOCK_W * BLOCK_C8 * 8 fp16 elements.
            // Threads writing beyond this boundary are in the padded region
            // (TOTAL_SPATIAL > BLOCK_W) and can skip the instruction.
            {
                const int lane_id = ck_tile::get_lane_id();
                load_active = (warp_lds_offset + lane_id * 8
                               < TC::BLOCK_W * TC::BLOCK_C8 * 8);
            }

            // Create buffer resource from the same base pointer the tile_window uses.
            auto elem_space_size = input_dram_desc.get_element_space_size();
            input_rsrc = ck_tile::make_builtin_buffer_resource(
                input_base,
                static_cast<uint32_t>(elem_space_size * sizeof(_Float16)));

            // Per-thread DRAM byte offset.
            input_voffset = static_cast<ck_tile::index_t>(dram_elem_offset * sizeof(_Float16));

            // LDS destination pointer: base + lane-0 LDS offset in fp16 elements.
            // Cast to LDS address space (address_space(3)) required by buffer_load_lds intrinsic.
            auto* lds_fp16_base = reinterpret_cast<CK_TILE_LDS_ADDR _Float16*>(
                reinterpret_cast<uintptr_t>(reinterpret_cast<_Float16*>(&input_lds[0])));
            store_input_lds = lds_fp16_base + warp_lds_offset;
        } // tmp_dram_window goes out of scope — no persistent register cost

        // Row stride in bytes: advancing one row in the input tensor.
        row_stride_bytes = static_cast<ck_tile::index_t>(wi * bc.C * sizeof(_Float16));

        // Precompute per-thread MFMA LDS read offsets for each kw slice.
        // This saves rgisters as we don't need to store the heavy tile window state.
        {
            auto mfma_buf_tmp = MfmaBuf{
                reinterpret_cast<_Float16*>(input_lds_ptr),
                static_cast<ck_tile::index_t>(TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16)};
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
        } // mfma_window_tmp goes out of scope — no persistent VGPR cost
    }

    __device__ void fetch_tile_to_lds(int lds_buffer_index)
    {
        if(load_active)
        {
            input_voffset += row_stride_bytes;
        }
        prefetch_tile_to_lds(lds_buffer_index);
    }

    __device__ void prefetch_tile_to_lds(int lds_buffer_index)
    {
        if(load_active)
        {
            // Compute LDS destination for the selected double-buffer slot.
            CK_TILE_LDS_ADDR _Float16* lds_dest =
                store_input_lds + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16;

            // Async DRAM→LDS load: 8 × fp16 = 16 bytes per thread.
            // oob_conditional_check=true: use is_valid from pad transform to force OOB for
            // threads in padding regions. Without this, threads in the padding zone but within
            // the buffer's address range would load real data instead of zeros.
            ck_tile::amd_async_buffer_load<_Float16, 8,
                ck_tile::amd_buffer_coherence_enum::coherence_default, true>(
                lds_dest,
                input_rsrc,
                input_voffset,  // per-thread byte offset (VGPR)
                0,              // wave offset (SGPR)
                ck_tile::number<0>{},  // immediate offset
                is_valid);      // pad-transform validity flag
        }
    }

    // Read a given kw slice for this thread from LDS into registers.
    // Uses precomputed element offsets — no tile_window state needed.
    __device__ void read_from_lds(ck_tile::fp16x4_t& input_reg, int slice, int lds_buffer_index) const
    {
        const _Float16* base = reinterpret_cast<const _Float16*>(input_lds_ptr)
                               + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16;
        auto buf = MfmaBuf{const_cast<_Float16*>(base),
                           static_cast<ck_tile::index_t>(TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16)};
        input_reg = buf.template get<ck_tile::fp16x4_t>(mfma_lds_offsets[slice], 0, true);
    }
};

} // namespace direct_conv
} // namespace ck_tile
