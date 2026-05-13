// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/load_tile_transpose.hpp"
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
template <int KH, int KW, typename VecType = fp16x4_t>
struct WeightAccessor
{
    using value_type = VecType;
    VecType weights[KH * KW];

    static constexpr auto desc_ = ck_tile::make_naive_tensor_descriptor_packed(
        ck_tile::make_tuple(ck_tile::number<KH>{}, ck_tile::number<KW>{}));

    template <int R, int S>
    __device__ __forceinline__ VecType get() const
    {
        constexpr auto coord = ck_tile::make_tensor_coordinate(
            desc_, ck_tile::make_tuple(ck_tile::number<R>{}, ck_tile::number<S>{}));
        return weights[coord.get_offset()];
    }

    template <int R, int S>
    __device__ __forceinline__ VecType get_transposed() const
    {
        return get<KH - 1 - R, KW - 1 - S>();
    }
};

// -----------------------------------------------------------------------
// WeightAccessor8 — register-resident filter weight buffer with 8-element
// vector per filter position. Used by 8c (Toeplitz) and 32c kernels where
// the MFMA B-operand is fp16x8_t/bf16x8_t (mfma_f32_16x16x32).
//
// Note: WeightAccessor8 is a convenience alias — WeightAccessor with an
// 8-element VecType provides the same functionality. This alias is kept
// for readability at call sites that distinguish 4-element vs 8-element.
// -----------------------------------------------------------------------
template <int KH, int KW, typename VecType = fp16x8_t>
using WeightAccessor8 = WeightAccessor<KH, KW, VecType>;

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
template <typename TC, auto cfg, bool Padded, typename BlockCoords_, typename ElementType = _Float16>
__device__ void weight_load_to_lds(const BlockCoords_& bc,
                                   uint4* weight_lds,
                                   const ElementType* __restrict__ wei,
                                   const int c_per_group,
                                   const int k_per_group)
{
    constexpr auto weight_dram_desc = TC::Weight::MakeDramReadDescriptor();
    constexpr auto weight_dram_dist = TC::Weight::MakeDramReadTileDistribution();

    // When Padded=true, check at runtime whether padding is actually needed.
    // When Padded=false, skip the padded path entirely (dead code elimination).
    if constexpr(Padded)
    {
        if(TC::GROUP_SIZE != c_per_group || TC::GROUP_SIZE != k_per_group)
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
            const ElementType* wei_block =
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
                reinterpret_cast<ElementType*>(weight_lds), weight_lds_desc);

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
            return;
        }
    }

    // Unpadded path: c_per_group == GROUP_SIZE && k_per_group == GROUP_SIZE.
    // Reached when Padded=false, or when Padded=true but no padding is needed.
    // Use efficient async_load_tile (buffer_load_lds) with the flat 2D descriptor.
    {
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
            reinterpret_cast<ElementType*>(weight_lds), weight_lds_desc);
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

// Shared Dgrad weight read function for grouped convolution kernels.
//
// Reads weights from LDS into registers using load_tile_transpose.
// The distribution and descriptor are per-warp (single-warp encoding);
// the warp offset into LDS is computed here to avoid mixing K-stride
// and C-stride factors in a single flat dimension.
//
// The LDS layout is [K_total][KH*KW][C]. The per-warp descriptor views
// a [16, 16] slice (16 K rows × 16 C cols) with strides [KH_KW*GROUP_SIZE, 1].
// After transpose, each thread holds the MFMA B-operand for its K position.
//
// For 16c (fp16x4_t): one transpose read per filter position.
// For 32c (fp16x8_t): two transpose reads per filter position, with the second
// reading from K rows offset by 16 (covering K[16:31] within the group).
//
// TC must provide:
//   TC::Weight::MakeLdsReadDescriptorDgrad()   — per-warp 2D [16, 16] descriptor
//   TC::Weight::MakeLdsReadTileDistributionDgrad() — per-warp input distribution
//   TC::GROUP_SIZE, TC::KH_KW
//   TC::BLOCK_GROUPS — number of conv groups per workgroup
//
// WavesPerGroup: number of waves per conv group (1 for 16c, 2 for 32c).
//   For 32c, wave_half selects which 16 C columns to read.
//
// WeightAccessorT must provide:
//   value_type — fp16x4_t (16c) or fp16x8_t (32c)
//   weights[]  — register array indexed by filter position
template <typename TC, int KH, int KW, typename WeightAccessorT, int WavesPerGroup = 1,
          typename ElementType = _Float16>
__device__ void weight_read_dgrad(WeightAccessorT& wa, uint4* weight_lds)
{
    constexpr int K_STRIDE = KH * KW * TC::GROUP_SIZE; // stride per K row in fp16
    constexpr int KH_KW = KH * KW;

    // Compute per-warp base pointer.
    const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);

    ElementType* warp_base;
    if constexpr(WavesPerGroup == 1)
    {
        // 16c: each warp = one conv group.
        // wave_group = warp_id, reads K[0:GROUP_SIZE-1] of this group.
        warp_base = reinterpret_cast<ElementType*>(weight_lds)
                  + warp_id * TC::GROUP_SIZE * K_STRIDE;
    }
    else
    {
        // 32c: 2 waves per group. wave_group selects group, wave_half selects C half.
        const int wave_group = warp_id / WavesPerGroup;
        const int wave_half = warp_id % WavesPerGroup;
        // wave_half * 16 offsets into the C dimension (C[0:15] or C[16:31]).
        warp_base = reinterpret_cast<ElementType*>(weight_lds)
                  + wave_group * TC::GROUP_SIZE * K_STRIDE
                  + wave_half * 16;
    }

    constexpr auto lds_desc = TC::Weight::template MakeLdsReadDescriptorDgrad<WavesPerGroup>();
    auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        warp_base, lds_desc);

    using DgradDist = decltype(TC::Weight::MakeLdsReadTileDistributionDgrad());
    constexpr DgradDist dgrad_dist = TC::Weight::MakeLdsReadTileDistributionDgrad();

    // Window dimensions match the descriptor: [GROUP_SIZE, GROUP_SIZE] for 16c,
    // [32, 16] for 32c (where each wave_half reads 32 K_out × 16 C).
    using VecType = typename std::remove_reference_t<WeightAccessorT>::value_type;
    // Distinguish 16c (4-element vec, 8 bytes) from 32c (8-element vec, 16 bytes).
    constexpr int WIN_DIM0 = (sizeof(VecType) == 4 * sizeof(ElementType))
                                 ? TC::GROUP_SIZE : 32;
    constexpr int WIN_DIM1 = (sizeof(VecType) == 4 * sizeof(ElementType))
                                 ? TC::GROUP_SIZE : 16;

    auto lds_window = ck_tile::make_tile_window(
        lds_view,
        ck_tile::make_tuple(ck_tile::number<WIN_DIM0>{},
                            ck_tile::number<WIN_DIM1>{}),
        {0, 0},
        dgrad_dist);

    // Derive the output (post-transpose) distribution from the input distribution.
    using InputDstrEncode = typename DgradDist::DstrEncode;
    using OutputDstrEncode = typename ck_tile::OutputTileDistributionTraits<
        InputDstrEncode, ElementType>::TransposedDstrEncode;
    auto out_tensor = ck_tile::make_static_distributed_tensor<ElementType>(
        ck_tile::make_static_tile_distribution(OutputDstrEncode{}));

    static_for<KH_KW>(
        [&]<int khw>()
        {
            // Offset selects filter position khw within the [K][KH*KW][C] LDS layout.
            // Each filter position is GROUP_SIZE fp16 elements apart.
            constexpr int filter_offset = khw * TC::GROUP_SIZE;

            ck_tile::load_tile_transpose_with_offset(
                out_tensor, lds_window, filter_offset);

            // For 16c: thread buffer has 4 fp16 -> fp16x4_t.
            // For 32c: thread buffer has 8 fp16 -> fp16x8_t (2 ds_read calls handled by Y dim).
            wa.weights[khw] = out_tensor.get_thread_buffer()
                                  .template get_as<VecType>(ck_tile::number<0>{});
        });
}

} // namespace direct_conv
} // namespace ck_tile
