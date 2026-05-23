// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

/// @brief Implementation base for WCNN forward pipeline — provides LDS/DRAM window helpers.
///
/// Analogous to GemmPipelineAgBgCrImplBase but adapted for WCNN:
///   - No col-major / row-major / load-transpose logic
///   - A (input) DRAM and LDS windows are 2D [HW, C]
///   - B (weight) DRAM and LDS windows are 2D [KYX, C]
///   - No tuple overloads (WCNN has single input/weight, not multi-source)
///
/// @tparam Problem  WcnnFwdPipelineProblem type.
/// @tparam Policy   Policy class providing LDS descriptors and tile distributions.
template <typename Problem, typename Policy>
struct WcnnPipelineImplBase
{
    using ADataType = typename Problem::ADataType;
    using BDataType = typename Problem::BDataType;

    static constexpr index_t HPerBlock = Problem::BlockWcnnShape::HPerBlock;
    static constexpr index_t WPerBlock = Problem::BlockWcnnShape::WPerBlock;
    static constexpr index_t CPerBlock = Problem::BlockWcnnShape::CPerBlock;
    static constexpr index_t KPerBlock = Problem::BlockWcnnShape::KPerBlock;
    static constexpr index_t HPerWcnn  = Problem::BlockWcnnShape::HPerWcnn;
    static constexpr index_t WPerWcnn  = Problem::BlockWcnnShape::WPerWcnn;
    static constexpr index_t FilterY   = Problem::FilterY;
    static constexpr index_t FilterX   = Problem::FilterX;
    static constexpr index_t BlockSize = Problem::BlockWcnnShape::BlockSize;

    static constexpr index_t HWPerBlock  = HPerBlock * WPerBlock;
    static constexpr index_t YXPerBlock  = FilterY * FilterX;
    static constexpr index_t KYXPerBlock = KPerBlock * YXPerBlock;

    using BlockWcnn = remove_cvref_t<decltype(Policy::template GetBlockWcnn<Problem>())>;

    // ======================== LocalPrefill / LocalPrefetch ========================

    template <typename DstTileWindow, typename SrcBlockTile>
    CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                     const SrcBlockTile& src_block_tile) const
    {
        store_tile(lds_tile_window, src_block_tile);
    }

    // ======================== GetABLdsTensorView ========================

    CK_TILE_DEVICE auto GetABLdsTensorView(void* p_smem) const
    {
        // A (input) tile in LDS
        ADataType* __restrict__ p_a_lds = static_cast<ADataType*>(p_smem);
        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned = integer_least_multiple(
            sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16);

        // B (weight) tile in LDS
        BDataType* __restrict__ p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        return make_tuple(std::move(a_lds_block), std::move(b_lds_block));
    }

    // ======================== CopyADramWindow ========================

    template <typename ADramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    CopyADramWindow(const ADramBlockWindowTmp& a_dram_block_window_tmp) const
    {
        // Create DRAM copy window with tile distribution for cooperative thread loading
        // Input block window: [HW, C] (2D, H and W merged by kernel)
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<HWPerBlock>{}, number<CPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        return a_copy_dram_window;
    }

    // ======================== MakeALdsWindows ========================

    template <typename ALdsTensorView, typename ALdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto MakeALdsWindows(const ALdsTensorView& a_lds_block_view,
                                                  const ALdsLoadTileDistr&) const
    {
        // LDS descriptor is 2D [HW, C] (H and W merged for LDS layout)
        auto a_copy_lds_window = make_tile_window(
            a_lds_block_view, make_tuple(number<HWPerBlock>{}, number<CPerBlock>{}), {0, 0});

        // The LDS data is stored in interleaved order matching MakeABlockWindow:
        // HW = (h_sub, w_sub, h_wcnn, w_wcnn) where
        //   h_sub = HPerBlock/HPerWcnn, w_sub = WPerBlock/WPerWcnn
        // We reconstruct the 3D (H, W, C) view accounting for this interleaving.
        constexpr index_t HSubTiles = HPerBlock / HPerWcnn;
        constexpr index_t WSubTiles = WPerBlock / WPerWcnn;

        // Step 1: Unmerge HW into (h_sub, w_sub, h_wcnn, w_wcnn)
        const auto a_lds_5d_view =
            transform_tensor_view(a_lds_block_view,
                                  make_tuple(make_unmerge_transform(make_tuple(number<HSubTiles>{},
                                                                               number<WSubTiles>{},
                                                                               number<HPerWcnn>{},
                                                                               number<WPerWcnn>{})),
                                             make_pass_through_transform(number<CPerBlock>{})),
                                  make_tuple(sequence<0>{}, sequence<1>{}),
                                  make_tuple(sequence<0, 1, 2, 3>{}, sequence<4>{}));

        // Step 2: Merge (h_sub, h_wcnn) → H and (w_sub, w_wcnn) → W
        const auto a_lds_block_view_3d = transform_tensor_view(
            a_lds_5d_view,
            make_tuple(make_merge_transform(make_tuple(number<HSubTiles>{}, number<HPerWcnn>{})),
                       make_merge_transform(make_tuple(number<WSubTiles>{}, number<WPerWcnn>{})),
                       make_pass_through_transform(number<CPerBlock>{})),
            make_tuple(sequence<0, 2>{}, sequence<1, 3>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        // LDS compute window with distribution for warp conv (3D)
        auto a_lds_wcnn_window = make_tile_window(
            a_lds_block_view_3d,
            make_tuple(number<HPerBlock>{}, number<WPerBlock>{}, number<CPerBlock>{}),
            {0, 0, 0},
            ALdsLoadTileDistr{});

        return make_tuple(std::move(a_copy_lds_window), std::move(a_lds_wcnn_window));
    }

    // ======================== GetAWindows ========================

    template <typename ADramBlockWindowTmp, typename ALdsTensorView, typename ALdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto GetAWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                              const ALdsTensorView& a_lds_block_view,
                                              const ALdsLoadTileDistr& a_lds_load_tile_distr) const
    {
        auto a_copy_dram_window = CopyADramWindow(a_dram_block_window_tmp);

        auto [a_copy_lds_window, a_lds_wcnn_window] =
            MakeALdsWindows(a_lds_block_view, a_lds_load_tile_distr);

        return make_tuple(std::move(a_copy_dram_window),
                          std::move(a_copy_lds_window),
                          std::move(a_lds_wcnn_window));
    }

    // ======================== CopyBDramWindow ========================

    template <typename BDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    CopyBDramWindow(const BDramBlockWindowTmp& b_dram_block_window_tmp) const
    {
        // Weight block window: [KYX, C] (2D, K and YX merged by kernel)
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<KYXPerBlock>{}, number<CPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        return b_copy_dram_window;
    }

    // ======================== MakeBLdsWindows ========================

    template <typename BLdsTensorView, typename BLdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto MakeBLdsWindows(const BLdsTensorView& b_lds_block_view,
                                                  const BLdsLoadTileDistr&) const
    {
        // LDS descriptor is 2D [KYX, C] (K, Y, X merged for LDS layout)
        auto b_copy_lds_window = make_tile_window(
            b_lds_block_view, make_tuple(number<KYXPerBlock>{}, number<CPerBlock>{}), {0, 0});

        // Transform 2D [KYX, C] → 3D [K, YX, C] by unmerging KYX dimension
        const auto b_lds_block_view_3d = transform_tensor_view(
            b_lds_block_view,
            make_tuple(
                make_unmerge_transform(make_tuple(number<KPerBlock>{}, number<YXPerBlock>{})),
                make_pass_through_transform(number<CPerBlock>{})),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0, 1>{}, sequence<2>{}));

        // LDS compute window with distribution for warp conv
        auto b_lds_wcnn_window = make_tile_window(
            b_lds_block_view_3d,
            make_tuple(number<KPerBlock>{}, number<YXPerBlock>{}, number<CPerBlock>{}),
            {0, 0, 0},
            BLdsLoadTileDistr{});

        return make_tuple(std::move(b_copy_lds_window), std::move(b_lds_wcnn_window));
    }

    // ======================== GetBWindows ========================

    template <typename BDramBlockWindowTmp, typename BLdsTensorView, typename BLdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto GetBWindows(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                              const BLdsTensorView& b_lds_block_view,
                                              const BLdsLoadTileDistr& b_lds_load_tile_distr) const
    {
        auto b_copy_dram_window = CopyBDramWindow(b_dram_block_window_tmp);

        auto [b_copy_lds_window, b_lds_wcnn_window] =
            MakeBLdsWindows(b_lds_block_view, b_lds_load_tile_distr);

        return make_tuple(std::move(b_copy_dram_window),
                          std::move(b_copy_lds_window),
                          std::move(b_lds_wcnn_window));
    }
};

} // namespace ck_tile
