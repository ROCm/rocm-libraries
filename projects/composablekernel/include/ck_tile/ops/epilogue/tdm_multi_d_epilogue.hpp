// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"

namespace ck_tile {

// TDM epilogue with fused D tensors (E = CDElementwise(C, D0, D1, ...)).
//
// This is a separate type from TdmEpilogue on purpose: TdmEpilogue keeps rejecting D tensors so
// that no existing TdmEpilogue user (grouped convolution, mx flatmm, ...) silently starts to
// accept Ds. The LDS layout, the LDS size and the TDM stores are the same as TdmEpilogueStaged;
// only the register stage differs: every D tile is loaded with the accumulator distribution (one
// slab of it when staged) and fused on the fp32 accumulator before the cast to ODataType.
//
// Restrictions:
//  - NumDTensor > 0 (use TdmEpilogue for plain GEMM),
//  - row-major E and row-major Ds,
//  - no ScaleM/ScaleN,
//  - output window must use memory_operation_enum::set (no split-K accumulation). The TDM store
//    always overwrites, so any other memory operation traps at run time. The inherited
//    SupportsSplitK == false keeps UniversalGemmKernel from instantiating its atomic-add
//    branch; a trap rather than a static_assert covers kernels that dispatch on k_batch at run
//    time without reading that marker.
//
// Row-broadcast Ds (bit i of DsRowBroadcastMask_ set): D_i is an N-vector added to every row, e.g.
// a bias, passed as a row-major D with row stride 0 (GemmKernelMultiD rejects any other stride).
// The accumulator layout gives each lane a strip of rows in one column, so a full-tile load of
// such a D issues one scalar load per element, all of the same few values. A row-broadcast D is
// instead read once, as the N-vector each wave needs, and broadcast over M in registers.
//
// The Problem is reused from CShuffleEpilogueProblem.
template <typename Problem_,
          index_t MIterPerStage_      = 0,
          index_t NumStageBuffers_    = 2,
          index_t DsRowBroadcastMask_ = 0>
struct TdmMultiDEpilogueStaged
    : public TdmEpilogueStaged<Problem_, MIterPerStage_, NumStageBuffers_>
{
    using Base          = TdmEpilogueStaged<Problem_, MIterPerStage_, NumStageBuffers_>;
    using Problem       = typename Base::Problem;
    using AccDataType   = typename Base::AccDataType;
    using ODataType     = typename Base::ODataType;
    using DsDataType    = typename Base::DsDataType;
    using DsLayout      = typename Base::DsLayout;
    using ELayout       = typename Base::ELayout;
    using CDElementwise = typename Base::CDElementwise;
    using EmptyScale    = typename Base::EmptyScale;

    static constexpr index_t kBlockSize = Base::kBlockSize;
    static constexpr index_t kMPerBlock = Base::kMPerBlock;
    static constexpr index_t kNPerBlock = Base::kNPerBlock;
    static constexpr index_t NumDTensor = Base::NumDTensor;

    // Marker used by kernels to check that a TDM pipeline with D tensors is paired with this
    // epilogue.
    static constexpr bool kIsTdmMultiDEpilogue = true;

    static_assert(NumDTensor > 0, "TdmMultiDEpilogue requires at least one D tensor");
    static_assert(NumDTensor == DsLayout::size() && NumDTensor == DsDataType::size(),
                  "DsLayout and DsDataType must have NumDTensor elements");
    static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                  "TdmMultiDEpilogue supports only row-major E");

    CK_TILE_HOST_DEVICE static constexpr bool AllDsRowMajor()
    {
        bool all_row = true;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
            all_row        = all_row && std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>;
        });
        return all_row;
    }
    static_assert(AllDsRowMajor(), "TdmMultiDEpilogue supports only row-major D tensors");

    // Bit i set: D_i is a row-broadcast N-vector (row stride 0); see the file comment.
    static constexpr index_t kDsRowBroadcastMask = DsRowBroadcastMask_;
    static_assert(kDsRowBroadcastMask >= 0 && kDsRowBroadcastMask < (index_t{1} << NumDTensor),
                  "DsRowBroadcastMask may only set bits of existing D tensors");

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr bool IsRowBroadcastD()
    {
        return ((kDsRowBroadcastMask >> I) & 1) != 0;
    }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM = EmptyScale,
              typename ScaleN = EmptyScale>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        ignore = scale_m;
        ignore = scale_n;
        static_assert(std::is_same_v<ScaleM, EmptyScale> && std::is_same_v<ScaleN, EmptyScale>,
                      "ScaleM and ScaleN must be EmptyScale with TdmMultiDEpilogue");
        static_assert(kBlockSize % get_warp_size() == 0, "BlockSize must be multiple of WarpSize");

        constexpr auto kOutMemOp =
            remove_cvref_t<decltype(out_dram_window.get_bottom_tensor_view())>::DstInMemOp;

        if constexpr(kOutMemOp != memory_operation_enum::set)
        {
            // The TDM store cannot accumulate; reaching this branch means a split-K launch that
            // the host side should have rejected.
            ignore = out_dram_window;
            ignore = o_acc_tile;
            ignore = ds_dram_windows;
            ignore = p_smem;
            __builtin_trap();
        }
        else if constexpr(Base::kStaged)
        {
            using AccDstr = remove_cvref_t<decltype(o_acc_tile.get_tile_distribution())>;
            constexpr auto stage_dstr =
                Base::template MakeStageDistribution<typename AccDstr::DstrEncode>();
            // Row-broadcast Ds are the same for every slab: read them once.
            const auto row_ds = load_row_broadcast_ds(ds_dram_windows, stage_dstr);

            this->run_staged(
                out_dram_window,
                o_acc_tile,
                p_smem,
                [&](const auto& stage_tile, auto& in_lds_window, auto stage) {
                    // D tiles are loaded per slab to keep their live ranges short.
                    // Rows past M are outside the buffer range and read as zero, as in the
                    // full-tile path.
                    const auto d_tiles = generate_tuple(
                        [&](auto i) {
                            if constexpr(IsRowBroadcastD<i.value>())
                                return row_ds[i];
                            else
                            {
                                const auto d_origin = ds_dram_windows[i].get_window_origin();
                                return load_tile(make_tile_window(
                                    ds_dram_windows[i].get_bottom_tensor_view(),
                                    make_tuple(number<Base::kStageRows>{}, number<kNPerBlock>{}),
                                    {d_origin[0] + stage.value * Base::kStageRows, d_origin[1]},
                                    stage_dstr));
                            }
                        },
                        number<NumDTensor>{});

                    fuse_and_store_lds(stage_tile, d_tiles, in_lds_window);
                });
        }
        else
        {
            // Same LDS layout and TDM store distribution as TdmEpilogue (E is row-major).
            auto o_lds_block =
                impl::tdm_row_major_lds_view<ODataType, kMPerBlock, kNPerBlock>(p_smem);

            const auto acc_dstr = o_acc_tile.get_tile_distribution();

            auto in_lds_window =
                make_tile_window(o_lds_block,
                                 make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                 {0, 0},
                                 acc_dstr);

            auto out_lds_window =
                make_tile_window(o_lds_block,
                                 make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                 {0, 0},
                                 impl::tdm_wave_linear_distr<kBlockSize, kMPerBlock, kNPerBlock>());

            TDMConfig tdm_config;

            // Wait for all outstanding TDM loads of the main loop before LDS is reused.
            s_wait_tensorcnt_barrier<0 /*tensor_cnt*/, 0 /*lgkmcnt*/>();

            // D tiles are loaded after the barrier to keep their live ranges short.
            const auto row_ds  = load_row_broadcast_ds(ds_dram_windows, acc_dstr);
            const auto d_tiles = generate_tuple(
                [&](auto i) {
                    if constexpr(IsRowBroadcastD<i.value>())
                        return row_ds[i];
                    else
                        return load_tile(make_tile_window(ds_dram_windows[i], acc_dstr));
                },
                number<NumDTensor>{});

            fuse_and_store_lds(o_acc_tile, d_tiles, in_lds_window);
            block_sync_lds();

            store_tile_tdm(tdm_config, out_dram_window, out_lds_window);
            // The store reads p_smem asynchronously; retire it before the caller reuses LDS.
            s_wait_tensorcnt();
        }
    }

    private:
    // For each row-broadcast D, the N-vector of this block's columns, in the distribution of
    // `acc_dstr` reduced over M: each lane holds the values of the columns it owns, and every lane
    // that owns a column reads it. Other Ds get a placeholder.
    template <typename DsDramWindows, typename AccDstr>
    CK_TILE_DEVICE static auto load_row_broadcast_ds(const DsDramWindows& ds_dram_windows,
                                                     const AccDstr&)
    {
        constexpr auto n_dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                typename remove_cvref_t<AccDstr>::DstrEncode{}, sequence<0>{}));
        return generate_tuple(
            [&](auto i) {
                if constexpr(IsRowBroadcastD<i.value>())
                {
                    // Every row of D_i is the same N-vector: read row 0. The view keeps the
                    // kernel's N padding, so columns past N read as zero as in the full path.
                    const auto& d_view = ds_dram_windows[i].get_bottom_tensor_view();
                    const auto n_view  = transform_tensor_view(
                        d_view,
                        make_tuple(make_freeze_transform(number<0>{}),
                                   make_pass_through_transform(
                                       d_view.get_tensor_descriptor().get_length(number<1>{}))),
                        make_tuple(sequence<0>{}, sequence<1>{}),
                        make_tuple(sequence<>{}, sequence<0>{}));
                    return load_tile(make_tile_window(n_view,
                                                      make_tuple(number<kNPerBlock>{}),
                                                      {ds_dram_windows[i].get_window_origin()[1]},
                                                      n_dstr));
                }
                else
                    return number<0>{};
            },
            number<NumDTensor>{});
    }

    // E = CDElementwise(C, D0, D1, ...), written to LDS as ODataType.
    template <typename OAccTile, typename DTiles, typename InLdsWindow>
    CK_TILE_DEVICE static void fuse_and_store_lds(const OAccTile& o_acc_tile,
                                                  const DTiles& d_tiles,
                                                  InLdsWindow& in_lds_window)
    {
        auto e_tile = make_static_distributed_tensor<ODataType>(o_acc_tile.get_tile_distribution());

        if constexpr(kDsRowBroadcastMask == 0)
        {
            tile_elementwise_inout_unpack(
                CDElementwise{},
                concat_tuple_of_reference(
                    tie(e_tile, o_acc_tile),
                    generate_tie([&](auto i) -> const auto& { return d_tiles[i]; },
                                 number<NumDTensor>{})));
        }
        else
        {
            // A row-broadcast D is indexed by the column alone.
            constexpr auto spans = OAccTile::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j = make_tuple(idx0, idx1);
                    constexpr auto j   = make_tuple(idx1);
                    unpack(
                        [&](const auto&... ds) {
                            CDElementwise{}(e_tile(i_j), o_acc_tile[i_j], ds...);
                        },
                        generate_tuple(
                            [&](auto i) {
                                if constexpr(IsRowBroadcastD<i.value>())
                                    return d_tiles[i][j];
                                else
                                    return d_tiles[i][i_j];
                            },
                            number<NumDTensor>{}));
                });
            });
        }

        store_tile(in_lds_window, e_tile);
    }
};

// Full-tile TdmMultiDEpilogue
template <typename Problem_, index_t DsRowBroadcastMask_ = 0>
using TdmMultiDEpilogue = TdmMultiDEpilogueStaged<Problem_, 0, 2, DsRowBroadcastMask_>;

// TdmMultiDEpilogueStaged with as many slab buffers as fit in GemmPipeline_'s LDS.
// Takes the pipeline type for the reason given at TdmEpilogueStagedForPipeline.
template <typename Problem_,
          index_t MIterPerStage_,
          typename GemmPipeline_,
          index_t DsRowBroadcastMask_ = 0>
struct TdmMultiDEpilogueStagedForPipeline
    : public TdmMultiDEpilogueStaged<Problem_,
                                     MIterPerStage_,
                                     tdm_stage_buffers_for_lds<Problem_, MIterPerStage_>(
                                         GemmPipeline_::GetSmemSize()),
                                     DsRowBroadcastMask_>
{
};

} // namespace ck_tile
