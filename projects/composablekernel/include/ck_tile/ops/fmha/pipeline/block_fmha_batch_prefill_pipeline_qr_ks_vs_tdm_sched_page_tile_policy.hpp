// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_policy.hpp"

namespace ck_tile {

namespace detail {
// KV-split partial outputs: FP32 output planes that a merge kernel combines.
template <typename Problem>
inline constexpr bool fmha_tdm_is_split_partial_v =
    std::is_same_v<remove_cvref_t<typename Problem::ODataType>, float>;
} // namespace detail

// Policy of the gfx1250 batch-prefill page-tile pipeline: the scheduled-TDM policy and the
// decisions that only the paged pipeline makes. The paged pipeline applies a bias, ALiBi, a logits
// soft cap and sinks itself, so the problem needs only this policy's geometry.
template <typename Geometry_,
          typename Tuning_,
          typename TileSchedule_ = FmhaTdmSchedDefaultScheduleFor<Geometry_>>
struct BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTilePolicy
    : BlockFmhaPipelineQRKSVSTdmSchedPolicy<Geometry_, Tuning_, TileSchedule_>
{
    using Base     = BlockFmhaPipelineQRKSVSTdmSchedPolicy<Geometry_, Tuning_, TileSchedule_>;
    using Geometry = typename Base::Geometry;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedProblem()
    {
        return Base::template IsSupportedGeometryProblem<Problem>();
    }

    // Steady-loop shape for D64/V64, N128, one page per wave slice (P >= 64), no sink and no bias.
    // It is used for every grid; the benefit is largest with one wave per SIMD (grids of at
    // most one workgroup per CU), where every scalar stall in the tile loop is on the critical
    // path. This configuration
    //  - uses the countdown loop (fewer long-lived scalars),
    //  - forms every LDS read base of a tile once, before the WMMA streams (PrepareTileBases), and
    //  - waits for the next K tile at the barrier in load_next_k, which publishes it before
    //    its first reader, the last PV stage, instead of a separate wait + barrier after QK.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseLeanD64Loop()
    {
        return Geometry::kHeadDimQK == 64 && Geometry::kDv == 64 && Geometry::kN == 128 &&
               Problem::kPageBlockSize >= 64 && !Problem::kHasSink &&
               Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCountdownLoop()
    {
        constexpr bool kCountdown =
            Base::template UseCountdownLoop<Problem>() || UseLeanD64Loop<Problem>();
        // The countdown loop keeps no tile index, which ALiBi and the sink tiles read.
        static_assert(!kCountdown ||
                      (!Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS));
        return kCountdown;
    }

    // The lean D64 loop waits for the K tile of the previous iteration in load_next_k, after the
    // softmax, instead of after QK (or, with DeferTensorReady, inside the softmax).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool WaitTensorReadyAtLoadNextK()
    {
        return UseLeanD64Loop<Problem>() && !Base::template DeferTensorReady<Problem>();
    }

    // Lean TDM issue (per-wave descriptor inputs derived once, see make_lean_issue_state): the D64
    // lean loop, and the vectorized P64 D128 tiles, which take only the issue path and keep the
    // D128 general loop.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseLeanTdmIssue()
    {
        return UseLeanD64Loop<Problem>() ||
               (detail::fmha_tdm_is_vectorized_kv<Problem>::value && Geometry::kHeadDimQK == 128 &&
                Geometry::kDv == 128 && Geometry::kN == 128 && Problem::kPageBlockSize == 64 &&
                !Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
    }

    // Mask predicate of a score at (row, col). The unsigned forms make one unsigned compare per
    // element instead of three compares joined by SALU ORs of lane masks. The causal mask also
    // skips the row-validity test: rows at or past seqlen_q are never written out (the O view is
    // bounded by seqlen_q), so their scores may be masked arbitrarily; not for P16 D128, where it
    // raises SGPR spills (32 -> 75) and is ~0.5% slower overall.
    template <typename Problem, typename Mask>
    CK_TILE_DEVICE static bool IsMaskedOut(const Mask& mask, index_t row, index_t col)
    {
        constexpr bool kCausal = std::is_same_v<Mask, SimplifiedGenericAttentionMask<true>>;
        if constexpr(Problem::kHasSink)
            return mask.IsOutOfSinkBound(row, col);
        else if constexpr(kCausal &&
                          !(Problem::kPageBlockSize == 16 && Geometry::kHeadDimQK == 128))
            return mask.IsOutOfBoundUnsignedStoredRow(row, col);
        else if constexpr(kCausal && Base::kUseUnsignedCausalMask)
            return mask.IsOutOfBoundUnsigned(row, col);
        else
            return mask.IsOutOfBound(row, col);
    }

    // LDS read bases of one tile: QK stage 0's next K access (formed by the pipeline), and in the
    // lean D64 loop the QK tail stage's V access 0 and the last PV stage's next-K access 0.
    struct TileBases
    {
        FmhaTdmPreparedKRead::Address qk_stage0{};
        FmhaTdmPreparedKRead::Address qk_v0{};
        FmhaTdmPreparedKRead::Address pv_k0{};
    };

    // Forms the policy's bases of the tile, called with qk_stage0 formed: in the lean D64 loop,
    // from the V read window and the next K tile (next_k_ptr), ahead of every WMMA, and keeps each
    // in a VGPR: otherwise the compiler recomputes them next to their first read, where the
    // VALU-write -> LDS-read wait must also drain the WMMA issued in between.
    template <typename Problem, typename KLdsView, typename KPointer, typename VWindow>
    CK_TILE_DEVICE static void PrepareTileBases(TileBases& bases,
                                                const KLdsView& k_lds_view,
                                                KPointer next_k_ptr,
                                                const VWindow& v_window)
    {
        if constexpr(UseLeanD64Loop<Problem>())
        {
            bases.qk_v0 = FmhaTdmPreparedKRead::Prepare<0>(v_window);
            auto pv_k_window =
                make_tile_window(k_lds_view,
                                 make_tuple(number<Geometry::kQkSuColumns>{},
                                            number<Problem::BlockFmhaShape::kQKHeaddim>{}),
                                 {0, 0},
                                 Base::template MakeKSuRegTileDistribution<Problem>());
            pv_k_window.set_bottom_tensor_view_data_ptr(next_k_ptr);
            bases.pv_k0 = FmhaTdmPreparedKRead::Prepare<0>(pv_k_window);
            asm volatile("" : "+v"(bases.qk_stage0.byte_address));
            asm volatile("" : "+v"(bases.qk_v0.byte_address));
            asm volatile("" : "+v"(bases.pv_k0.byte_address));
        }
    }

    // QK stage with the tile's bases: the earlier stages read the next K tile from qk_stage0; the
    // lean D64 loop's tail stage reads V from qk_v0.
    template <index_t Stage,
              typename Mode,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow>
    CK_TILE_DEVICE static void RunQkStage(const BlockGemm& gemm,
                                          CBlockTensor& c_block_tensor,
                                          const ABlockTensor& a_block_tensor,
                                          const BBlockTensor& b_block_tensor,
                                          NextBBlockTensor& next_b_block_tensor,
                                          const BTileWindow& b_lds_window,
                                          const TileBases& bases)
    {
        if constexpr(Stage < Geometry::kQkStages - 1)
            Base::template RunQkScheduledStage<Stage, Mode>(gemm,
                                                            c_block_tensor,
                                                            a_block_tensor,
                                                            b_block_tensor,
                                                            next_b_block_tensor,
                                                            b_lds_window,
                                                            &bases.qk_stage0);
        else if constexpr(UseLeanD64Loop<typename Mode::SelectedProblem>())
            Base::template RunQkScheduledStage<Stage, Mode>(gemm,
                                                            c_block_tensor,
                                                            a_block_tensor,
                                                            b_block_tensor,
                                                            next_b_block_tensor,
                                                            b_lds_window,
                                                            &bases.qk_v0,
                                                            bool_constant<true>{});
        else
            Base::template RunQkScheduledStage<Stage, Mode>(gemm,
                                                            c_block_tensor,
                                                            a_block_tensor,
                                                            b_block_tensor,
                                                            next_b_block_tensor,
                                                            b_lds_window);
    }

    // PV stage with the tile's bases. In the lean D64 loop, V stage s reads rows
    // [(s+1)*kK1, (s+2)*kK1) of the buffer the QK tail stage started on (the row offset folds into
    // the ds_load immediates), and the last stage reads the next K tile from pv_k0.
    template <index_t Stage,
              typename Mode,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename WmmaEmitter,
              typename RescaleEmitter>
    CK_TILE_DEVICE static void RunPvStage(NextBBlockTensor& next_b_block_tensor,
                                          const BTileWindow& b_lds_window,
                                          WmmaEmitter& emit_wmma,
                                          RescaleEmitter& emit_rescale,
                                          const TileBases& bases)
    {
        using Problem = typename Mode::SelectedProblem;
        if constexpr(UseLeanD64Loop<Problem>())
        {
            using VDataType                    = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t kRow             = (Stage + 1) * Problem::BlockFmhaShape::kK1;
            FmhaTdmPreparedKRead::Address base = bases.pv_k0;
            if constexpr(Stage < Geometry::kPvStages - 1)
            {
                base = bases.qk_v0;
                if constexpr(detail::fmha_tdm_is_vectorized_kv<Problem>::value)
                {
                    // [rows/8][D][8] V tile: the row offset from the V LDS descriptor.
                    constexpr index_t kRowOffset =
                        Base::template MakeVLdsBlockDescriptor<Problem>().calculate_offset(
                            make_multi_index(kRow, 0));
                    base.byte_address += kRowOffset * sizeof(VDataType);
                }
                else
                    base.byte_address += kRow * Base::kVPhysicalStride * sizeof(VDataType);
            }
            Base::template RunPvScheduledStageWithWmma<Stage, Mode>(next_b_block_tensor,
                                                                    b_lds_window,
                                                                    emit_wmma,
                                                                    emit_rescale,
                                                                    &base,
                                                                    bool_constant<true>{});
        }
        else
            Base::template RunPvScheduledStageWithWmma<Stage, Mode>(
                next_b_block_tensor, b_lds_window, emit_wmma, emit_rescale);
    }
};

// QK column group of a page-tile problem: D64 M64 N128 tiles with K64 PV stages (block tile
// kK1 == 64) walk QK in 64-column groups as well; every other problem uses two WMMA column tiles.
template <typename Problem, typename Shape = remove_cvref_t<typename Problem::BlockFmhaShape>>
inline constexpr index_t kFmhaBatchPrefillPageTileQkSuColumnsFor =
    (Shape::kM0 == 64 && Shape::kN0 == 128 && Shape::kQKHeaddim == 64 && Shape::kN1 == 64 &&
     Shape::kK1 == 64)
        ? 64
        : kFmhaTdmSchedDefaultQkSuColumns<typename Shape::Gemm0WarpTile>;

template <typename Problem>
using FmhaBatchPrefillPageTileGeometryFor =
    FmhaTdmSchedGeometryFor<Problem, kFmhaBatchPrefillPageTileQkSuColumnsFor<Problem>>;

// Code-generation options per problem, as measured on gfx1250 (BF16, vectorized-KV batch-prefill
// page tile, dense and causal):
//  - M128 tiles (P64/P128 x D64/D128): all four (scalar row clamp 1);
//  - P128 D64 M64 tiles: the v_maximum3 row max and the scalar row clamp 2 (3.5 % less time than
//    without them; clamp 1 recovers 3 %). Not for P64 D64 M64, where the earlier TDM issue showed a
//    stale-fragment event (see kDrainLdsBeforeTdmIssue), nor for D128 M64 (about 1 %);
//  - the defaults for every other problem, which keeps their code unchanged, including the KV-split
//    partial outputs (the split routes were measured without the options).
template <typename Problem, typename Geometry = FmhaBatchPrefillPageTileGeometryFor<Problem>>
using FmhaBatchPrefillPageTileOptionsFor = std::conditional_t<
    detail::fmha_tdm_is_vectorized_kv<Problem>::value &&
        !detail::fmha_tdm_is_split_partial_v<Problem> && Geometry::kM == 128,
    FmhaTdmSchedTuningOptions<true, true, true, 1>,
    std::conditional_t<detail::fmha_tdm_is_vectorized_kv<Problem>::value &&
                           !detail::fmha_tdm_is_split_partial_v<Problem> && Geometry::kM == 64 &&
                           Geometry::kHeadDimQK == 64 && Problem::kPageBlockSize == 128,
                       FmhaTdmSchedTuningOptions<false, false, true, 2>,
                       FmhaTdmSchedTuningOptions<>>>;

template <typename Problem>
using FmhaBatchPrefillPageTilePolicyFor = BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTilePolicy<
    FmhaBatchPrefillPageTileGeometryFor<Problem>,
    FmhaTdmSchedDefaultTuning<FmhaBatchPrefillPageTileGeometryFor<Problem>,
                              FmhaBatchPrefillPageTileOptionsFor<Problem>>>;

} // namespace ck_tile
