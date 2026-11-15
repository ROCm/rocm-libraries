// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_policy.hpp"

namespace ck_tile {

namespace detail {
// KV-split partial outputs: FP32 output planes that a merge kernel combines. The page-tile kernel
// has an FP32 output exactly when it is a KV split (FmhaBatchPrefillQrTdmPageTileKernel).
template <typename Problem>
inline constexpr bool fmha_tdm_is_split_partial_v =
    std::is_same_v<remove_cvref_t<typename Problem::ODataType>, float>;

// D128/V128 N128 page tiles of 64-token or larger pages, without sink or bias, other than the
// KV-split partial kernels: the policy forms their V and next-K read bases once per tile
// (UsePreparedTileBases) and, with the WMMA operand guard, reads PV stage 0's V in the softmax
// (FmhaBatchPrefillPageTileScheduleFor). The split partial kernels keep their schedule: with
// both, the causal M128 P64 split ran 18 % longer on the gtest shapes.
template <typename Problem, typename Geometry>
inline constexpr bool fmha_page_tile_d128_prepared_bases_v =
    Geometry::kHeadDimQK == 128 && Geometry::kDv == 128 && Geometry::kN == 128 &&
    Problem::kPageBlockSize >= 64 && !Problem::kHasSink &&
    Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS && !fmha_tdm_is_split_partial_v<Problem>;
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

    // The tile's V and next-K read bases are formed once, ahead of every WMMA (PrepareTileBases):
    // the lean D64 loop, and the D128/V128 N128 tiles of pages of 64 tokens or more, without sink
    // or bias. Formed next to their first reads instead, each base is a VALU write between WMMAs
    // and the read that uses it must drain the WMMA pipe (s_wait_alu depctr_va_vdst(0)): six such
    // drains per KV tile in the D128 P64 vectorized loop, 3-5% of its time on small grids.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UsePreparedTileBases()
    {
        return UseLeanD64Loop<Problem>() ||
               detail::fmha_page_tile_d128_prepared_bases_v<Problem, Geometry>;
    }

    // LDS read bases of one tile: QK stage 0's next K access (formed by the pipeline), and with
    // UsePreparedTileBases the QK tail stage's V access 0 and the last PV stage's next-K access 0.
    struct TileBases
    {
        FmhaTdmPreparedKRead::Address qk_stage0{};
        FmhaTdmPreparedKRead::Address qk_v0{};
        FmhaTdmPreparedKRead::Address pv_k0{};
    };

    // Forms the policy's bases of the tile, called with qk_stage0 formed: with
    // UsePreparedTileBases, from the V read window and the next K tile (next_k_ptr), ahead of every
    // WMMA, and keeps each in a VGPR: otherwise the compiler recomputes them next to their first
    // read, where the VALU-write -> LDS-read wait must also drain the WMMA issued in between.
    template <typename Problem, typename KLdsView, typename KPointer, typename VWindow>
    CK_TILE_DEVICE static void PrepareTileBases(TileBases& bases,
                                                const KLdsView& k_lds_view,
                                                KPointer next_k_ptr,
                                                const VWindow& v_window)
    {
        if constexpr(UsePreparedTileBases<Problem>())
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

    // PV stage 0's V fragments (V0) are read in the softmax when the schedule's QK tail stage reads
    // nothing (FmhaBatchPrefillPageTileScheduleFor): after that stage's RAW fence, with no WMMA in
    // flight, so they need no drain of the WMMA operand guard.
    static constexpr bool kReadsV0InSoftmax =
        !kFmhaTdmSchedQkTailReads<typename Base::Schedule::LocalPlacement>;

    // V0, from the tile's prepared V base, at the start of the softmax's second part (after the row
    // maxima): its loads overlap the exponentials.
    template <typename Problem, typename VTensor, typename VWindow>
    CK_TILE_DEVICE static void
    LoadV0InSoftmax(VTensor& v_tile, const VWindow& v_window, const TileBases& bases)
    {
        static_assert(kReadsV0InSoftmax && UsePreparedTileBases<Problem>());
        __builtin_amdgcn_sched_barrier(0);
        static_for<0, Geometry::kVStageLoadCount, 1>{}([&](auto access) {
            FmhaTdmAffineVRead::Load<decltype(access)::value, Base::kVPhysicalStride>(
                v_tile, v_window, bases.qk_v0);
        });
    }

    // Guarded row max (FA4-style) for the BF16 D128 tiles with prepared tile bases and no logits
    // soft cap. The row max is reduced only while a row of the wave has no finite base: on the
    // first tile, and on the tiles after one in which a row had no visible key (a sliding window's
    // left edge). Later tiles weight their scores against the stored base, so they reduce no row
    // max and rescale no output. After PV, a row whose weight sum of the tile exceeds 2^8 is
    // rebased by that sum, which bounds every weight of the row in the tile (weights are not
    // negative): output, row sum and base. The guard runs after the P conversion, so P must keep
    // the FP32 exponent range (BF16, not FP16); a weight past ~2^110 above the base in one tile (a
    // score jump of ~76 natural-log units) still overflows the weights or the output. The other
    // tiles keep the per-tile row max: in the vectorized layout their inline-asm operands (the WMMA
    // operand guard's kept operands, the page-id queue's loads) fill the low 256 VGPRs, and the
    // guarded max spills 16-32 VGPRs there (PV stage 0's V where the QK tail stage reads it, the
    // page-id queue's operands with a soft cap).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseGuardedRowMax()
    {
        return Base::kEarlyOutputRescale &&
               std::is_same_v<remove_cvref_t<typename Problem::PDataType>, bf16_t> &&
               detail::fmha_page_tile_d128_prepared_bases_v<Problem, Geometry> &&
               !Problem::kHasLogitsSoftCap;
    }

    // P>=64 page ids in VGPR lanes (load_page_id_batch of the page-tile window: one vector load
    // per 32 tiles instead of one LDS publication per tile) for the tiles without sink tokens, KV
    // split or logits soft cap. Sink tiles are not contiguous; the two batch VGPRs made the
    // KV-split partial P64 D128 M128 tiles spill (their inline-asm operands fill the low 256 VGPRs)
    // and took the soft-cap linear P128 D128 M128 tiles past 512 VGPRs (one wave per SIMD).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UsePageIdBatch()
    {
        return Problem::kPageBlockSize >= 64 && !Problem::kHasSink &&
               !detail::fmha_tdm_is_split_partial_v<Problem> && !Problem::kHasLogitsSoftCap;
    }

    // Masked tiles whose KV loop forms the mask's interior key-tile range once per query tile and
    // drops the split tail predicate (see the pipeline): the tiles without sink tokens, KV split or
    // logits soft cap. A KV-split partial's key range ends at the split's end, which the mask does
    // not cover; the soft-cap D128 M128 tiles spill VGPRs with these trims (their inline-asm
    // operands fill the low 256 VGPRs), and the P16 D128 ALiBi sink tiles spill more.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseMaskLoopTrims()
    {
        return !Problem::kHasSink && !detail::fmha_tdm_is_split_partial_v<Problem> &&
               !Problem::kHasLogitsSoftCap;
    }

    static constexpr float kGuardedRowSumBound = 256.0f;

    // Guarded row max, max path: the row max (Part01), then the output and the row sums rescaled
    // by the change of the max.
    template <typename Problem, typename ScoreTensor, typename RowTensor, typename Fragments>
    CK_TILE_DEVICE static void RunGuardedRowMaxPart01(ScoreTensor& score,
                                                      RowTensor& row_max,
                                                      RowTensor& row_sum,
                                                      Fragments& output,
                                                      float log2e_scale)
    {
        float delta_m0;
        float delta_m1;
        Base::template RunSplitSoftmaxPart01<Problem>(
            score, row_max, log2e_scale, delta_m0, delta_m1);
        const float scale_m0 = Base::ScoreSoftmax::Exp2(delta_m0);
        const float scale_m1 = Base::ScoreSoftmax::Exp2(delta_m1);
        static_for<0, Base::OutputFragments::kNumFragments, 1>{}([&](auto ordinal) {
            Base::template RescaleOutputFragment<decltype(ordinal)::value>(
                output, scale_m0, scale_m1);
        });
        auto& sums = row_sum.get_thread_buffer();
        sums[number<0>{}] *= scale_m0;
        if constexpr(Base::ScoreMapping::kNumMIter == 2)
            sums[number<1>{}] *= scale_m1;
    }

    // Guarded row max, weights: exp2 of the scores against the stored base (0 for a row without a
    // finite base, whose weights are 0); the tile's row sums are added to the row sums and returned
    // for the guard. The sums are merged across the lane pair every tile: the permlane is
    // convergent, so the pre-ISel IR sinking cannot move the sums, nor the weights they read, out
    // of the softmax to their first use after PV (128 FP32 weights live across PV take the D128
    // M128 tiles past 512 VGPRs).
    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static array<float, 2> RunGuardedRowMaxPart2(ScoreTensor& score,
                                                                const RowTensor& row_max,
                                                                RowTensor& row_sum,
                                                                float log2e_scale)
    {
        using Softmax      = typename Base::ScoreSoftmax;
        const auto base_of = [](float value) {
            return value <= -numeric<float>::infinity() ? 0.0f : value;
        };
        const auto& maxes = row_max.get_thread_buffer();
        auto& sums        = row_sum.get_thread_buffer();
        array<float, 2> tile_sums{0.0f, 0.0f};
        const float base_m0 = base_of(maxes[number<0>{}]);
        tile_sums(0)        = Softmax::MergeRowSum(
            Softmax::template RunPart2LocalSum<0>(score, base_m0, log2e_scale),
            Softmax::template RunPart2LocalSum<1>(score, base_m0, log2e_scale));
        sums[number<0>{}] += tile_sums[0];
        if constexpr(Base::ScoreMapping::kNumMIter == 2)
        {
            const float base_m1 = base_of(maxes[number<1>{}]);
            tile_sums(1)        = Softmax::MergeRowSum(
                Softmax::template RunPart2LocalSum<2>(score, base_m1, log2e_scale),
                Softmax::template RunPart2LocalSum<3>(score, base_m1, log2e_scale));
            sums[number<1>{}] += tile_sums[1];
        }
        return tile_sums;
    }

    // Guarded row max, after PV: rebases the rows whose weight sum of the tile exceeds the bound.
    // Returns whether a row of the wave still has no finite base, so the next tile reduces the row
    // max (once every row has one, it keeps it).
    template <typename Problem, typename RowTensor, typename Fragments>
    CK_TILE_DEVICE static bool RunGuardedRowMaxGuard(RowTensor& row_max,
                                                     RowTensor& row_sum,
                                                     Fragments& output,
                                                     const array<float, 2>& tile_sums,
                                                     float log2e_scale,
                                                     bool base_pending)
    {
        auto& maxes = row_max.get_thread_buffer();
        auto& sums  = row_sum.get_thread_buffer();
        const bool unsafe =
            !(tile_sums[0] <= kGuardedRowSumBound) || !(tile_sums[1] <= kGuardedRowSumBound);
        if(__builtin_amdgcn_ballot_w32(unsafe) != 0)
        {
            const auto base_of = [](float value) {
                return value <= -numeric<float>::infinity() ? 0.0f : value;
            };
            const float bound_m0 = tile_sums[0];
            const float bound_m1 = tile_sums[1];
            const bool rebase_m0 = !(bound_m0 <= kGuardedRowSumBound);
            const bool rebase_m1 = !(bound_m1 <= kGuardedRowSumBound);
            const float scale_m0 = rebase_m0 ? 1.0f / bound_m0 : 1.0f;
            const float scale_m1 = rebase_m1 ? 1.0f / bound_m1 : 1.0f;
            static_for<0, Base::OutputFragments::kNumFragments, 1>{}([&](auto ordinal) {
                Base::template RescaleOutputFragment<decltype(ordinal)::value>(
                    output, scale_m0, scale_m1);
            });
            sums[number<0>{}] *= scale_m0;
            if(rebase_m0)
                maxes[number<0>{}] =
                    base_of(maxes[number<0>{}]) + __builtin_amdgcn_logf(bound_m0) / log2e_scale;
            if constexpr(Base::ScoreMapping::kNumMIter == 2)
            {
                sums[number<1>{}] *= scale_m1;
                if(rebase_m1)
                    maxes[number<1>{}] =
                        base_of(maxes[number<1>{}]) + __builtin_amdgcn_logf(bound_m1) / log2e_scale;
            }
        }
        if(!base_pending)
            return false;
        bool no_base = maxes[number<0>{}] <= -numeric<float>::infinity();
        if constexpr(Base::ScoreMapping::kNumMIter == 2)
            no_base = no_base || maxes[number<1>{}] <= -numeric<float>::infinity();
        return __builtin_amdgcn_ballot_w32(no_base) != 0;
    }

    // QK stage with the tile's bases: the earlier stages read the next K tile from qk_stage0; with
    // UsePreparedTileBases the tail stage reads V from qk_v0.
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
        else if constexpr(UsePreparedTileBases<typename Mode::SelectedProblem>())
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

    // PV stage with the tile's bases. With UsePreparedTileBases, V stage s reads rows
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
        if constexpr(UsePreparedTileBases<Problem>())
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
//  - the WMMA operand guard for every vectorized-KV problem, which requires it; the linear page
//    tile keeps its unguarded refills (see the hazard note of the scheduled-TDM policy) and takes
//    the LDS drain before every K/V TDM issue instead (kDrainLdsBeforeTdmIssue);
//  - M128 tiles (P64/P128 x D64/D128): the other four (scalar row clamp 1);
//  - P128 D64 M64 tiles: the v_maximum3 row max and the scalar row clamp 2 (3.5 % less time than
//    without them; clamp 1 recovers 3 %). Not for P64 D64 M64, where the earlier TDM issue showed a
//    stale-fragment event (see kDrainLdsBeforeTdmIssue), nor for D128 M64 (about 1 %);
//  - none of the four for every other problem, which keeps their code unchanged, including the
//    KV-split partial outputs (the split routes were measured without them).
template <typename Problem, typename Geometry = FmhaBatchPrefillPageTileGeometryFor<Problem>>
using FmhaBatchPrefillPageTileOptionsFor = std::conditional_t<
    !detail::fmha_tdm_is_vectorized_kv<Problem>::value,
    FmhaTdmSchedTuningOptions<false, false, false, 0, false, true>,
    std::conditional_t<
        !detail::fmha_tdm_is_split_partial_v<Problem> && Geometry::kM == 128,
        FmhaTdmSchedTuningOptions<true, true, true, 1, true>,
        std::conditional_t<!detail::fmha_tdm_is_split_partial_v<Problem> && Geometry::kM == 64 &&
                               Geometry::kHeadDimQK == 64 && Problem::kPageBlockSize == 128,
                           FmhaTdmSchedTuningOptions<false, false, true, 2, true>,
                           FmhaTdmSchedTuningOptions<false, false, false, 0, true>>>>;

// Scheduled-TDM tuning of a page-tile problem: the default tuning of its geometry, with the
// page-tile code-generation options (read through FmhaTdmSchedTuningOptionsOf).
template <typename BaseTuning, typename Options_>
struct FmhaBatchPrefillPageTileTuning : BaseTuning
{
    using Options = Options_;
};

// Grouped drains of a guarded page-tile problem. The D128 tiles with prepared tile bases read PV
// stage 0's V in the softmax, so their QK tail stage refills nothing and one drain covers a whole
// stage (D128 P64: 20 instead of 27 full drains per KV tile). With V0 read in the QK tail stage it
// stays alive across that stage and the softmax, and whole-stage groups spill it (32 VGPRs): the
// kept operands take the low 256 registers.
template <typename Problem, typename Geometry = FmhaBatchPrefillPageTileGeometryFor<Problem>>
using FmhaBatchPrefillPageTileGroupedDrainFor =
    std::conditional_t<detail::fmha_page_tile_d128_prepared_bases_v<Problem, Geometry>,
                       FmhaTdmSchedGroupedDrainPlacement<
                           Geometry,
                           Geometry::kQkWmmasPerStage / Geometry::kQkMIter,
                           Geometry::kPvWmmasPerStage - kFmhaTdmSchedPvLastMRow<Geometry>,
                           false>,
                       FmhaTdmSchedGroupedDrainPlacement<Geometry>>;

// Schedule of a page-tile problem: with the WMMA operand guard (the vectorized KV layout) the
// WMMA-first rows run with grouped drains; otherwise the shared default schedule.
template <typename Problem, typename Geometry = FmhaBatchPrefillPageTileGeometryFor<Problem>>
using FmhaBatchPrefillPageTileScheduleFor = std::conditional_t<
    FmhaBatchPrefillPageTileOptionsFor<Problem>::kWmmaOperandGuard &&
        std::is_same_v<typename FmhaTdmSchedDefaultScheduleFor<Geometry>::LocalPlacement,
                       FmhaTdmSchedWmmaFirstPlacement<FmhaTdmSchedScheduleFor<Geometry>>>,
    FmhaTdmSchedSequentialTileSchedule<FmhaBatchPrefillPageTileGroupedDrainFor<Problem, Geometry>>,
    FmhaTdmSchedDefaultScheduleFor<Geometry>>;

template <typename Problem>
using FmhaBatchPrefillPageTilePolicyFor = BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTilePolicy<
    FmhaBatchPrefillPageTileGeometryFor<Problem>,
    FmhaBatchPrefillPageTileTuning<
        FmhaTdmSchedDefaultTuning<FmhaBatchPrefillPageTileGeometryFor<Problem>>,
        FmhaBatchPrefillPageTileOptionsFor<Problem>>,
    FmhaBatchPrefillPageTileScheduleFor<Problem>>;

} // namespace ck_tile
