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

// D64/V64 M128 N128 page tiles other than the KV-split partial kernels: with the WMMA operand
// guard, they read each stage's K or V in groups of four
// (FmhaBatchPrefillPageTileQuadD64N128ReadSchedule).
template <typename Problem, typename Geometry>
inline constexpr bool fmha_page_tile_d64_quad_reads_v =
    Geometry::kM == 128 && Geometry::kN == 128 && Geometry::kHeadDimQK == 64 &&
    Geometry::kDv == 64 && !fmha_tdm_is_split_partial_v<Problem>;

// D128 M128 page tiles without prepared tile bases: the guarded problems whose code spills on
// gfx1250 (VGPR spills in 288 of their 354 generated instances, none in any other guarded
// instance). They keep the shared grouped-drain placement: the full drain that
// FmhaBatchPrefillPageTileGroupedDrainPlacement adds at the end of each QK stage moved their
// register allocation, and spill refills of Q fragments then landed on the operands of WMMAs in
// flight (100 instances; devlog O1 of bp-review-fixes).
template <typename Problem, typename Geometry>
inline constexpr bool fmha_page_tile_d128_m128_spills_v =
    Geometry::kHeadDimQK == 128 && Geometry::kM == 128 &&
    !fmha_page_tile_d128_prepared_bases_v<Problem, Geometry>;
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
//  - the LDS drain also for every vectorized D64 M64 tile, KV-split partials included, where the
//    stale-fragment event was most frequent (kDrainLdsBeforeTdmIssue of the scheduled-TDM policy);
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
        std::conditional_t<
            !detail::fmha_tdm_is_split_partial_v<Problem> && Geometry::kM == 64 &&
                Geometry::kHeadDimQK == 64 && Problem::kPageBlockSize == 128,
            FmhaTdmSchedTuningOptions<false, false, true, 2, true, true>,
            FmhaTdmSchedTuningOptions<false,
                                      false,
                                      false,
                                      0,
                                      true,
                                      Geometry::kM == 64 && Geometry::kHeadDimQK == 64>>>>;

// Scheduled-TDM tuning of a page-tile problem: the default tuning of its geometry, with the
// page-tile code-generation options (read through FmhaTdmSchedTuningOptionsOf).
template <typename BaseTuning, typename Options_>
struct FmhaBatchPrefillPageTileTuning : BaseTuning
{
    using Options = Options_;
};

// Reads of the guarded D64/V64 M128 N128 page tiles: each stage's eight K or V loads in two groups
// of four, the loads of WMMAs w .. w + 3 in the second half of WMMA w (w = 0, 4). The scheduled-TDM
// default pairs them behind every even WMMA (FmhaTdmSchedPairedD64N128ReadSchedule). The read
// after a WMMA waits for the WMMA's issue, so on latency-bound launches fewer, larger groups save
// about 2 % of a KV block. The stage tail waits hold for any placement of a stage's loads.
template <typename Geometry>
struct FmhaBatchPrefillPageTileQuadD64N128ReadSchedule
    : FmhaTdmSchedEvenlyInterleavedReadSchedule<Geometry>
{
    using Base = FmhaTdmSchedEvenlyInterleavedReadSchedule<Geometry>;
    using Kind = FmhaTdmSchedLoadKind;
    static_assert(Geometry::kKSuLoadCount == 8 && Geometry::kVStageLoadCount == 8 &&
                  Base::kQkWmmasPerStage == 8 && Base::kPvWmmasPerStage == 8);

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkRowHalf(Visitor&& visitor)
    {
        static_assert(Wmma >= 0 && Wmma < Base::kQkWmmasPerStage);
        constexpr auto kind = Stage == Base::kNumQkStages - 1 ? Kind::VRead : Kind::KRead;
        if constexpr(SecondHalf && Wmma % 4 == 0)
            static_for<Wmma, Wmma + 4, 1>{}(
                [&](auto access) { visitor(std::integral_constant<Kind, kind>{}, access); });
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        static_assert(Wmma >= 0 && Wmma < Base::kPvWmmasPerStage);
        constexpr auto kind = Stage == Base::kNumPvStages - 1 ? Kind::KRead : Kind::VRead;
        if constexpr(SecondHalf && Wmma % 4 == 0)
            static_for<Wmma, Wmma + 4, 1>{}(
                [&](auto access) { visitor(std::integral_constant<Kind, kind>{}, access); });
    }
};

// Grouped-drain placement of the page tile: FmhaTdmSchedGroupedDrainPlacement with a Drain point,
// the full drain, where that placement has its KeepPending point (after the RAW fence of a QK
// stage's last row). The fence waits for the last WMMA's result only. Under the hazard note's
// model (WMMAs do not retire in issue order) it does not prove that the earlier pairs of the group
// have read their K slices, which the reads after it may refill; the drain does, so the guard
// does not depend on the waits the compiler adds under expert scheduling. The pairs kept to that
// point are KeepPending's (those since the previous drain), or the drain's own where the row is a
// drain row too. Everything else is the shared placement's. The page-tile policy selects it for
// every guarded problem except the spilling D128 M128 tiles (fmha_page_tile_d128_m128_spills_v),
// whose stage-end release keeps relying on the waits that expert scheduling adds.
template <typename Geometry_,
          index_t QkGroup_        = 4,
          index_t PvGroup_        = 4,
          bool QkTailReads_       = true,
          typename ReadPlacement_ = FmhaTdmSchedScheduleFor<Geometry_>>
struct FmhaBatchPrefillPageTileGroupedDrainPlacement
    : FmhaTdmSchedGroupedDrainPlacement<Geometry_, QkGroup_, PvGroup_, QkTailReads_, ReadPlacement_>
{
    using Base  = FmhaTdmSchedGroupedDrainPlacement<Geometry_,
                                                    QkGroup_,
                                                    PvGroup_,
                                                    QkTailReads_,
                                                    ReadPlacement_>;
    using Point = typename Base::Point;
    using Kind  = typename Base::Kind;
    template <Point P>
    using At = std::integral_constant<Point, P>;

    // The QK rows that end with a full drain in place of the shared placement's KeepPending point.
    CK_TILE_HOST_DEVICE static constexpr bool QkStageEndDrain(index_t r)
    {
        return Base::kQkGroup > 1 && r == Base::kQkWmmasPerStage - 1;
    }
    CK_TILE_HOST_DEVICE static constexpr FmhaTdmSchedRowRange QkKeptPairs(index_t r, bool drain)
    {
        return QkStageEndDrain(r) ? Base::QkKeptPairs(r, Base::QkDrainRow(r))
                                  : Base::QkKeptPairs(r, drain);
    }

    // Base::RunQkRow with one Drain point at a stage's last row (no KeepPending point).
    template <typename Mode,
              index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename ReadEmitter,
              typename FenceEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    RunQkRow(WmmaEmitter& emit_wmma, ReadEmitter& emit_read, FenceEmitter& emit_fence)
    {
        emit_wmma(number<Stage>{}, number<Wmma>{});
        emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::AfterWmma>{});
        if constexpr(Base::kQkTailReads || Stage != Base::kNumQkStages - 1)
        {
            if constexpr(Base::QkDrainRow(Wmma) || QkStageEndDrain(Wmma))
                emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::Drain>{});
            if constexpr(Wmma % Base::kQkMIter == 0 && Wmma > 0)
                QkReads<Stage, Wmma - 1>(emit_read, emit_fence);
            if constexpr(Wmma % Base::kQkMIter != Base::kQkMIter - 1 ||
                         Wmma == Base::kQkWmmasPerStage - 1)
                QkReads<Stage, Wmma>(emit_read, emit_fence);
        }
        emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::AfterTokens>{});
    }

    private:
    // Base::QkReads (private there).
    template <index_t Stage, index_t Row, typename ReadEmitter, typename FenceEmitter>
    CK_TILE_HOST_DEVICE static constexpr void QkReads(ReadEmitter& emit_read,
                                                      FenceEmitter& emit_fence)
    {
        auto read = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value != Kind::IgnoredLegacy)
                emit_read(kind, access);
        };
        Base::ReadPlacement::template VisitQkRowHalf<Stage, Row, false>(read);
        emit_fence(number<Stage>{}, number<Row>{}, At<Point::BetweenTokenHalves>{});
        Base::ReadPlacement::template VisitQkRowHalf<Stage, Row, true>(read);
    }
};

// The shared placement traits for the page-tile placement: a grouped-drain placement, whose QK
// tail stage reads PV stage 0's V unless QkTailReads is false.
template <typename Geometry,
          index_t QkGroup,
          index_t PvGroup,
          bool QkTailReads,
          typename ReadPlacement>
inline constexpr bool
    kIsFmhaTdmSchedGroupedDrain<FmhaBatchPrefillPageTileGroupedDrainPlacement<Geometry,
                                                                              QkGroup,
                                                                              PvGroup,
                                                                              QkTailReads,
                                                                              ReadPlacement>> =
        true;
template <typename Geometry,
          index_t QkGroup,
          index_t PvGroup,
          bool QkTailReads,
          typename ReadPlacement>
inline constexpr bool
    kFmhaTdmSchedQkTailReads<FmhaBatchPrefillPageTileGroupedDrainPlacement<Geometry,
                                                                           QkGroup,
                                                                           PvGroup,
                                                                           QkTailReads,
                                                                           ReadPlacement>> =
        QkTailReads;

// Grouped drains of a guarded page-tile problem. The D128 tiles with prepared tile bases read PV
// stage 0's V in the softmax, so their QK tail stage refills nothing and one drain covers a whole
// stage (D128 P64: 20 instead of 27 full drains per KV tile). With V0 read in the QK tail stage it
// stays alive across that stage and the softmax, and whole-stage groups spill it (32 VGPRs): the
// kept operands take the low 256 registers. The D64 M128 tiles read in groups of four. The spilling
// D128 M128 tiles keep the shared placement (fmha_page_tile_d128_m128_spills_v).
template <typename Problem, typename Geometry = FmhaBatchPrefillPageTileGeometryFor<Problem>>
using FmhaBatchPrefillPageTileGroupedDrainFor = std::conditional_t<
    detail::fmha_page_tile_d128_prepared_bases_v<Problem, Geometry>,
    FmhaBatchPrefillPageTileGroupedDrainPlacement<Geometry,
                                                  Geometry::kQkWmmasPerStage / Geometry::kQkMIter,
                                                  Geometry::kPvWmmasPerStage -
                                                      kFmhaTdmSchedPvLastMRow<Geometry>,
                                                  false>,
    std::conditional_t<
        detail::fmha_page_tile_d64_quad_reads_v<Problem, Geometry>,
        FmhaBatchPrefillPageTileGroupedDrainPlacement<
            Geometry,
            4,
            4,
            true,
            FmhaBatchPrefillPageTileQuadD64N128ReadSchedule<Geometry>>,
        std::conditional_t<detail::fmha_page_tile_d128_m128_spills_v<Problem, Geometry>,
                           FmhaTdmSchedGroupedDrainPlacement<Geometry>,
                           FmhaBatchPrefillPageTileGroupedDrainPlacement<Geometry>>>>;

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
