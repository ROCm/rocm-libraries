// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule_d192.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule_types.hpp"

namespace ck_tile {

// A placement supplies read events for each row half. It may preserve non-read slots to keep
// the midpoint scheduling point at its original position.
template <typename Geometry, typename Placement>
struct FmhaTdmSchedExplicitRowReadSchedule
{
    using Kind = FmhaTdmSchedLoadKind;

    static_assert(std::is_same_v<Geometry, typename Placement::RequiredGeometry>,
                  "The explicit row placement requires a matching geometry");
    static_assert(Geometry::kQkStages == Placement::kNumQkStages &&
                      Geometry::kPvStages == Placement::kNumPvStages &&
                      Geometry::kQkWmmasPerStage == Placement::kQkWmmasPerStage &&
                      Geometry::kPvWmmasPerStage == Placement::kPvWmmasPerStage,
                  "The explicit row placement must match the geometry's stage and WMMA counts");

    static constexpr index_t kNumQkStages     = Placement::kNumQkStages;
    static constexpr index_t kNumPvStages     = Placement::kNumPvStages;
    static constexpr index_t kQkWmmasPerStage = Placement::kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerStage = Placement::kPvWmmasPerStage;

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkRowHalf(Visitor&& visitor)
    {
        static_assert(Stage >= 0 && Stage < kNumQkStages);
        static_assert(Wmma >= 0 && Wmma < kQkWmmasPerStage);
        Placement::template VisitQkReadRowHalf<Stage, Wmma, SecondHalf>(visitor);
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        static_assert(Stage >= 0 && Stage < kNumPvStages);
        static_assert(Wmma >= 0 && Wmma < kPvWmmasPerStage);
        Placement::template VisitPvReadRowHalf<Stage, Wmma, SecondHalf>(visitor);
    }
};

template <index_t NumWmma, index_t NumLoads>
struct FmhaTdmSchedActiveRows
{
    static_assert(NumWmma > 0 && NumLoads > 0);
    static constexpr index_t kMaxLoadsPerRow = (NumLoads + NumWmma - 1) / NumWmma;
    struct Row
    {
        index_t accesses[kMaxLoadsPerRow]{};
        index_t size = 0;
    };
    Row rows[NumWmma]{};

    CK_TILE_HOST_DEVICE constexpr FmhaTdmSchedActiveRows()
    {
        for(index_t wmma = 0; wmma < NumWmma; ++wmma)
        {
            const index_t first = wmma * NumLoads / NumWmma;
            const index_t last  = (wmma + 1) * NumLoads / NumWmma;
            for(index_t access = first; access < last; ++access)
                rows[wmma].accesses[rows[wmma].size++] = access;
        }
    }
};

// Spread reads proportionally across WMMA rows, keeping each row's ordinal order.
template <typename Geometry>
struct FmhaTdmSchedEvenlyInterleavedReadSchedule
{
    using Kind                                = FmhaTdmSchedLoadKind;
    static constexpr index_t kNumQkStages     = Geometry::kQkStages;
    static constexpr index_t kNumPvStages     = Geometry::kPvStages;
    static constexpr index_t kQkWmmasPerStage = Geometry::kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerStage = Geometry::kPvWmmasPerStage;

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQkRows()
    {
        static_assert(Stage >= 0 && Stage < kNumQkStages);
        constexpr index_t loads =
            Stage == kNumQkStages - 1 ? Geometry::kVStageLoadCount : Geometry::kKSuLoadCount;
        return FmhaTdmSchedActiveRows<kQkWmmasPerStage, loads>{};
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr auto MakePvRows()
    {
        static_assert(Stage >= 0 && Stage < kNumPvStages);
        constexpr index_t loads =
            Stage == kNumPvStages - 1 ? Geometry::kKSuLoadCount : Geometry::kVStageLoadCount;
        return FmhaTdmSchedActiveRows<kPvWmmasPerStage, loads>{};
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkRowHalf(Visitor&& visitor)
    {
        static_assert(Wmma >= 0 && Wmma < kQkWmmasPerStage);
        constexpr auto row  = MakeQkRows<Stage>().rows[Wmma];
        constexpr auto kind = Stage == kNumQkStages - 1 ? Kind::VRead : Kind::KRead;
        static_for<SecondHalf ? row.size / 2 : 0, SecondHalf ? row.size : row.size / 2, 1>{}(
            [&](auto i) {
                visitor(std::integral_constant<Kind, kind>{}, number<row.accesses[i]>{});
            });
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        static_assert(Wmma >= 0 && Wmma < kPvWmmasPerStage);
        constexpr auto row  = MakePvRows<Stage>().rows[Wmma];
        constexpr auto kind = Stage == kNumPvStages - 1 ? Kind::KRead : Kind::VRead;
        static_for<SecondHalf ? row.size / 2 : 0, SecondHalf ? row.size : row.size / 2, 1>{}(
            [&](auto i) {
                visitor(std::integral_constant<Kind, kind>{}, number<row.accesses[i]>{});
            });
    }
};

// M128 N128 D64 Dv64 (the page tile's D64 geometry): eight K or V loads per stage, one per WMMA.
// The evenly interleaved schedule issues each WMMA's load in that WMMA's second half; this one
// issues the loads of WMMAs w and w + 1 together in the second half of every even WMMA w.
template <typename Geometry>
struct FmhaTdmSchedPairedD64N128ReadSchedule : FmhaTdmSchedEvenlyInterleavedReadSchedule<Geometry>
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
        if constexpr(!SecondHalf)
            Base::template VisitQkRowHalf<Stage, Wmma, false>(visitor);
        else if constexpr(Wmma % 2 == 0)
        {
            visitor(std::integral_constant<Kind, kind>{}, number<Wmma>{});
            visitor(std::integral_constant<Kind, kind>{}, number<Wmma + 1>{});
        }
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        static_assert(Wmma >= 0 && Wmma < Base::kPvWmmasPerStage);
        constexpr auto kind = Stage == Base::kNumPvStages - 1 ? Kind::KRead : Kind::VRead;
        if constexpr(!SecondHalf)
            Base::template VisitPvRowHalf<Stage, Wmma, false>(visitor);
        else if constexpr(Wmma % 2 == 0)
        {
            visitor(std::integral_constant<Kind, kind>{}, number<Wmma>{});
            visitor(std::integral_constant<Kind, kind>{}, number<Wmma + 1>{});
        }
    }
};

template <typename Geometry>
using FmhaTdmSchedScheduleFor = std::conditional_t<
    std::is_same_v<Geometry, FmhaTdmSchedBf16D192N128Geometry>,
    FmhaTdmSchedExplicitRowReadSchedule<Geometry, FmhaTdmSchedGfx125Bf16D192ReadPlacement>,
    std::conditional_t<Geometry::kM == 128 && Geometry::kN == 128 && Geometry::kHeadDimQK == 64 &&
                           Geometry::kDv == 64,
                       FmhaTdmSchedPairedD64N128ReadSchedule<Geometry>,
                       FmhaTdmSchedEvenlyInterleavedReadSchedule<Geometry>>>;

// Resolve the output-rescale placement from the selected geometry and policy.
template <typename Problem, typename Policy>
struct FmhaTdmSchedMode
{
    using SelectedProblem = Problem;
    using SelectedPolicy  = Policy;
    using Geometry        = typename Policy::Geometry;

    static constexpr bool kStreamOutputRescale =
        std::is_same_v<Geometry, FmhaTdmSchedBf16D192N128Geometry>;
    static constexpr bool kEarlyOutputRescale = Policy::kEarlyOutputRescale;
    static constexpr bool kPostSoftmaxFragmentRescale =
        !kEarlyOutputRescale && !kStreamOutputRescale;
    static constexpr bool kDeferTensorReady = Policy::template DeferTensorReady<Problem>();
    static constexpr bool kIsLegal =
        !(kEarlyOutputRescale && kStreamOutputRescale) &&
        Policy::Schedule::kNumQkStages == Geometry::kQkStages &&
        Policy::Schedule::kNumPvStages == Geometry::kPvStages &&
        Policy::Schedule::kQkWmmasPerStage == Geometry::kQkWmmasPerStage &&
        Policy::Schedule::kPvWmmasPerStage == Geometry::kPvWmmasPerStage;
};

// A placement executes real row operations. A future placement can issue
// unequal numbers of reads on either side of a WMMA without changing Policy.
template <typename ReadPlacement_>
struct FmhaTdmSchedWmmaFirstPlacement
{
    using ReadPlacement                       = ReadPlacement_;
    using Kind                                = FmhaTdmSchedLoadKind;
    using Point                               = FmhaTdmSchedSchedulePoint;
    static constexpr index_t kNumQkStages     = ReadPlacement::kNumQkStages;
    static constexpr index_t kNumPvStages     = ReadPlacement::kNumPvStages;
    static constexpr index_t kQkWmmasPerStage = ReadPlacement::kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerStage = ReadPlacement::kPvWmmasPerStage;

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
        emit_fence(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterWmma>{});
        auto read = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value != Kind::IgnoredLegacy &&
                         !(Mode::SelectedPolicy::kLeadFirstLoad &&
                           (Stage == 0 || Stage == kNumQkStages - 1) &&
                           decltype(access)::value == 0))
                emit_read(kind, access);
        };
        ReadPlacement::template VisitQkRowHalf<Stage, Wmma, false>(read);
        emit_fence(number<Stage>{},
                   number<Wmma>{},
                   std::integral_constant<Point, Point::BetweenTokenHalves>{});
        ReadPlacement::template VisitQkRowHalf<Stage, Wmma, true>(read);
        emit_fence(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterTokens>{});
    }

    template <typename Mode,
              index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename ReadEmitter,
              typename FenceEmitter,
              typename RescaleEmitter>
    CK_TILE_HOST_DEVICE static constexpr void RunPvRow(WmmaEmitter& emit_wmma,
                                                       ReadEmitter& emit_read,
                                                       FenceEmitter& emit_fence,
                                                       RescaleEmitter& emit_rescale)
    {
        emit_wmma(number<Stage>{}, number<Wmma>{});
        if constexpr(Mode::kStreamOutputRescale && Stage == 0 &&
                     Wmma + 2 < Mode::SelectedPolicy::OutputFragments::kNumFragments)
            emit_rescale(number<Wmma + 2>{});
        emit_fence(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterWmma>{});
        auto read = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value != Kind::IgnoredLegacy &&
                         !(Mode::SelectedPolicy::kLeadFirstLoad &&
                           (Stage == 0 || Stage == kNumPvStages - 1) &&
                           decltype(access)::value == 0))
                emit_read(kind, access);
        };
        ReadPlacement::template VisitPvRowHalf<Stage, Wmma, false>(read);
        emit_fence(number<Stage>{},
                   number<Wmma>{},
                   std::integral_constant<Point, Point::BetweenTokenHalves>{});
        ReadPlacement::template VisitPvRowHalf<Stage, Wmma, true>(read);
        emit_fence(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterTokens>{});
    }
};

// Issue each row's next-stage reads before its current-stage WMMA. The D64/N64
// default has one read per row, keeping PV access1 after WMMA0 prepares its
// address. Other geometries and read mappings need their address dependencies
// checked before using this placement. Lead-load and streaming-rescale modes
// require their own placement and are rejected here.
template <typename ReadPlacement_>
struct FmhaTdmSchedReadFirstPlacement : FmhaTdmSchedWmmaFirstPlacement<ReadPlacement_>
{
    using ReadPlacement = ReadPlacement_;
    using Point         = FmhaTdmSchedSchedulePoint;

    template <bool Pv,
              typename Mode,
              index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename ReadEmitter,
              typename FenceEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    RunRow(WmmaEmitter& emit_wmma, ReadEmitter& emit_read, FenceEmitter& emit_fence)
    {
        using Geometry = typename Mode::Geometry;
        static_assert(Geometry::kHeadDimQK == 64 && Geometry::kDv == 64 && Geometry::kN == 64 &&
                          std::is_same_v<ReadPlacement, FmhaTdmSchedScheduleFor<Geometry>>,
                      "read-first placement requires the D64/V64 N64 default read mapping");
        static_assert(!Mode::SelectedPolicy::kLeadFirstLoad && !Mode::kStreamOutputRescale,
                      "read-first placement requires ordinary stage reads and rescale");
        if constexpr(Pv)
            ReadPlacement::template VisitPvRowHalf<Stage, Wmma, false>(emit_read);
        else
            ReadPlacement::template VisitQkRowHalf<Stage, Wmma, false>(emit_read);
        emit_fence(number<Stage>{},
                   number<Wmma>{},
                   std::integral_constant<Point, Point::BetweenTokenHalves>{});
        if constexpr(Pv)
            ReadPlacement::template VisitPvRowHalf<Stage, Wmma, true>(emit_read);
        else
            ReadPlacement::template VisitQkRowHalf<Stage, Wmma, true>(emit_read);
        emit_fence(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterTokens>{});
        emit_wmma(number<Stage>{}, number<Wmma>{});
        emit_fence(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterWmma>{});
    }

    template <typename Mode,
              index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename ReadEmitter,
              typename FenceEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    RunQkRow(WmmaEmitter& emit_wmma, ReadEmitter& emit_read, FenceEmitter& emit_fence)
    {
        RunRow<false, Mode, Stage, Wmma>(emit_wmma, emit_read, emit_fence);
    }

    template <typename Mode,
              index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename ReadEmitter,
              typename FenceEmitter,
              typename RescaleEmitter>
    CK_TILE_HOST_DEVICE static constexpr void RunPvRow(WmmaEmitter& emit_wmma,
                                                       ReadEmitter& emit_read,
                                                       FenceEmitter& emit_fence,
                                                       RescaleEmitter&)
    {
        RunRow<true, Mode, Stage, Wmma>(emit_wmma, emit_read, emit_fence);
    }
};

// The first PV row of a stage's last M iteration: rows [kFmhaTdmSchedPvLastMRow, kPvWmmasPerStage)
// read the last M iteration's P and V operands.
template <typename Geometry>
inline constexpr index_t kFmhaTdmSchedPvLastMRow =
    (Geometry::kPvMIter - 1) * Geometry::kPvWmmasPerStage / Geometry::kPvMIter;

// Half-open range [begin, end) of rows, or of QK m0/m1 pairs.
struct FmhaTdmSchedRowRange
{
    index_t begin;
    index_t end;
};

// WMMA-first rows with the grouped drains of the WMMA operand guard (see the hazard note of the
// scheduled-TDM policy). The rows of a stage's last M iteration issue their reads one row late,
// behind the next row's WMMA. A Drain point, a full drain, comes before those reads every QkGroup
// QK m0/m1 pairs, and on every PvGroup-th last-M PV row and the stage's last one. The stage's last
// row also issues its own reads and, in QK, has a KeepPending point after its RAW fence. The policy
// implements both points: the operand registers of the WMMAs that a drain covers stay allocated
// until it (QkKeptPairs; PvKeptRows for the PV WMMA emitter). No streaming output rescale.
// Without QkTailReads the QK tail stage issues no reads and has no Drain or KeepPending point: the
// caller reads PV stage 0's V after the stage's RAW fence (kFmhaTdmSchedQkTailReads). ReadPlacement
// assigns the reads to the rows.
template <typename Geometry_,
          index_t QkGroup_        = 4,
          index_t PvGroup_        = 4,
          bool QkTailReads_       = true,
          typename ReadPlacement_ = FmhaTdmSchedScheduleFor<Geometry_>>
struct FmhaTdmSchedGroupedDrainPlacement
{
    using Geometry      = Geometry_;
    using ReadPlacement = ReadPlacement_;
    using Kind          = FmhaTdmSchedLoadKind;
    using Point         = FmhaTdmSchedSchedulePoint;
    template <Point P>
    using At = std::integral_constant<Point, P>;

    static constexpr index_t kNumQkStages     = ReadPlacement::kNumQkStages;
    static constexpr index_t kNumPvStages     = ReadPlacement::kNumPvStages;
    static constexpr index_t kQkWmmasPerStage = ReadPlacement::kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerStage = ReadPlacement::kPvWmmasPerStage;
    static constexpr index_t kQkGroup         = QkGroup_;
    static constexpr index_t kPvGroup         = PvGroup_;
    static constexpr bool kQkTailReads        = QkTailReads_;
    static constexpr index_t kQkMIter         = Geometry::kQkMIter;
    static constexpr index_t kPvLastMRow      = kFmhaTdmSchedPvLastMRow<Geometry>;
    static_assert(kQkMIter == 1 || kQkMIter == 2);
    static_assert(kQkGroup >= 1 && kPvGroup >= 1);

    // QK row r is M iteration r % kQkMIter of m0/m1 pair r / kQkMIter; a pair's last-M row is the
    // last reader of its K slice.
    CK_TILE_HOST_DEVICE static constexpr bool QkDrainRow(index_t r)
    {
        return r % kQkMIter == 0 && r > 0 && (r / kQkMIter) % kQkGroup == 0;
    }
    // Pairs whose last-M K slices stay allocated at the Drain (drain) or KeepPending point of row
    // r: the pairs since the previous drain.
    CK_TILE_HOST_DEVICE static constexpr FmhaTdmSchedRowRange QkKeptPairs(index_t r, bool drain)
    {
        const index_t p = r / kQkMIter;
        if(kQkGroup == 1)
            return {p, p};
        return drain ? FmhaTdmSchedRowRange{p - kQkGroup, p}
                     : FmhaTdmSchedRowRange{(p / kQkGroup) * kQkGroup, p};
    }

    // PV: a last-M row waits when it is the stage's last row or every kPvGroup rows after the
    // first.
    CK_TILE_HOST_DEVICE static constexpr bool PvWaitRow(index_t r)
    {
        return r == kPvWmmasPerStage - 1 || (r > kPvLastMRow && (r - kPvLastMRow) % kPvGroup == 0);
    }
    // First last-M row kept allocated at the wait of row w: the previous wait row (conservative,
    // that row's own drain already covered its WMMA) or kPvLastMRow.
    CK_TILE_HOST_DEVICE static constexpr index_t PvPendingFrom(index_t w)
    {
        for(index_t q = w - 1; q > kPvLastMRow; --q)
            if(PvWaitRow(q))
                return q;
        return kPvLastMRow;
    }
    // Rows whose operands the PV WMMA emitter keeps allocated at row r: before its WMMA, the rows
    // pending at the wait of row r - 1; after the RAW fence of the stage's last row, the rows
    // pending at it.
    CK_TILE_HOST_DEVICE static constexpr FmhaTdmSchedRowRange PvKeptRows(index_t r,
                                                                         bool after_fence)
    {
        if(!after_fence && r >= kPvLastMRow + 2 && PvWaitRow(r - 1))
            return {PvPendingFrom(r - 1), r - 1};
        if(after_fence && r == kPvWmmasPerStage - 1 && r > kPvLastMRow)
            return {PvPendingFrom(r), r};
        return {r, r};
    }

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
        if constexpr(kQkTailReads || Stage != kNumQkStages - 1)
        {
            if constexpr(QkDrainRow(Wmma))
                emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::Drain>{});
            if constexpr(kQkGroup > 1 && Wmma == kQkWmmasPerStage - 1)
                emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::KeepPending>{});
            // The previous row's reads when it is a last-M row, then this row's own unless it
            // is one.
            if constexpr(Wmma % kQkMIter == 0 && Wmma > 0)
                QkReads<Stage, Wmma - 1>(emit_read, emit_fence);
            if constexpr(Wmma % kQkMIter != kQkMIter - 1 || Wmma == kQkWmmasPerStage - 1)
                QkReads<Stage, Wmma>(emit_read, emit_fence);
        }
        emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::AfterTokens>{});
    }

    template <typename Mode,
              index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename ReadEmitter,
              typename FenceEmitter,
              typename RescaleEmitter>
    CK_TILE_HOST_DEVICE static constexpr void RunPvRow(WmmaEmitter& emit_wmma,
                                                       ReadEmitter& emit_read,
                                                       FenceEmitter& emit_fence,
                                                       RescaleEmitter&)
    {
        static_assert(!Mode::kStreamOutputRescale,
                      "the grouped drains do not place the streaming output rescale");
        emit_wmma(number<Stage>{}, number<Wmma>{});
        emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::AfterWmma>{});
        if constexpr(Wmma > kPvLastMRow)
        {
            if constexpr(PvWaitRow(Wmma))
                emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::Drain>{});
            PvReads<Stage, Wmma - 1>(emit_read, emit_fence);
        }
        if constexpr(Wmma < kPvLastMRow || Wmma == kPvWmmasPerStage - 1)
            PvReads<Stage, Wmma>(emit_read, emit_fence);
        emit_fence(number<Stage>{}, number<Wmma>{}, At<Point::AfterTokens>{});
    }

    private:
    // One row's reads in the read placement's order (the emitters skip a lead-first access).
    template <index_t Stage, index_t Row, typename ReadEmitter, typename FenceEmitter>
    CK_TILE_HOST_DEVICE static constexpr void QkReads(ReadEmitter& emit_read,
                                                      FenceEmitter& emit_fence)
    {
        auto read = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value != Kind::IgnoredLegacy)
                emit_read(kind, access);
        };
        ReadPlacement::template VisitQkRowHalf<Stage, Row, false>(read);
        emit_fence(number<Stage>{}, number<Row>{}, At<Point::BetweenTokenHalves>{});
        ReadPlacement::template VisitQkRowHalf<Stage, Row, true>(read);
    }

    template <index_t Stage, index_t Row, typename ReadEmitter, typename FenceEmitter>
    CK_TILE_HOST_DEVICE static constexpr void PvReads(ReadEmitter& emit_read,
                                                      FenceEmitter& emit_fence)
    {
        auto read = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value != Kind::IgnoredLegacy)
                emit_read(kind, access);
        };
        ReadPlacement::template VisitPvRowHalf<Stage, Row, false>(read);
        emit_fence(number<Stage>{}, number<Row>{}, At<Point::BetweenTokenHalves>{});
        ReadPlacement::template VisitPvRowHalf<Stage, Row, true>(read);
    }
};

// Whether a placement is the grouped-drain placement, whose points the policy implements.
template <typename Placement>
inline constexpr bool kIsFmhaTdmSchedGroupedDrain = false;
template <typename Geometry,
          index_t QkGroup,
          index_t PvGroup,
          bool QkTailReads,
          typename ReadPlacement>
inline constexpr bool kIsFmhaTdmSchedGroupedDrain<
    FmhaTdmSchedGroupedDrainPlacement<Geometry, QkGroup, PvGroup, QkTailReads, ReadPlacement>> =
    true;

// Whether a placement's QK tail stage reads PV stage 0's V (all but the grouped-drain placement
// without QkTailReads, whose caller reads it after the QK stage).
template <typename Placement>
inline constexpr bool kFmhaTdmSchedQkTailReads = true;
template <typename Geometry,
          index_t QkGroup,
          index_t PvGroup,
          bool QkTailReads,
          typename ReadPlacement>
inline constexpr bool kFmhaTdmSchedQkTailReads<
    FmhaTdmSchedGroupedDrainPlacement<Geometry, QkGroup, PvGroup, QkTailReads, ReadPlacement>> =
    QkTailReads;

// The tile schedule owns phase order. The context exposes only the concrete
// work of one sequence-K tile; its large tensors stay in the pipeline scope.
template <typename BeginOp,
          typename QkOp,
          typename ScoreOp,
          typename SoftmaxOp,
          typename NextKOp,
          typename PvOp,
          typename AdvanceOp>
struct FmhaTdmSchedTileOps
{
    BeginOp& begin;
    QkOp& qk;
    ScoreOp& score;
    SoftmaxOp& softmax;
    NextKOp& next_k;
    PvOp& pv;
    AdvanceOp& advance;

    CK_TILE_HOST_DEVICE constexpr void Begin() { begin(); }
    CK_TILE_HOST_DEVICE constexpr void Qk() { qk(); }
    CK_TILE_HOST_DEVICE constexpr void Score() { score(); }
    CK_TILE_HOST_DEVICE constexpr decltype(auto) Softmax() { return softmax(); }
    CK_TILE_HOST_DEVICE constexpr void NextK() { next_k(); }
    template <typename PTile>
    CK_TILE_HOST_DEVICE constexpr void Pv(const PTile& p_tile)
    {
        pv(p_tile);
    }
    CK_TILE_HOST_DEVICE constexpr void Advance() { advance(); }
};

template <typename BeginOp,
          typename QkOp,
          typename ScoreOp,
          typename SoftmaxOp,
          typename NextKOp,
          typename PvOp,
          typename AdvanceOp>
CK_TILE_HOST_DEVICE constexpr auto MakeFmhaTdmSchedTileOps(BeginOp& begin,
                                                           QkOp& qk,
                                                           ScoreOp& score,
                                                           SoftmaxOp& softmax,
                                                           NextKOp& next_k,
                                                           PvOp& pv,
                                                           AdvanceOp& advance)
{
    return FmhaTdmSchedTileOps<BeginOp, QkOp, ScoreOp, SoftmaxOp, NextKOp, PvOp, AdvanceOp>{
        begin, qk, score, softmax, next_k, pv, advance};
}

template <typename LocalPlacement_>
struct FmhaTdmSchedSequentialTileSchedule
{
    using LocalPlacement                      = LocalPlacement_;
    using ReadPlacement                       = typename LocalPlacement::ReadPlacement;
    static constexpr index_t kNumQkStages     = LocalPlacement::kNumQkStages;
    static constexpr index_t kNumPvStages     = LocalPlacement::kNumPvStages;
    static constexpr index_t kQkWmmasPerStage = LocalPlacement::kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerStage = LocalPlacement::kPvWmmasPerStage;

    template <typename Ops>
    CK_TILE_HOST_DEVICE static constexpr void Run(Ops& ops)
    {
        ops.Begin();
        ops.Qk();
        ops.Score();
        auto p_tile = ops.Softmax();
        ops.NextK();
        ops.Pv(p_tile);
        ops.Advance();
    }

    template <typename Mode, index_t Stage, typename ReadEmitter>
    CK_TILE_HOST_DEVICE static constexpr void RunQkReadPrelude(ReadEmitter&& emit_read)
    {
        if constexpr(Mode::SelectedPolicy::kLeadFirstLoad &&
                     (Stage == 0 ||
                      (Stage == kNumQkStages - 1 && kFmhaTdmSchedQkTailReads<LocalPlacement>)))
            emit_read(number<0>{});
    }

    template <typename Mode, index_t Stage, typename ReadEmitter>
    CK_TILE_HOST_DEVICE static constexpr void RunPvReadPrelude(ReadEmitter&& emit_read)
    {
        if constexpr(Mode::SelectedPolicy::kLeadFirstLoad &&
                     (Stage == 0 || Stage == kNumPvStages - 1))
            emit_read(number<0>{});
    }

    template <typename Mode,
              typename Part01,
              typename WaitV,
              typename Rescale,
              typename Part2Fragments>
    CK_TILE_HOST_DEVICE static constexpr void RunSplitSoftmax(Part01&& part01,
                                                              WaitV&& wait_v,
                                                              Rescale&& rescale,
                                                              Part2Fragments&& part2_fragments)
    {
        static_assert(Mode::kIsLegal);
        part01();
        if constexpr(Mode::kDeferTensorReady)
            wait_v();
        if constexpr(Mode::kEarlyOutputRescale)
            static_for<0, Mode::SelectedPolicy::OutputFragments::kNumFragments, 1>{}(
                [&](auto ordinal) { rescale(ordinal); });
        part2_fragments(number<0>{});
        if constexpr(Mode::kPostSoftmaxFragmentRescale)
            static_for<0, Mode::SelectedPolicy::OutputFragments::kNumFragments, 1>{}(
                [&](auto ordinal) { rescale(ordinal); });
    }

    template <typename Mode, index_t Stage, typename RescaleEmitter>
    CK_TILE_HOST_DEVICE static constexpr void RunPvStagePrelude(RescaleEmitter&& emit_rescale)
    {
        if constexpr(Mode::kStreamOutputRescale && Stage == 0)
        {
            emit_rescale(number<0>{});
            emit_rescale(number<1>{});
        }
    }

    template <typename Mode, typename TensorDrain, typename DsDrain>
    CK_TILE_HOST_DEVICE static constexpr void RunEpilogue(TensorDrain&& tensor_drain,
                                                          DsDrain&& ds_drain)
    {
        if constexpr((Mode::SelectedPolicy::kKPrefetchTensorCount != 0 ||
                      Mode::SelectedPolicy::kVPrefetchTensorCount != 0) &&
                     Mode::SelectedPolicy::kPrefetchTailDrain)
            tensor_drain();
        using Policy   = typename Mode::SelectedPolicy;
        using Geometry = typename Policy::Geometry;
        if constexpr(Policy::template GetPvTailDsCount<Geometry::kPvStages - 1>() != 0)
            ds_drain();
    }
};

template <typename Geometry>
using FmhaTdmSchedDefaultScheduleFor = FmhaTdmSchedSequentialTileSchedule<
    std::conditional_t<Geometry::kHeadDimQK == 64 && Geometry::kN == 64,
                       FmhaTdmSchedReadFirstPlacement<FmhaTdmSchedScheduleFor<Geometry>>,
                       FmhaTdmSchedWmmaFirstPlacement<FmhaTdmSchedScheduleFor<Geometry>>>>;

} // namespace ck_tile
