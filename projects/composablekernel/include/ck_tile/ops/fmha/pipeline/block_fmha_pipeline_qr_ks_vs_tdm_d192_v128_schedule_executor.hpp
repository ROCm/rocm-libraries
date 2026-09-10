// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule.hpp"

namespace ck_tile {

enum class FmhaTdmV128SchedulePoint : index_t
{
    AfterWmma,
    BetweenTokenHalves,
    AfterTokens,
};

using FmhaD192SchedulePoint = FmhaTdmV128SchedulePoint;

template <typename Schedule_>
struct FmhaTdmV128ScheduleExecutor
{
    using Schedule = Schedule_;
    using Point    = FmhaTdmV128SchedulePoint;

    template <index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename TokenEmitter,
              typename PointEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    ExecuteQkRow(WmmaEmitter& emit_wmma, TokenEmitter& emit_token, PointEmitter& emit_point)
    {
        static_assert(Stage >= 0 && Stage < Schedule::kNumQkStages);
        static_assert(Wmma >= 0 && Wmma < Schedule::kQkWmmasPerStage);
        emit_wmma(number<Stage>{}, number<Wmma>{});
        emit_point(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterWmma>{});
        Schedule::template VisitQkRowHalf<Stage, Wmma, false>(emit_token);
        emit_point(number<Stage>{},
                   number<Wmma>{},
                   std::integral_constant<Point, Point::BetweenTokenHalves>{});
        Schedule::template VisitQkRowHalf<Stage, Wmma, true>(emit_token);
        emit_point(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterTokens>{});
    }

    template <index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename TokenEmitter,
              typename PointEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    ExecutePvRow(WmmaEmitter& emit_wmma, TokenEmitter& emit_token, PointEmitter& emit_point)
    {
        static_assert(Stage >= 0 && Stage < Schedule::kNumPvStages);
        static_assert(Wmma >= 0 && Wmma < Schedule::kPvWmmasPerStage);
        emit_wmma(number<Stage>{}, number<Wmma>{});
        emit_point(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterWmma>{});
        Schedule::template VisitPvRowHalf<Stage, Wmma, false>(emit_token);
        emit_point(number<Stage>{},
                   number<Wmma>{},
                   std::integral_constant<Point, Point::BetweenTokenHalves>{});
        Schedule::template VisitPvRowHalf<Stage, Wmma, true>(emit_token);
        emit_point(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterTokens>{});
    }

    template <index_t Stage, typename WmmaEmitter, typename TokenEmitter, typename PointEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    ExecuteQkStage(WmmaEmitter& emit_wmma, TokenEmitter& emit_token, PointEmitter& emit_point)
    {
        static_for<0, Schedule::kQkWmmasPerStage, 1>{}(
            [&](auto wmma) { ExecuteQkRow<Stage, wmma>(emit_wmma, emit_token, emit_point); });
    }

    template <index_t Stage, typename WmmaEmitter, typename TokenEmitter, typename PointEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    ExecutePvStage(WmmaEmitter& emit_wmma, TokenEmitter& emit_token, PointEmitter& emit_point)
    {
        static_for<0, Schedule::kPvWmmasPerStage, 1>{}(
            [&](auto wmma) { ExecutePvRow<Stage, wmma>(emit_wmma, emit_token, emit_point); });
    }
};

using BlockFmhaPipelineQRKSVSTdmD192V128ScheduleExecutor =
    FmhaTdmV128ScheduleExecutor<BlockFmhaPipelineQRKSVSTdmD192V128Schedule>;

} // namespace ck_tile
