// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule.hpp"

namespace ck_tile {

enum class FmhaD192SchedulePoint : index_t
{
    AfterWmma,
    BetweenTokenHalves,
    AfterTokens,
};

struct BlockFmhaPipelineQRKSVSTdmD192V128ScheduleExecutor
{
    using Schedule = BlockFmhaPipelineQRKSVSTdmD192V128Schedule;
    using Point    = FmhaD192SchedulePoint;

    template <index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename TokenEmitter,
              typename PointEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    ExecuteQkRow(WmmaEmitter& emit_wmma, TokenEmitter& emit_token, PointEmitter& emit_point)
    {
        emit_wmma(number<Stage>{}, number<Wmma>{});
        emit_point(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterWmma>{});
        Schedule::VisitQkRowHalf<Stage, Wmma, false>(emit_token);
        emit_point(number<Stage>{},
                   number<Wmma>{},
                   std::integral_constant<Point, Point::BetweenTokenHalves>{});
        Schedule::VisitQkRowHalf<Stage, Wmma, true>(emit_token);
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
        emit_wmma(number<Stage>{}, number<Wmma>{});
        emit_point(
            number<Stage>{}, number<Wmma>{}, std::integral_constant<Point, Point::AfterWmma>{});
        Schedule::VisitPvRowHalf<Stage, Wmma, false>(emit_token);
        emit_point(number<Stage>{},
                   number<Wmma>{},
                   std::integral_constant<Point, Point::BetweenTokenHalves>{});
        Schedule::VisitPvRowHalf<Stage, Wmma, true>(emit_token);
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

} // namespace ck_tile
