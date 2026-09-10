// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule.hpp"
#include "ck_tile/ops/fmha/pipeline/fmha_tdm_v128_config.hpp"

namespace ck_tile {

enum class FmhaTdmV128LoadKind : index_t
{
    KRead,
    VRead,
    IgnoredLegacy,
};

// Retain callback identity on inactive slots without making active load ordinals token-relative.
template <FmhaD192ScheduleToken Token, index_t Ordinal>
struct FmhaTdmV128IgnoredLegacy
    : std::integral_constant<FmhaTdmV128LoadKind, FmhaTdmV128LoadKind::IgnoredLegacy>
{
    template <typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void EmitSoftmax(Visitor& visitor)
    {
        if constexpr((Token >= FmhaD192ScheduleToken::P2M0 &&
                      Token <= FmhaD192ScheduleToken::P2M3) ||
                     Token == FmhaD192ScheduleToken::ORescale)
        {
            visitor(std::integral_constant<FmhaD192ScheduleToken, Token>{}, number<Ordinal>{});
        }
    }
};

template <typename Geometry>
struct FmhaTdmV128LegacySchedule
{
    using Legacy = BlockFmhaPipelineQRKSVSTdmD192V128Schedule;
    using Token  = FmhaD192ScheduleToken;
    using Kind   = FmhaTdmV128LoadKind;

    static_assert(std::is_same_v<Geometry, LegacyD192Geometry>,
                  "The legacy row table is valid only for BF16 D192");
    static constexpr index_t kNumQkStages     = Legacy::kNumQkStages;
    static constexpr index_t kNumPvStages     = Legacy::kNumPvStages;
    static constexpr index_t kQkWmmasPerStage = Legacy::kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerStage = Legacy::kPvWmmasPerStage;

    template <bool IsQk, index_t Stage, typename TokenConstant, typename Ordinal, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void Translate(TokenConstant, Ordinal, Visitor& visitor)
    {
        constexpr auto token = TokenConstant::value;
        if constexpr(token >= Token::KM0 && token <= Token::KM3)
        {
            static_assert(IsQk ? Stage < kNumQkStages - 1 : Stage == kNumPvStages - 1);
            constexpr index_t msb = static_cast<index_t>(token) - static_cast<index_t>(Token::KM0);
            constexpr index_t per_msb = Geometry::kKSuLoadCount / 4;
            constexpr index_t local   = Ordinal::value - (IsQk ? Stage * per_msb : 0);
            static_assert(local >= 0 && local < per_msb);
            visitor(std::integral_constant<Kind, Kind::KRead>{}, number<msb * per_msb + local>{});
        }
        else if constexpr(token >= Token::VM0 && token <= Token::VM3)
        {
            static_assert(IsQk ? Stage == kNumQkStages - 1 : Stage < kNumPvStages - 1);
            constexpr index_t msb = static_cast<index_t>(token) - static_cast<index_t>(Token::VM0);
            constexpr index_t per_msb = Geometry::kVStageLoadCount / 4;
            constexpr index_t local   = Ordinal::value - (IsQk ? 0 : Stage * per_msb);
            static_assert(local >= 0 && local < per_msb);
            visitor(std::integral_constant<Kind, Kind::VRead>{}, number<msb * per_msb + local>{});
        }
        else
        {
            visitor(FmhaTdmV128IgnoredLegacy<token, Ordinal::value>{}, number<Ordinal::value>{});
        }
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkRowHalf(Visitor&& visitor)
    {
        auto translate = [&](auto token, auto ordinal) {
            Translate<true, Stage>(token, ordinal, visitor);
        };
        Legacy::template VisitQkRowHalf<Stage, Wmma, SecondHalf>(translate);
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        auto translate = [&](auto token, auto ordinal) {
            Translate<false, Stage>(token, ordinal, visitor);
        };
        Legacy::template VisitPvRowHalf<Stage, Wmma, SecondHalf>(translate);
    }
};

template <index_t NumWmma, index_t NumLoads>
struct FmhaTdmV128ActiveRows
{
    static_assert(NumWmma > 0 && NumLoads > 0);
    static constexpr index_t kMaxLoadsPerRow = (NumLoads + NumWmma - 1) / NumWmma;
    struct Row
    {
        index_t accesses[kMaxLoadsPerRow]{};
        index_t size = 0;
    };
    Row rows[NumWmma]{};

    CK_TILE_HOST_DEVICE constexpr FmhaTdmV128ActiveRows()
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

// New geometries use active operations only; the D192 adapter above keeps its exact old slots.
template <typename Geometry>
struct FmhaTdmV128ActiveSchedule
{
    using Kind                                = FmhaTdmV128LoadKind;
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
        return FmhaTdmV128ActiveRows<kQkWmmasPerStage, loads>{};
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr auto MakePvRows()
    {
        static_assert(Stage >= 0 && Stage < kNumPvStages);
        constexpr index_t loads =
            Stage == kNumPvStages - 1 ? Geometry::kKSuLoadCount : Geometry::kVStageLoadCount;
        return FmhaTdmV128ActiveRows<kPvWmmasPerStage, loads>{};
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

template <typename Geometry>
using FmhaTdmV128ScheduleFor = std::conditional_t<std::is_same_v<Geometry, LegacyD192Geometry>,
                                                  FmhaTdmV128LegacySchedule<Geometry>,
                                                  FmhaTdmV128ActiveSchedule<Geometry>>;

} // namespace ck_tile
