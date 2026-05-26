// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm_decode/pipeline/gemm_decode_problem.hpp"

namespace ck_tile {

// Tile distribution policies for warp-per-scalar dense GEMM.
//
// Adapted from ops/warp_decode/pipeline/warp_decode_policy.hpp; identical
// shape, trimmed comments. P0 uses kWarpsPerBlock = 1, so the
// MakeOutputTileDistribution and MakeXBroadcastTileDistribution forms collapse
// to the same encoding. Both are kept distinct so adding waves_per_wg > 1
// (P4) does not require re-deriving the broadcast.
struct GemmDecodePolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeTileDistribution()
    {
        return MakeOutputTileDistribution<Problem>();
    }

    // P0=warp_id  -> output row, P1=lane_id -> reduction (K) dimension.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOutputTileDistribution()
    {
        constexpr index_t V             = Problem::kVector;
        constexpr index_t WarpsPerBlock = Problem::kWarpsPerBlock;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<WarpsPerBlock>,
                                             sequence<get_warp_size(), V>>,
                                       tuple<sequence<1>, sequence<2>>,
                                       tuple<sequence<0>, sequence<0>>,
                                       sequence<2>,
                                       sequence<1>>{});
    }

    // P0=warp_id replicated, P1=lane_id -> K. Used when several warps share
    // one activation row in a future cross-wave-reuse variant (P4).
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBroadcastTileDistribution()
    {
        constexpr index_t V             = Problem::kVector;
        constexpr index_t WarpsPerBlock = Problem::kWarpsPerBlock;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<WarpsPerBlock>,
                                       tuple<sequence<1>, sequence<get_warp_size(), V>>,
                                       tuple<sequence<0>, sequence<2>>,
                                       tuple<sequence<0>, sequence<0>>,
                                       sequence<2>,
                                       sequence<1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 0;
    }
};

} // namespace ck_tile
