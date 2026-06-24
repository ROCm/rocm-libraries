// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"

namespace ck_tile {

struct WarpDecodePolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeTileDistribution()
    {
        return MakeOutputTileDistribution<Problem>();
    }

    // Output compute maps P0=get_warp_id() to adjacent output rows and
    // P1=get_lane_id() to the reduction dimension.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOutputTileDistribution()
    {
        constexpr index_t V = Problem::kVector;
        constexpr index_t WarpsPerBlock = Problem::kWarpsPerBlock;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<WarpsPerBlock>,
                    sequence<get_warp_size(), V>
                >,
                tuple<sequence<1>, sequence<2>>,
                tuple<sequence<0>, sequence<0>>,
                sequence<2>,
                sequence<1>
            >{});
    }

    // X is shared by all output rows in an LDS gate/up block. Replicate it
    // across P0 instead of assigning P0 to a tensor dimension.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBroadcastTileDistribution()
    {
        constexpr index_t V = Problem::kVector;
        constexpr index_t WarpsPerBlock = Problem::kWarpsPerBlock;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WarpsPerBlock>,
                tuple<
                    sequence<1>,
                    sequence<get_warp_size(), V>
                >,
                tuple<sequence<0>, sequence<2>>,
                tuple<sequence<0>, sequence<0>>,
                sequence<2>,
                sequence<1>
            >{});
    }

    // CTA copy maps all warps and lanes onto distinct vector segments.
    template <typename Problem, index_t CopyVector>
    CK_TILE_DEVICE static constexpr auto MakeBlockCopyTileDistribution()
    {
        constexpr index_t WarpsPerBlock = Problem::kWarpsPerBlock;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<1>,
                    sequence<WarpsPerBlock, get_warp_size(), CopyVector>
                >,
                tuple<sequence<2>, sequence<2>>,
                tuple<sequence<0>, sequence<1>>,
                sequence<2>,
                sequence<2>
            >{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 0;
    }
};

} // namespace ck_tile
