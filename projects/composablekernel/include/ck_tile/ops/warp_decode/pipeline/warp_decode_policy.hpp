// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"

namespace ck_tile {

struct WarpDecodePolicy
{
    // Make distribution for a [1, WAVE_SIZE] tile.
    // This is used for X, W, and Intermediate tiles within the loop.
    // M dimension is always size 1. N dimension is WAVE_SIZE.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<1, 1>,                   // M: [Iterations=1, Warps=1]
                    sequence<1, get_warp_size(), 1>   // N: [Iterations=1, Lanes=WAVE_SIZE, Vector=1]
                >,
                tuple<
                    sequence<1>, // P0 (Warp ID) maps to M's factor 1
                    sequence<2>  // P1 (Lane ID) maps to N's factor 1
                >,
                tuple<
                    sequence<1>, // M factor 1
                    sequence<1>  // N factor 1
                >,
                sequence<1, 2, 2>, // Y0 -> M(0), Y1 -> N(0), Y2 -> N(2)
                sequence<0, 0, 2>  // Y0 -> M(0), Y1 -> N(0), Y2 -> N(2)
            >{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 0;
    }
};

} // namespace ck_tile
