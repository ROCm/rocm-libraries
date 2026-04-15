// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"

namespace ck_tile {

struct WarpDecodePolicy
{
    // [1, WAVE_SIZE * V] tile: each thread gets V consecutive elements.
    // V comes from Problem::kVector (default 1, use 2 for pk_fp4_t).
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeTileDistribution()
    {
        constexpr index_t V = Problem::kVector;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<1>,                    // M: [Warps=1]
                    sequence<get_warp_size(), V>    // N: [Lanes=WAVE_SIZE, Vector=V]
                >,
                tuple<sequence<1>, sequence<2>>,
                tuple<sequence<0>, sequence<0>>,
                sequence<2>,
                sequence<1>
            >{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 0;
    }
};

} // namespace ck_tile
