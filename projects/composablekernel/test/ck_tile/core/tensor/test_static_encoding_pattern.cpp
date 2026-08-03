// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"

namespace {

using namespace ck_tile;

template <index_t XPerTile, index_t NumWaveGroups>
using ThreadRakedPattern =
    tile_distribution_encoding_pattern_2d<256,
                                          16,
                                          XPerTile,
                                          2,
                                          tile_distribution_pattern::thread_raked,
                                          NumWaveGroups>;

template <index_t XPerTile, index_t NumWaveGroups>
constexpr bool check_packed_iteration_order()
{
    constexpr auto regular =
        ThreadRakedPattern<XPerTile, NumWaveGroups>::make_2d_static_tile_distribution();
    constexpr auto packed =
        ThreadRakedPattern<XPerTile,
                           NumWaveGroups>::template make_2d_static_tile_distribution<true>();

    static_assert(regular.NDimP == 2 && regular.NDimY == 3);
    static_assert(packed.NDimP == 2 && packed.NDimY == 3);

    constexpr auto origin = array<index_t, 5>{0, 0, 0, 0, 0};
    constexpr auto y1     = array<index_t, 5>{0, 0, 0, 1, 0};
    constexpr auto y2     = array<index_t, 5>{0, 0, 0, 0, 1};

    constexpr auto regular_origin =
        regular.get_ps_ys_to_xs_adaptor().calculate_bottom_index(origin);
    constexpr auto regular_y1    = regular.get_ps_ys_to_xs_adaptor().calculate_bottom_index(y1);
    constexpr auto regular_y2    = regular.get_ps_ys_to_xs_adaptor().calculate_bottom_index(y2);
    constexpr auto packed_origin = packed.get_ps_ys_to_xs_adaptor().calculate_bottom_index(origin);
    constexpr auto packed_y1     = packed.get_ps_ys_to_xs_adaptor().calculate_bottom_index(y1);
    constexpr auto packed_y2     = packed.get_ps_ys_to_xs_adaptor().calculate_bottom_index(y2);

    // The regular layout keeps the historical <Y2, X2, X1> order. X2 is
    // contiguous in the source tensor, but X1 is the physical-minor
    // distributed-buffer dimension.
    static_assert(regular_y1[1] - regular_origin[1] == 1);
    static_assert(regular_y2[1] - regular_origin[1] ==
                  ThreadRakedPattern<XPerTile, NumWaveGroups>::X2);

    // Packed storage needs the source-contiguous X2 coordinate in the
    // physical-minor position. Otherwise logical offsets alias after they are
    // divided by PackedSize.
    static_assert(packed_y1[1] - packed_origin[1] ==
                  ThreadRakedPattern<XPerTile, NumWaveGroups>::X2);
    static_assert(packed_y2[1] - packed_origin[1] == 1);

    constexpr auto packed_d_origin =
        packed.get_ys_to_d_descriptor().calculate_offset(array<index_t, 3>{0, 0, 0});
    constexpr auto packed_d_y1 =
        packed.get_ys_to_d_descriptor().calculate_offset(array<index_t, 3>{0, 1, 0});
    constexpr auto packed_d_y2 =
        packed.get_ys_to_d_descriptor().calculate_offset(array<index_t, 3>{0, 0, 1});

    static_assert(packed_d_y1 - packed_d_origin == ThreadRakedPattern<XPerTile, NumWaveGroups>::X2);
    static_assert(packed_d_y2 - packed_d_origin == 1);

    return true;
}

#if defined(__HIP_DEVICE_COMPILE__)
// X=128 is the production gfx1100 A-tile shape. The host pass deliberately
// uses wave64, where this shape has X2=1 and therefore no reorderable Y2/X2/X1
// triplet; validate it only against the actual wave32 device contract.
static_assert(check_packed_iteration_order<128, 1>());
static_assert(check_packed_iteration_order<128, 2>());
#endif

// X=256 retains X2>1 in both host-wave64 and device-wave32 compilation, so it
// protects the generic packed branch in both passes.
static_assert(check_packed_iteration_order<256, 1>());
static_assert(check_packed_iteration_order<256, 2>());

} // namespace

TEST(StaticEncodingPattern, PackedThreadRakedIterationOrder) { SUCCEED(); }
