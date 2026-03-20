// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file tile_distribution_encoding_calculator.hpp
 * @brief Generate TileDistributionEncodings describing register mappings for any MmaOp
 * @details Defines TileDistrEncCalc, which takes an amdgcn_mma type (MmaOp) and provides
 * ABC warp distribution encodings for mapping matrix fragment coordinates to register coordinates
 * (lane, vector item) and vice versa.
 */

#pragma once

#include "ck_tile/core/tensor/tile_distribution.hpp"

namespace ck_tile::core::arch::mma {
template <typename MmaOp,
          bool CTranspose        = false,
          index_t SFactor        = 1,
          index_t AttrNumAccessA = MmaOp::kAKNumAccess,
          index_t AttrNumAccessB = MmaOp::kBKNumAccess>
struct TileDistrEncCalc
{
    static constexpr index_t NumAccessA = std::max(MmaOp::kAKNumAccess, AttrNumAccessA);
    static constexpr index_t NumAccessB = std::max(MmaOp::kBKNumAccess, AttrNumAccessB);

    static_assert(AttrNumAccessA >= MmaOp::kAKNumAccess,
                  "Requesting smaller NumAccessA than required by builtin.");
    static_assert(AttrNumAccessB >= MmaOp::kBKNumAccess,
                  "Requesting smaller NumAccessB than required by builtin.");

    static_assert(MmaOp::kABKPerLane % NumAccessA == 0);
    static_assert(MmaOp::kABKPerLane % NumAccessB == 0);
    static_assert(SFactor == 1, "Swizzle not implementeed yet."); // TODO: Implement Swizzle.
    static_assert(CTranspose == false,
                  "CTranspose not implemented yet."); // TODO: Implement CTranspose.

    template <index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEncoding = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<MmaOp::kM>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio>>,
        tuple<sequence<0, 2, 1>>,
        tuple<sequence<0, 1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    static constexpr auto get_cwarp_dstr_encoding()
    {
        if constexpr(MmaOp::CBlockDimInVecDim)
        {
            return tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<MmaOp::kCMBlocks,
                               MmaOp::kCMNumAccess,
                               MmaOp::kM / MmaOp::kCMBlocks / MmaOp::kCMPerLane,
                               MmaOp::kCMPerLane / MmaOp::kCMNumAccess>,
                      sequence<MmaOp::kCNBlocks, MmaOp::kN / MmaOp::kCNBlocks>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<2, 1>>,
                sequence<1, 2, 1, 1>,
                sequence<0, 0, 1, 3>>{};
        }
        else
        {
            return tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<MmaOp::kCMBlocks,
                               MmaOp::kCMNumAccess,
                               MmaOp::kM / MmaOp::kCMBlocks / MmaOp::kCMPerLane,
                               MmaOp::kCMPerLane / MmaOp::kCMNumAccess>,
                      sequence<MmaOp::kCNBlocks, MmaOp::kN>>,
                tuple<sequence<1, 2, 1, 2>>,
                tuple<sequence<0, 0, 2, 1>>,
                sequence<1, 1>,
                sequence<1, 3>>{};
        }
    }

    using AWarpDstrEncoding =
        ABWarpDstrEncoding<MmaOp::kARepeat, NumAccessA, MmaOp::kCompressionRatio>;
    using BWarpDstrEncoding = ABWarpDstrEncoding<MmaOp::kBRepeat, NumAccessB>;
    using CWarpDstrEncoding = decltype(get_cwarp_dstr_encoding());
};
} // namespace ck_tile::core::arch::mma
