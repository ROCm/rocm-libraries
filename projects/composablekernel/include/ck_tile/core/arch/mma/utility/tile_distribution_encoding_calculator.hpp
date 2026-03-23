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
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"

namespace ck_tile::core::arch::mma {
template <typename MmaOp,
          bool CTranspose        = false,
          index_t SFactor        = 1,
          index_t AttrNumAccessA = MmaOp::kAKNumAccess,
          index_t AttrNumAccessB = MmaOp::kBKNumAccess>
struct TileDistrEncCalc
{
    private:
    static constexpr index_t NumAccessA = std::max(MmaOp::kAKNumAccess, AttrNumAccessA);
    static constexpr index_t NumAccessB = std::max(MmaOp::kBKNumAccess, AttrNumAccessB);

    static_assert(AttrNumAccessA >= MmaOp::kAKNumAccess,
                  "Requesting smaller NumAccessA than required by builtin.");
    static_assert(AttrNumAccessB >= MmaOp::kBKNumAccess,
                  "Requesting smaller NumAccessB than required by builtin.");

    static_assert(MmaOp::kABKPerLane % NumAccessA == 0);
    static_assert(MmaOp::kABKPerLane % NumAccessB == 0);
    static_assert(SFactor == 1, "Swizzle not implemented yet."); // TODO: Implement Swizzle.

    template <index_t MajorDimSize, index_t Repeat, index_t NumAccess, index_t CompressionRatio = 1>
    using ABWarpDstrEnc = tile_distribution_encoding<
        sequence<Repeat>,
        tuple<sequence<MajorDimSize>,
              sequence<NumAccess,
                       MmaOp::kK / MmaOp::kABKPerLane,
                       MmaOp::kABKPerLane / NumAccess / CompressionRatio>>,
        tuple<sequence<0, 2, 1>>,
        tuple<sequence<0, 1, 0>>,
        sequence<2, 2>,
        sequence<0, 2>>;

    static constexpr auto get_cwarp_dstr_encoding()
    {
        // We unmerge the M and N dimensions in the same way every time.
        using MSubDims = sequence<MmaOp::kCMBlocks,
                                  MmaOp::kCMNumAccess,
                                  MmaOp::kM / MmaOp::kCMBlocks / MmaOp::kCMPerLane,
                                  MmaOp::kCMPerLane / MmaOp::kCMNumAccess>;
        using NSubDims = sequence<MmaOp::kCNBlocks, MmaOp::kN / MmaOp::kCNBlocks>;

        // In case of CTranspose, all we do is swap the M and N dimension.
        using MatDims =
            std::conditional_t<CTranspose, tuple<NSubDims, MSubDims>, tuple<MSubDims, NSubDims>>;
        constexpr int MInx = CTranspose ? 2 : 1;
        constexpr int NInx = CTranspose ? 1 : 2;

        // For MFMA intrinsics with blocks, the block dimensions might be in the Lane dim or in the
        // Vec dim, so we get different merge orderings.
        if constexpr(MmaOp::CBlockDimInVecDim)
        {
            return tile_distribution_encoding<sequence<1>,
                                              MatDims,
                                              tuple<sequence<MInx, NInx>>,
                                              tuple<sequence<2, 1>>,
                                              sequence<MInx, NInx, MInx, MInx>,
                                              sequence<0, 0, 1, 3>>{};
        }
        else
        {
            return tile_distribution_encoding<sequence<1>,
                                              MatDims,
                                              tuple<sequence<MInx, NInx, MInx, NInx>>,
                                              tuple<sequence<0, 0, 2, 1>>,
                                              sequence<MInx, MInx>,
                                              sequence<1, 3>>{};
        }
    }

    using AEnc_ = ABWarpDstrEnc<MmaOp::kM, MmaOp::kARepeat, NumAccessA, MmaOp::kCompressionRatio>;
    using BEnc_ = ABWarpDstrEnc<MmaOp::kN, MmaOp::kBRepeat, NumAccessB>;

    public:
    // When using CTranspose, the A and B matrices are swapped.
    using AWarpDstrEncoding = std::conditional_t<CTranspose, BEnc_, AEnc_>;
    using BWarpDstrEncoding = std::conditional_t<CTranspose, AEnc_, BEnc_>;
    using CWarpDstrEncoding = decltype(get_cwarp_dstr_encoding());

    // Some additional consistency checks
    static_assert(TileDistrEncRegMap<AWarpDstrEncoding>::num_lanes == MmaOp::WaveSize);
    static_assert(TileDistrEncRegMap<BWarpDstrEncoding>::num_lanes == MmaOp::WaveSize);
    static_assert(TileDistrEncRegMap<CWarpDstrEncoding>::num_lanes == MmaOp::WaveSize);

    static_assert(TileDistrEncRegMap<AWarpDstrEncoding>::num_vector_items ==
                  vector_traits<typename MmaOp::AVecType>::vector_size);
    static_assert(TileDistrEncRegMap<BWarpDstrEncoding>::num_vector_items ==
                  vector_traits<typename MmaOp::BVecType>::vector_size);
    static_assert(TileDistrEncRegMap<CWarpDstrEncoding>::num_vector_items ==
                  vector_traits<typename MmaOp::CVecType>::vector_size);
};
} // namespace ck_tile::core::arch::mma
