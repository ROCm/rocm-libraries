// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <limits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_e_permute_kernel.hpp"

namespace ck_tile {

/// @brief Compile-time part of an A-stationary GEMM: data types, K and the workgroup geometry.
///        A workgroup keeps an A tile [MPerBlock, K] in LDS; its Waves waves form RowGroups x
///        (Waves / RowGroups) and walk [MPerBlock / RowGroups, NPerWave] output tiles over N.
///        StrideM0 > 0 fixes the E row stride at compile time (0: run time); the store address
///        math is then folded into immediates.
template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename EDataType_,
          index_t kMPerBlock_,
          index_t kK_,
          index_t kNPerWave_,
          index_t kWaves_,
          index_t kRowGroups_,
          index_t kStrideM0_ = 0>
struct GemmAStationaryProblem
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using AccDataType = AccDataType_;
    using EDataType   = EDataType_;

    static constexpr index_t kMPerBlock = kMPerBlock_;
    static constexpr index_t kK         = kK_;
    static constexpr index_t kNPerWave  = kNPerWave_;
    static constexpr index_t kWaves     = kWaves_;
    static constexpr index_t kRowGroups = kRowGroups_;
    static constexpr index_t kStrideM0  = kStrideM0_;
};

/// @brief Run-time arguments: A [M, K] with row stride stride_A, the weight in the pipeline's flat
///        layout, and E viewed as [M0, M1, N0, N1] (m = m0 * M1 + m1, n = n0 * N1 + n1).
struct GemmAStationaryKernelArgs
{
    const void* a_ptr;
    const void* b_flat_ptr;
    void* e_ptr;
    index_t K;
    index_t stride_A;
    index_t k_batch; // read by IsSupportedArgument only
    GemmEPermuteDesc e;
};

/// @brief What the A-stationary pipelines share: geometry, tile addressing and argument checks.
template <typename Problem_>
struct GemmPipelineAStationaryBase
{
    using Problem   = remove_cvref_t<Problem_>;
    using ADataType = typename Problem::ADataType;
    using EDataType = typename Problem::EDataType;

    static constexpr index_t kMPerBlock   = Problem::kMPerBlock;
    static constexpr index_t kK           = Problem::kK;
    static constexpr index_t kNPerWave    = Problem::kNPerWave;
    static constexpr index_t kWaves       = Problem::kWaves;
    static constexpr index_t kRowGroups   = Problem::kRowGroups;
    static constexpr index_t kStrideM0    = Problem::kStrideM0;
    static constexpr index_t kColGroups   = kWaves / kRowGroups;
    static constexpr index_t kRowsPerWave = kMPerBlock / kRowGroups;

    static_assert(kWaves % kRowGroups == 0 && kMPerBlock % kRowGroups == 0);

    CK_TILE_HOST_DEVICE static constexpr index_t WaveTiles(index_t N)
    {
        return N / (kNPerWave * kColGroups);
    }
    /// First GEMM column of wave tile i of column group col_group.
    CK_TILE_HOST_DEVICE static constexpr index_t TileN0(index_t col_group, index_t i)
    {
        return (col_group + kColGroups * i) * kNPerWave;
    }
    /// A row m0 * M1 + m1; the rows of a tile are M1 * stride_A apart.
    CK_TILE_DEVICE static const ADataType*
    ARow0(const GemmAStationaryKernelArgs& kargs, index_t m0, index_t m1)
    {
        return static_cast<const ADataType*>(kargs.a_ptr) +
               (long_index_t{m0} * kargs.e.M1 + m1) * kargs.stride_A;
    }
    /// E element (m0, m1, n); the rows of a tile are stride_M0 apart. n must not cross an n0.
    CK_TILE_DEVICE static EDataType*
    ETile(const GemmAStationaryKernelArgs& kargs, index_t m0, index_t m1, index_t n)
    {
        const auto& e = kargs.e;
        return static_cast<EDataType*>(kargs.e_ptr) + long_index_t{m1} * e.stride_M1 +
               long_index_t{n / e.N1} * e.stride_N0 + n % e.N1 + long_index_t{m0} * e.stride_M0;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const GemmAStationaryKernelArgs& kargs)
    {
        const auto reject = [](const char* reason) {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("GemmAStationary: ", reason);
            }
            return false;
        };
        const auto& e = kargs.e;
        if(kargs.k_batch != 1)
        {
            return reject("split-K (k_batch != 1) is not supported");
        }
        if(kargs.K != kK)
        {
            return reject("K must equal the compile-time K of the kernel");
        }
        if(kStrideM0 > 0 && e.stride_M0 != kStrideM0)
        {
            return reject("stride_M0 must equal the compile-time E row stride of the kernel");
        }
        if(std::min({e.M0, e.M1, e.N0, e.N1}) < 1 ||
           std::min({kargs.stride_A, e.stride_M0, e.stride_M1, e.stride_N0}) < 0)
        {
            return reject("needs M0, M1, N0, N1 >= 1 and strides >= 0");
        }
        // N and the workgroup count are index_t on the device.
        constexpr long_index_t index_max = std::numeric_limits<index_t>::max();
        const long_index_t n             = long_index_t{e.N0} * e.N1;
        if(n > index_max || long_index_t{e.M1} * integer_divide_ceil(e.M0, kMPerBlock) > index_max)
        {
            return reject("N0 * N1 and the workgroup count must fit in index_t");
        }
        if(gemm_e_permute_span(e) < 0)
        {
            return reject("strides must be positive and must not make E overlap itself");
        }
        // A wave tile must not cross an n0 boundary, and the column groups tile N exactly.
        if(e.N1 % kNPerWave != 0 || n % (kNPerWave * kColGroups) != 0)
        {
            return reject("N1 must be a multiple of NPerWave and N of NPerWave * column groups");
        }
        // 16-byte transfers: pointers and every row and tile stride.
        const auto unaligned = [](long_index_t bytes) { return bytes % 16 != 0; };
        if(unaligned(reinterpret_cast<uintptr_t>(kargs.a_ptr)) ||
           unaligned(reinterpret_cast<uintptr_t>(kargs.b_flat_ptr)) ||
           unaligned(reinterpret_cast<uintptr_t>(kargs.e_ptr)) ||
           unaligned(long_index_t{kargs.stride_A} * sizeof(ADataType)) ||
           unaligned(long_index_t{e.stride_M0} * sizeof(EDataType)) ||
           unaligned(long_index_t{e.stride_M1} * sizeof(EDataType)) ||
           unaligned(long_index_t{e.stride_N0} * sizeof(EDataType)))
        {
            return reject("pointers and strides must be 16-byte aligned");
        }
        // 32-bit per-lane offsets from a tile's first A row and from a wave tile's first E row.
        constexpr long_index_t limit = long_index_t{1} << 31;
        if(((kMPerBlock - 1) * long_index_t{e.M1} * kargs.stride_A + kK) * sizeof(ADataType) >=
               limit ||
           long_index_t{kRowsPerWave} * e.stride_M0 * sizeof(EDataType) >= limit)
        {
            return reject("the A tile and E wave tile spans must be below 2 GB");
        }
        return true;
    }
};

} // namespace ck_tile
