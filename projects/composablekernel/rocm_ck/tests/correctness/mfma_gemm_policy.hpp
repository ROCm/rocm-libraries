// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// FP32 MFMA GEMM tile configuration and policy.
// Adapted from CK tutorial 03_mfma_16x16x16 for FP32×FP32→FP32.

#pragma once

#include "block_gemm_asmem_bsmem_creg_v1_accum.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"

namespace rocm_ck::test {

// Tile shape: 128×128 output per block, 16 K per LDS load iteration
static constexpr ck_tile::index_t kMfmaBlockSize = 256;
static constexpr ck_tile::index_t kMfmaBlockM    = 128;
static constexpr ck_tile::index_t kMfmaBlockN    = 128;
static constexpr ck_tile::index_t kMfmaBlockK    = 16;
static constexpr ck_tile::index_t kKPack         = 4; // 16 / sizeof(float)

// Simple shape struct (same interface as tutorial 03's local TileGemmShape)
struct MfmaBlockGemmShape
{
    static constexpr ck_tile::index_t kM = kMfmaBlockM;
    static constexpr ck_tile::index_t kN = kMfmaBlockN;
    static constexpr ck_tile::index_t kK = kMfmaBlockK;
};

// Block GEMM problem
struct MfmaBlockGemmProblem
{
    using ADataType      = float;
    using BDataType      = float;
    using CDataType      = float;
    using BlockGemmShape = MfmaBlockGemmShape;
    static constexpr ck_tile::index_t kBlockSize = kMfmaBlockSize;
};

// Policy: FP32 MFMA warp GEMM, 4×1 warp grid
struct MfmaBlockGemmPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        constexpr ck_tile::index_t kMWarp = 4;
        constexpr ck_tile::index_t kNWarp = 1;
        return ck_tile::make_tuple(
            ck_tile::WarpGemmMfmaF32F32F32M16N16K16<>{}, kMWarp, kNWarp);
    }
};

using MfmaBlockGemm =
    ck_tile::BlockGemmASmemBSmemCRegV1<MfmaBlockGemmProblem, MfmaBlockGemmPolicy>;

template <typename AccumPolicy>
using MfmaBlockGemmAccum =
    BlockGemmASmemBSmemCRegV1Accum<MfmaBlockGemmProblem, MfmaBlockGemmPolicy, AccumPolicy>;

template <typename AccumPolicy>
using MfmaBlockGemmAccumK4 =
    BlockGemmASmemBSmemCRegV1AccumK4<MfmaBlockGemmProblem, MfmaBlockGemmPolicy, AccumPolicy>;

template <typename AccumPolicy = AccumTwoSum>
using MfmaBlockGemmVeltkampK4 =
    BlockGemmASmemBSmemCRegV1VeltkampK4<MfmaBlockGemmProblem, MfmaBlockGemmPolicy, AccumPolicy>;

// LDS descriptors — PADDING_K_FIRST for A, plain for B
CK_TILE_HOST_DEVICE static constexpr auto MakeALdsDesc()
{
    using namespace ck_tile;
    constexpr auto desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK / kKPack>{}, number<kKPack>{}),
        make_tuple(number<(kMfmaBlockK / kKPack + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
        number<kKPack>{},
        number<1>{});
    return transform_tensor_descriptor(
        desc_0,
        make_tuple(make_pass_through_transform(kMfmaBlockM),
                   make_merge_transform(make_tuple(kMfmaBlockK / kKPack, kKPack))),
        make_tuple(sequence<0>{}, sequence<1, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
}

CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsDesc()
{
    using namespace ck_tile;
    constexpr auto desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK / kKPack>{}, number<kKPack>{}),
        make_tuple(number<kMfmaBlockK>{}, number<kKPack>{}, number<1>{}),
        number<kKPack>{},
        number<1>{});
    return transform_tensor_descriptor(
        desc_0,
        make_tuple(make_pass_through_transform(kMfmaBlockN),
                   make_merge_transform(make_tuple(kMfmaBlockK / kKPack, kKPack))),
        make_tuple(sequence<0>{}, sequence<1, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
}

// DRAM tile distributions for coalesced global loads
CK_TILE_HOST_DEVICE static constexpr auto MakeADramDist()
{
    using namespace ck_tile;
    constexpr index_t K1 = kKPack;
    constexpr index_t K0 = kMfmaBlockK / K1;
    constexpr index_t M2 = get_warp_size() / K0;
    constexpr index_t M1 = kMfmaBlockSize / get_warp_size();
    constexpr index_t M0 = kMfmaBlockM / (M2 * M1);

    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                   tuple<sequence<1>, sequence<1, 2>>,
                                   tuple<sequence<1>, sequence<2, 0>>,
                                   sequence<1, 2>,
                                   sequence<0, 1>>{});
}

CK_TILE_HOST_DEVICE static constexpr auto MakeBDramDist()
{
    using namespace ck_tile;
    constexpr index_t K1 = kKPack;
    constexpr index_t K0 = kMfmaBlockK / K1;
    constexpr index_t N2 = get_warp_size() / K0;
    constexpr index_t N1 = kMfmaBlockSize / get_warp_size();
    constexpr index_t N0 = kMfmaBlockN / (N2 * N1);

    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                   tuple<sequence<1>, sequence<1, 2>>,
                                   tuple<sequence<1>, sequence<2, 0>>,
                                   sequence<1, 2>,
                                   sequence<0, 1>>{});
}

// LDS size in bytes
CK_TILE_HOST_DEVICE static constexpr auto ComputeLdsSizes()
{
    constexpr auto a_desc = MakeALdsDesc();
    constexpr auto b_desc = MakeBLdsDesc();

    constexpr ck_tile::index_t a_bytes =
        ck_tile::integer_divide_ceil(
            static_cast<ck_tile::index_t>(sizeof(float)) * a_desc.get_element_space_size(), 16) *
        16;
    constexpr ck_tile::index_t b_bytes =
        static_cast<ck_tile::index_t>(sizeof(float)) * b_desc.get_element_space_size();

    return ck_tile::make_tuple(
        ck_tile::number<a_bytes>{}, ck_tile::number<b_bytes>{}, ck_tile::number<a_bytes + b_bytes>{});
}

static constexpr ck_tile::index_t kALdsBytes = decltype(ComputeLdsSizes().template at<0>())::value;
static constexpr ck_tile::index_t kBLdsBytes = decltype(ComputeLdsSizes().template at<1>())::value;
static constexpr ck_tile::index_t kLdsBytes  = decltype(ComputeLdsSizes().template at<2>())::value;

} // namespace rocm_ck::test
