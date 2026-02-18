// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include <cassert>
#include <cstdint>

namespace {

using namespace ck_tile;

/**
 * @struct RegisterIdxPair
 * @brief Small helper struct to hold a pair of lane and vector index values
 */
struct RegisterIdxPair
{
    uint32_t lane;
    uint32_t vecIdx;
};

/**
 * @class TileDistrEncodingMap
 * @brief Unified, static methods for register mapping
 *
 * Uses tile_distribution_encoding to compute the mapping
 * from matrix coordinates (m, k) or (m, n) to (lane, vecIdx) pairs.
 *
 * @tparam TileDistrEnc tile_distribution_encoding type
 * @tparam NumLanes Number of lanes in the wave
 * @tparam NumVecItems Number of vector items per lane
 */
template <typename TileDistrEnc, index_t NumLanes, index_t NumVecItems>
struct TileDistrEncodingMap
{
    static constexpr auto distr            = make_static_tile_distribution(TileDistrEnc{});
    static constexpr auto ps_ys_to_xs_adaptor = distr.get_ps_ys_to_xs_adaptor();

    /**
     * @brief Convert (lane, vecIdx) to matrix coordinates
     */
    CK_TILE_HOST_DEVICE static auto
    calc_matrix_indices_from_lane_vector(index_t lane_idx, index_t vector_idx)
    {
        // Unmerge the Y dimension index into its hidden indices
        array<index_t, TileDistrEnc::NDimY> y_hidden_idx;
        index_t vec_tmp = vector_idx;
        for(index_t i = TileDistrEnc::NDimY - 1; i >= 0; --i)
        {
            y_hidden_idx[i] = vec_tmp % TileDistrEnc::detail::ys_lengths_[i];
            vec_tmp /= TileDistrEnc::detail::ys_lengths_[i];
        }

        const auto ps_ys_idx = container_concat(array<index_t, 1>{lane_idx}, y_hidden_idx);
        const auto coord     = make_tensor_adaptor_coordinate(ps_ys_to_xs_adaptor, ps_ys_idx);
        return coord.get_bottom_index();
    }

    /**
     * @brief Convert matrix coordinates to (lane, vecIdx)
     */
    CK_TILE_HOST_DEVICE static RegisterIdxPair
    calc_lane_vector_from_matrix_indices(index_t major_idx, index_t minor_idx)
    {
        for(index_t l = 0; l < NumLanes; ++l)
        {
            for(index_t v = 0; v < NumVecItems; ++v)
            {
                auto res = calc_matrix_indices_from_lane_vector(l, v);
                if(res[0] == major_idx && res[1] == minor_idx)
                {
                    return {static_cast<uint32_t>(l), static_cast<uint32_t>(v)};
                }
            }
        }
        assert(false && "calc_lane_vector_from_matrix_indices: no mapping found");
        return {0, 0};
    }
};

/**
 * @class RegisterMapTraits
 * @brief Traits class that defines tile_distribution_encoding for each MmaOp
 * @tparam MmaOp amdgcn_mma specialization
 */
template <typename MmaOp>
struct RegisterMapTraits
{
    static_assert(sizeof(MmaOp) == 0, "RegisterMapTraits requires a specialization");
};

/**
 * @class RegisterMap
 * @brief Uses specialized RegisterMapTraits to get the encoding
 * @tparam MmaOp amdgcn_mma specialization
 */
template <typename MmaOp>
struct RegisterMap
{
    using Traits = RegisterMapTraits<MmaOp>;

    using AMap =
        TileDistrEncodingMap<typename Traits::AWarpDstrEncoding, Traits::WaveSize, Traits::AVecSize>;
    using BMap =
        TileDistrEncodingMap<typename Traits::BWarpDstrEncoding, Traits::WaveSize, Traits::BVecSize>;
    using CMap =
        TileDistrEncodingMap<typename Traits::CWarpDstrEncoding, Traits::WaveSize, Traits::CVecSize>;

    CK_TILE_HOST_DEVICE static RegisterIdxPair A2RegisterMap(const uint32_t m, const uint32_t k)
    {
        return AMap::calc_lane_vector_from_matrix_indices(static_cast<index_t>(m),
                                                          static_cast<index_t>(k));
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair B2RegisterMap(const uint32_t k, const uint32_t n)
    {
        return BMap::calc_lane_vector_from_matrix_indices(static_cast<index_t>(n),
                                                          static_cast<index_t>(k));
    }

    CK_TILE_HOST_DEVICE static RegisterIdxPair C2RegisterMap(const uint32_t m, const uint32_t n)
    {
        return CMap::calc_lane_vector_from_matrix_indices(static_cast<index_t>(m),
                                                          static_cast<index_t>(n));
    }
};

// ====================== Specializations per target =====================

/**
 * @brief RegisterMapTraits for GFX12 WMMA 16x16x16_F16_F16_F32_GFX12
 */
template <typename CtrlFlags, typename CompilerTarget>
struct RegisterMapTraits<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    ck_tile::core::arch::enable_if_target_family_gfx12_t<CompilerTarget>>>
{
    using MmaOp = ck_tile::core::arch::mma::amdgcn_mma<
        ck_tile::fp16_t,
        ck_tile::fp16_t,
        ck_tile::fp32_t,
        16u,
        16u,
        16u,
        CtrlFlags,
        CompilerTarget,
        ck_tile::core::arch::enable_if_target_family_gfx12_t<CompilerTarget>>;

    using MmaTraits = ck_tile::core::arch::mma::MmaOpTraits<MmaOp>;
    static constexpr index_t WaveSize =
        static_cast<index_t>(MmaTraits::CompilerTarget::WAVE_SIZE_ID);
    static constexpr index_t AVecSize =
        sizeof(typename MmaTraits::AVecType) / sizeof(typename MmaTraits::ADataType);
    static constexpr index_t BVecSize =
        sizeof(typename MmaTraits::BVecType) / sizeof(typename MmaTraits::BDataType);
    static constexpr index_t CVecSize =
        sizeof(typename MmaTraits::CVecType) / sizeof(typename MmaTraits::CDataType);

    // TODO: express the encoding in terms of amdgcn constants
    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<16>, sequence<4, 2, 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<2, 0>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<16>, sequence<4, 2, 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 0>>,
        sequence<2, 2>,
        sequence<2, 0>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<2, 8>, sequence<16>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<0, 0>>,
        sequence<1>,
        sequence<1>>;
};

/**
 * @brief RegisterMapTraits for GFX9 MFMA 16x16x16_F16_F16_F32_GFX9
 */
template <typename CtrlFlags, typename CompilerTarget>
struct RegisterMapTraits<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    ck_tile::core::arch::enable_if_target_family_gfx9_t<CompilerTarget>>>
{
    using MmaOp = ck_tile::core::arch::mma::amdgcn_mma<
        ck_tile::fp16_t,
        ck_tile::fp16_t,
        ck_tile::fp32_t,
        16u,
        16u,
        16u,
        CtrlFlags,
        CompilerTarget,
        ck_tile::core::arch::enable_if_target_family_gfx9_t<CompilerTarget>>;

    using MmaTraits = ck_tile::core::arch::mma::MmaOpTraits<MmaOp>;
    static constexpr index_t WaveSize =
        static_cast<index_t>(MmaTraits::CompilerTarget::WAVE_SIZE_ID);
    static constexpr index_t AVecSize =
        sizeof(typename MmaTraits::AVecType) / sizeof(typename MmaTraits::ADataType);
    static constexpr index_t BVecSize =
        sizeof(typename MmaTraits::BVecType) / sizeof(typename MmaTraits::BDataType);
    static constexpr index_t CVecSize =
        sizeof(typename MmaTraits::CVecType) / sizeof(typename MmaTraits::CDataType);

    // TODO: express the encoding in terms of amdgcn constants
    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<16>, sequence<4, 4>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<16>, sequence<4, 4>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<4, 4>, sequence<16>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<0, 0>>,
        sequence<1>,
        sequence<1>>;
};

/**
 * @brief RegisterMapTraits for GFX11 WMMA 16x16x16_F16_F16_F32_GFX11
 */
template <typename CtrlFlags, typename CompilerTarget>
struct RegisterMapTraits<ck_tile::core::arch::mma::amdgcn_mma<
    ck_tile::fp16_t,
    ck_tile::fp16_t,
    ck_tile::fp32_t,
    16u,
    16u,
    16u,
    CtrlFlags,
    CompilerTarget,
    ck_tile::core::arch::enable_if_target_family_gfx11_t<CompilerTarget>>>
{
    using MmaOp = ck_tile::core::arch::mma::amdgcn_mma<
        ck_tile::fp16_t,
        ck_tile::fp16_t,
        ck_tile::fp32_t,
        16u,
        16u,
        16u,
        CtrlFlags,
        CompilerTarget,
        ck_tile::core::arch::enable_if_target_family_gfx11_t<CompilerTarget>>;

    using MmaTraits = ck_tile::core::arch::mma::MmaOpTraits<MmaOp>;
    static constexpr index_t WaveSize =
        static_cast<index_t>(MmaTraits::CompilerTarget::WAVE_SIZE_ID);
    static constexpr index_t AVecSize =
        sizeof(typename MmaTraits::AVecType) / sizeof(typename MmaTraits::ADataType);
    static constexpr index_t BVecSize =
        sizeof(typename MmaTraits::BVecType) / sizeof(typename MmaTraits::BDataType);
    static constexpr index_t CVecSize =
        sizeof(typename MmaTraits::CVecType) / sizeof(typename MmaTraits::CDataType);

    // TODO: express the encoding in terms of amdgcn constants
    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<2>,
        tuple<sequence<16>, sequence<16>>,
        tuple<sequence<0, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<0>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<2>,
        tuple<sequence<16>, sequence<16>>,
        tuple<sequence<0, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<0>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<8, 2>, sequence<16>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 0>>,
        sequence<1>,
        sequence<0>>;
};

// ========================================================================

} // namespace

