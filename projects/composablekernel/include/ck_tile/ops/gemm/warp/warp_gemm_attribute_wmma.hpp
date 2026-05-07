// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

namespace ck_tile {

// TODO: currently only support 16 bit input, which means only support tr16_b128; will use ADataType
// to determine the layout in the future
template <typename Impl>
struct AWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kAMBlock, Impl::kAMLane>,
              sequence<Impl::kAK0PerLane, Impl::kABKLane, Impl::kAK1PerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct BWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kBNBlock, Impl::kBNLane>,
              sequence<Impl::kBK0PerLane, Impl::kABKLane, Impl::kBK1PerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct CWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCMBlock, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNBlock, Impl::kCNLane>>,
        tuple<typename Impl::kCPs2RHssMajor>,
        tuple<typename Impl::kCPs2RHssMinor>,
        typename Impl::kCYs2RHsMajor,
        typename Impl::kCYs2RHsMinor>;
};

template <typename Impl>
struct CTransposedWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCNBlock, Impl::kCNLane>,
              sequence<Impl::kCMBlock, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        tuple<typename Impl::kCTPs2RHssMajor>,
        tuple<typename Impl::kCTPs2RHssMinor>,
        typename Impl::kCTYs2RHsMajor,
        typename Impl::kCTYs2RHsMinor>;
};

template <typename WarpGemmAttributeWmmaImpl_, bool kTransC = false>
struct WarpGemmAttributeWmma
{
    using Impl = remove_cvref_t<WarpGemmAttributeWmmaImpl_>;

    // When kTransC is true and A/B types differ, we need an impl with swapped types
    using TransposedImpl =
        std::conditional_t<kTransC &&
                               !std::is_same_v<typename Impl::ADataType, typename Impl::BDataType>,
                           WarpGemmAttributeWmmaImpl<WmmaTraits<typename Impl::TraitsType::ArchType,
                                                                typename Impl::BDataType,
                                                                typename Impl::ADataType,
                                                                typename Impl::CDataType,
                                                                Impl::kM,
                                                                Impl::kN,
                                                                Impl::kK>>,
                           Impl>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM      = Impl::kM;
    static constexpr index_t kN      = Impl::kN;
    static constexpr index_t kK      = Impl::kK;
    static constexpr index_t kCMLane = Impl::kCMLane;

    static_assert(Impl::kAK0PerLane * Impl::kAK1PerLane == Impl::kBK0PerLane * Impl::kBK1PerLane);
    static constexpr index_t kKPerThread = Impl::kAK0PerLane * Impl::kAK1PerLane;
    static constexpr index_t kKPack      = Impl::kAK1PerLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    // 16 bit input, kAMLane = 16, kABK0PerLane = 4, kABKLane = 2, kABK1PerLane = 2
    // 8  bit input, kAMLane = 16, kABK0PerLane = 2, kABKLane = 2, kABK1PerLane = 4
    using AWarpDstrEncoding = typename AWarpDstrEncodingTrait<Impl>::type;
    using BWarpDstrEncoding = typename BWarpDstrEncodingTrait<Impl>::type;

    // kCM0PerLane = 1, kCMLane = 2, kCM1PerLane = 2, kCNLane = 16
    using CWarpDstrEncoding =
        std::conditional_t<kTransC,
                           typename CTransposedWarpDstrEncodingTrait<Impl>::type,
                           typename CWarpDstrEncodingTrait<Impl>::type>;

    // c_vec += a_vec * b_vec
    template <typename... Params>
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            TransposedImpl{}.template operator()<Params..., SwapReuse_<true>>(c_vec, b_vec, a_vec);
        }
        else
        {
            Impl{}.template operator()<Params...>(c_vec, a_vec, b_vec);
        }
    }

    // c_vec = a_vec * b_vec
    template <typename... Params>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            return TransposedImpl{}.template operator()<Params..., SwapReuse_<true>>(b_vec, a_vec);
        }
        else
        {
            return Impl{}.template operator()<Params...>(a_vec, b_vec);
        }
    }

    // c_vec += a_vec * b_vec
    template <typename... Params>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const int32_t& a_scale,
                                   const BVecType& b_vec,
                                   const int32_t& b_scale) const
    {
        if constexpr(kTransC)
        {
            TransposedImpl{}.template operator()<Params..., SwapReuse_<true>>(
                c_vec, b_vec, b_scale, a_vec, a_scale);
        }
        else
        {
            Impl{}.template operator()<Params...>(c_vec, a_vec, a_scale, b_vec, b_scale);
        }
    }

    // c_vec = a_vec * b_vec
    template <typename... Params>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const int32_t& a_scale,
                                       const BVecType& b_vec,
                                       const int32_t& b_scale) const
    {
        if constexpr(kTransC)
        {
            return TransposedImpl{}.template operator()<Params..., SwapReuse_<true>>(
                b_vec, b_scale, a_vec, a_scale);
        }
        else
        {
            return Impl{}.template operator()<Params...>(a_vec, a_scale, b_vec, b_scale);
        }
    }
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          index_t M_Warp_Tile,
          index_t N_Warp_Tile,
          index_t K_Warp_Tile>
CK_TILE_HOST bool check_wmma_supported()
{
    if(is_gfx120_supported())
    {
        return has_wmma_traits_v<gfx120_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else if(is_gfx125_supported())
    {
        return has_wmma_traits_v<gfx125_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else if(is_gfx11_supported())
    {
        return has_wmma_traits_v<gfx11_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else if(is_gfx13_supported())
    {
        return has_wmma_traits_v<gfx13_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else
    {
        return false;
    }
}

} // namespace ck_tile
