// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/page_block_navigator.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"

namespace ck_tile {

namespace detail {

template <typename Problem>
inline constexpr bool is_qr_tdm_pingpong_qkv_v =
    (std::is_same_v<typename Problem::QDataType, bf16_t> &&
     std::is_same_v<typename Problem::KDataType, bf16_t> &&
     std::is_same_v<typename Problem::VDataType, bf16_t>) ||
    (std::is_same_v<typename Problem::QDataType, half_t> &&
     std::is_same_v<typename Problem::KDataType, half_t> &&
     std::is_same_v<typename Problem::VDataType, half_t>) ||
    (std::is_same_v<typename Problem::QDataType, fp8_t> &&
     std::is_same_v<typename Problem::KDataType, fp8_t> &&
     std::is_same_v<typename Problem::VDataType, fp8_t>);

// One pad per row, expressed as a row stride: the layout the TDM writer produces with a
// padding interval of one row, with affine reader address math (no merge/unmerge div/mod).
template <typename DataType, index_t Rows, index_t Cols, typename PaddingConfig>
CK_TILE_HOST_DEVICE constexpr auto make_qr_tdm_pingpong_lds_descriptor()
{
    static_assert(numeric_traits<DataType>::PackedSize == 1);
    constexpr index_t kAccessElements = kQrTdmLdsAccessBytes / sizeof(DataType);
    constexpr index_t kPadElements    = PaddingConfig::kPadBytes / sizeof(DataType);

    static_assert(!PaddingConfig::kEnabled ||
                      (PaddingConfig::kIntervalBytes == Cols * sizeof(DataType) &&
                       PaddingConfig::kPadBytes % sizeof(DataType) == 0),
                  "qr_tdm_pingpong pads each LDS row once");
    return make_naive_tensor_descriptor(make_tuple(number<Rows>{}, number<Cols>{}),
                                        make_tuple(number<Cols + kPadElements>{}, number<1>{}),
                                        number<kAccessElements>{},
                                        number<1>{});
}

// Rows padded once need a power-of-2 row width, like the TDM padding interval.
template <typename Problem>
inline constexpr bool is_qr_tdm_pingpong_padding_enabled_v =
    is_qr_tdm_padding_supported_arch_v && Problem::BlockFmhaShape::IsVLayoutRowMajor &&
    is_power_of_two_integer(Problem::BlockFmhaShape::kSubQKHeaddim *
                            sizeof(typename Problem::KDataType)) &&
    is_power_of_two_integer(Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType)) &&
    Problem::BlockFmhaShape::kSubQKHeaddim * sizeof(typename Problem::KDataType) >= 16 &&
    Problem::BlockFmhaShape::kSubQKHeaddim * sizeof(typename Problem::KDataType) <= 1024 &&
    Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType) >= 16 &&
    Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType) <= 1024;

// 16 B per Q and K row (row-strided ds_load_b128), 32 B per V row (transposed ds_load_tr).
template <typename Problem, bool Enabled = is_qr_tdm_pingpong_padding_enabled_v<Problem>>
struct QrTdmPingPongPadding
{
    using Q = LdsPaddingConfig<false, 0, 0>;
    using K = LdsPaddingConfig<false, 0, 0>;
    using V = LdsPaddingConfig<false, 0, 0>;
};

template <typename Problem>
struct QrTdmPingPongPadding<Problem, true>
{
    using Shape = typename Problem::BlockFmhaShape;

    using Q = LdsPaddingConfig<true,
                               Shape::kSubQKHeaddim * sizeof(typename Problem::QDataType),
                               kQrTdmLdsAccessBytes>;
    using K = LdsPaddingConfig<true,
                               Shape::kSubQKHeaddim * sizeof(typename Problem::KDataType),
                               kQrTdmLdsAccessBytes>;
    using V = LdsPaddingConfig<true,
                               Shape::kN1 * sizeof(typename Problem::VDataType),
                               2 * kQrTdmLdsAccessBytes>;
};

// [K buffer 0][K buffer 1][S][V, aliased by Q]. Q is read once before the main loop and
// V's first write follows the loop-head barrier, so Q sits on V and K's first transfer can
// be issued right behind Q's. S only takes space with the gemm_0 warps on N. d128
// fp16/bf16, padded: K0=0, K1=17408, Q/V0=34816, end=53248 (warps on M, M=64); the
// 16-row short-query tile: S=34800, Q/V0=39168, end=57600.
template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct QrTdmPingPongLdsArenaLayout
{
    using Shape = typename Problem::BlockFmhaShape;

    static constexpr index_t kArenaAlignment  = 256;
    static constexpr index_t kRegionAlignment = 256;

    static constexpr index_t kQBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::QDataType,
                                            Shape::kM0,
                                            Shape::kSubQKHeaddim,
                                            QPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::QDataType);
    static constexpr index_t kKBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::KDataType,
                                            Shape::kN0,
                                            Shape::kSubQKHeaddim,
                                            KPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::KDataType);
    static constexpr index_t kVBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::VDataType,
                                            Shape::kN0,
                                            Shape::kN1,
                                            VPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::VDataType);

    static constexpr index_t kSBytes =
        BlockFmhaPipelineQRKSVSTdmDefaultPolicy::template GetSmemSizeS<Problem>();
    static constexpr index_t kSAlignment =
        BlockFmhaPipelineQRKSVSTdmDefaultPolicy::template GetSmemNPackS<Problem>() *
        sizeof(typename Problem::SaccDataType);

    static constexpr index_t kK0Offset = 0;
    static constexpr index_t kK1Offset =
        integer_least_multiple(kK0Offset + kKBytes, kRegionAlignment);
    static constexpr index_t kSOffset = integer_least_multiple(kK1Offset + kKBytes, kSAlignment);
    static constexpr index_t kV0Offset =
        integer_least_multiple(kSOffset + kSBytes, kRegionAlignment);
    static constexpr index_t kQOffset = kV0Offset;
    static constexpr index_t kArenaBytes =
        integer_least_multiple(max(kQOffset + kQBytes, kV0Offset + kVBytes), kArenaAlignment);

    static_assert(!KPadding::kEnabled || (kK1Offset - kK0Offset) % KPadding::kIntervalBytes == 0);
    static_assert(kArenaBytes <= 128 * 1024);
};

} // namespace detail

// qr_tdm_pingpong: the qr_tdm policy with a single V buffer, whole-tile K staged in two
// buffers, and Q on V (see QrTdmPingPongLdsArenaLayout).
struct BlockFmhaPipelineQRKSVSTdmPingPongPolicy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    template <typename Problem>
    using LdsPaddingConfigQ = typename detail::QrTdmPingPongPadding<Problem>::Q;

    template <typename Problem>
    using LdsPaddingConfigK = typename detail::QrTdmPingPongPadding<Problem>::K;

    template <typename Problem>
    using LdsPaddingConfigV = typename detail::QrTdmPingPongPadding<Problem>::V;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmPingPongLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::QDataType,
                                                           Problem::BlockFmhaShape::kM0,
                                                           Problem::BlockFmhaShape::kSubQKHeaddim,
                                                           LdsPaddingConfigQ<Problem>>();
    }

    // The whole K tile; the pipeline walks its kK0 slices with the read window.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::KDataType,
                                                           Problem::BlockFmhaShape::kN0,
                                                           Problem::BlockFmhaShape::kSubQKHeaddim,
                                                           LdsPaddingConfigK<Problem>>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::VDataType,
                                                           Problem::BlockFmhaShape::kN0,
                                                           Problem::BlockFmhaShape::kN1,
                                                           LdsPaddingConfigV<Problem>>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return LdsArenaLayout<Problem>::kArenaBytes;
    }

    // S comes back out of LDS (gemm_0 warps on N) to be softmaxed and fed into gemm_1
    // as its A operand, so it takes P's distribution. An encoding built from the QK
    // gemm's warp layout would permute P's columns within a row once the warps sit on
    // N: the row sums (and LSE) stay right while each key meets the wrong value row.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        return MakePRegTileDistribution<Problem>();
    }

    // Row sum of P on the matrix unit: P (M x kN0) times ones (kN0 x one warp tile), with
    // the PV gemm's types and warp gemm, so the sum lands replicated in o_acc's layout.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetRowSumBlockGemm()
    {
        using PVBlockGemm = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        using WarpGemm    = typename PVBlockGemm::WarpGemm;

        static_assert(Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{}) == 1,
                      "one warp tile of columns covers the row sum only with the gemm_1 "
                      "warps all on M");

        using GemmProblem = BlockGemmProblem<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            Problem::kBlockSize,
            TileGemmShape<
                sequence<Problem::BlockFmhaShape::kM0, WarpGemm::kN, Problem::BlockFmhaShape::kK1>,
                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                typename Problem::VDataType,
                                                typename Problem::OaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::KMN>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }
};

// qr_tdm_pingpong on 5D paged KV (see Paged5DTileNavigator), one page per K tile: each warp's
// TDM box copies whole rows of a page into unpadded LDS, which keeps the page layout, and the
// reads go through that layout. V needs no transposed read: a page holds V token-minor.
struct BlockFmhaPipelineQRKSVSTdmPingPongPaged5DPolicy : BlockFmhaPipelineQRKSVSTdmPingPongPolicy
{
    template <typename Problem>
    using Page = Paged5DTileNavigator<typename Problem::KDataType,
                                      typename Problem::VDataType,
                                      Problem::BlockFmhaShape::kN0,
                                      Problem::BlockFmhaShape::kQKHeaddim>;

    template <typename Problem>
    using LdsPaddingConfigK = detail::LdsPaddingConfig<false, 0, 0>;

    template <typename Problem>
    using LdsPaddingConfigV = detail::LdsPaddingConfig<false, 0, 0>;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmPingPongLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return LdsArenaLayout<Problem>::kArenaBytes;
    }

    // Rows x Cols, each warp taking whole rows: one TDM box per warp
    template <typename Problem, index_t Rows, index_t Cols>
    CK_TILE_HOST_DEVICE static constexpr auto MakeWarpRowsDramTileDistribution()
    {
        constexpr index_t warpNum = Problem::kBlockSize / get_warp_size();
        static_assert(Rows % warpNum == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, Rows / warpNum>, sequence<Cols>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        using P = Page<Problem>;
        return MakeWarpRowsDramTileDistribution<Problem, P::kKRows, P::kPageSize * P::kX>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using P = Page<Problem>;
        return MakeWarpRowsDramTileDistribution<Problem, P::kVRows, P::kHeadDim * P::kX>();
    }

    // see Paged5DTileNavigator::make_v_tail_view: warp w copies the X tokens of rows
    // [w * kHeadDim, (w + 1) * kHeadDim), the same bytes as its row of the full V box
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVTailDramTileDistribution()
    {
        using P = Page<Problem>;
        static_assert(Problem::kBlockSize / get_warp_size() == P::kVRows);
        return MakeWarpRowsDramTileDistribution<Problem, P::kVRows * P::kHeadDim, P::kX>();
    }

    template <typename DataType, index_t Rows, index_t Cols>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDenseLdsDescriptor()
    {
        constexpr index_t kX = 16 / sizeof(DataType);
        return make_naive_tensor_descriptor(make_tuple(number<Rows>{}, number<Cols>{}),
                                            make_tuple(number<Cols>{}, number<1>{}),
                                            number<kX>{},
                                            number<1>{});
    }

    // the TDM-written K / V buffers, a page each
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        using P = Page<Problem>;
        return MakeDenseLdsDescriptor<typename Problem::KDataType,
                                      P::kKRows,
                                      P::kPageSize * P::kX>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        using P = Page<Problem>;
        return MakeDenseLdsDescriptor<typename Problem::VDataType,
                                      P::kVRows,
                                      P::kHeadDim * P::kX>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVTailLdsBlockDescriptor()
    {
        using P = Page<Problem>;
        return MakeDenseLdsDescriptor<typename Problem::VDataType,
                                      P::kVRows * P::kHeadDim,
                                      P::kX>();
    }

    // (Inner, Outer * X) over [Outer][Inner][X]. A WMMA operand lane holds 8 consecutive k, which
    // stay inside one X block, so the reads keep the vector.
    template <index_t Outer, index_t Inner, index_t X>
    CK_TILE_HOST_DEVICE static constexpr auto MakePageReadDescriptor()
    {
        constexpr auto page =
            make_naive_tensor_descriptor(make_tuple(number<Outer>{}, number<Inner>{}, number<X>{}),
                                         make_tuple(number<Inner * X>{}, number<X>{}, number<1>{}),
                                         number<X>{},
                                         number<1>{});

        return transform_tensor_descriptor(
            page,
            make_tuple(
                make_pass_through_transform(number<Inner>{}),
                make_merge_transform_v3_division_mod(make_tuple(number<Outer>{}, number<X>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // K as (token, d)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsReadDescriptor()
    {
        using P = Page<Problem>;
        return MakePageReadDescriptor<P::kKRows, P::kPageSize, P::kX>();
    }

    // V as (d, token)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadDescriptor()
    {
        using P = Page<Problem>;
        return MakePageReadDescriptor<P::kVRows, P::kHeadDim, P::kX>();
    }

    // the PV gemm's B operand, read untransposed
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        using PVBlockGemm = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        return make_static_tile_distribution(
            decltype(PVBlockGemm::MakeBBlockDistributionEncode()){});
    }
};

} // namespace ck_tile
