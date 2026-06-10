// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Sparge-only fork of upstream BlockFmhaPipelineQXKSVSCustomPolicy
// (include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp).
//
// Why fork
// --------
// Sparge selects a mixed K/V dtype configuration (int8 Q,K + fp16 V) per the
// SpargeAttn paper's per-block quantization scheme. The upstream FMHA policy
// assumes K and V share the same byte size when computing per-buffer LDS slot
// strides and when picking the QK warp_gemm. Two changes are required for
// sparge int8 BLOCKSCALE:
//
//   1. GEMM0 warp_gemm dispatch must yield an int8/int8/int32 mfma variant and
//      the BlockGemm accumulator type must be int32 (not the float SaccDataType
//      that tracks the softmax dtype). Upstream's GetQKBlockGemm hard-codes
//      SwizzleA-flavored entries which the WarpGemmDispatcher does not cover
//      for int8/int8/int32, so the int8 branch names
//      WarpGemmMfma_i32_32x32x16_i8_i8_CTransposed explicitly.
//
//   2. K and V share each per-buffer LDS slot (K is consumed by gemm0 then V is
//      written over the same slot for gemm1). When sizeof(V) > sizeof(K) the
//      per-buffer footprint must be max(K_bytes, V_bytes); otherwise V slot 1
//      overlaps K slot 2 (or runs OOB past the smem reservation) and silently
//      corrupts the int8 BLOCKSCALE output. When sizeof(K)==sizeof(V) the new
//      helpers reduce bit-identically to the upstream expressions.
//
// Scope of the fork
// -----------------
// To keep the upstream file at HEAD (PR #7999 should not pull in the FMHA infra
// reviewer pool, and the mxfp4/mxfp8/fp8/fp16 paths are at risk if upstream
// changes), this file replicates the relevant subset of the upstream policy and
// the sparge pipeline targets it explicitly via
// BlockFmhaPipelineQRKSVSAsyncSpargeDefaultPolicy below.
//
// This fork is intentionally a near-verbatim copy of the upstream policy with
// the two changes above applied. Keep this file in lock-step with upstream
// behaviorally for non-int8 / non-mixed-dtype callers — the K==V code path
// (e.g. fp16 NO_SCALE sparge) is byte-identical to upstream.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_mx_areg_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_mx_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

// Sparge fork of BlockFmhaPipelineQXCustomPolicy<true>. Adds the int8/int8/int32
// QK warp_gemm branch; everything else mirrors the upstream QLoadOnce=true
// policy. (Sparge always uses QLoadOnce=true; the QLoadOnce=false specialization
// is not forked.)
template <bool QLoadOnce_>
struct BlockFmhaPipelineQXSpargeCustomPolicy;

template <>
struct BlockFmhaPipelineQXSpargeCustomPolicy</* QLoadOnce = */ true>
{
    static constexpr bool QLoadOnce = true;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return 0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t MaxVectorSize =
            16 * numeric_traits<QDataType>::PackedSize / sizeof(QDataType);

        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeABlockTileDistribution<
            Problem::BlockFmhaShape::kM0,
            Problem::BlockFmhaShape::kSubQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQScaleRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeAScaleBlockTileDistribution<
            Problem::BlockFmhaShape::kM0,
            Problem::BlockFmhaShape::kSubQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKScaleRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::MakeBScaleBlockTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kNumGemm0Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr auto QScaleEnum = []() {
            if constexpr(is_detected<detail::has_qscale_enum_type, Problem>{})
                return Problem::QScaleEnum;
            else
                return ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE;
        }();

        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
        {
            constexpr auto warp_gemm = []() {
                static_assert(std::is_same_v<typename Problem::QDataType, pk_fp4_t> ==
                              std::is_same_v<typename Problem::KDataType, pk_fp4_t>);
                constexpr auto AttrNumAccess = std::is_same_v<typename Problem::QDataType, pk_fp4_t>
                                                   ? WGAttrNumAccessEnum::Single
                                                   : WGAttrNumAccessEnum::Double;
                return WarpGemmDispatcher<typename Problem::KDataType,
                                          typename Problem::QDataType,
                                          typename Problem::SaccDataType,
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                          true,  // TransposeC
                                          false, // SwizzleA
                                          false,
                                          AttrNumAccess>{};
            }();

            // Ensure that QKBlockGemm's C (S) can be used as KVBlockGemm's A (P)
            constexpr index_t TargetCMPerLane = [] {
                constexpr auto AttrNumAccess =
                    ck_tile::is_any_of<typename Problem::PDataType, pk_fp4_t, pk_fp6x16_t>::value
                        ? WGAttrNumAccessEnum::Single
                        : WGAttrNumAccessEnum::Double;
                using WarpGemm =
                    WarpGemmDispatcher<typename Problem::VDataType,
                                       typename Problem::PDataType,
                                       typename Problem::OaccDataType,
                                       Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                                       Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                                       Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                                       true,  // TransposeC
                                       false, // SwizzleA
                                       false,
                                       AttrNumAccess>;
                return WarpGemm::WarpGemmAttribute::Impl::kABKPerLane /
                       WarpGemm::WarpGemmAttribute::AttrNumAccessV;
            }();

            using BlockGemmPolicy = BlockGemmMxARegBSmemCRegV1CustomPolicy<
                typename Problem::QDataType,
                typename Problem::KDataType,
                typename Problem::SaccDataType,
                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                decltype(warp_gemm)>;

            return BlockGemmMxARegBSmemCRegV1<GemmProblem, BlockGemmPolicy, TargetCMPerLane>{};
        }
        else
        {
            // Sparge int8 path: under int8 Q/K, the MFMA accumulator is int32
            // even though Problem::SaccDataType stays float (it tracks the
            // softmax dtype). Override the BlockGemm CDataType to int32 for
            // the int8 path so MakeCBlockTile produces an int32 tile that
            // matches the warp gemm. Mirrors sage's GemmAccDataType pattern.
            using QKGemmAccDataType =
                std::conditional_t<std::is_same_v<typename Problem::QDataType, int8_t> &&
                                       std::is_same_v<typename Problem::KDataType, int8_t>,
                                   int32_t,
                                   typename Problem::SaccDataType>;

            using QKGemmProblem =
                BlockGemmProblem<typename Problem::QDataType,
                                 typename Problem::KDataType,
                                 QKGemmAccDataType,
                                 Problem::kNumGemm0Warps * get_warp_size(),
                                 TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                        Problem::BlockFmhaShape::kN0,
                                                        Problem::BlockFmhaShape::kK0>,
                                               typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                               typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

            constexpr auto warp_gemm = []() {
                if constexpr(get_warp_size() == 64 &&
                             std::is_same_v<typename Problem::QDataType, fp8_t> &&
                             std::is_same_v<typename Problem::KDataType, fp8_t> &&
                             std::is_same_v<typename Problem::SaccDataType, float> &&
                             Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}) == 32 &&
                             Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}) == 32 &&
                             Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}) == 32)
                {
                    constexpr index_t swizzle_factor = 4;
                    return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<
                        swizzle_factor>{};
                }
                else if constexpr(get_warp_size() == 64 &&
                                  std::is_same_v<typename Problem::QDataType, int8_t> &&
                                  std::is_same_v<typename Problem::KDataType, int8_t> &&
                                  Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}) == 32 &&
                                  Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}) == 32 &&
                                  Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}) == 16)
                {
                    // Sparge int8 GEMM0 for BLOCKSCALE bm0=64. The
                    // WarpGemmDispatcher's int8/int8/int32 entries do not
                    // cover SwizzleA=true, so name the plain TransposeC=true
                    // variant explicitly here; mirrors sage's int8 branch.
                    return WarpGemmMfma_i32_32x32x16_i8_i8_CTransposed{};
                }
                else
                {
                    constexpr bool SwizzleA =
                        Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}) == 32;
                    return WarpGemmDispatcher<
                        typename Problem::KDataType,
                        typename Problem::QDataType,
                        typename Problem::SaccDataType,
                        Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                        Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                        Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                        true, // TransposeC
                        SwizzleA>{};
                }
            }();

            using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
                typename Problem::QDataType,
                typename Problem::KDataType,
                QKGemmAccDataType,
                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                decltype(warp_gemm)>;

            if constexpr(1 < Problem::kNumGemm0Warps)
                return BlockGemmARegBSmemCRegV2<QKGemmProblem, BlockGemmPolicy>{};
            else
                return BlockGemmARegBSmemCRegOneWarpV1<QKGemmProblem, BlockGemmPolicy>{};
        }
    }
};

// Sparge fork of BlockFmhaPipelineQXKSVSCustomPolicy. Inherits the int8-aware
// QLoadOnce=true GetQKBlockGemm and overrides only the K/V LDS sizing helpers
// (GetSmemSizeKV + the K-store, K-load, V outer-stride descriptors) so that the
// per-buffer slot stride is max(sizeof(K),sizeof(V)) bytes. When K and V share
// the same byte size, every offset is bit-identical to upstream.
template <bool QLoadOnce_, bool AsyncCopy_, index_t NumPrefetchK_, index_t NumPrefetchV_>
struct BlockFmhaPipelineQXKSVSSpargeCustomPolicy : BlockFmhaPipelineQXSpargeCustomPolicy<QLoadOnce_>
{
    static constexpr bool AsyncCopy = AsyncCopy_;

    static constexpr index_t NumPrefetchK = NumPrefetchK_;
    static constexpr index_t NumPrefetchV = NumPrefetchK_;

    static constexpr index_t NumKVLdsBuffers = max(NumPrefetchK, NumPrefetchV);

    using QXPolicy = BlockFmhaPipelineQXSpargeCustomPolicy<QLoadOnce_>;

    // ------------------------------------------------------------------
    // LdsBufferSequence and helpers below are byte-for-byte the upstream
    // BlockFmhaPipelineQXKSVSCustomPolicy implementations. Kept here so the
    // sparge pipeline can resolve every method through the sparge policy
    // without crossing back into the upstream chain at runtime.
    // ------------------------------------------------------------------

    template <index_t k_prefetches_, index_t v_prefetches_, index_t k_loops_, index_t v_loops_>
    struct LdsBufferSequence
    {
        static constexpr index_t num_lds_buffers_ = max(k_prefetches_, v_prefetches_);
        static constexpr index_t ceil_ = ((v_loops_ - 1) / num_lds_buffers_) * num_lds_buffers_;

        static constexpr auto Make()
        {
            return transform_sequences(
                [&](auto i) {
                    if(i < k_loops_)
                        return i % num_lds_buffers_;
                    else
                        return ((num_lds_buffers_ - 1) + (i - k_loops_ + ceil_ - (v_loops_ - 1))) %
                               num_lds_buffers_;
                },
                typename arithmetic_sequence_gen<0, k_loops_ + v_loops_, 1>::type{});
        };

        using type = remove_cvref_t<decltype(Make())>;
    };

    // clang-format off
    template<> struct
    LdsBufferSequence<3, 3, 4, 4> { using type = sequence<1, 2, 0, 1,   0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 4, 2> { using type = sequence<1, 2, 0, 1,   2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 2, 4> { using type = sequence<1, 2,         0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 3, 3> { using type = sequence<1, 2, 0,      1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 3, 4> { using type = sequence<1, 2, 0,      0, 1, 2, 0>; };

    template<> struct
    LdsBufferSequence<3, 3, 2, 2> { using type = sequence<1, 2,         1, 0>;};
    // clang-format on

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsBufferSequence()
    {
        using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

        constexpr index_t kN0        = BlockFmhaShape::kN0;
        constexpr index_t kK0        = BlockFmhaShape::kK0;
        constexpr index_t kK1        = BlockFmhaShape::kK1;
        constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;

        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        return typename LdsBufferSequence<NumPrefetchK, NumPrefetchV, k0_loops, k1_loops>::type{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 * numeric_traits<KDataType>::PackedSize / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        if constexpr(AsyncCopy)
        {
#if defined(__gfx950__)
            constexpr index_t MaxLoadSizeInBytes = 4 * 4; // dwordx4
#else
            constexpr index_t MaxLoadSizeInBytes = 4; // dword
#endif

            return MaxLoadSizeInBytes * numeric_traits<KDataType>::PackedSize / sizeof(KDataType);
        }
        else
        {
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

            constexpr index_t MaxVectorSize =
                16 * numeric_traits<KDataType>::PackedSize / sizeof(KDataType);
            constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

            return min(MaxVectorSize, ElemPerThread);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t kBlockSize   = Problem::kBlockSize;
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock   = Problem::BlockFmhaShape::kK1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        constexpr index_t kMaxVecLoad  = min(
            total_pixels,
            static_cast<index_t>(16 * numeric_traits<VDataType>::PackedSize / sizeof(VDataType)));

        return kMaxVecLoad;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        using VLayout                  = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t kBlockSize   = Problem::kBlockSize;
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock   = Problem::BlockFmhaShape::kK1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        constexpr index_t kMaxVecLoad  = min(
            total_pixels,
            static_cast<index_t>(16 * numeric_traits<VDataType>::PackedSize / sizeof(VDataType)));

        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t kMinVecLoad =
                4 * numeric_traits<VDataType>::PackedSize / sizeof(VDataType);

            constexpr index_t kVecLoad = ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                                             ? kMaxVecLoad
                                             : (total_pixels / kMinVecLoad);

            return kVecLoad;
        }
        else
        {
            return kMaxVecLoad;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBias()
    {
        using BlockGemm = remove_cvref_t<decltype(QXPolicy::template GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        return WG::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentRandVal()
    {
        using BlockGemm = remove_cvref_t<decltype(QXPolicy::template GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        using CWarpDstr       = typename WG::CWarpDstr;

        constexpr auto c_warp_y_lengths = CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::RandValOutputDataType);
        return min(MaxVectorSize, c_warp_y_lengths.get(number<CWarpDstr::NDimY - 1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::ODataType);
        return min(MaxVectorSize, WG::WarpGemmAttribute::Impl::kCM1PerLane);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleSmemElementSpaceSize()
    {
        // this function assume K/V can share smem
        constexpr index_t SingleKSize = [&]() {
            if constexpr(!AsyncCopy)
            {
                return MakeKLdsBlockDescriptor<Problem>().get_element_space_size();
            }
            else
            {
                constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
                constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
                constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
                constexpr index_t WarpSize   = ck_tile::get_warp_size();

                constexpr index_t KPack   = GetSmemKPackK<Problem>();
                constexpr index_t KVector = GetAlignmentK<Problem>();
                constexpr index_t kPad    = KPack;

                static_assert(WarpSize * KVector >= kKPerBlock &&
                              WarpSize * KVector % kKPerBlock == 0);
                constexpr index_t LanesPerK  = kKPerBlock / KVector;
                constexpr index_t LaneGroups = WarpSize / LanesPerK;
                constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

                return NumIssues * NumWarps * (WarpSize * KVector + kPad);
            }
        }();

        constexpr index_t SingleVSize = [&]() {
            using VDataType         = remove_cvref_t<typename Problem::VDataType>;
            constexpr index_t Banks = get_n_lds_banks();
            constexpr index_t PixelsPerRow =
                Banks * 4 * numeric_traits<VDataType>::PackedSize / sizeof(VDataType);
            constexpr index_t kKPack = GetSmemKPackK<Problem>();
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr index_t NPerRow    = PixelsPerRow / kKPack;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        return max(SingleKSize, SingleVSize);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
        constexpr index_t kKPack     = GetSmemKPackK<Problem>();

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kNPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kNPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kNPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    // Per-buffer K/V slot stride in bytes. K and V share each per-buffer LDS
    // slot (K consumed by gemm0 then V written over the same slot for gemm1),
    // so the per-buffer footprint must accommodate the wider of the two element
    // types. When sizeof(K) == sizeof(V) this reduces to the legacy expression.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSingleSmemSlotBytes()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t k_bytes = GetSingleSmemElementSpaceSize<Problem>() * sizeof(KDataType) /
                                    numeric_traits<KDataType>::PackedSize;
        constexpr index_t v_bytes = GetSingleSmemElementSpaceSize<Problem>() * sizeof(VDataType) /
                                    numeric_traits<VDataType>::PackedSize;
        return k_bytes > v_bytes ? k_bytes : v_bytes;
    }

    // Per-buffer K slot stride expressed in K-element units (K LDS descriptors).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSingleSmemSlotKElems()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return GetSingleSmemSlotBytes<Problem>() * numeric_traits<KDataType>::PackedSize /
               sizeof(KDataType);
    }

    // Per-buffer V slot stride expressed in V-element units (V LDS descriptor).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSingleSmemSlotVElems()
    {
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return GetSingleSmemSlotBytes<Problem>() * numeric_traits<VDataType>::PackedSize /
               sizeof(VDataType);
    }

    template <typename Problem, index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeKLdsStoreBlockDescriptor(number<IBuf> = number<0>{})
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>();
        constexpr index_t KVector = GetAlignmentK<Problem>();
        constexpr index_t kPad    = KPack;

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector;
        constexpr index_t LaneGroups = WarpSize / LanesPerK;
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
            make_tuple(number<NumIssues>{},
                       number<LaneGroups>{},
                       number<NumWarps>{},
                       number<LanesPerK>{},
                       number<KVector>{}),
            make_tuple(number<NumWarps*(WarpSize * KVector + kPad)>{},
                       number<kKPerBlock>{},
                       number<WarpSize * KVector + kPad>{},
                       number<KVector>{},
                       number<1>{}),
            number<IBuf * GetSingleSmemSlotKElems<Problem>()>{},
            number<KVector>{},
            number<1>{});

        constexpr auto k_lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<NumIssues>{}),
                       make_pass_through_transform(number<NumWarps>{}),
                       make_merge_transform(make_tuple(
                           number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        return k_lds_block_desc_issues_warps_lanes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsLoadBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
        constexpr index_t WarpSize   = ck_tile::get_warp_size();

        constexpr index_t KPack   = GetSmemKPackK<Problem>();
        constexpr index_t KVector = GetAlignmentK<Problem>();
        constexpr index_t kPad    = KPack;

        static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
        constexpr index_t LanesPerK  = kKPerBlock / KVector;
        constexpr index_t LaneGroups = WarpSize / LanesPerK;
        constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));
        // Per-buffer K outer stride in K-element units (matches the K slot
        // stride used by MakeKLdsStoreBlockDescriptor and aligns with V's
        // byte stride when sizeof(V) > sizeof(K)).
        constexpr index_t BufferSize = GetSingleSmemSlotKElems<Problem>();

        constexpr auto k_lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumKVLdsBuffers>{},
                                                    number<NumIssues>{},
                                                    number<NumWarps>{},
                                                    number<LaneGroups>{},
                                                    number<kKPerBlock / KPack>{},
                                                    number<KPack>{}),
                                         make_tuple(number<BufferSize>{},
                                                    number<NumWarps*(WarpSize * KVector + kPad)>{},
                                                    number<WarpSize * KVector + kPad>{},
                                                    number<kKPerBlock>{},
                                                    number<KPack>{},
                                                    number<1>{}),
                                         number<KPack>{},
                                         number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<NumKVLdsBuffers>{},
                                                number<NumIssues>{},
                                                number<LaneGroups>{},
                                                number<NumWarps>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<0, 1, 3, 2>{}, sequence<4, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        using VDataType         = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t Banks = get_n_lds_banks();
        constexpr index_t PixelsPerRow =
            Banks * 4 * numeric_traits<VDataType>::PackedSize / sizeof(VDataType);
        constexpr index_t kKPack = GetSmemKPackV<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NumKVLdsBuffers>{},
                       number<kKPerBlock / kKPack>{},
                       number<kNPerBlock / NPerRow>{},
                       number<NPerRow>{},
                       number<kKPack>{}),
            make_tuple(number<GetSingleSmemSlotVElems<Problem>()>{},
                       number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       number<PixelsPerRow + kKPack>{},
                       number<kKPack>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(
                    number<NumKVLdsBuffers>{}, number<kNPerBlock / NPerRow>{}, number<NPerRow>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return v_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV()
    {
        // TODO: assume Q is in register
        return QXPolicy::template GetSmemSizeQ<Problem>() +
               GetSingleSmemSlotBytes<Problem>() * NumKVLdsBuffers;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        if constexpr(AsyncCopy)
        {
            return GetSmemSizeKV<Problem>() + GetSmemSizeDropout<Problem>(0);
        }
        else
        {
            return ck_tile::max(GetSmemSizeKV<Problem>(), GetSmemSizeDropout<Problem>(0));
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr std::
        enable_if_t<std::is_convertible_v<decltype(Problem::kHasDropout), bool>, ck_tile::index_t>
        GetSmemSizeDropout(int)
    {
        if constexpr(Problem::kHasDropout)
        {
            constexpr auto gemm_0 = QXPolicy::template GetQKBlockGemm<Problem>();
            constexpr auto config =
                decltype(gemm_0)::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WG                    = remove_cvref_t<decltype(config.template at<0>())>;
            constexpr index_t MWarp     = config.template at<1>();
            constexpr index_t kMPerStep = MWarp * WG::kM;
            constexpr index_t kNPerStep = WG::kN;

            return (kMPerStep + 1) * kNPerStep * sizeof(uint8_t);
        }
        else
        {
            return 0;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout(...)
    {
        return 0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        if constexpr(!AsyncCopy)
        {
            using KDataType = remove_cvref_t<typename Problem::KDataType>;

            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

            constexpr index_t MaxVectorSize =
                16 * numeric_traits<KDataType>::PackedSize / sizeof(KDataType);
            constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

            constexpr index_t K1 = min(MaxVectorSize, ElemPerThread);
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N0 = kNPerBlock / (N2 * N1);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else
        {
            constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t NumWarps   = Problem::BlockFmhaShape::NumWarps;
            constexpr index_t WarpSize   = ck_tile::get_warp_size();

            constexpr index_t KVector = GetAlignmentK<Problem>();

            static_assert(WarpSize * KVector >= kKPerBlock && WarpSize * KVector % kKPerBlock == 0);
            constexpr index_t LanesPerK  = kKPerBlock / KVector;
            constexpr index_t LaneGroups = WarpSize / LanesPerK;
            constexpr index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
            static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

            constexpr index_t N0 = NumIssues;
            constexpr index_t N1 = LaneGroups;
            constexpr index_t N2 = NumWarps;
            constexpr index_t K0 = LanesPerK;
            constexpr index_t K1 = KVector;

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<2>, sequence<1, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();
            constexpr index_t N0 = kNPerBlock / N1;

            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
            constexpr index_t kKPack       = GetSmemKPackV<Problem>();
            constexpr index_t K3           = total_pixels / N1;
            constexpr index_t K2           = kKPack / K3;
            if constexpr(total_pixels % N1 != 0 || kKPack % K3 != 0)
            {
                static_assert(kNPerBlock % 16 == 0);
                constexpr index_t kNPack = kNPerBlock % 32 == 0 ? 32 : 16;
                constexpr index_t K0     = kBlockSize / get_warp_size();
                constexpr index_t N2     = 2;
                constexpr index_t N1_m   = kNPack / N2;
                constexpr index_t N0_m   = kNPerBlock / kNPack;
                constexpr index_t K1     = get_warp_size() / N1_m;
                constexpr index_t K2_m   = kKPerBlock / K1 / K0;
                return make_static_tile_distribution(
                    tile_distribution_encoding<
                        sequence<1>,
                        tuple<sequence<N0_m, N1_m, N2>, sequence<K0, K1, K2_m>>,
                        tuple<sequence<2>, sequence<2, 1>>,
                        tuple<sequence<0>, sequence<1, 1>>,
                        sequence<1, 2, 1>,
                        sequence<0, 2, 2>>{});
            }
            else if constexpr(get_warp_size() % (K2 * N0) == 0)
            {
                constexpr index_t K1 = get_warp_size() / (K2 * N0);
                constexpr index_t K0 = kBlockSize / get_warp_size();
                static_assert(kKPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                               tuple<sequence<2>, sequence<2, 1, 2>>,
                                               tuple<sequence<0>, sequence<1, 0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1   = (K2 * N0) / get_warp_size();
                constexpr index_t K2_m = K2 / K1;
                constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
                static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                               tuple<sequence<2, 2>, sequence<1, 2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
        }
        else
        {
            constexpr index_t K1 = GetAlignmentV<Problem>();
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            static_assert(N2 != 0, "N2 is zero, which will lead to a division by zero error.");
            static_assert(N1 != 0, "N1 is zero, which will lead to a division by zero error.");
            constexpr index_t N0 = kNPerBlock / (N2 * N1);
            static_assert(N0 != 0);

            constexpr auto dstr = make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
            if constexpr(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                         kNPerBlock * kKPerBlock)
            {
                return dstr;
            }
            else
            {
                static_assert(kKPerBlock % 16 == 0);
                constexpr index_t kKPerIter = kKPerBlock % 32 == 0 ? 32 : 16;
                constexpr index_t K0_m      = kKPerBlock / kKPerIter;
                constexpr index_t K2        = 2;
                constexpr index_t K1_m      = kKPerIter / K2;
                constexpr index_t N2_m      = get_warp_size() / K1_m;
                constexpr index_t N0_m      = kNPerBlock / (N2_m * N1);
                constexpr auto dstr_m       = make_static_tile_distribution(
                    tile_distribution_encoding<
                              sequence<1>,
                              tuple<sequence<N0_m, N1, N2_m>, sequence<K0_m, K1_m, K2>>,
                              tuple<sequence<1>, sequence<1, 2>>,
                              tuple<sequence<1>, sequence<2, 1>>,
                              sequence<2, 1, 2>,
                              sequence<0, 0, 2>>{});
                static_assert(container_reduce(dstr_m.get_lengths(),
                                               std::multiplies<index_t>{},
                                               1) == kNPerBlock * kKPerBlock);
                return dstr_m;
            }
        }
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasDramTileDistribution()
    {
        return BlockGemm::MakeCBlockTile().get_tile_distribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegBlockDescriptor()
    {
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t N1           = GetAlignmentV<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        constexpr index_t K3           = total_pixels / N1;
        constexpr index_t kKPack       = GetSmemKPackV<Problem>();
        constexpr index_t K2           = kKPack / K3;
        if constexpr(total_pixels % N1 != 0 || kKPack % K3 != 0)
        {
            static_assert(kNPerBlock % 16 == 0);
            constexpr index_t kNPack = kNPerBlock % 32 == 0 ? 32 : 16;
            constexpr index_t K0     = kBlockSize / get_warp_size();
            constexpr index_t N2     = 2;
            constexpr index_t N1_m   = kNPack / N2;
            constexpr index_t N0_m   = kNPerBlock / kNPack;
            constexpr index_t K1     = get_warp_size() / N1_m;
            constexpr index_t K2_m   = kKPerBlock / K1 / K0;
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0_m, N1_m, N2>, sequence<K0, K1, K2_m>>,
                                           tuple<sequence<2>, sequence<2, 1>>,
                                           tuple<sequence<0>, sequence<1, 1>>,
                                           sequence<1, 1, 2>,
                                           sequence<0, 2, 2>>{});
        }
        else if constexpr(get_warp_size() % (K2 * N0) == 0)
        {
            constexpr index_t K1 = get_warp_size() / (K2 * N0);
            constexpr index_t K0 = kBlockSize / get_warp_size();

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                           tuple<sequence<2>, sequence<2, 1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
        else
        {
            constexpr index_t K1   = (K2 * N0) / get_warp_size();
            constexpr index_t K2_m = K2 / K1;
            constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                           tuple<sequence<2, 2>, sequence<1, 2>>,
                                           tuple<sequence<0, 1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePScaleRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;

        return BlockGemm::template MakeAScaleBlockTileDistribution<Problem::BlockFmhaShape::kM0,
                                                                   Problem::BlockFmhaShape::kN0>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVScaleRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;

        return BlockGemm::MakeBScaleBlockTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kNumGemm1Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN1,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        constexpr auto QScaleEnum = []() {
            if constexpr(is_detected<detail::has_qscale_enum_type, Problem>{})
                return Problem::QScaleEnum;
            else
                return ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE;
        }();

        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
        {
            constexpr auto warp_gemm = []() {
                static_assert(
                    ck_tile::is_any_of<typename Problem::PDataType, pk_fp4_t, pk_fp6x16_t>::value ==
                    std::is_same_v<typename Problem::VDataType, pk_fp4_t>);
                constexpr auto AttrNumAccess =
                    ck_tile::is_any_of<typename Problem::PDataType, pk_fp4_t, pk_fp6x16_t>::value
                        ? WGAttrNumAccessEnum::Single
                        : WGAttrNumAccessEnum::Double;
                return WarpGemmDispatcher<typename Problem::VDataType,
                                          typename Problem::PDataType,
                                          typename Problem::OaccDataType,
                                          Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                                          Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                                          Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                                          true,
                                          false,
                                          false,
                                          AttrNumAccess>{};
            }();

            using BlockGemmPolicy = BlockGemmMxARegBSmemCRegV1CustomPolicy<
                typename Problem::PDataType,
                typename Problem::VDataType,
                typename Problem::OaccDataType,
                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                decltype(warp_gemm)>;

            return BlockGemmMxARegBSmemCRegV1<GemmProblem, BlockGemmPolicy>{};
        }
        else
        {
            constexpr auto warp_gemm = []() {
                if constexpr(get_warp_size() == 64 &&
                             std::is_same_v<typename Problem::PDataType, fp8_t> &&
                             std::is_same_v<typename Problem::VDataType, fp8_t> &&
                             std::is_same_v<typename Problem::OaccDataType, float> &&
                             Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}) == 32 &&
                             Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}) == 32 &&
                             Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 32)
                {
                    return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<>{};
                }
                else
                {
                    return WarpGemmDispatcher<
                        typename Problem::VDataType,
                        typename Problem::PDataType,
                        typename Problem::OaccDataType,
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                        Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                        true>{};
                }
            }();

            using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
                typename Problem::PDataType,
                typename Problem::VDataType,
                typename Problem::OaccDataType,
                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                decltype(warp_gemm)>;

            return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
        }
    }
};

// Sparge-specific default policy alias. Sparge entry points should reference
// this instead of upstream BlockFmhaPipelineQRKSVSAsyncDefaultPolicy.
using BlockFmhaPipelineQRKSVSAsyncSpargeDefaultPolicy =
    BlockFmhaPipelineQXKSVSSpargeCustomPolicy</* QLoadOnce  = */ true,
                                              /* AsyncCopy  = */ true,
                                              /* NumPrefetchK = */ 3,
                                              /* NumPrefetchV = */ 3>;

} // namespace ck_tile
