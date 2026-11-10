// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v4_spec.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v4_tuning.hpp"

namespace ck_tile {

/**
 * @brief The GEMM traits of a GemmPipelineAgBgCrCompTDMV4 instance: RCR TileGemmUniversalTraits
 *        plus the shape-dependent code paths rocKE derives from its compile-time M, N and K.
 *
 * EdgeM / EdgeN (kPadM / kPadN): the last M (N) tile is partial, its TDM rows clipped and its D
 * vectors masked. WideA / WideB / WideD (LargeTensors if any): A's, B's or D's byte offsets pass
 * 2^31 and are i64. OddTiles: 1 if K / tile_k is odd, 0 if even, -1 if either (the odd tile then
 * runs after the loop, which may cost registers: rocKE knows the count at compile time).
 * TailSteps: the K steps of a last, partial tile when tile_k does not divide K (0 without one),
 * as rocKE's MxGemmSpec.tail_steps. StoreWidth: D columns per vector store. The edge and wide
 * paths are supersets of the ones they replace (an edge kernel runs whole tiles, a wide one
 * narrow offsets) and any store width dividing N works, so one instance serves every problem its
 * traits cover; OddTiles (unless -1) and TailSteps must be the problem's.
 */
template <bool EdgeM_         = false,
          bool EdgeN_         = false,
          bool WideA_         = false,
          bool WideB_         = false,
          bool WideD_         = false,
          index_t OddTiles_   = -1,
          index_t TailSteps_  = 0,
          index_t StoreWidth_ = 1>
struct MxTdmV4GemmTraits : TileGemmUniversalTraits<EdgeM_,
                                                   EdgeN_,
                                                   false,
                                                   true,
                                                   tensor_layout::gemm::RowMajor,
                                                   tensor_layout::gemm::ColumnMajor,
                                                   tensor_layout::gemm::RowMajor,
                                                   false,
                                                   false,
                                                   false,
                                                   1,
                                                   false,
                                                   16,
                                                   DataCachePrefetchKind::None,
                                                   DataCachePrefetchKind::None,
                                                   false,
                                                   WideA_ || WideB_ || WideD_>
{
    static constexpr bool kEdgeM         = EdgeM_;
    static constexpr bool kEdgeN         = EdgeN_;
    static constexpr bool kWideA         = WideA_;
    static constexpr bool kWideB         = WideB_;
    static constexpr bool kWideD         = WideD_;
    static constexpr index_t kOddTiles   = OddTiles_;
    static constexpr index_t kTailSteps  = TailSteps_;
    static constexpr bool kHasTail       = TailSteps_ != 0;
    static constexpr index_t kStoreWidth = StoreWidth_;
};

/**
 * @brief Problem of GemmPipelineAgBgCrCompTDMV4: D = alpha * A @ B^T + beta * C.
 *
 * A is M x K and B is N x K, both K-contiguous (RCR); C and D are M x N row-major. A and B are
 * pk_fp4_t, fp8_t or bf8_t of the same width, with one E8M0 scale per 32 K values in the K-major
 * layout [ceil(K / 128)][M or N][4] (see host/mx_scale_kmajor.hpp). ODataType (C and D) is fp8_t
 * (saturating), half_t or bf16_t.
 *
 * BlockGemmShape is a TileGemmShape: block tile (tile_m, tile_n, tile_k), block warps 2x2x1,
 * 4x1x1 or 1x4x1 and warp tile 32x16x128 (FP4) or 16x16x128: the scaled WMMA atom.
 *
 * An MxGemmPipelineProblem with float accumulators, PassThrough element functions and E8M0
 * scales per 32 K values, plus ODataType; kBlockSize follows its convention (warps x
 * get_warp_size()). Traits is an MxTdmV4GemmTraits.
 */
template <typename ADataType_,
          typename BDataType_,
          typename ODataType_,
          typename BlockGemmShape_,
          typename Traits_ = MxTdmV4GemmTraits<>>
struct MxGemmTdmV4Problem
    : MxGemmPipelineProblem<ADataType_, BDataType_, float, BlockGemmShape_, Traits_>
{
    using ODataType   = remove_cvref_t<ODataType_>;
    using AccDataType = float;

    static_assert(
        std::is_same_v<typename MxGemmTdmV4Problem::ALayout, tensor_layout::gemm::RowMajor> &&
            std::is_same_v<typename MxGemmTdmV4Problem::BLayout,
                           tensor_layout::gemm::ColumnMajor> &&
            std::is_same_v<typename MxGemmTdmV4Problem::CLayout, tensor_layout::gemm::RowMajor>,
        "comp_tdm_v4: A and B are K-contiguous and C row-major (RCR)");
};

namespace mx_tdm_v4 {

/// The MxTdmV4Dtype of an A or B type; an out-of-range value (rejected as "dtype") otherwise.
template <typename T>
CK_TILE_HOST_DEVICE constexpr MxTdmV4Dtype dtype_of()
{
    if constexpr(std::is_same_v<T, pk_fp4_t>)
        return MxTdmV4Dtype::Fp4;
    else if constexpr(std::is_same_v<T, fp8_t>)
        return MxTdmV4Dtype::Fp8;
    else if constexpr(std::is_same_v<T, bf8_t>)
        return MxTdmV4Dtype::Bf8;
    else
        return static_cast<MxTdmV4Dtype>(-1);
}

/// The MxTdmV4OutDtype of C and D; an out-of-range value (rejected as "out_dtype") otherwise.
template <typename T>
CK_TILE_HOST_DEVICE constexpr MxTdmV4OutDtype out_dtype_of()
{
    if constexpr(std::is_same_v<T, fp8_t>)
        return MxTdmV4OutDtype::Fp8E4M3;
    else if constexpr(std::is_same_v<T, half_t>)
        return MxTdmV4OutDtype::Fp16;
    else if constexpr(std::is_same_v<T, bf16_t>)
        return MxTdmV4OutDtype::Bf16;
    else
        return static_cast<MxTdmV4OutDtype>(-1);
}

/// Instantiating this with a reason other than Ok fails, and the compiler names the reason.
template <MxTdmV4Reason R>
struct InvalidTuning;
template <>
struct InvalidTuning<MxTdmV4Reason::Ok>
{
    static constexpr bool value = true;
};

} // namespace mx_tdm_v4

/**
 * @brief Policy of GemmPipelineAgBgCrCompTDMV4: the tuning (see MxTdmV4DefaultTuning) and
 *        everything derived from it and the Problem's tile, at compile time.
 */
template <typename Tuning = MxTdmV4DefaultTuning>
struct GemmPipelineAgBgCrCompTDMV4Policy
{
    using TuningType = Tuning;

    /// The compile-time part of the spec: M, N and K are 0.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr MxTdmV4Spec GetSpec()
    {
        using Shape              = typename Problem::BlockGemmShape;
        using WarpTile           = typename Shape::WarpTile;
        using Warps              = typename Shape::BlockWarps;
        constexpr index_t atom_m = WarpTile::at(number<0>{});
        constexpr index_t atom_n = WarpTile::at(number<1>{});
        constexpr index_t atom_k = WarpTile::at(number<2>{});
        constexpr bool warp_k_ok = Warps::at(number<2>{}) == 1 && atom_k == mx_tdm_v4::kAtomK;

        MxTdmV4Spec s{};
        s.out                    = mx_tdm_v4::out_dtype_of<typename Problem::ODataType>();
        s.tile_m                 = Shape::kM;
        s.tile_n                 = Shape::kN;
        s.tile_k                 = Shape::kK;
        s.warp_m                 = warp_k_ok ? Warps::at(number<0>{}) : 0;
        s.warp_n                 = warp_k_ok ? Warps::at(number<1>{}) : 0;
        s.mma                    = (atom_m == 32 && atom_n == 16)   ? MxTdmV4Mma::M32N16
                                   : (atom_m == 16 && atom_n == 16) ? MxTdmV4Mma::M16N16
                                                                    : static_cast<MxTdmV4Mma>(-1);
        s.a_dtype                = mx_tdm_v4::dtype_of<typename Problem::ADataType>();
        s.b_dtype                = mx_tdm_v4::dtype_of<typename Problem::BDataType>();
        s.cluster_x              = Tuning::kClusterDimX;
        s.cluster_y              = Tuning::kClusterDimY;
        s.cluster_z              = Tuning::kClusterDimZ;
        s.gl2_distance           = Tuning::kGl2PrefetchDistance;
        s.signal_wmmas_in        = Tuning::kClusterSignalWmmas;
        s.local_prefetch         = Tuning::kLocalPrefetch;
        s.half_plr               = Tuning::kHalfPlr;
        s.half_b_order           = Tuning::kHalfBOrder;
        s.lds_segment_interleave = Tuning::kLdsSegmentInterleave;
        return s;
    }

    /// The spec of one problem, for the run-time checks.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr MxTdmV4Spec
    GetSpec(long_index_t M, long_index_t N, long_index_t K)
    {
        MxTdmV4Spec s = GetSpec<Problem>();
        s.M           = M;
        s.N           = N;
        s.K           = K;
        return s;
    }

    /// The first compile-time rule the Problem and tuning break, or Ok.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr MxTdmV4Reason GetReason()
    {
        return mx_tdm_v4_validate(GetSpec<Problem>(), false);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr MxTdmV4LdsLayout GetLdsLayout()
    {
        return mx_tdm_v4_lds_layout(GetSpec<Problem>());
    }

    /// Bytes of LDS: both buffers.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return GetLdsLayout<Problem>().lds_bytes;
    }

    /// The K loop's s_wait_dscnt count i (0 or 1), clamped to the 6-bit immediate.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetDscntWait(index_t i)
    {
        return mx_tdm_v4::min_i(GetSpec<Problem>().dscnt_wait(i), mx_tdm_v4::kDscntMax);
    }

    /// Reads of the other buffer left in flight at the cluster late wait, clamped.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLatePendingReads()
    {
        return mx_tdm_v4::min_i(GetSpec<Problem>().late_pending_reads(), mx_tdm_v4::kDscntMax);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSignalWmmas()
    {
        return GetSpec<Problem>().signal_wmmas();
    }

    /// The cluster the kernel must be launched with: (x, y, z) = (N tiles, M tiles, 1).
    CK_TILE_HOST_DEVICE static constexpr bool HasCluster()
    {
        return !(Tuning::kClusterDimX == 1 && Tuning::kClusterDimY == 1);
    }
};

/// rocKE's validate() at compile time: nullptr if Problem and Tuning build, else the reason code
/// (see mx_tdm_v4_reason_code). The shape rules are the pipeline's GetUnsupportedReason.
template <typename Problem, typename Tuning = MxTdmV4DefaultTuning>
CK_TILE_HOST_DEVICE constexpr const char* MxTdmV4Validate()
{
    constexpr MxTdmV4Reason r =
        GemmPipelineAgBgCrCompTDMV4Policy<Tuning>::template GetReason<Problem>();
    return r == MxTdmV4Reason::Ok ? nullptr : mx_tdm_v4_reason_code(r);
}

/// The rocKE name suffix of a tuning: the cluster, then its non-default values (see
/// mx_tdm_v4_tuning_name), without the parts a CK kernel name already has.
template <typename Tuning>
CK_TILE_HOST std::string MxTdmV4TuningName()
{
    MxTdmV4Spec s{};
    s.cluster_x              = Tuning::kClusterDimX;
    s.cluster_y              = Tuning::kClusterDimY;
    s.cluster_z              = Tuning::kClusterDimZ;
    s.gl2_distance           = Tuning::kGl2PrefetchDistance;
    s.signal_wmmas_in        = Tuning::kClusterSignalWmmas;
    s.local_prefetch         = Tuning::kLocalPrefetch;
    s.half_plr               = Tuning::kHalfPlr;
    s.half_b_order           = Tuning::kHalfBOrder;
    s.lds_segment_interleave = Tuning::kLdsSegmentInterleave;
    return mx_tdm_v4_tuning_name(s, false);
}

} // namespace ck_tile
