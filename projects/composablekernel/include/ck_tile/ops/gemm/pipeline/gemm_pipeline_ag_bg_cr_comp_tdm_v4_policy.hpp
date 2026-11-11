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

    /// Threads of the device pass: the waves times 32.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetBlockSize()
    {
        return GetSpec<Problem>().block_size();
    }

    /// WMMA atoms per wave along M and N.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetWaveRepeatM()
    {
        return GetSpec<Problem>().wave_repeat_m();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetWaveRepeatN()
    {
        return GetSpec<Problem>().wave_repeat_n();
    }

    /// 128-wide K steps per K tile.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetKSteps()
    {
        return GetSpec<Problem>().k_steps();
    }

    /// Floats of one WMMA C fragment per lane.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetCFragLen()
    {
        return GetSpec<Problem>().c_frag_len();
    }

    /// K values per byte of A and B (2 for FP4).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetKPerByte()
    {
        return GetSpec<Problem>().k_per_byte();
    }

    /// Bytes of a 128-wide K step in an A or B row.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetStepBytes()
    {
        return GetSpec<Problem>().step_bytes();
    }

    /// 16-byte LDS reads per lane per A or B row fragment, and the byte offset of read i.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetChunksPerLane()
    {
        return GetSpec<Problem>().chunks_per_lane();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetChunkOffset(index_t i)
    {
        return GetSpec<Problem>().chunk_offset(i);
    }

    /// LDS reads of a K step, clamped to the 6-bit s_wait_dscnt immediate.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetReadsPerStep()
    {
        return mx_tdm_v4::min_i(GetSpec<Problem>().reads_per_step(), mx_tdm_v4::kDscntMax);
    }

    /// Whether the next K step's fragments are read while this one's WMMAs run.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool GetLocalPrefetch()
    {
        return GetSpec<Problem>().local_prefetch == 1;
    }

    /// Whether the B fragments are read in two halves, and whether the second half is read late.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool GetHalfB()
    {
        return GetSpec<Problem>().half_b();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool GetHalfBLate()
    {
        return GetSpec<Problem>().half_b_late();
    }

    /// Bytes between the two LDS buffers.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsBufferStride()
    {
        return GetLdsLayout<Problem>().buffer_stride;
    }

    /// Bytes of an A or B row of one K tile in LDS, padding excluded.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsRowBytes()
    {
        return GetLdsLayout<Problem>().row_bytes;
    }

    /// The LDS placement of A's or B's data or scales.
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr MxTdmV4Operand GetLdsOperand()
    {
        constexpr MxTdmV4LdsLayout l = GetLdsLayout<Problem>();
        if constexpr(IsScale)
            return IsA ? l.a_scale : l.b_scale;
        else
            return IsA ? l.a : l.b;
    }

    /// Whether the TDM pads A's or B's data or scales in LDS: a data load in iterate mode is
    /// unpadded (its rows land at iterate_stride instead).
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr bool GetLdsPadEnable()
    {
        constexpr MxTdmV4Operand op = GetLdsOperand<Problem, IsA, IsScale>();
        return (IsScale || op.iterate_rows == 0) && op.pad_bytes > 0;
    }

    /// The TDM LDS padding of A's or B's data or scales; the zero config without padding.
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr TDMLdsPaddingConfig GetLdsPaddingConfig()
    {
        constexpr MxTdmV4Operand op = GetLdsOperand<Problem, IsA, IsScale>();
        if constexpr(GetLdsPadEnable<Problem, IsA, IsScale>())
            return make_tdm_lds_padding_config(op.block_bytes, op.pad_bytes);
        else
            return TDMLdsPaddingConfig{};
    }

    /// The cluster workgroups that share this one's A (same cluster y) or B (same cluster x)
    /// tile and so receive its TDM loads; 0 without a cluster. Grid x walks N tiles and y M
    /// tiles.
    template <bool IsA>
    CK_TILE_DEVICE static uint32_t GetTDMWorkgroupMask(index_t cluster_wx, index_t cluster_wy)
    {
        if constexpr(!HasCluster())
        {
            ignore = cluster_wx;
            ignore = cluster_wy;
            return 0;
        }
        else if constexpr(IsA)
        {
            ignore = cluster_wx;
            return cluster_mask_same_y(Tuning::kClusterDimX, cluster_wy);
        }
        else
        {
            ignore = cluster_wy;
            return cluster_mask_same_x(Tuning::kClusterDimX, Tuning::kClusterDimY, cluster_wx);
        }
    }

    /// The C block tile: per thread the WMMA C fragments of its M and N repeats, (rm, rn) outer
    /// and the fragment slot inner. A lane's M repeats are adjacent rows and its N repeats
    /// adjacent columns, so one slot across the N repeats is one vector store of D.
    ///   M: (wave_m, slot / 8, lane / 16, slot % 8, rm)   N: (wave_n, lane % 16, rn)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeCBlockTileDistribution()
    {
        constexpr MxTdmV4Spec s = GetSpec<Problem>();
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<s.warp_m, s.atom_m() / 16, 2, 8, s.wave_repeat_m()>,
                      sequence<s.warp_n, s.atom_n(), s.wave_repeat_n()>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<0, 0>, sequence<2, 1>>,
                sequence<1, 2, 1, 1>,
                sequence<4, 2, 1, 3>>{});
    }

    /// The C block tile's elements with the N repeat innermost: (rm, slot / 8, slot % 8, rn), so
    /// a thread's adjacent columns are adjacent in its buffer, as D is stored and C loaded.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeCStoreTileDistribution()
    {
        constexpr MxTdmV4Spec s = GetSpec<Problem>();
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<s.warp_m, s.atom_m() / 16, 2, 8, s.wave_repeat_m()>,
                      sequence<s.warp_n, s.atom_n(), s.wave_repeat_n()>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<0, 0>, sequence<2, 1>>,
                sequence<1, 1, 1, 2>,
                sequence<4, 1, 3, 2>>{});
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
