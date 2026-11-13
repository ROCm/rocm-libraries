// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_mx_asmem_bsmem_creg_tdm_scheduled.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_spec.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_tuning.hpp"
#include "ck_tile/ops/gemm/pipeline/tdm_multicast_direction.hpp"

namespace ck_tile {

/**
 * @brief The GEMM traits of a MxGemmPipelineAgBgCrCompTDMScheduled instance: RCR
 * TileGemmUniversalTraits plus the shape-dependent code paths rocKE derives from its compile-time
 * M, N and K.
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
struct MxTdmScheduledGemmTraits : TileGemmUniversalTraits<EdgeM_,
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
 * @brief Problem of MxGemmPipelineAgBgCrCompTDMScheduled: E = A @ B^T.
 *
 * A is M x K and B is N x K, both K-contiguous (RCR); E is M x N row-major. A and B are
 * pk_fp4_t, fp8_t or bf8_t of the same width, with one E8M0 scale per 32 K values in the K-major
 * layout [ceil(K / 128)][M or N][4] (see host/mx_scale_kmajor.hpp). ODataType (E) is fp8_t
 * (saturating), half_t or bf16_t.
 *
 * BlockGemmShape is a TileGemmShape: block tile (tile_m, tile_n, tile_k), block warps 2x2x1,
 * 4x1x1 or 1x4x1 and warp tile 32x16x128 (FP4) or 16x16x128: the scaled WMMA atom.
 *
 * An MxGemmPipelineProblem with float accumulators, PassThrough element functions and E8M0
 * scales per 32 K values, plus ODataType; kBlockSize follows its convention (warps x
 * get_warp_size()). Traits is an MxTdmScheduledGemmTraits.
 */
template <typename ADataType_,
          typename BDataType_,
          typename ODataType_,
          typename BlockGemmShape_,
          typename Traits_ = MxTdmScheduledGemmTraits<>>
struct MxGemmTdmScheduledProblem
    : MxGemmPipelineProblem<ADataType_, BDataType_, float, BlockGemmShape_, Traits_>
{
    using ODataType   = remove_cvref_t<ODataType_>;
    using AccDataType = float;

    static_assert(std::is_same_v<typename MxGemmTdmScheduledProblem::ALayout,
                                 tensor_layout::gemm::RowMajor> &&
                      std::is_same_v<typename MxGemmTdmScheduledProblem::BLayout,
                                     tensor_layout::gemm::ColumnMajor> &&
                      std::is_same_v<typename MxGemmTdmScheduledProblem::CLayout,
                                     tensor_layout::gemm::RowMajor>,
                  "comp_tdm_scheduled: A and B are K-contiguous and C row-major (RCR)");
};

namespace mx_tdm_scheduled {

/// The MxTdmScheduledDtype of an A or B type; an out-of-range value (rejected as "dtype")
/// otherwise.
template <typename T>
CK_TILE_HOST_DEVICE constexpr MxTdmScheduledDtype dtype_of()
{
    if constexpr(std::is_same_v<T, pk_fp4_t>)
        return MxTdmScheduledDtype::Fp4;
    else if constexpr(std::is_same_v<T, fp8_t>)
        return MxTdmScheduledDtype::Fp8;
    else if constexpr(std::is_same_v<T, bf8_t>)
        return MxTdmScheduledDtype::Bf8;
    else
        return static_cast<MxTdmScheduledDtype>(-1);
}

/// The MxTdmScheduledOutDtype of C and D; an out-of-range value (rejected as "out_dtype")
/// otherwise.
template <typename T>
CK_TILE_HOST_DEVICE constexpr MxTdmScheduledOutDtype out_dtype_of()
{
    if constexpr(std::is_same_v<T, fp8_t>)
        return MxTdmScheduledOutDtype::Fp8E4M3;
    else if constexpr(std::is_same_v<T, half_t>)
        return MxTdmScheduledOutDtype::Fp16;
    else if constexpr(std::is_same_v<T, bf16_t>)
        return MxTdmScheduledOutDtype::Bf16;
    else
        return static_cast<MxTdmScheduledOutDtype>(-1);
}

/// Instantiating this with a reason other than Ok fails, and the compiler names the reason.
template <MxTdmScheduledReason R>
struct InvalidTuning;
template <>
struct InvalidTuning<MxTdmScheduledReason::Ok>
{
    static constexpr bool value = true;
};

} // namespace mx_tdm_scheduled

/**
 * @brief Policy of MxGemmPipelineAgBgCrCompTDMScheduled: the tuning (see
 * MxTdmScheduledDefaultTuning) and everything derived from it and the Problem's tile, at compile
 * time.
 */
template <typename Tuning = MxTdmScheduledDefaultTuning>
struct MxGemmPipelineAgBgCrCompTDMScheduledPolicy
{
    using TuningType = Tuning;

    /// The compile-time part of the spec: M, N and K are 0.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr MxTdmScheduledSpec GetSpec()
    {
        using Shape              = typename Problem::BlockGemmShape;
        using WarpTile           = typename Shape::WarpTile;
        using Warps              = typename Shape::BlockWarps;
        constexpr index_t atom_m = WarpTile::at(number<0>{});
        constexpr index_t atom_n = WarpTile::at(number<1>{});
        constexpr index_t atom_k = WarpTile::at(number<2>{});
        constexpr bool warp_k_ok =
            Warps::at(number<2>{}) == 1 && atom_k == mx_tdm_scheduled::kAtomK;

        MxTdmScheduledSpec s{};
        s.out                    = mx_tdm_scheduled::out_dtype_of<typename Problem::ODataType>();
        s.tile_m                 = Shape::kM;
        s.tile_n                 = Shape::kN;
        s.tile_k                 = Shape::kK;
        s.warp_m                 = warp_k_ok ? Warps::at(number<0>{}) : 0;
        s.warp_n                 = warp_k_ok ? Warps::at(number<1>{}) : 0;
        s.mma                    = (atom_m == 32 && atom_n == 16)   ? MxTdmScheduledMma::M32N16
                                   : (atom_m == 16 && atom_n == 16) ? MxTdmScheduledMma::M16N16
                                                                    : static_cast<MxTdmScheduledMma>(-1);
        s.a_dtype                = mx_tdm_scheduled::dtype_of<typename Problem::ADataType>();
        s.b_dtype                = mx_tdm_scheduled::dtype_of<typename Problem::BDataType>();
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
    CK_TILE_HOST_DEVICE static constexpr MxTdmScheduledSpec
    GetSpec(long_index_t M, long_index_t N, long_index_t K)
    {
        MxTdmScheduledSpec s = GetSpec<Problem>();
        s.M                  = M;
        s.N                  = N;
        s.K                  = K;
        return s;
    }

    /// The first compile-time rule the Problem and tuning break, or Ok.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr MxTdmScheduledReason GetReason()
    {
        return mx_tdm_scheduled_validate(GetSpec<Problem>(), false);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr MxTdmScheduledLdsLayout GetLdsLayout()
    {
        return mx_tdm_scheduled_lds_layout(GetSpec<Problem>());
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
        return mx_tdm_scheduled::min_i(GetSpec<Problem>().dscnt_wait(i),
                                       mx_tdm_scheduled::kDscntMax);
    }

    /// Reads of the other buffer left in flight at the cluster late wait, clamped.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLatePendingReads()
    {
        return mx_tdm_scheduled::min_i(GetSpec<Problem>().late_pending_reads(),
                                       mx_tdm_scheduled::kDscntMax);
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
        return mx_tdm_scheduled::min_i(GetSpec<Problem>().reads_per_step(),
                                       mx_tdm_scheduled::kDscntMax);
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
    CK_TILE_HOST_DEVICE static constexpr MxTdmScheduledOperand GetLdsOperand()
    {
        constexpr MxTdmScheduledLdsLayout l = GetLdsLayout<Problem>();
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
        constexpr MxTdmScheduledOperand op = GetLdsOperand<Problem, IsA, IsScale>();
        return (IsScale || op.iterate_rows == 0) && op.pad_bytes > 0;
    }

    /// The TDM LDS padding of A's or B's data or scales as the default TDM policy gives it:
    /// (pad_enable, pad_amount, pad_interval), zeros without padding.
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfig()
    {
        constexpr MxTdmScheduledOperand op = GetLdsOperand<Problem, IsA, IsScale>();
        constexpr bool enable              = GetLdsPadEnable<Problem, IsA, IsScale>();
        constexpr TDMLdsPaddingConfig padding =
            enable ? make_tdm_lds_padding_config(op.block_bytes, op.pad_bytes)
                   : TDMLdsPaddingConfig{};
        return make_tuple(number<enable>{},
                          number<static_cast<index_t>(padding.pad_amount)>{},
                          number<static_cast<index_t>(padding.pad_interval)>{});
    }

    /// A's or B's data in LDS as the TDM loads write it, over [tile row, K] in elements of the
    /// operand: component c's op.rows rows start op.stride bytes after component c - 1's. The
    /// offset is from the operand's base.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsTdmBlockDescriptor()
    {
        constexpr auto runs       = MakeLdsRunsDescriptor<Problem, IsA, true>();
        constexpr auto lengths    = runs.get_lengths();
        constexpr index_t comps   = lengths[number<0>{}];
        constexpr index_t outer   = lengths[number<1>{}];
        constexpr index_t inner   = lengths[number<2>{}];
        constexpr index_t k_elems = lengths[number<3>{}];
        return transform_tensor_descriptor(
            runs,
            make_tuple(
                make_merge_transform(make_tuple(number<comps>{}, number<outer>{}, number<inner>{})),
                make_pass_through_transform(number<k_elems>{})),
            make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    /// The DRAM distribution of one TDM component load of A or B: one box of the component's
    /// rows (an iterate-mode block of them) by the K tile, the same for every wave; a wave's
    /// component is its window's origin. With IsScale, of A's or B's K-major scales: the tile's
    /// K steps by the component's rows.
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDramTdmTileDistribution()
    {
        constexpr MxTdmScheduledOperand op = GetLdsOperand<Problem, IsA, IsScale>();
        constexpr index_t box_rows         = op.iterate_rows > 0 ? op.iterate_rows : op.rows;
        constexpr index_t len0             = IsScale ? GetKSteps<Problem>() : box_rows;
        constexpr index_t len1             = IsScale ? op.rows : GetSpec<Problem>().tile_k;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<mx_tdm_scheduled::kWaves>,
                                       tuple<sequence<len0>, sequence<len1>>,
                                       tuple<sequence<0>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{},
            bool_constant<true>{});
    }

    /// A's or B's data in LDS as a descriptor over [tile row in fragment order, K word]. Row t =
    /// (wave * repeats + r) * atom + l is tile row wave * atom * repeats + l * repeats + r (as
    /// MRow / NRow), so one WMMA fragment is atom consecutive rows; the offset, in 4-byte words,
    /// is mx_tdm_scheduled_row_offset's less the operand's base.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        constexpr MxTdmScheduledSpec s     = GetSpec<Problem>();
        constexpr MxTdmScheduledOperand op = GetLdsOperand<Problem, IsA>();
        constexpr index_t row_bytes        = GetLdsRowBytes<Problem>();
        constexpr index_t warps            = IsA ? s.warp_m : s.warp_n;
        constexpr index_t atom             = IsA ? s.atom_m() : s.atom_n();
        constexpr index_t repeats          = IsA ? s.wave_repeat_m() : s.wave_repeat_n();
        constexpr index_t rows             = warps * atom * repeats;
        constexpr index_t words            = row_bytes / 4;
        constexpr auto naive               = MakeLdsRunsDescriptor<Problem, IsA>();
        constexpr index_t inner            = naive.get_lengths()[number<2>{}];
        constexpr auto by_row              = transform_tensor_descriptor(
            naive,
            make_tuple(make_merge_transform(make_tuple(
                           number<rows / op.rows>{}, number<op.rows / inner>{}, number<inner>{})),
                       make_pass_through_transform(number<words>{})),
            make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        constexpr auto by_wave = transform_tensor_descriptor(
            by_row,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<warps>{}, number<atom>{}, number<repeats>{})),
                       make_pass_through_transform(number<words>{})),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0, 1, 2>{}, sequence<3>{}));
        return transform_tensor_descriptor(
            by_wave,
            make_tuple(make_merge_transform(
                           make_tuple(number<warps>{}, number<repeats>{}, number<atom>{})),
                       make_pass_through_transform(number<words>{})),
            make_tuple(sequence<0, 2, 1>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    /// A's or B's data in LDS as runs of rows, over [component, run, run row, K]: K in 4-byte
    /// words, or with InElements in elements of the operand.
    template <typename Problem, bool IsA, bool InElements = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsRunsDescriptor()
    {
        constexpr MxTdmScheduledSpec s     = GetSpec<Problem>();
        constexpr MxTdmScheduledOperand op = GetLdsOperand<Problem, IsA>();
        constexpr index_t row_bytes        = GetLdsRowBytes<Problem>();
        constexpr index_t warps            = IsA ? s.warp_m : s.warp_n;
        constexpr index_t atom             = IsA ? s.atom_m() : s.atom_n();
        constexpr index_t repeats          = IsA ? s.wave_repeat_m() : s.wave_repeat_n();
        constexpr index_t rows             = warps * atom * repeats;
        // A component's rows split into (outer, inner) runs: iterate-mode blocks, padded blocks
        // of whole rows, or one run (rows padded within, or unpadded).
        constexpr bool iterate         = op.iterate_rows > 0;
        constexpr bool padded          = !iterate && op.pad_bytes > 0;
        constexpr bool block_pad       = padded && op.block_bytes >= row_bytes;
        constexpr index_t inner        = iterate     ? op.iterate_rows
                                         : block_pad ? op.block_bytes / row_bytes
                                                     : op.rows;
        constexpr index_t outer_stride = iterate     ? op.iterate_stride
                                         : block_pad ? op.block_bytes + op.pad_bytes
                                                     : 0;
        constexpr index_t row_stride   = padded && !block_pad
                                             ? row_bytes + row_bytes / op.block_bytes * op.pad_bytes
                                             : row_bytes;
        static_assert(rows % op.rows == 0 && op.rows % inner == 0 && row_bytes % 4 == 0 &&
                          op.stride % 4 == 0 && outer_stride % 4 == 0 && row_stride % 4 == 0,
                      "comp_tdm_scheduled: the LDS layout is not a whole number of words per run");
        constexpr auto unit = [](index_t bytes) {
            return InElements ? bytes * GetKPerByte<Problem>() : bytes / 4;
        };
        return make_naive_tensor_descriptor(make_tuple(number<rows / op.rows>{},
                                                       number<op.rows / inner>{},
                                                       number<inner>{},
                                                       number<unit(row_bytes)>{}),
                                            make_tuple(number<unit(op.stride)>{},
                                                       number<unit(outer_stride)>{},
                                                       number<unit(row_stride)>{},
                                                       number<1>{}),
                                            number < InElements ? 1 : 4 > {},
                                            number<1>{});
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        return MakeLdsBlockDescriptor<Problem, true>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        return MakeLdsBlockDescriptor<Problem, false>();
    }

    /// 16-row halves of an A or B WMMA fragment: 2 for A of the 32x16 atom, else 1.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetFragmentHalves()
    {
        return IsA ? GetSpec<Problem>().atom_m() / 16 : 1;
    }

    /// One wave's read of an A or B fragment of one K step, in words: lane l reads row l % 16 of
    /// each half and, per 32-byte chunk, the 16 bytes l / 16 selects.
    ///   rows: (half, l % 16)   K words: (chunk, l / 16, 4)
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsReadDistribution()
    {
        constexpr index_t chunks = GetChunksPerLane<Problem>();
        if constexpr(GetFragmentHalves<Problem, IsA>() == 2)
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                           tuple<sequence<2, 16>, sequence<chunks, 2, 4>>,
                                           tuple<sequence<2, 1>>,
                                           tuple<sequence<1, 1>>,
                                           sequence<1, 2, 2>,
                                           sequence<0, 0, 2>>{});
        else
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                           tuple<sequence<16>, sequence<chunks, 2, 4>>,
                                           tuple<sequence<2, 1>>,
                                           tuple<sequence<1, 0>>,
                                           sequence<2, 2>,
                                           sequence<0, 2>>{});
    }

    /// Word offset of fragment-order row t of A's or B's data from the operand's base, as
    /// MakeLdsBlockDescriptor maps it.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsRowOffset(index_t t)
    {
        constexpr MxTdmScheduledSpec s     = GetSpec<Problem>();
        constexpr MxTdmScheduledOperand op = GetLdsOperand<Problem, IsA>();
        constexpr index_t atom             = IsA ? s.atom_m() : s.atom_n();
        constexpr index_t repeats          = IsA ? s.wave_repeat_m() : s.wave_repeat_n();
        const index_t wave                 = t / (repeats * atom);
        const index_t r                    = t / atom % repeats;
        const index_t l                    = t % atom;
        const index_t row                  = wave * atom * repeats + l * repeats + r;
        return (mx_tdm_scheduled_row_offset<index_t>(op, GetLdsRowBytes<Problem>(), row) -
                op.base) /
               4;
    }

    /// Whether a fragment read can take its repeat and half offsets as constants: for every
    /// lane's first row, moving by a repeat r and a half h adds the offset of (r, h) from row 0.
    /// Also spot-checks MakeLdsBlockDescriptor against mx_tdm_scheduled_row_offset (a full check is
    /// past the constexpr step limit).
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr bool IsLdsReadLinear()
    {
        constexpr MxTdmScheduledSpec s = GetSpec<Problem>();
        constexpr index_t warps        = IsA ? s.warp_m : s.warp_n;
        constexpr index_t atom         = IsA ? s.atom_m() : s.atom_n();
        constexpr index_t repeats      = IsA ? s.wave_repeat_m() : s.wave_repeat_n();
        constexpr index_t halves       = GetFragmentHalves<Problem, IsA>();
        constexpr index_t rows         = warps * repeats * atom;
        constexpr auto desc            = MakeLdsBlockDescriptor<Problem, IsA>();
        const auto offset              = [](index_t t) { return GetLdsRowOffset<Problem, IsA>(t); };
        for(const index_t t : {index_t{1}, atom - 1, atom, rows / 2 + 1, rows - 1})
            if(desc.calculate_offset(make_multi_index(t, 0)) != offset(t))
                return false;
        const index_t wave_offset = offset(repeats * atom);
        for(index_t wave = 0; wave < warps; ++wave)
            for(index_t l = 0; l < 16; ++l)
            {
                const index_t t0 = wave * repeats * atom + l;
                const index_t o0 = offset(t0);
                if(o0 != wave * wave_offset + offset(l))
                    return false;
                for(index_t r = 0; r < repeats; ++r)
                    for(index_t h = 0; h < halves; ++h)
                        if(offset(t0 + r * atom + h * 16) != o0 + offset(r * atom) + offset(h * 16))
                            return false;
            }
        return true;
    }

    /// A's or B's E8M0 scales in LDS as the TDM loads write them, over [K step, tile row] in
    /// 4-byte packs: component c's op.rows rows of every step start op.stride bytes after
    /// component c - 1's, each step's rows follow the last step's, padded per block.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsScaleTdmBlockDescriptor()
    {
        constexpr MxTdmScheduledSpec s     = GetSpec<Problem>();
        constexpr MxTdmScheduledOperand op = GetLdsOperand<Problem, IsA, true>();
        constexpr index_t warps            = IsA ? s.warp_m : s.warp_n;
        constexpr index_t atom             = IsA ? s.atom_m() : s.atom_n();
        constexpr index_t repeats          = IsA ? s.wave_repeat_m() : s.wave_repeat_n();
        constexpr index_t comps            = warps * atom * repeats / op.rows;
        constexpr index_t steps            = GetKSteps<Problem>();
        constexpr index_t run              = steps * op.rows;
        constexpr index_t block            = op.pad_bytes > 0 ? op.block_bytes / 4 : run;
        static_assert(
            warps * atom * repeats % op.rows == 0 && run % block == 0 && op.stride % 4 == 0 &&
                (op.block_bytes + op.pad_bytes) % 4 == 0,
            "comp_tdm_scheduled: the scale layout is not a whole number of padded blocks");
        constexpr auto naive = make_naive_tensor_descriptor(
            make_tuple(number<comps>{}, number<run / block>{}, number<block>{}),
            make_tuple(number<op.stride / 4>{},
                       number<(op.block_bytes + op.pad_bytes) / 4>{},
                       number<1>{}),
            number<4>{},
            number<1>{});
        constexpr auto linear = transform_tensor_descriptor(
            naive,
            make_tuple(make_pass_through_transform(number<comps>{}),
                       make_merge_transform(make_tuple(number<run / block>{}, number<block>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        constexpr auto by_step = transform_tensor_descriptor(
            linear,
            make_tuple(make_pass_through_transform(number<comps>{}),
                       make_unmerge_transform(make_tuple(number<steps>{}, number<op.rows>{}))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<1>{}, sequence<0, 2>{}));
        return transform_tensor_descriptor(
            by_step,
            make_tuple(make_pass_through_transform(number<steps>{}),
                       make_merge_transform(make_tuple(number<comps>{}, number<op.rows>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    /// A's or B's E8M0 scales in LDS as words, top dimensions (K step, wave * atom + lane row,
    /// repeat): MakeLdsScaleTdmBlockDescriptor with the tile row split per repeat.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsScaleDescriptor()
    {
        constexpr MxTdmScheduledSpec s = GetSpec<Problem>();
        constexpr index_t warps        = IsA ? s.warp_m : s.warp_n;
        constexpr index_t atom         = IsA ? s.atom_m() : s.atom_n();
        constexpr index_t repeats      = IsA ? s.wave_repeat_m() : s.wave_repeat_n();
        constexpr index_t steps        = GetKSteps<Problem>();
        return transform_tensor_descriptor(
            MakeLdsScaleTdmBlockDescriptor<Problem, IsA>(),
            make_tuple(
                make_pass_through_transform(number<steps>{}),
                make_unmerge_transform(make_tuple(number<warps * atom>{}, number<repeats>{}))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}));
    }

    /// Lanes reading distinct scale rows: all 32 for A of the 32x16 atom, else 16 (lanes l and
    /// l + 16 read the same row).
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetScaleLanes()
    {
        return IsA && GetSpec<Problem>().atom_m() == 32 ? 32 : 16;
    }

    /// One wave's read of A's or B's scales of one K step: lane l reads its row's words of all
    /// repeats.   (K step: 1, rows: lanes, repeats)
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsScaleReadDistribution()
    {
        constexpr MxTdmScheduledSpec s = GetSpec<Problem>();
        constexpr index_t repeats      = IsA ? s.wave_repeat_m() : s.wave_repeat_n();
        if constexpr(GetScaleLanes<Problem, IsA>() == 32)
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                           tuple<sequence<1>, sequence<32>, sequence<repeats>>,
                                           tuple<sequence<2>>,
                                           tuple<sequence<0>>,
                                           sequence<1, 3>,
                                           sequence<0, 0>>{});
        else
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<2>,
                                           tuple<sequence<1>, sequence<16>, sequence<repeats>>,
                                           tuple<sequence<0, 2>>,
                                           tuple<sequence<0, 0>>,
                                           sequence<1, 3>,
                                           sequence<0, 0>>{});
    }

    /// Whether a scale read can take its K step as a constant word offset and its repeats as
    /// consecutive words, checked against mx_tdm_scheduled_scale_offset. Also spot-checks
    /// MakeLdsScaleDescriptor.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr bool IsLdsScaleReadLinear()
    {
        constexpr MxTdmScheduledSpec s = GetSpec<Problem>();
        constexpr index_t warps        = IsA ? s.warp_m : s.warp_n;
        constexpr index_t atom         = IsA ? s.atom_m() : s.atom_n();
        constexpr index_t repeats      = IsA ? s.wave_repeat_m() : s.wave_repeat_n();
        constexpr index_t steps        = GetKSteps<Problem>();
        constexpr auto desc            = MakeLdsScaleDescriptor<Problem, IsA>();
        const auto offset              = [](index_t step, index_t row) {
            constexpr MxTdmScheduledOperand op = GetLdsOperand<Problem, IsA, true>();
            return (mx_tdm_scheduled_scale_offset<index_t>(op, step, row) - op.base) / 4;
        };
        const index_t step_words = offset(1, 0);
        for(const index_t t : {index_t{1}, atom - 1, atom, warps * atom - 1})
            for(const index_t step : {index_t{0}, steps - 1})
                if(desc.calculate_offset(make_multi_index(step, t, repeats - 1)) !=
                   offset(step, t * repeats + repeats - 1))
                    return false;
        for(index_t t = 0; t < warps * atom; ++t)
            for(index_t step = 0; step < steps; ++step)
                for(index_t r = 0; r < repeats; ++r)
                    if(offset(step, t * repeats + r) !=
                       offset(0, t * repeats) + step * step_words + r)
                        return false;
        return true;
    }

    /// The cluster workgroups that share this one's A (kM: same M tile) or B (kN: same N tile)
    /// and so receive its TDM loads; 0 without a cluster. As the default TDM policy's, but grid
    /// x walks N tiles and y M tiles, and the cluster shape is the tuning's.
    template <MultiCastDirection Direction, typename Problem>
    CK_TILE_DEVICE static uint16_t GetTDMWorkgroupMask(dim3 block_id_in_cluster)
    {
        static_assert(Direction != MultiCastDirection::kMN,
                      "comp_tdm_scheduled multicasts A along M and B along N");
        uint32_t mask = 0;
        if constexpr(!HasCluster())
            ignore = block_id_in_cluster;
        else if constexpr(Direction == MultiCastDirection::kM)
            mask = cluster_mask_same_y(Tuning::kClusterDimX, block_id_in_cluster.y);
        else
            mask = cluster_mask_same_x(
                Tuning::kClusterDimX, Tuning::kClusterDimY, block_id_in_cluster.x);
        // A cluster has at most 16 workgroups; without this the 16-bit mask costs an s_and.
        __builtin_assume(mask <= 0xFFFFu);
        return static_cast<uint16_t>(mask);
    }

    /// The C block tile: per thread the WMMA C fragments of its M and N repeats, (rm, rn) outer
    /// and the fragment slot inner. A lane's M repeats are adjacent rows and its N repeats
    /// adjacent columns, so one slot across the N repeats is one vector store of D.
    ///   M: (wave_m, slot / 8, lane / 16, slot % 8, rm)   N: (wave_n, lane % 16, rn)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeCBlockTileDistribution()
    {
        constexpr MxTdmScheduledSpec s = GetSpec<Problem>();
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
        constexpr MxTdmScheduledSpec s = GetSpec<Problem>();
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

    /// The block GEMM: a wave's LDS fragment reads and WMMAs on this policy's LDS layout.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return BlockGemmMxAsmemBsmemCRegTDMScheduled<Problem,
                                                     MxGemmPipelineAgBgCrCompTDMScheduledPolicy>{};
    }

    /// The cluster the kernel must be launched with: (x, y, z) = (N tiles, M tiles, 1).
    CK_TILE_HOST_DEVICE static constexpr bool HasCluster()
    {
        return !(Tuning::kClusterDimX == 1 && Tuning::kClusterDimY == 1);
    }
};

/// rocKE's validate() at compile time: nullptr if Problem and Tuning build, else the reason code
/// (see mx_tdm_scheduled_reason_code). The shape rules are the pipeline's GetUnsupportedReason.
template <typename Problem, typename Tuning = MxTdmScheduledDefaultTuning>
CK_TILE_HOST_DEVICE constexpr const char* MxTdmScheduledValidate()
{
    constexpr MxTdmScheduledReason r =
        MxGemmPipelineAgBgCrCompTDMScheduledPolicy<Tuning>::template GetReason<Problem>();
    return r == MxTdmScheduledReason::Ok ? nullptr : mx_tdm_scheduled_reason_code(r);
}

/// The rocKE name suffix of a tuning: the cluster, then its non-default values (see
/// mx_tdm_scheduled_tuning_name), without the parts a CK kernel name already has.
template <typename Tuning>
CK_TILE_HOST std::string MxTdmScheduledTuningName()
{
    MxTdmScheduledSpec s{};
    s.cluster_x              = Tuning::kClusterDimX;
    s.cluster_y              = Tuning::kClusterDimY;
    s.cluster_z              = Tuning::kClusterDimZ;
    s.gl2_distance           = Tuning::kGl2PrefetchDistance;
    s.signal_wmmas_in        = Tuning::kClusterSignalWmmas;
    s.local_prefetch         = Tuning::kLocalPrefetch;
    s.half_plr               = Tuning::kHalfPlr;
    s.half_b_order           = Tuning::kHalfBOrder;
    s.lds_segment_interleave = Tuning::kLdsSegmentInterleave;
    return mx_tdm_scheduled_tuning_name(s, false);
}

} // namespace ck_tile
