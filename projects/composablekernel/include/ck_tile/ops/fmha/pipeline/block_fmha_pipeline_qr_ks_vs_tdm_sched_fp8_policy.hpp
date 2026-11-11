// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_load.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_softmax.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"

namespace ck_tile {

// Tuning that also selects the segmented D256 LDS layout; other instances keep their type.
template <typename Tuning_>
struct FmhaTdmSchedFp8LdsSegTuning : Tuning_
{
};

template <typename Tuning>
inline constexpr bool kFmhaTdmSchedFp8IsLdsSegTuning = false;
template <typename Tuning>
inline constexpr bool kFmhaTdmSchedFp8IsLdsSegTuning<FmhaTdmSchedFp8LdsSegTuning<Tuning>> = true;

template <typename Geometry_,
          typename Tuning_,
          typename TileSchedule_ = FmhaTdmSchedFp8DefaultScheduleFor<Geometry_>>
struct BlockFmhaPipelineQRKSVSTdmSchedFp8Policy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Geometry   = Geometry_;
    using Tuning     = Tuning_;
    using Schedule   = TileSchedule_;
    using Layout     = FmhaTdmSchedFp8Layout<Geometry>;
    using KLoad      = FmhaTdmSchedFp8Load<Geometry::kKSuLoadCount, Geometry::kKLdsReadBytes>;
    using VLoad      = FmhaTdmSchedFp8TransposeLoad<Geometry::kVStageLoadCount>;
    static constexpr index_t kBlockPerCu = Tuning::kBlockPerCu;

    static constexpr index_t kLdsRows         = Geometry::kN;
    static constexpr index_t kKValidWidth     = Geometry::kHeadDimQK;
    static constexpr index_t kKPhysicalStride = Layout::kKPhysicalStride;
    static constexpr index_t kVLogicalWidth   = Geometry::kDv;
    static constexpr index_t kVPhysicalStride = Layout::kVPhysicalStride;

    using KPadding =
        std::conditional_t<Layout::kKPadByTdm,
                           detail::LdsPaddingConfig<true, kKValidWidth, Layout::kRowPadBytes>,
                           detail::LdsPaddingConfig<false, 0, 0>>;
    // Width of the K TDM box: the row, or the padded row whose tail reads zeros past hdim_q.
    static constexpr index_t kKDramBoxWidth = Layout::kKPadByTdm ? kKValidWidth : kKPhysicalStride;
    // D128/D256 Q rows take the K row padding so per-lane row reads spread across LDS banks.
    static constexpr bool kPadQRows = kKValidWidth >= 128 && Layout::kKPadByTdm;
    using VPadding = detail::LdsPaddingConfig<true, kVLogicalWidth, Layout::kRowPadBytes>;
    static_assert(sizeof(typename Geometry::KDataType) == 1 &&
                      sizeof(typename Geometry::VDataType) == 1 &&
                      kKPhysicalStride == kKValidWidth + Layout::kRowPadBytes &&
                      kVPhysicalStride == kVLogicalWidth + Layout::kRowPadBytes,
                  "TDM row padding must match the LDS read stride");
    static constexpr index_t kKFootprintElements = kLdsRows * kKPhysicalStride;
    static constexpr index_t kVFootprintElements = kLdsRows * kVPhysicalStride;
    static constexpr index_t kKFootprintBytes =
        kKFootprintElements * sizeof(typename Geometry::KDataType);
    static constexpr index_t kVFootprintBytes =
        kVFootprintElements * sizeof(typename Geometry::VDataType);
    // The output rescale is fused into softmax part 2; taking the early slot skips the post pass.
    static constexpr bool kEarlyOutputRescale      = true;
    static constexpr bool kLeadFirstLoad           = false;
    static constexpr index_t kKPrefetchTensorCount = Tuning::kKWait;
    static constexpr index_t kVPrefetchTensorCount = Tuning::kVWait;
    static constexpr bool kPrefetchTailDrain       = Tuning::kTailDrain;
    static_assert(kKPrefetchTensorCount == 0 || kKPrefetchTensorCount == 1);
    static_assert(kVPrefetchTensorCount == 0 || kVPrefetchTensorCount == 1);

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQkTailDsCount()
    {
        return Tuning::template GetQkTailDsCount<Stage>();
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetPvTailDsCount()
    {
        return Tuning::template GetPvTailDsCount<Stage>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitQkStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kQkStages);
        s_wait_dscnt<kFmhaTdmSchedFp8DsWait<GetQkTailDsCount<Stage>()>>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitPvStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        s_wait_dscnt<kFmhaTdmSchedFp8DsWait<GetPvTailDsCount<Stage>()>>();
    }

    // FP8 Q is staged in the V1 buffer, which D192 widens to its unpadded rows.
    static constexpr index_t kQFootprintBytes =
        Geometry::kM * (kPadQRows ? kKPhysicalStride : kKValidWidth);
    static constexpr index_t kV1FootprintBytes = max(kVFootprintBytes, kQFootprintBytes);
    // Segmented: K/V lower halves packed from 0 (Q fills the gap), upper halves a segment away.
    static constexpr bool kLdsSegLayout   = kFmhaTdmSchedFp8IsLdsSegTuning<Tuning_>;
    static constexpr index_t kLdsSegHalf  = kLdsRows / 2 * kKPhysicalStride;
    static constexpr index_t kLdsSegShift = kLdsSegLayout ? 4 * kLdsSegHalf : 0;
    static_assert(!kLdsSegLayout ||
                  (Geometry::kHeadDimQK == 256 && Geometry::kWaves == 4 &&
                   kKPhysicalStride == kVPhysicalStride && kQFootprintBytes == 2 * kLdsSegHalf &&
                   kLdsSegShift + kLdsSegHalf >= 65536));
    static constexpr index_t kLdsOffsetK0 = 0;
    static constexpr index_t kLdsOffsetK1 =
        kLdsSegLayout ? kLdsSegHalf : kLdsOffsetK0 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV0 =
        kLdsSegLayout ? 2 * kLdsSegHalf : kLdsOffsetK1 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV1 =
        kLdsSegLayout ? 3 * kLdsSegHalf : kLdsOffsetV0 + kVFootprintBytes;
    static constexpr index_t kLdsArenaSize =
        kLdsSegLayout ? 9 * kLdsSegHalf : kLdsOffsetV1 + kV1FootprintBytes;

    // fp32 S/O fragments are input-type independent, so the 16-bit softmax/output maps apply.
    using OutputFragments = FmhaTdmSchedOutputFragmentsFor<Geometry::kPvNIter / 2>;

    static_assert(kFmhaTdmSchedFp8GeometryOk<Geometry>);
    using ScoreSoftmax = FmhaTdmSchedSplitSoftmaxFor<Geometry::kQkStages>;
    using Softmax      = FmhaTdmSchedFp8Softmax<Geometry, OutputFragments>;
    using ScoreMapping = typename ScoreSoftmax::Mapping;
    static_assert(ScoreMapping::kNumMIter == Geometry::kQkMIter &&
                      ScoreMapping::kNumNIter == Geometry::kN / Geometry::kQkWmmaN &&
                      ScoreMapping::kNumSu == Geometry::kQkStages &&
                      ScoreMapping::kNIterPerSu == Geometry::kQkSuNIter &&
                      ScoreMapping::kNumMsb == Geometry::kQkMIter * Geometry::kQkSuNIter &&
                      ScoreMapping::kElementsPerWmma ==
                          Geometry::QkWmma::kCM0PerLane * Geometry::QkWmma::kCM1PerLane,
                  "split-softmax score mapping does not match the QK geometry");
    static_assert(OutputFragments::kNumFragments == Geometry::kPvMIter * Geometry::kPvNIter &&
                      OutputFragments::kNumDmsb % Geometry::kPvMIter == 0 &&
                      OutputFragments::kElementsPerFragment ==
                          Geometry::PvWmma::kCM0PerLane * Geometry::PvWmma::kCM1PerLane,
                  "output fragments do not match the PV geometry");
    // One PV stage covers the KV tile, so every PV row is an output fragment.
    static_assert(Geometry::kPvWmmasPerStage == OutputFragments::kNumFragments);

    static_assert(kLdsOffsetK0 % 16 == 0 && kLdsOffsetK1 % 16 == 0 && kLdsOffsetV0 % 16 == 0 &&
                  kLdsOffsetV1 % 16 == 0);

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemmSu()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Geometry::kQkSuColumns,
                                                    Problem::BlockFmhaShape::kQKHeaddim>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::QDataType,
                                            typename Problem::KDataType,
                                            typename Problem::SaccDataType,
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                            true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::SaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKSuRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemmSu<Problem>())>;
        static_assert(BlockGemm::NIterPerWarp == Geometry::kQkSuNIter &&
                      BlockGemm::KIterPerWarp == Geometry::kQkKIter);
        return make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
    }

    // Marks a GEMM whose WMMA scale operands stay neutral.
    struct NoScale
    {
    };

    // K64 has no scaled WMMA: zero-pad to K128, whose low eight dwords keep the K64 lane order.
    template <typename CWarpTensor, typename AWarpTensor, typename BWarpTensor>
    CK_TILE_DEVICE static void RunPaddedScaledWmma(CWarpTensor& c_warp_tensor,
                                                   const AWarpTensor& a_warp_tensor,
                                                   const BWarpTensor& b_warp_tensor,
                                                   int32_t a_scale,
                                                   int32_t b_scale)
    {
        using QDataType     = typename Geometry::QDataType;
        using PaddedGemm    = WarpGemmDispatcher<QDataType, QDataType, float, 16, 16, 128, true>;
        using Half          = int32_t __attribute__((ext_vector_type(8)));
        constexpr auto zero = number<0>{};
        constexpr auto one  = number<1>{};
        static_assert(sizeof(typename AWarpTensor::DataType) == 1 &&
                      AWarpTensor::get_thread_buffer_size() == sizeof(Half) &&
                      BWarpTensor::get_thread_buffer_size() == sizeof(Half));

        typename PaddedGemm::AWarpTensor a;
        typename PaddedGemm::BWarpTensor b;
        typename PaddedGemm::CWarpTensor c;
        a.get_thread_buffer().template set_as<Half>(
            zero, bit_cast<Half>(a_warp_tensor.get_thread_buffer()));
        a.get_thread_buffer().template set_as<Half>(one, Half{});
        b.get_thread_buffer().template set_as<Half>(
            zero, bit_cast<Half>(b_warp_tensor.get_thread_buffer()));
        b.get_thread_buffer().template set_as<Half>(one, Half{});
        c.get_thread_buffer() = c_warp_tensor.get_thread_buffer();
        PaddedGemm{}(c, a, b, a_scale, b_scale);
        c_warp_tensor.get_thread_buffer() = c.get_thread_buffer();
    }

    template <index_t WmmaOrdinal,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename AScaleFn = NoScale,
              typename BScaleFn = NoScale>
    CK_TILE_DEVICE static void RunQkSuWmma(CBlockTensor& c_block_tensor,
                                           const ABlockTensor& a_block_tensor,
                                           const BBlockTensor& b_block_tensor,
                                           const AScaleFn& a_scale = {},
                                           const BScaleFn& b_scale = {})
    {
        using WarpGemm    = typename BlockGemm::WarpGemm;
        using AWarpDstr   = typename WarpGemm::AWarpDstr;
        using BWarpDstr   = typename WarpGemm::BWarpDstr;
        using CWarpDstr   = typename WarpGemm::CWarpDstr;
        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_assert(WmmaOrdinal >= 0 && WmmaOrdinal < BlockGemm::NIterPerWarp *
                                                            BlockGemm::KIterPerWarp *
                                                            BlockGemm::MIterPerWarp);
        constexpr index_t n_iter =
            WmmaOrdinal / (BlockGemm::KIterPerWarp * BlockGemm::MIterPerWarp);
        constexpr index_t k_iter =
            (WmmaOrdinal / BlockGemm::MIterPerWarp) % BlockGemm::KIterPerWarp;
        constexpr index_t m_iter = WmmaOrdinal % BlockGemm::MIterPerWarp;

        AWarpTensor a_warp_tensor;
        a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, k_iter>{}, a_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

        BWarpTensor b_warp_tensor;
        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<n_iter, k_iter>{}, b_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

        CWarpTensor c_warp_tensor;
        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

        if constexpr(std::is_same_v<AScaleFn, NoScale>)
            WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        else if constexpr(WarpGemm::kK == 128)
            WarpGemm{}(c_warp_tensor,
                       a_warp_tensor,
                       b_warp_tensor,
                       a_scale(number<m_iter>{}, number<k_iter>{}),
                       b_scale(number<n_iter>{}, number<k_iter>{}));
        else
            RunPaddedScaledWmma(c_warp_tensor,
                                a_warp_tensor,
                                b_warp_tensor,
                                a_scale(number<m_iter>{}, number<k_iter>{}),
                                b_scale(number<n_iter>{}, number<k_iter>{}));
        c_block_tensor.set_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    }

    template <FmhaTdmSchedSchedulePoint Point>
    CK_TILE_DEVICE static void EmitSchedulePoint()
    {
        if constexpr(Point == FmhaTdmSchedSchedulePoint::AfterTokens)
            __builtin_amdgcn_sched_barrier(0);
        else
            __builtin_amdgcn_sched_barrier(LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::TRANS);
    }

    // Fence emitter of the placement executor: (stage, wmma, point).
    struct SchedulePointEmitter
    {
        template <typename Stage, typename Wmma, typename Point>
        CK_TILE_DEVICE void operator()(Stage, Wmma, Point) const
        {
            EmitSchedulePoint<Point::value>();
        }
    };

    // QK stage WMMAs interleaved with reads of the next K group (last stage: the PV V tile).
    template <index_t Stage,
              typename Mode,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename AScaleFn = NoScale,
              typename BScaleFn = NoScale>
    CK_TILE_DEVICE static void RunQkScheduledStage(const BlockGemm&,
                                                   CBlockTensor& c_block_tensor,
                                                   const ABlockTensor& a_block_tensor,
                                                   const BBlockTensor& b_block_tensor,
                                                   NextBBlockTensor& next_b_block_tensor,
                                                   const BTileWindow& b_lds_window,
                                                   const AScaleFn& a_scale = {},
                                                   const BScaleFn& b_scale = {})
    {
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumQkStages);
        static_assert((Stage < SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount) ||
                      (Stage == SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount));

        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
            {
                static_assert(Stage < Geometry::kQkStages - 1);
                KLoad::template LoadInstruction<decltype(access)::value>(next_b_block_tensor,
                                                                         b_lds_window);
            }
            else
            {
                static_assert(Stage == Geometry::kQkStages - 1);
                VLoad::template LoadAccess<decltype(access)::value>(next_b_block_tensor,
                                                                    b_lds_window);
            }
        };
        SchedulePointEmitter emit_point;
        auto emit_wmma = [&](auto, auto wmma) {
            RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
                c_block_tensor, a_block_tensor, b_block_tensor, a_scale, b_scale);
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        WaitQkStageTail<Stage>();
    }

    // Marks a lead read that has no group after it.
    struct NoLeadGroup
    {
    };

    // QK stage reading the next group's second half, then the first half of the group after.
    template <index_t Stage,
              typename Mode,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename LeadBBlockTensor,
              typename LeadTileWindow,
              typename AScaleFn = NoScale,
              typename BScaleFn = NoScale>
    CK_TILE_DEVICE static void RunQkScheduledLeadStage(const BlockGemm&,
                                                       CBlockTensor& c_block_tensor,
                                                       const ABlockTensor& a_block_tensor,
                                                       const BBlockTensor& b_block_tensor,
                                                       NextBBlockTensor& next_b_block_tensor,
                                                       const BTileWindow& b_lds_window,
                                                       LeadBBlockTensor& lead_b_block_tensor,
                                                       const LeadTileWindow& lead_lds_window,
                                                       const AScaleFn& a_scale = {},
                                                       const BScaleFn& b_scale = {})
    {
        using SelectedSchedule  = typename Mode::SelectedPolicy::Schedule;
        using Executor          = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind              = FmhaTdmSchedLoadKind;
        constexpr index_t kHalf = Geometry::kKSuLoadCount / 2;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumQkStages - 1 &&
                      BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount);

        auto emit_token = [&](auto kind, auto access) {
            static_assert(decltype(kind)::value == Kind::KRead);
            constexpr index_t a = decltype(access)::value;
            if constexpr(a < kHalf)
                KLoad::template LoadInstruction<a + kHalf>(next_b_block_tensor, b_lds_window);
            else if constexpr(!std::is_same_v<LeadTileWindow, NoLeadGroup>)
                KLoad::template LoadInstruction<a - kHalf>(lead_b_block_tensor, lead_lds_window);
        };
        SchedulePointEmitter emit_point;
        auto emit_wmma = [&](auto, auto wmma) {
            RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
                c_block_tensor, a_block_tensor, b_block_tensor, a_scale, b_scale);
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        WaitQkStageTail<Stage>();
    }

    // Marks a PV stage without streamed V reads (all but D256).
    struct NoStreamedV
    {
    };

    // PV stage WMMAs interleaved with next-K group 0 reads and, at D256, the streamed V chunks.
    template <index_t Stage,
              typename Mode,
              bool KDwordStore,
              typename NextKTensor,
              typename KTileWindow,
              typename VTensor,
              typename VTileWindow,
              typename WmmaEmitter>
    CK_TILE_DEVICE static void RunPvStageWithWmma(NextKTensor& next_k_tensor,
                                                  const KTileWindow& k_lds_window,
                                                  VTensor& v_tensor,
                                                  const VTileWindow& v_lds_window,
                                                  WmmaEmitter& emit_wmma)
    {
        using SelectedSchedule   = typename Mode::SelectedPolicy::Schedule;
        using Executor           = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind               = FmhaTdmSchedLoadKind;
        constexpr bool kStreamsV = !std::is_same_v<VTileWindow, NoStreamedV>;
        static_assert(Stage == SelectedSchedule::kNumPvStages - 1 &&
                      KTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount);
        if constexpr(kStreamsV)
            static_assert(VTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount);
        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
                KLoad::template LoadInstruction<decltype(access)::value, KDwordStore>(next_k_tensor,
                                                                                      k_lds_window);
            else
            {
                static_assert(kStreamsV && decltype(access)::value >= 0, "V reads need a V window");
                VLoad::template LoadAccess<decltype(access)::value>(v_tensor, v_lds_window);
            }
        };
        SchedulePointEmitter emit_point;
        // The output rescale is fused into softmax, so PV rows emit none.
        auto no_rescale = [](auto) {};
        Executor::template ExecutePvStage<Stage>(emit_wmma, emit_token, emit_point, no_rescale);
        WaitPvStageTail<Stage>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCountdownLoop()
    {
        return !Problem::FmhaMask::IsMasking && !Problem::kHasSink &&
               Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool DeferTensorReady()
    {
        return !Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{}) == 1;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedProblem()
    {
        using Shape = remove_cvref_t<typename Problem::BlockFmhaShape>;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        constexpr bool is_wave32 = get_warp_size() == Geometry::kWaveSize;
#else
        constexpr bool is_wave32 = true;
#endif
        return (Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                (Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS &&
                 Problem::QScaleEnum != BlockAttentionQuantScaleEnum::MX)) &&
               std::is_same_v<FmhaTdmSchedGeometryFor<Problem, Geometry::kQkSuColumns>, Geometry> &&
               Shape::Gemm1BlockWarps::at(number<0>{}) == Shape::Gemm0BlockWarps::at(number<0>{}) &&
               Shape::Gemm1BlockWarps::at(number<1>{}) == Shape::Gemm0BlockWarps::at(number<1>{}) &&
               Shape::Gemm1BlockWarps::at(number<2>{}) == Shape::Gemm0BlockWarps::at(number<2>{}) &&
               std::is_same_v<remove_cvref_t<typename Problem::SaccDataType>, float> &&
               std::is_same_v<remove_cvref_t<typename Problem::OaccDataType>, float> &&
               Shape::kK0 == Geometry::kHeadDimQK && Shape::kSubQKHeaddim == Geometry::kHeadDimQK &&
               Shape::IsVLayoutRowMajor && Shape::NumWarps == Geometry::kWaves && is_wave32;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK0() { return kLdsOffsetK0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK1() { return kLdsOffsetK1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV0() { return kLdsOffsetV0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV1() { return kLdsOffsetV1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsArenaSize() { return kLdsArenaSize; }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_fp8_sched requires a full-head K TDM load");
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(kLdsRows % warp_num == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, kLdsRows / warp_num>, sequence<kKDramBoxWidth>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackK<Problem>();
        return make_naive_tensor_descriptor(make_tuple(number<kLdsRows>{}, number<kKValidWidth>{}),
                                            make_tuple(number<kKPhysicalStride>{}, number<1>{}),
                                            number<kKPack>{},
                                            number<1>{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        static_assert(LoadOnce, "qr_tdm_fp8_sched requires a full-head K LDS descriptor");
        return MakeKLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackV<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<kVLogicalWidth>{}),
            make_tuple(number<kVPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        static_assert(!Xor, "the V TDM writer requires an affine V LDS descriptor");
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        using BlockGemm = remove_cvref_t<decltype(BasePolicy::template GetPVBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t kMWarp        = Geometry::kWaves;
        constexpr index_t kNWarp        = 1;
        constexpr index_t kNIterPerWarp = Geometry::kPvNIter;
        constexpr index_t kKIterPerWarp = Geometry::kPvStageKIter;
        static_assert(kNIterPerWarp == kVLogicalWidth / (kNWarp * WarpGemm::kN) &&
                      kKIterPerWarp == Geometry::kPvStageK / WarpGemm::kK);

        constexpr auto outer_encoding = tile_distribution_encoding<
            sequence<kMWarp>,
            tuple<sequence<kNIterPerWarp, kNWarp>, sequence<kKIterPerWarp>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<2, 1>,
            sequence<0, 0>>{};
        constexpr auto block_encoding = detail::make_embed_tile_distribution_encoding(
            outer_encoding, typename WarpGemm::BWarpDstrEncoding{});
        using ReadEncoding =
            typename InputTileDistributionTraits<decltype(block_encoding),
                                                 typename Problem::VDataType>::TransposedDstrEncode;

        return make_static_tile_distribution(ReadEncoding{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_fp8_sched requires a full-head K LDS allocation");
        return kKFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        return kVFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        return GetLdsArenaSize();
    }

    // Without kPadQRows, Q reuses the qr_tdm LDS layout, which pads only 16-bit types.
    template <typename Problem>
    using QPadding = std::conditional_t<kPadQRows,
                                        KPadding,
                                        typename BasePolicy::template LdsPaddingConfigQ<Problem>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        static_assert(Problem::BlockFmhaShape::kSubQKHeaddim == kKValidWidth);
        constexpr auto desc = [] {
            if constexpr(kPadQRows)
                return detail::make_qr_tdm_row_major_lds_descriptor<typename Problem::QDataType,
                                                                    Problem::BlockFmhaShape::kM0,
                                                                    kKValidWidth,
                                                                    QPadding<Problem>,
                                                                    detail::kQrTdmLdsAccessBytes>();
            else
                return BasePolicy::template MakeQLdsBlockDescriptor<Problem>();
        }();
        static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                          kV1FootprintBytes,
                      "Q LDS descriptor overflows the V1 buffer");
        return desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        using Padding = detail::EncodedTdmPadding<QPadding<Problem>>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        using Padding = detail::EncodedTdmPadding<KPadding>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        using Padding = detail::EncodedTdmPadding<VPadding>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }
};

// Every quantization mode fits 512 VGPRs without mainloop spills, so all run two per CU.
inline constexpr index_t kFmhaTdmSchedFp8BlockPerCu = 2;
// The elementwise bias tile does not fit two workgroups' VGPRs, so it runs one per CU.
inline constexpr index_t kFmhaTdmSchedFp8BiasBlockPerCu = 1;
// D256 exceeds half of a CU's VGPRs and LDS, so it runs one workgroup per CU.
inline constexpr index_t kFmhaTdmSchedFp8D256BlockPerCu = 1;
// D192 MX scales (six bytes per Q/K row) spill at two per CU, so it runs one per CU.
inline constexpr index_t kFmhaTdmSchedFp8D192MxBlockPerCu = 1;
// D192 merged-GQA decode runs one per CU where two overflow the LDS (FP32 O) or VGPRs (sink).
inline constexpr index_t kFmhaTdmSchedFp8D192MergedBlockPerCu = 1;
template <typename Problem>
inline constexpr bool kFmhaTdmSchedFp8D192MergedOnePerCu =
    Problem::BlockFmhaShape::kQKHeaddim == 192 && Problem::kMergeNumHeadGroupsSeqLenQ &&
    (((Problem::QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD ||
       Problem::QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE) &&
      sizeof(typename Problem::ODataType) == 4) ||
     (Problem::kHasSink && Problem::FmhaMask::IsMasking));
// Target 2 per SIMD; D128/D192 are LDS-bound to 1 until the one-wavefront LDS footprint shrinks.
template <index_t HeadDim>
inline constexpr index_t kFmhaTdmSchedFp8OneWaveBlockPerCu = HeadDim == 256 ? 1 : 2;

// Dense four-wavefront D256 splits each K/V tile across LDS segments by wavefront parity.
template <typename Problem>
inline constexpr bool kFmhaTdmSchedFp8LdsSegLayout =
    Problem::BlockFmhaShape::kQKHeaddim == 256 && FmhaTdmSchedGeometryFor<Problem>::kWaves == 4 &&
    (Problem::QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ||
     Problem::QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR ||
     Problem::QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD) &&
    !Problem::kMergeNumHeadGroupsSeqLenQ && !Problem::kHasSink && !Problem::FmhaMask::IsMasking &&
    Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;

template <typename Problem>
using FmhaTdmSchedFp8TuningFor = FmhaTdmSchedDerivedTuning<
    FmhaTdmSchedGeometryFor<Problem>,
    Problem::BlockFmhaShape::kM0 == 32
        ? kFmhaTdmSchedFp8OneWaveBlockPerCu<Problem::BlockFmhaShape::kQKHeaddim>
    : Problem::BlockFmhaShape::kQKHeaddim == 256 ? kFmhaTdmSchedFp8D256BlockPerCu
    : Problem::BlockFmhaShape::kQKHeaddim == 192 &&
            Problem::QScaleEnum == BlockAttentionQuantScaleEnum::MX
        ? kFmhaTdmSchedFp8D192MxBlockPerCu
    : kFmhaTdmSchedFp8D192MergedOnePerCu<Problem>          ? kFmhaTdmSchedFp8D192MergedBlockPerCu
    : Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? kFmhaTdmSchedFp8BlockPerCu
                                                           : kFmhaTdmSchedFp8BiasBlockPerCu>;

template <typename Problem>
using FmhaTdmSchedFp8PolicyFor = BlockFmhaPipelineQRKSVSTdmSchedFp8Policy<
    FmhaTdmSchedGeometryFor<Problem>,
    std::conditional_t<kFmhaTdmSchedFp8LdsSegLayout<Problem>,
                       FmhaTdmSchedFp8LdsSegTuning<FmhaTdmSchedFp8TuningFor<Problem>>,
                       FmhaTdmSchedFp8TuningFor<Problem>>>;

} // namespace ck_tile
