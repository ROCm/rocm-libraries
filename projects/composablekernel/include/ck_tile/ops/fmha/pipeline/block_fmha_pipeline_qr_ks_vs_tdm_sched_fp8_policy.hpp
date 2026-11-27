// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_common_policy.hpp"
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
struct BlockFmhaPipelineQRKSVSTdmSchedFp8Policy
    : BlockFmhaPipelineQRKSVSTdmSchedCommonPolicy<
          BlockFmhaPipelineQRKSVSTdmSchedFp8Policy<Geometry_, Tuning_, TileSchedule_>,
          Geometry_,
          Tuning_,
          TileSchedule_,
          FmhaTdmSchedFp8Layout<Geometry_>>
{
    using Base =
        BlockFmhaPipelineQRKSVSTdmSchedCommonPolicy<BlockFmhaPipelineQRKSVSTdmSchedFp8Policy,
                                                    Geometry_,
                                                    Tuning_,
                                                    TileSchedule_,
                                                    FmhaTdmSchedFp8Layout<Geometry_>>;
    using BasePolicy      = typename Base::BasePolicy;
    using Geometry        = Geometry_;
    using Tuning          = Tuning_;
    using Schedule        = TileSchedule_;
    using Layout          = typename Base::Layout;
    using OutputFragments = typename Base::OutputFragments;
    using NoScale         = typename Base::NoScale;
    using Base::kKFootprintBytes;
    using Base::kKPhysicalStride;
    using Base::kKValidWidth;
    using Base::kLdsRows;
    using Base::kVFootprintBytes;
    using Base::kVLogicalWidth;
    using Base::kVPhysicalStride;
    using KLoad = FmhaTdmSchedFp8Load<Geometry::kKSuLoadCount, Geometry::kKLdsReadBytes>;
    using VLoad = FmhaTdmSchedFp8TransposeLoad<Geometry::kVStageLoadCount>;

    using KPadding =
        std::conditional_t<Layout::kKPadByTdm,
                           detail::LdsPaddingConfig<true, kKValidWidth, Layout::kRowPadBytes>,
                           detail::LdsPaddingConfig<false, 0, 0>>;
    // Width of the K TDM box: the row, or the padded row whose tail reads zeros past hdim_q.
    static constexpr index_t kKDramBoxWidth = Layout::kKPadByTdm ? kKValidWidth : kKPhysicalStride;
    // LDS K rows hold hdim_q elements at the padded stride.
    static constexpr index_t kKLdsWidth = kKValidWidth;
    // D128/D256 Q rows take the K row padding so per-lane row reads spread across LDS banks.
    static constexpr bool kPadQRows = kKValidWidth >= 128 && Layout::kKPadByTdm;
    using VPadding = detail::LdsPaddingConfig<true, kVLogicalWidth, Layout::kRowPadBytes>;
    static_assert(sizeof(typename Geometry::KDataType) == 1 &&
                      sizeof(typename Geometry::VDataType) == 1 &&
                      kKPhysicalStride == kKValidWidth + Layout::kRowPadBytes &&
                      kVPhysicalStride == kVLogicalWidth + Layout::kRowPadBytes,
                  "TDM row padding must match the LDS read stride");
    // The output rescale is fused into softmax part 2; taking the early slot skips the post pass.
    static constexpr bool kEarlyOutputRescale = true;
    static constexpr bool kLeadFirstLoad      = false;

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitQkStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kQkStages);
        s_wait_dscnt<kFmhaTdmSchedFp8DsWait<Base::template GetQkTailDsCount<Stage>()>>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitPvStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        s_wait_dscnt<kFmhaTdmSchedFp8DsWait<Base::template GetPvTailDsCount<Stage>()>>();
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

    static_assert(kFmhaTdmSchedFp8GeometryOk<Geometry>);
    using ScoreSoftmax = typename Base::ScoreSoftmax;
    using Softmax      = FmhaTdmSchedFp8Softmax<Geometry, OutputFragments>;
    // One PV stage covers the KV tile, so every PV row is an output fragment.
    static_assert(Geometry::kPvWmmasPerStage == OutputFragments::kNumFragments);

    static_assert(kLdsOffsetK0 % 16 == 0 && kLdsOffsetK1 % 16 == 0 && kLdsOffsetV0 % 16 == 0 &&
                  kLdsOffsetV1 % 16 == 0);

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

    // K128 WMMA from two adjacent K64 steps, low step in the low eight dwords.
    template <typename CWarpTensor, typename AWarpTensor, typename BWarpTensor>
    CK_TILE_DEVICE static void RunPairedScaledWmma(CWarpTensor& c_warp_tensor,
                                                   const AWarpTensor& a_lo,
                                                   const AWarpTensor& a_hi,
                                                   const BWarpTensor& b_lo,
                                                   const BWarpTensor& b_hi,
                                                   int32_t a_scale,
                                                   int32_t b_scale)
    {
        using QDataType     = typename Geometry::QDataType;
        using PairedGemm    = WarpGemmDispatcher<QDataType, QDataType, float, 16, 16, 128, true>;
        using Half          = int32_t __attribute__((ext_vector_type(8)));
        constexpr auto zero = number<0>{};
        constexpr auto one  = number<1>{};
        static_assert(sizeof(typename AWarpTensor::DataType) == 1 &&
                      AWarpTensor::get_thread_buffer_size() == sizeof(Half) &&
                      BWarpTensor::get_thread_buffer_size() == sizeof(Half));

        typename PairedGemm::AWarpTensor a;
        typename PairedGemm::BWarpTensor b;
        typename PairedGemm::CWarpTensor c;
        a.get_thread_buffer().template set_as<Half>(zero, bit_cast<Half>(a_lo.get_thread_buffer()));
        a.get_thread_buffer().template set_as<Half>(one, bit_cast<Half>(a_hi.get_thread_buffer()));
        b.get_thread_buffer().template set_as<Half>(zero, bit_cast<Half>(b_lo.get_thread_buffer()));
        b.get_thread_buffer().template set_as<Half>(one, bit_cast<Half>(b_hi.get_thread_buffer()));
        c.get_thread_buffer() = c_warp_tensor.get_thread_buffer();
        PairedGemm{}(c, a, b, a_scale, b_scale);
        c_warp_tensor.get_thread_buffer() = c.get_thread_buffer();
    }

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
        typename Base::SchedulePointEmitter emit_point;
        auto emit_wmma = [&](auto, auto wmma) {
            Base::template RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
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
        typename Base::SchedulePointEmitter emit_point;
        auto emit_wmma = [&](auto, auto wmma) {
            Base::template RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
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
        typename Base::SchedulePointEmitter emit_point;
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
        return (Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                (Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS &&
                 Problem::QScaleEnum != BlockAttentionQuantScaleEnum::MX)) &&
               Base::template IsSupportedSchedProblem<Problem>() &&
               Shape::kK0 == Geometry::kHeadDimQK;
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
                      "FP8 qr_tdm_sched policy geometry does not match the problem");
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
};

// Dense four-wavefront D256 splits each K/V tile across LDS segments by wavefront parity.
template <typename Problem>
inline constexpr bool kFmhaTdmSchedFp8LdsSegLayout =
    Problem::BlockFmhaShape::kQKHeaddim == 256 && FmhaTdmSchedGeometryFor<Problem>::kWaves == 4 &&
    (Problem::QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ||
     Problem::QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR ||
     Problem::QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD) &&
    !Problem::kMergeNumHeadGroupsSeqLenQ && !Problem::kHasSink && !Problem::FmhaMask::IsMasking &&
    Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;

// Workgroups per CU come from the generated traits (the codegen holds the occupancy rules).
template <typename Problem>
using FmhaTdmSchedFp8TuningFor =
    FmhaTdmSchedDerivedTuning<FmhaTdmSchedGeometryFor<Problem>, Problem::kBlockPerCu>;

template <typename Problem>
using FmhaTdmSchedFp8PolicyFor = BlockFmhaPipelineQRKSVSTdmSchedFp8Policy<
    FmhaTdmSchedGeometryFor<Problem>,
    std::conditional_t<kFmhaTdmSchedFp8LdsSegLayout<Problem>,
                       FmhaTdmSchedFp8LdsSegTuning<FmhaTdmSchedFp8TuningFor<Problem>>,
                       FmhaTdmSchedFp8TuningFor<Problem>>>;

} // namespace ck_tile
