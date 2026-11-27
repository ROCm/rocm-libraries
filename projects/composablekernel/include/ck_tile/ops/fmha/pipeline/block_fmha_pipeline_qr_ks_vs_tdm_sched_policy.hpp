// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_load.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_common_policy.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_prepared_k_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_prepared_v_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_affine_k_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_affine_v_read.hpp"

namespace ck_tile {

template <typename Geometry_,
          typename Tuning_,
          typename TileSchedule_ = FmhaTdmSchedDefaultScheduleFor<Geometry_>>
struct BlockFmhaPipelineQRKSVSTdmSchedPolicy
    : BlockFmhaPipelineQRKSVSTdmSchedCommonPolicy<
          BlockFmhaPipelineQRKSVSTdmSchedPolicy<Geometry_, Tuning_, TileSchedule_>,
          Geometry_,
          Tuning_,
          TileSchedule_,
          FmhaTdmSchedLayout<Geometry_>>
{
    using Base = BlockFmhaPipelineQRKSVSTdmSchedCommonPolicy<BlockFmhaPipelineQRKSVSTdmSchedPolicy,
                                                             Geometry_,
                                                             Tuning_,
                                                             TileSchedule_,
                                                             FmhaTdmSchedLayout<Geometry_>>;
    using BasePolicy      = typename Base::BasePolicy;
    using Geometry        = Geometry_;
    using Tuning          = Tuning_;
    using Schedule        = TileSchedule_;
    using Layout          = typename Base::Layout;
    using OutputFragments = typename Base::OutputFragments;
    using ScoreSoftmax    = typename Base::ScoreSoftmax;
    using ScoreMapping    = typename Base::ScoreMapping;
    using Base::kKFootprintBytes;
    using Base::kKPhysicalStride;
    using Base::kLdsRows;
    using Base::kVFootprintBytes;
    using Base::kVPhysicalStride;
    using Base::kVPrefetchTensorCount;
    using KLoad = FmhaTdmSchedLoad<Geometry::kKSuLoadCount>;
    using VLoad = FmhaTdmSchedTransposeLoad<Geometry::kVStageLoadCount>;
    static_assert(Schedule::kNumQkStages == Geometry::kQkStages &&
                  Schedule::kNumPvStages == Geometry::kPvStages &&
                  Schedule::kQkWmmasPerStage == Geometry::kQkWmmasPerStage &&
                  Schedule::kPvWmmasPerStage == Geometry::kPvWmmasPerStage);

    static constexpr index_t kVPadAmount   = Layout::kVPadAmount;
    static constexpr index_t kVPadInterval = Layout::kVPadInterval;
    // K rows are written whole by TDM (no TDM padding), so the K box and LDS rows span the stride.
    static constexpr index_t kKDramBoxWidth = kKPhysicalStride;
    static constexpr index_t kKLdsWidth     = kKPhysicalStride;
    // V rows get kVPadAmount-encoded padding at each row boundary.
    static constexpr index_t kVRowBytes = Geometry::kDv * sizeof(typename Geometry::VDataType);
    using KPadding                      = detail::LdsPaddingConfig<false, 0, 0>;
    using VPadding = detail::LdsPaddingConfig<true, kVRowBytes, (kVPadAmount + 1) * 4>;
    static_assert(detail::EncodedTdmPadding<VPadding>::kPadInterval == kVPadInterval &&
                  detail::EncodedTdmPadding<VPadding>::kPadAmount == kVPadAmount);
    // Named geometry knobs. Values frozen to the current D128/D192 formulas.
    static constexpr bool kEarlyOutputRescale = Geometry::kHeadDimQK == 128;
    static constexpr bool kLeadFirstLoad      = Geometry::kHeadDimQK == 128;
    static constexpr bool kUseUnsignedCausalMask =
        Geometry::kHeadDimQK == 128 || Geometry::kHeadDimQK == 192;

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitQkStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kQkStages);
        s_wait_dscnt<Base::template GetQkTailDsCount<Stage>()>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitPvStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        s_wait_dscnt<Base::template GetPvTailDsCount<Stage>()>();
    }

    static constexpr index_t kLdsOffsetK0  = 0;
    static constexpr index_t kLdsOffsetK1  = kLdsOffsetK0 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV0  = kLdsOffsetK1 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV1  = kLdsOffsetV0 + kVFootprintBytes;
    static constexpr index_t kLdsArenaSize = kLdsOffsetV1 + kVFootprintBytes;

    static_assert(FmhaTdmSchedSupportedGeometry<Geometry>::value);

    // Q is used only during the prologue. Its descriptor may span both K buffers
    // when the KV tile is 64 rows; the LDS barrier precedes the K loads that overwrite it.
    static_assert(Geometry::kM * Geometry::kHeadDimQK * sizeof(typename Geometry::QDataType) <=
                  2 * kKFootprintBytes);
    static_assert(kLdsOffsetK0 % 16 == 0 && kLdsOffsetK1 % 16 == 0 && kLdsOffsetV0 % 16 == 0 &&
                  kLdsOffsetV1 % 16 == 0);

    template <index_t Stage,
              typename Mode,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow>
    CK_TILE_DEVICE static void
    RunQkScheduledStage(const BlockGemm&,
                        CBlockTensor& c_block_tensor,
                        const ABlockTensor& a_block_tensor,
                        const BBlockTensor& b_block_tensor,
                        NextBBlockTensor& next_b_block_tensor,
                        const BTileWindow& b_lds_window,
                        const FmhaTdmPreparedKRead::Address* prepared_stage0 = nullptr)
    {
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumQkStages);
        static_assert((Stage < SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount) ||
                      (Stage == SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount));
        constexpr bool kUseAffineKRead = Stage < Geometry::kQkStages - 1;

        const auto affine_k0 = [&]() {
            if constexpr(kUseAffineKRead)
            {
                if constexpr(Stage == 0)
                {
                    if(prepared_stage0 != nullptr)
                        return *prepared_stage0;
                }
                return FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            }
            else
                return FmhaTdmPreparedKRead::Address{};
        }();
        const auto affine_v0 = [&]() {
            if constexpr(Stage == Geometry::kQkStages - 1)
                return FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            else
                return FmhaTdmPreparedKRead::Address{};
        }();
        constexpr bool kLeadFirstLoadThisStage = Mode::SelectedPolicy::kLeadFirstLoad &&
                                                 (Stage == 0 || Stage == Geometry::kQkStages - 1);
        if constexpr(kLeadFirstLoadThisStage)
        {
            SelectedSchedule::template RunQkReadPrelude<Mode, Stage>([&](auto event) {
                static_assert(decltype(event)::value == 0);
                if constexpr(Stage == 0)
                    FmhaTdmAffineKRead::Load<Stage, 0, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
                else
                    FmhaTdmAffineVRead::Load<0, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_v0);
            });
            __builtin_amdgcn_sched_barrier(0);
        }
        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
            {
                static_assert(Stage < Geometry::kQkStages - 1);
                if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
                {
                    // Access0 was issued before WMMA0; do not issue it twice.
                }
                else
                    FmhaTdmAffineKRead::Load<Stage, decltype(access)::value, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
            }
            else if constexpr(decltype(kind)::value == Kind::VRead)
            {
                static_assert(Stage == Geometry::kQkStages - 1);
                if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
                {
                    // Access0 was issued before WMMA0; do not issue it twice.
                }
                else
                    FmhaTdmAffineVRead::Load<decltype(access)::value, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_v0);
            }
        };
        typename Base::SchedulePointEmitter emit_point;

        auto emit_wmma = [&](auto, auto wmma) {
            Base::template RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
                c_block_tensor, a_block_tensor, b_block_tensor);
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        WaitQkStageTail<Stage>();
    }

    template <index_t Stage,
              typename Mode,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename WmmaEmitter,
              typename RescaleEmitter>
    CK_TILE_DEVICE static void RunPvScheduledStageWithWmma(NextBBlockTensor& next_b_block_tensor,
                                                           const BTileWindow& b_lds_window,
                                                           WmmaEmitter& emit_wmma,
                                                           RescaleEmitter& emit_rescale)
    {
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumPvStages);
        static_assert((Stage < SelectedSchedule::kNumPvStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount) ||
                      (Stage == SelectedSchedule::kNumPvStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount));

        FmhaTdmPreparedKRead::Address affine_k0{};
        FmhaTdmPreparedKRead::Address prepared_v1{};
        constexpr bool kLeadFirstLoadThisStage = Mode::SelectedPolicy::kLeadFirstLoad &&
                                                 (Stage == 0 || Stage == Geometry::kPvStages - 1);
        if constexpr(kLeadFirstLoadThisStage)
        {
            SelectedSchedule::template RunPvReadPrelude<Mode, Stage>([&](auto event) {
                static_assert(decltype(event)::value == 0);
                if constexpr(Stage == 0)
                {
                    prepared_v1 = FmhaTdmPreparedKRead::Prepare<1>(b_lds_window);
                    __builtin_amdgcn_sched_barrier(0);
                    VLoad::template LoadAccess<0>(next_b_block_tensor, b_lds_window);
                }
                else
                {
                    affine_k0 = FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
                    FmhaTdmAffineKRead::Load<Stage, 0, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
                }
            });
            __builtin_amdgcn_sched_barrier(0);
        }
        auto emit_prepared_wmma = [&](auto stage, auto wmma) {
            if constexpr(!kLeadFirstLoadThisStage && Stage == 0 && decltype(wmma)::value == 0)
            {
                prepared_v1 = FmhaTdmPreparedKRead::Prepare<1>(b_lds_window);
                __builtin_amdgcn_sched_barrier(0);
            }
            if constexpr(!kLeadFirstLoadThisStage && Stage == Geometry::kPvStages - 1 &&
                         decltype(wmma)::value == 0)
            {
                affine_k0 = FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            }
            emit_wmma(stage, wmma);
        };
        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
            {
                static_assert(Stage == Geometry::kPvStages - 1);
                if constexpr(Geometry::kN == 64)
                    KLoad::template LoadInstruction<decltype(access)::value>(next_b_block_tensor,
                                                                             b_lds_window);
                else
                    FmhaTdmAffineKRead::Load<Stage, decltype(access)::value, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
            }
            else if constexpr(decltype(kind)::value == Kind::VRead)
            {
                static_assert(Stage < Geometry::kPvStages - 1);
                VLoad::template LoadAccess<decltype(access)::value>(next_b_block_tensor,
                                                                    b_lds_window);
            }
        };
        typename Base::SchedulePointEmitter emit_point;

        auto emit_prepared_token = [&](auto kind, auto access) {
            if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
            {
                // Access0 was issued before WMMA0; do not issue it twice.
            }
            else if constexpr(Stage == 0 && decltype(kind)::value == Kind::VRead &&
                              decltype(access)::value == 1)
                FmhaTdmPreparedVRead::Load<1>(next_b_block_tensor, b_lds_window, prepared_v1);
            else
                emit_token(kind, access);
        };

        Executor::template ExecutePvStage<Stage>(
            emit_prepared_wmma, emit_prepared_token, emit_point, emit_rescale);
        WaitPvStageTail<Stage>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCountdownLoop()
    {
        return Geometry::kHeadDimQK == 128 && !Problem::FmhaMask::IsMasking && !Problem::kHasSink &&
               Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool DeferTensorReady()
    {
        // Defer the V tensor wait only for the supported geometry and problem cases.
        return (Geometry::kHeadDimQK == 128 ||
                (std::is_same_v<Geometry, FmhaTdmSchedBf16D192N64Geometry> &&
                 !Problem::FmhaMask::IsMasking)) &&
               !Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{}) == 1;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCompilerMax()
    {
        // Let the allocator place score tuples above the inline-asm low-VGPR range.
        // D64/D128 and D192 dense batch use compiler-scheduled maximum chains.
        return Geometry::kHeadDimQK <= 128 ||
               (Geometry::kHeadDimQK == 192 && !Problem::kIsGroupMode &&
                !Problem::FmhaMask::IsMasking);
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(
        ScoreTensor& score, RowTensor& row_max, float log2e_scale, float& delta_m0, float& delta_m1)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        const float old_max_m0 = row_max.get_thread_buffer()[number<0>{}];
        const float old_max_m1 = row_max.get_thread_buffer()[number<1>{}];

        constexpr bool kUseCompilerMax = UseCompilerMax<Problem>();
        const auto p0_m0 = Softmax::template RunPart0<0, ScoreTensor, kUseCompilerMax>(
            score, old_max_m0, log2e_scale);
        const auto p0_m1 = Softmax::template RunPart0<1, ScoreTensor, kUseCompilerMax>(
            score, old_max_m0, log2e_scale);
        const auto p0_m2 = Softmax::template RunPart0<2, ScoreTensor, kUseCompilerMax>(
            score, old_max_m1, log2e_scale);
        const auto p0_m3 = Softmax::template RunPart0<3, ScoreTensor, kUseCompilerMax>(
            score, old_max_m1, log2e_scale);

        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking;
        const auto p1_m0            = Softmax::template RunPart1<kValidateMax>(
            p0_m0.local_max, p0_m1.local_max, p0_m0.old_max_log2e, log2e_scale);
        const auto p1_m1 = Softmax::template RunPart1<kValidateMax>(
            p0_m2.local_max, p0_m3.local_max, p0_m2.old_max_log2e, log2e_scale);

        row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
        row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
        delta_m0                                 = p1_m0.delta;
        delta_m1                                 = p1_m1.delta;
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart2AndGetScale(ScoreTensor& score,
                                                               const RowTensor& row_max,
                                                               RowTensor& row_sum,
                                                               float log2e_scale,
                                                               float delta_m0,
                                                               float delta_m1,
                                                               float& output_scale_m0,
                                                               float& output_scale_m1)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking;

        const auto validated_max = [](float value) {
            if constexpr(kValidateMax)
            {
                return value <= -numeric<float>::infinity() ? 0.0f : value;
            }
            else
            {
                return value;
            }
        };

        const float row_max_m0 = row_max.get_thread_buffer()[number<0>{}];
        const float row_max_m1 = row_max.get_thread_buffer()[number<1>{}];
        const float old_sum_m0 = row_sum.get_thread_buffer()[number<0>{}];
        const float old_sum_m1 = row_sum.get_thread_buffer()[number<1>{}];

        const float sum_m0 =
            Softmax::template RunPart2LocalSum<0>(score, validated_max(row_max_m0), log2e_scale);
        const float sum_m1 =
            Softmax::template RunPart2LocalSum<1>(score, validated_max(row_max_m0), log2e_scale);
        const float sum_m2 =
            Softmax::template RunPart2LocalSum<2>(score, validated_max(row_max_m1), log2e_scale);
        const float sum_m3 =
            Softmax::template RunPart2LocalSum<3>(score, validated_max(row_max_m1), log2e_scale);

        row_sum.get_thread_buffer()[number<0>{}] =
            Softmax::UpdateRowSum(old_sum_m0, Softmax::MergeRowSum(sum_m0, sum_m1), delta_m0);
        row_sum.get_thread_buffer()[number<1>{}] =
            Softmax::UpdateRowSum(old_sum_m1, Softmax::MergeRowSum(sum_m2, sum_m3), delta_m1);

        output_scale_m0 = Softmax::Exp2(delta_m0);
        output_scale_m1 = Softmax::Exp2(delta_m1);
    }

    template <index_t Ordinal, typename Fragments>
    CK_TILE_DEVICE static void
    RescaleOutputFragment(Fragments& output, float scale_m0, float scale_m1)
    {
        static_assert(Ordinal >= 0 && Ordinal < OutputFragments::kNumFragments);
        constexpr index_t d_msb = Ordinal / OutputFragments::kNumN;
        const float scale       = d_msb < 2 ? scale_m0 : scale_m1;
        auto& fragment          = output.at(number<Ordinal>{});
        static_for<0, OutputFragments::kElementsPerFragment, 1>{}(
            [&](auto element) { fragment[decltype(element)::value] *= scale; });
    }

    template <typename Problem,
              bool WaitTensorAfterMax = false,
              typename ScoreTensor,
              typename RowTensor,
              typename Fragments>
    CK_TILE_DEVICE static void RunSplitSoftmaxFragments(ScoreTensor& score,
                                                        RowTensor& row_max,
                                                        RowTensor& row_sum,
                                                        Fragments& output,
                                                        float log2e_scale)
    {
        float delta_m0;
        float delta_m1;
        RunSplitSoftmaxPart01<Problem>(score, row_max, log2e_scale, delta_m0, delta_m1);

        if constexpr(WaitTensorAfterMax)
            s_wait_tensorcnt_barrier<kVPrefetchTensorCount>();

        const auto rescale_output = [&](float scale_m0, float scale_m1) {
            static_for<0, OutputFragments::kNumFragments, 1>{}([&](auto ordinal) {
                constexpr index_t d_msb = decltype(ordinal)::value / OutputFragments::kNumN;
                const float scale       = d_msb < 2 ? scale_m0 : scale_m1;
                auto& fragment          = output.at(ordinal);
                static_for<0, OutputFragments::kElementsPerFragment, 1>{}(
                    [&](auto element) { fragment[decltype(element)::value] *= scale; });
            });
        };
        if constexpr(kEarlyOutputRescale)
        {
            using Softmax = ScoreSoftmax;
            rescale_output(Softmax::Exp2(delta_m0), Softmax::Exp2(delta_m1));
        }

        float output_scale_m0;
        float output_scale_m1;
        RunSplitSoftmaxPart2AndGetScale<Problem>(score,
                                                 row_max,
                                                 row_sum,
                                                 log2e_scale,
                                                 delta_m0,
                                                 delta_m1,
                                                 output_scale_m0,
                                                 output_scale_m1);

        if constexpr(!kEarlyOutputRescale)
        {
            rescale_output(output_scale_m0, output_scale_m1);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedProblem()
    {
        using Shape = remove_cvref_t<typename Problem::BlockFmhaShape>;
        return Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               Base::template IsSupportedSchedProblem<Problem>() &&
               Shape::kK0 == Geometry::kQkWmmaK;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadBlockDescriptor()
    {
        return Base::template MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    static constexpr bool kUsePaddedQ = Geometry::kHeadDimQK <= 128 && Geometry::kM == 128;

    template <typename Problem>
    using QPadding = std::conditional_t<
        kUsePaddedQ<Problem>,
        detail::
            LdsPaddingConfig<true, Geometry::kHeadDimQK * sizeof(typename Geometry::QDataType), 16>,
        typename BasePolicy::template LdsPaddingConfigQ<Problem>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(), "V128 policy received an invalid problem");
        if constexpr(kUsePaddedQ<Problem>)
        {
            using Padding = QPadding<Problem>;
            constexpr auto desc =
                detail::make_qr_tdm_row_major_lds_descriptor<typename Problem::QDataType,
                                                             Geometry::kM,
                                                             Geometry::kHeadDimQK,
                                                             Padding,
                                                             detail::kQrTdmLdsAccessBytes>();
            static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                          2 * kKFootprintBytes);
            return desc;
        }
        else
        {
            constexpr auto desc = BasePolicy::template MakeQLdsBlockDescriptor<Problem>();
            static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                              2 * kKFootprintBytes,
                          "Q LDS descriptor overlaps the V buffer");
            return desc;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    }
};

// Workgroups per CU come from the generated traits (the codegen holds the occupancy rules).
template <typename Problem>
using FmhaTdmSchedPolicyFor = BlockFmhaPipelineQRKSVSTdmSchedPolicy<
    FmhaTdmSchedGeometryFor<Problem>,
    FmhaTdmSchedDerivedTuning<FmhaTdmSchedGeometryFor<Problem>, Problem::kBlockPerCu>>;

} // namespace ck_tile
