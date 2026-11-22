// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_load.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"
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
struct BlockFmhaPipelineQRKSVSTdmSchedPolicy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Geometry   = Geometry_;
    using Tuning     = Tuning_;
    using Schedule   = TileSchedule_;
    using Layout     = FmhaTdmSchedLayout<Geometry>;
    using KLoad      = FmhaTdmSchedLoad<Geometry::kKSuLoadCount>;
    using VLoad      = FmhaTdmSchedTransposeLoad<Geometry::kVStageLoadCount>;
    static_assert(Schedule::kNumQkStages == Geometry::kQkStages &&
                  Schedule::kNumPvStages == Geometry::kPvStages &&
                  Schedule::kQkWmmasPerStage == Geometry::kQkWmmasPerStage &&
                  Schedule::kPvWmmasPerStage == Geometry::kPvWmmasPerStage);
    static constexpr index_t kBlockPerCu = Tuning::kBlockPerCu;

    static constexpr index_t kLdsRows            = Geometry::kN;
    static constexpr index_t kKValidWidth        = Geometry::kHeadDimQK;
    static constexpr index_t kKPhysicalStride    = Layout::kKPhysicalStride;
    static constexpr index_t kVLogicalWidth      = Geometry::kDv;
    static constexpr index_t kVPhysicalStride    = Layout::kVPhysicalStride;
    static constexpr index_t kVPadAmount         = Layout::kVPadAmount;
    static constexpr index_t kVPadInterval       = Layout::kVPadInterval;
    static constexpr index_t kKFootprintElements = kLdsRows * kKPhysicalStride;
    static constexpr index_t kVFootprintElements = kLdsRows * kVPhysicalStride;
    static constexpr index_t kKFootprintBytes =
        kKFootprintElements * sizeof(typename Geometry::KDataType);
    static constexpr index_t kVFootprintBytes =
        kVFootprintElements * sizeof(typename Geometry::VDataType);
    // Named geometry knobs. Values frozen to the current D128/D192 formulas.
    static constexpr bool kEarlyOutputRescale      = Geometry::kHeadDimQK == 128;
    static constexpr bool kLeadFirstLoad           = Geometry::kHeadDimQK == 128;
    static constexpr index_t kKPrefetchTensorCount = Tuning::kKWait;
    static constexpr index_t kVPrefetchTensorCount = Tuning::kVWait;
    static constexpr bool kPrefetchTailDrain       = Tuning::kTailDrain;
    static_assert(kKPrefetchTensorCount == 0 || kKPrefetchTensorCount == 1);
    static_assert(kVPrefetchTensorCount == 0 || kVPrefetchTensorCount == 1);

    struct PreparedLdsReadContext
    {
        FmhaTdmPreparedKRead::Address qk_stage0{};
        FmhaTdmPreparedKRead::Address v0{};
        FmhaTdmPreparedKRead::Address pv_k0{};
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseEarlyLdsBases()
    {
        // The lead-first D128 placement uses its own reads. Masked D192 N128 retains the
        // ordinary path to limit register pressure; D64 already prepares its read-first bases.
        return !kLeadFirstLoad && Geometry::kHeadDimQK == 192 &&
               (Geometry::kN == 64 || !Problem::FmhaMask::IsMasking);
    }

    template <typename Problem, typename KTileView, typename VTileWindow>
    CK_TILE_DEVICE static void
    PrepareIterationLdsReads(PreparedLdsReadContext& prepared_context,
                             const KTileView& k_lds_read_view,
                             typename Geometry::KDataType* k_lds_write_ptr,
                             const VTileWindow& v_lds_read_window,
                             typename Geometry::KDataType* k_lds_read_ptr)
    {
        auto next_su_window = make_tile_window(
            k_lds_read_view,
            make_tuple(number<Geometry::kQkSuColumns>{}, number<Geometry::kHeadDimQK>{}),
            {Geometry::kQkSuColumns, 0},
            MakeKSuRegTileDistribution<Problem>());
        next_su_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
        prepared_context.qk_stage0 = FmhaTdmPreparedKRead::Prepare<0>(next_su_window);
        if constexpr(UseEarlyLdsBases<Problem>())
        {
            // Keep the tile's read bases in VGPRs before the WMMA stream. Otherwise a late
            // VALU address write can make its LDS-read dependency wait drain intervening WMMA.
            prepared_context.v0 = FmhaTdmPreparedKRead::Prepare<0>(v_lds_read_window);
            asm volatile("" : "+v"(prepared_context.qk_stage0.byte_address));
            asm volatile("" : "+v"(prepared_context.v0.byte_address));
            // N64's final PV K reads use the ordinary instruction path.
            if constexpr(Geometry::kN == 128)
            {
                auto pv_k_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<Geometry::kHeadDimQK>{}),
                    {0, 0},
                    MakeKSuRegTileDistribution<Problem>());
                pv_k_window.set_bottom_tensor_view_data_ptr(k_lds_read_ptr);
                prepared_context.pv_k0 = FmhaTdmPreparedKRead::Prepare<0>(pv_k_window);
                asm volatile("" : "+v"(prepared_context.pv_k0.byte_address));
            }
        }
    }

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
        s_wait_dscnt<GetQkTailDsCount<Stage>()>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitPvStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        s_wait_dscnt<GetPvTailDsCount<Stage>()>();
    }

    static constexpr index_t kLdsOffsetK0  = 0;
    static constexpr index_t kLdsOffsetK1  = kLdsOffsetK0 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV0  = kLdsOffsetK1 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV1  = kLdsOffsetV0 + kVFootprintBytes;
    static constexpr index_t kLdsArenaSize = kLdsOffsetV1 + kVFootprintBytes;

    using OutputFragments =
        FmhaTdmSchedOutputFragmentsFor<Geometry::kPvNIter / 2, Geometry::kPvMIter>;

    // The split-softmax score mapping and the PV output fragments are hand-written for one
    // decomposition; tie them to the derived geometry.
    static_assert(FmhaTdmSchedSupportedGeometry<Geometry>::value);
    using ScoreSoftmax = FmhaTdmSchedSplitSoftmaxFor<Geometry::kQkStages, Geometry::kQkMIter>;
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

    // Q is used only during the prologue. Its descriptor may span both K buffers
    // when the KV tile is 64 rows; the LDS barrier precedes the K loads that overwrite it.
    static_assert(Geometry::kM * Geometry::kHeadDimQK * sizeof(typename Geometry::QDataType) <=
                  2 * kKFootprintBytes);
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

    // WMMA-source keep-alive. Each scheduled stage issues the next stage's LDS reads right after
    // its WMMAs (FmhaTdmSchedWmmaFirstPlacement). A read may reuse the VGPRs of a source that the
    // last WMMAs have consumed; in expert scheduling mode the compiler then waits (s_wait_alu
    // va_vdst) until those WMMAs complete before the read is issued (clang 24 inserts these WAR
    // waits; clang 23 did not). Keeping the LDS-fed source (K in QK, V in PV) of the previous
    // kWmmaSourceKeepDepth WMMAs of the stage alive until the next WMMA makes the allocator give
    // the reads other VGPRs, so the reused registers belong to WMMAs that have long completed.
    // Depths as measured on gfx1250 (clang 23 and 24): 2 for D64 M64 tiles, 3 for D64 M128xN128,
    // D128 and D192 N64 tiles, none for D64 N64 tiles (no such waits to remove; the FP16 batch
    // tile would cross 256 VGPRs) and D192 N128 tiles (the FP16 tile would cross 512 VGPRs).
    static constexpr index_t kWmmaSourceKeepDepth =
        (Geometry::kHeadDimQK == 64 && Geometry::kN == 64) ||
                (Geometry::kHeadDimQK == 192 && Geometry::kN == 128)
            ? 0
        : (Geometry::kHeadDimQK == 64 && Geometry::kQkMIter == 1) ? 2
                                                                  : 3;

    // An empty use of a fragment's registers at this point, one 32-bit operand per dword: an
    // inline-asm VGPR operand must sit in the low VGPR range, and a single wide operand would also
    // need a contiguous aligned tuple there.
    template <typename Buffer>
    CK_TILE_DEVICE static void KeepFragmentAlive(const Buffer& buffer)
    {
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(sizeof(Buffer) % 4 == 0);
        constexpr index_t kDwords = sizeof(Buffer) / 4;
        using Dwords              = ext_vector_t<float, kDwords>;
        const Dwords dwords       = bit_cast<Dwords>(buffer);
#pragma unroll
        for(index_t i = 0; i < kDwords; ++i)
        {
            const float dword = dwords[i];
            asm volatile("" : : "v"(dword));
        }
#else
        (void)buffer;
#endif
    }

    // The B (K) slice that RunQkSuWmma<WmmaOrdinal> passes to the warp GEMM.
    template <index_t WmmaOrdinal, typename BlockGemm, typename BBlockTensor>
    CK_TILE_DEVICE static auto GetQkBSlice(const BBlockTensor& b_block_tensor)
    {
        using WarpGemm  = typename BlockGemm::WarpGemm;
        using BWarpDstr = typename WarpGemm::BWarpDstr;
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr index_t n_iter =
            WmmaOrdinal / (BlockGemm::KIterPerWarp * BlockGemm::MIterPerWarp);
        constexpr index_t k_iter =
            (WmmaOrdinal / BlockGemm::MIterPerWarp) % BlockGemm::KIterPerWarp;
        return b_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<n_iter, k_iter>{}, b_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
    }

    template <index_t WmmaOrdinal,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor>
    CK_TILE_DEVICE static void RunQkSuWmma(CBlockTensor& c_block_tensor,
                                           const ABlockTensor& a_block_tensor,
                                           const BBlockTensor& b_block_tensor)
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

        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        c_block_tensor.set_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    }

    template <index_t Stage,
              typename Mode,
              typename Problem,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow>
    CK_TILE_DEVICE static void RunQkScheduledStage(const BlockGemm&,
                                                   CBlockTensor& c_block_tensor,
                                                   const ABlockTensor& a_block_tensor,
                                                   const BBlockTensor& b_block_tensor,
                                                   NextBBlockTensor& next_b_block_tensor,
                                                   const BTileWindow& b_lds_window,
                                                   const PreparedLdsReadContext& prepared_context)
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
        constexpr bool kUsePreparedTailBase =
            Stage == Geometry::kQkStages - 1 &&
            Mode::SelectedPolicy::template UseEarlyLdsBases<Problem>();

        const auto affine_k0 = [&]() {
            if constexpr(kUseAffineKRead)
            {
                if constexpr(Stage == 0)
                    return prepared_context.qk_stage0;
                else
                    return FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            }
            else
                return FmhaTdmPreparedKRead::Address{};
        }();
        const auto affine_v0 = [&]() {
            if constexpr(kUsePreparedTailBase)
                return prepared_context.v0;
            else if constexpr(Stage == Geometry::kQkStages - 1)
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
        auto emit_point = [](auto, auto, auto point) {
            if constexpr(decltype(point)::value == FmhaTdmSchedSchedulePoint::AfterTokens)
                __builtin_amdgcn_sched_barrier(0);
            else
                __builtin_amdgcn_sched_barrier(0x0002 | 0x0400);
        };

        // K slices of the last kWmmaSourceKeepDepth WMMAs of this stage (keep_b the newest).
        using KeepSlice = remove_cvref_t<decltype(GetQkBSlice<0, BlockGemm>(b_block_tensor))>;
        [[maybe_unused]] KeepSlice keep_b;
        [[maybe_unused]] KeepSlice keep_b2;
        [[maybe_unused]] KeepSlice keep_b3;
        auto emit_wmma = [&](auto, auto wmma) {
            constexpr index_t w = decltype(wmma)::value;
            if constexpr(kWmmaSourceKeepDepth >= 1 && w > 0)
                KeepFragmentAlive(keep_b);
            if constexpr(kWmmaSourceKeepDepth >= 2 && w > 1)
                KeepFragmentAlive(keep_b2);
            if constexpr(kWmmaSourceKeepDepth >= 3 && w > 2)
                KeepFragmentAlive(keep_b3);
            RunQkSuWmma<w, BlockGemm>(c_block_tensor, a_block_tensor, b_block_tensor);
            if constexpr(kWmmaSourceKeepDepth >= 3 && w > 1)
                keep_b3 = keep_b2;
            if constexpr(kWmmaSourceKeepDepth >= 2 && w > 0)
                keep_b2 = keep_b;
            keep_b = GetQkBSlice<w, BlockGemm>(b_block_tensor);
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        constexpr index_t kQkW = SelectedSchedule::kQkWmmasPerStage;
        if constexpr(kWmmaSourceKeepDepth >= 1)
            KeepFragmentAlive(keep_b);
        if constexpr(kWmmaSourceKeepDepth >= 2 && kQkW > 1)
            KeepFragmentAlive(keep_b2);
        if constexpr(kWmmaSourceKeepDepth >= 3 && kQkW > 2)
            KeepFragmentAlive(keep_b3);
        WaitQkStageTail<Stage>();
    }

    template <index_t Stage,
              typename Mode,
              typename Problem,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename WmmaEmitter,
              typename RescaleEmitter>
    CK_TILE_DEVICE static void
    RunPvScheduledStageWithWmma(NextBBlockTensor& next_b_block_tensor,
                                const BTileWindow& b_lds_window,
                                WmmaEmitter& emit_wmma,
                                RescaleEmitter& emit_rescale,
                                const PreparedLdsReadContext& prepared_context)
    {
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        constexpr bool kUsePreparedBase =
            Mode::SelectedPolicy::template UseEarlyLdsBases<Problem>() &&
            (Stage < Geometry::kPvStages - 1 || Geometry::kN == 128);
        static_assert(!kUsePreparedBase || !Mode::SelectedPolicy::kLeadFirstLoad,
                      "prepared PV bases require the WMMA-first placement");
        // V stage s reads rows [(s+1)*kK1, (s+2)*kK1) of the QK tail's V buffer. Keep the
        // stage offset constant so every read uses an immediate offset from its prepared base.
        // N64's final K stage retains the ordinary instruction path.
        [[maybe_unused]] const auto prepared_base = [&]() {
            if constexpr(kUsePreparedBase && Stage < Geometry::kPvStages - 1)
            {
                auto base = prepared_context.v0;
                base.byte_address += (Stage + 1) * Problem::BlockFmhaShape::kK1 * kVPhysicalStride *
                                     sizeof(typename Geometry::VDataType);
                return base;
            }
            else if constexpr(kUsePreparedBase)
                return prepared_context.pv_k0;
            else
                return FmhaTdmPreparedKRead::Address{};
        }();
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
            if constexpr(!kUsePreparedBase && !kLeadFirstLoadThisStage && Stage == 0 &&
                         decltype(wmma)::value == 0)
            {
                prepared_v1 = FmhaTdmPreparedKRead::Prepare<1>(b_lds_window);
                __builtin_amdgcn_sched_barrier(0);
            }
            if constexpr(!kLeadFirstLoadThisStage && Stage == Geometry::kPvStages - 1 &&
                         decltype(wmma)::value == 0)
            {
                if constexpr(kUsePreparedBase)
                    affine_k0 = prepared_base;
                else
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
                if constexpr(kUsePreparedBase)
                    FmhaTdmAffineVRead::Load<decltype(access)::value, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, prepared_base);
                else
                    VLoad::template LoadAccess<decltype(access)::value>(next_b_block_tensor,
                                                                        b_lds_window);
            }
        };
        auto emit_point = [](auto, auto, auto point) {
            if constexpr(decltype(point)::value == FmhaTdmSchedSchedulePoint::AfterTokens)
                __builtin_amdgcn_sched_barrier(0);
            else
                __builtin_amdgcn_sched_barrier(0x0002 | 0x0400);
        };

        auto emit_prepared_token = [&](auto kind, auto access) {
            if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
            {
                // Access0 was issued before WMMA0; do not issue it twice.
            }
            else if constexpr(!kUsePreparedBase && Stage == 0 &&
                              decltype(kind)::value == Kind::VRead && decltype(access)::value == 1)
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
        // Derive the key origin from the prefetched V window only when no sink phase or bias
        // origin is needed. D192 N128 retains the index loop to limit register pressure.
        return !Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               !(Geometry::kHeadDimQK == 192 && Geometry::kN == 128);
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

    // Local max of MSB Msb for RunPart1 (lane max, its lane-half partner and the old max), with
    // the lane max formed on the score pairs (c, c + 1) as two-wide vectors.
    template <index_t Msb, typename ScoreTensor>
    CK_TILE_DEVICE static float PairLocalMax(const ScoreTensor& score, float old_max)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;
        array<typename Mapping::Pair, Mapping::kPairsPerMsb> pairs{};
        static_for<0, Mapping::kPairsPerMsb, 1>{}(
            [&](auto p) { pairs(p) = Mapping::template LoadPair<Msb, decltype(p)::value>(score); });
        const auto pair_max  = PairTreeMax(pairs);
        const float lane_max = max(pair_max[0], pair_max[1]);
        return Softmax::Max3(lane_max, Softmax::PermuteLaneX16(lane_max), old_max);
    }

    // ck_tile::max(float, float) is __builtin_fmaxf (llvm.maxnum); the same on two-wide vectors.
    template <typename Pair>
    CK_TILE_DEVICE static Pair PairMax(const Pair& x, const Pair& y)
    {
        return __builtin_elementwise_maxnum(x, y);
    }

    // Max of N pairs by max3 levels, lane by lane.
    template <typename Pair, index_t N>
    CK_TILE_DEVICE static Pair PairTreeMax(const array<Pair, N>& v)
    {
        if constexpr(N == 1)
            return v[number<0>{}];
        else if constexpr(N == 2)
            return PairMax(v[number<0>{}], v[number<1>{}]);
        else
        {
            constexpr index_t kTriples = N / 3;
            constexpr index_t kRest    = N % 3;
            array<Pair, kTriples + kRest> next{};
            static_for<0, kTriples, 1>{}([&](auto i) {
                constexpr index_t k = 3 * decltype(i)::value;
                next(i) = PairMax(PairMax(v[number<k>{}], v[number<k + 1>{}]), v[number<k + 2>{}]);
            });
            static_for<0, kRest, 1>{}([&](auto i) {
                constexpr index_t k          = decltype(i)::value;
                next(number<kTriples + k>{}) = v[number<3 * kTriples + k>{}];
            });
            return PairTreeMax(next);
        }
    }

    // Row maxima of the split softmax. Each MSB's score pairs (c, c + 1) reduce as two-wide vectors
    // before one horizontal max, for every geometry:
    // - On scalar max chains, the clang 24 SLP vectorizer pairs the two row halves' chains into
    //   <2 x float> maxima (gfx1250 has no packed FP32 maximum) and carries that pairing into the
    //   score PHIs and the mask selects: D64 M128 masked needs 336 VGPRs with 35 spilled, D128
    //   M128 masked 514 (two waves per SIMD -> one).
    // - It replaces the inline-asm v_max3 chains D192 used: inline-asm VGPR operands must sit in
    //   the low VGPR range, which the WMMA-source keep-alive (KeepFragmentAlive) also needs.
    // The maximum is associative and commutative, so the regrouped result is the same.
    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(
        ScoreTensor& score, RowTensor& row_max, float log2e_scale, float& delta_m0, float& delta_m1)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == Geometry::kQkMIter);

        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking;
        const float old_max_m0      = row_max.get_thread_buffer()[number<0>{}];
        const float local_m0        = PairLocalMax<0>(score, old_max_m0);
        const float local_m1        = PairLocalMax<1>(score, old_max_m0);
        const auto p1_m0            = Softmax::template RunPart1<kValidateMax>(
            local_m0, local_m1, old_max_m0 * log2e_scale, log2e_scale);
        row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
        delta_m0                                 = p1_m0.delta;
        if constexpr(Geometry::kQkMIter == 1)
        {
            // M64: each lane owns one row (Msb 0/1); there is no second row group.
            delta_m1 = 0.0f;
        }
        else
        {
            const float old_max_m1 = row_max.get_thread_buffer()[number<1>{}];
            const float local_m2   = PairLocalMax<2>(score, old_max_m1);
            const float local_m3   = PairLocalMax<3>(score, old_max_m1);
            const auto p1_m1       = Softmax::template RunPart1<kValidateMax>(
                local_m2, local_m3, old_max_m1 * log2e_scale, log2e_scale);
            row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
            delta_m1                                 = p1_m1.delta;
        }
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
        static_assert(RowTensor::get_thread_buffer_size() == Geometry::kQkMIter);

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

        if constexpr(Geometry::kQkMIter == 1)
        {
            // M64: one row group per lane.
            const float row_max_m0 = row_max.get_thread_buffer()[number<0>{}];
            const float old_sum_m0 = row_sum.get_thread_buffer()[number<0>{}];
            const float sum_m0     = Softmax::template RunPart2LocalSum<0>(
                score, validated_max(row_max_m0), log2e_scale);
            const float sum_m1 = Softmax::template RunPart2LocalSum<1>(
                score, validated_max(row_max_m0), log2e_scale);
            row_sum.get_thread_buffer()[number<0>{}] =
                Softmax::UpdateRowSum(old_sum_m0, Softmax::MergeRowSum(sum_m0, sum_m1), delta_m0);
            output_scale_m0 = Softmax::Exp2(delta_m0);
            output_scale_m1 = 1.0f;
        }
        else
        {
            const float row_max_m0 = row_max.get_thread_buffer()[number<0>{}];
            const float row_max_m1 = row_max.get_thread_buffer()[number<1>{}];
            const float old_sum_m0 = row_sum.get_thread_buffer()[number<0>{}];
            const float old_sum_m1 = row_sum.get_thread_buffer()[number<1>{}];

            const float sum_m0 = Softmax::template RunPart2LocalSum<0>(
                score, validated_max(row_max_m0), log2e_scale);
            const float sum_m1 = Softmax::template RunPart2LocalSum<1>(
                score, validated_max(row_max_m0), log2e_scale);
            const float sum_m2 = Softmax::template RunPart2LocalSum<2>(
                score, validated_max(row_max_m1), log2e_scale);
            const float sum_m3 = Softmax::template RunPart2LocalSum<3>(
                score, validated_max(row_max_m1), log2e_scale);

            row_sum.get_thread_buffer()[number<0>{}] =
                Softmax::UpdateRowSum(old_sum_m0, Softmax::MergeRowSum(sum_m0, sum_m1), delta_m0);
            row_sum.get_thread_buffer()[number<1>{}] =
                Softmax::UpdateRowSum(old_sum_m1, Softmax::MergeRowSum(sum_m2, sum_m3), delta_m1);

            output_scale_m0 = Softmax::Exp2(delta_m0);
            output_scale_m1 = Softmax::Exp2(delta_m1);
        }
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
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        constexpr bool is_wave32 = get_warp_size() == Geometry::kWaveSize;
#else
        constexpr bool is_wave32 = true;
#endif
        // The problem must describe exactly this policy's geometry (dtypes, block tile, wave
        // layout, WMMA tiles). Both GEMMs share one wave layout: P stays in the lanes that
        // produced S.
        return Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               std::is_same_v<FmhaTdmSchedGeometryFor<Problem, Geometry::kQkSuColumns>, Geometry> &&
               Shape::Gemm1BlockWarps::at(number<0>{}) == Shape::Gemm0BlockWarps::at(number<0>{}) &&
               Shape::Gemm1BlockWarps::at(number<1>{}) == Shape::Gemm0BlockWarps::at(number<1>{}) &&
               Shape::Gemm1BlockWarps::at(number<2>{}) == Shape::Gemm0BlockWarps::at(number<2>{}) &&
               std::is_same_v<remove_cvref_t<typename Problem::SaccDataType>, float> &&
               std::is_same_v<remove_cvref_t<typename Problem::OaccDataType>, float> &&
               Shape::kK0 == Geometry::kQkWmmaK && Shape::kSubQKHeaddim == Geometry::kHeadDimQK &&
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
                      "qr_tdm_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K TDM load");
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(kLdsRows % warp_num == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, kLdsRows / warp_num>, sequence<kKPhysicalStride>>,
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
                      "qr_tdm_sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackK<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<kKPhysicalStride>{}),
            make_tuple(number<kKPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K LDS descriptor");
        return MakeKLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackV<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<kVLogicalWidth>{}),
            make_tuple(number<kVPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadBlockDescriptor()
    {
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        static_assert(!Xor, "D192/V128 TDM writer requires an affine V LDS descriptor");
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
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
                      "qr_tdm_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K LDS allocation");
        return kKFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return kVFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return GetLdsArenaSize();
    }

    template <typename Problem>
    static constexpr bool kUsePaddedQ = Geometry::kHeadDimQK <= 128 && Geometry::kM == 128;

    template <typename Problem>
    using LdsPaddingConfigQ = std::conditional_t<
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
            using Padding = LdsPaddingConfigQ<Problem>;
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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        static_assert(IsSupportedProblem<Problem>(), "V128 policy received an invalid problem");
        using Padding = detail::EncodedTdmPadding<LdsPaddingConfigQ<Problem>>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return make_tuple(number<true>{}, number<kVPadAmount>{}, number<kVPadInterval>{});
    }
};

template <typename Geometry>
struct FmhaTdmSchedTuningSelector
{
    static constexpr index_t kBlockPerCu =
        Geometry::kHeadDimQK == 64 ? 3
                                   : (Geometry::kHeadDimQK == 192 && Geometry::kN == 64 ? 2 : 1);
    using type = FmhaTdmSchedDerivedTuning<Geometry, kBlockPerCu>;
};

template <typename Geometry>
using FmhaTdmSchedDefaultTuning = typename FmhaTdmSchedTuningSelector<Geometry>::type;

template <typename Problem>
using FmhaTdmSchedPolicyFor = BlockFmhaPipelineQRKSVSTdmSchedPolicy<
    FmhaTdmSchedGeometryFor<Problem>,
    FmhaTdmSchedDefaultTuning<FmhaTdmSchedGeometryFor<Problem>>>;

} // namespace ck_tile
