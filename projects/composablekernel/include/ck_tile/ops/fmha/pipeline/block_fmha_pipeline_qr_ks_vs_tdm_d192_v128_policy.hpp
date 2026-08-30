// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef CK_TILE_FMHA_GFX125_D192_CROSS_TILE
#define CK_TILE_FMHA_GFX125_D192_CROSS_TILE 0
#endif

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_load.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_output.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule_executor.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"

namespace ck_tile {

struct BlockFmhaPipelineQRKSVSTdmD192V128Policy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;

    static constexpr index_t kLdsRows         = 128;
    static constexpr index_t kKValidWidth     = 192;
    static constexpr index_t kKPhysicalStride = 200;
    static constexpr index_t kVLogicalWidth   = 128;
    static constexpr index_t kVPhysicalStride = 144;
    static constexpr index_t kVPadAmount      = 7;
    static constexpr index_t kVPadInterval    = 5;
    static constexpr index_t kLdsOffsetK0     = 0x00000;
    static constexpr index_t kLdsOffsetK1     = 0x10000;
    static constexpr index_t kLdsOffsetV0     = 0x20000;
    static constexpr index_t kLdsOffsetV1     = 0x30000;
    static constexpr index_t kLdsArenaSize    = 0x39000;
    static constexpr index_t kKFootprintBytes = kLdsRows * kKPhysicalStride * sizeof(bf16_t);
    static constexpr index_t kVFootprintBytes = kLdsRows * kVPhysicalStride * sizeof(bf16_t);
    static constexpr bool kUseFullHeadKSuQk   = true;
    static constexpr bool kUseSplitSoftmax    = true;
    static constexpr bool kSkipExactFullTilePredicate = true;
    static constexpr bool kUseCustomQkStageSchedule   = true;
    static constexpr bool kUseCustomPvStageSchedule   = true;
    static constexpr bool kUseOutputFragments         = true;
    static constexpr bool kUseCrossTile               = CK_TILE_FMHA_GFX125_D192_CROSS_TILE != 0;
    static constexpr auto kORescaleToken              = FmhaD192ScheduleToken::ORescale;

    using OutputFragments = FmhaD192OutputFragments;

    static_assert(kKFootprintBytes == 0xc800);
    static_assert(kVFootprintBytes == 0x9000);
    static_assert(kLdsOffsetK0 + kKFootprintBytes <= kLdsOffsetK1);
    static_assert(kLdsOffsetK1 + kKFootprintBytes <= kLdsOffsetV0);
    static_assert(kLdsOffsetV0 + kVFootprintBytes <= kLdsOffsetV1);
    static_assert(kLdsOffsetV1 + kVFootprintBytes == kLdsArenaSize);
    static_assert(kLdsOffsetK0 % 16 == 0 && kLdsOffsetK1 % 16 == 0 && kLdsOffsetV0 % 16 == 0 &&
                  kLdsOffsetV1 % 16 == 0);

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemmSu()
    {
        using GemmProblem = BlockGemmProblem<
            typename Problem::QDataType,
            typename Problem::KDataType,
            typename Problem::SaccDataType,
            Problem::kBlockSize,
            TileGemmShape<
                sequence<Problem::BlockFmhaShape::kM0, 32, Problem::BlockFmhaShape::kQKHeaddim>,
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
        static_assert(BlockGemm::NIterPerWarp == 2 && BlockGemm::KIterPerWarp == 6);
        return make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
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

    template <typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor>
    CK_TILE_DEVICE static void RunQkSu(const BlockGemm&,
                                       CBlockTensor& c_block_tensor,
                                       const ABlockTensor& a_block_tensor,
                                       const BBlockTensor& b_block_tensor)
    {
        static_assert(BlockGemm::KIterPerWarp == 6);
        static_assert(BlockGemm::MIterPerWarp == 2);
        static_assert(BlockGemm::NIterPerWarp == 2);

        static_assert(
            std::is_same_v<remove_cvref_t<decltype(BlockGemm::MakeABlockDistributionEncode())>,
                           remove_cvref_t<decltype(ABlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "D192 SU Q distribution is incompatible with the QK BlockGemm");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(BlockGemm::MakeBBlockDistributionEncode())>,
                           remove_cvref_t<decltype(BBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "D192 SU K distribution is incompatible with the QK BlockGemm");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(BlockGemm::MakeCBlockDistributionEncode())>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "D192 SU accumulator distribution is incompatible with the QK BlockGemm");

        static_for<0, BlockGemm::NIterPerWarp, 1>{}([&](auto n_iter) {
            static_for<0, BlockGemm::KIterPerWarp, 1>{}([&](auto k_iter) {
                static_for<0, BlockGemm::MIterPerWarp, 1>{}([&](auto m_iter) {
                    constexpr index_t ordinal =
                        n_iter * BlockGemm::KIterPerWarp * BlockGemm::MIterPerWarp +
                        k_iter * BlockGemm::MIterPerWarp + m_iter;
                    RunQkSuWmma<ordinal, BlockGemm>(c_block_tensor, a_block_tensor, b_block_tensor);
                });
            });
        });
    }

    template <index_t Stage,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename SoftmaxTokenEmitter>
    CK_TILE_DEVICE static void
    RunQkScheduledStageWithSoftmax(const BlockGemm&,
                                   CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor,
                                   NextBBlockTensor& next_b_block_tensor,
                                   const BTileWindow& b_lds_window,
                                   SoftmaxTokenEmitter& emit_softmax_token)
    {
        using Executor = BlockFmhaPipelineQRKSVSTdmD192V128ScheduleExecutor;
        using Token    = FmhaD192ScheduleToken;
        static_assert(Stage >= 0 && Stage < Executor::Schedule::kNumStages);
        static_assert(
            (Stage < Executor::Schedule::kNumStages - 1 && BTileWindow::NumAccessPerCoord == 24) ||
            (Stage == Executor::Schedule::kNumStages - 1 && BTileWindow::NumAccessPerCoord == 16));

        auto emit_wmma = [&](auto, auto wmma) {
            RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
                c_block_tensor, a_block_tensor, b_block_tensor);
        };
        auto emit_token = [&](auto token, auto ordinal) {
            constexpr auto token_value = decltype(token)::value;
            if constexpr(token_value >= Token::KM0 && token_value <= Token::KM3)
            {
                static_assert(Stage < Executor::Schedule::kNumStages - 1);
                constexpr index_t msb =
                    static_cast<index_t>(token_value) - static_cast<index_t>(Token::KM0);
                constexpr index_t loads_per_msb = BTileWindow::NumAccessPerCoord / 4;
                constexpr index_t local_ordinal = decltype(ordinal)::value - Stage * loads_per_msb;
                static_assert(local_ordinal >= 0 && local_ordinal < loads_per_msb);
                constexpr index_t access = msb * loads_per_msb + local_ordinal;
                FmhaD192Load::LoadInstruction<access>(next_b_block_tensor, b_lds_window);
            }
            else if constexpr(token_value >= Token::VM0 && token_value <= Token::VM3)
            {
                static_assert(Stage == Executor::Schedule::kNumStages - 1);
                constexpr index_t msb =
                    static_cast<index_t>(token_value) - static_cast<index_t>(Token::VM0);
                constexpr index_t loads_per_msb = BTileWindow::NumAccessPerCoord / 4;
                constexpr index_t access        = msb * loads_per_msb + decltype(ordinal)::value;
                FmhaD192TransposeLoad::LoadAccess<access>(next_b_block_tensor, b_lds_window);
            }
            else if constexpr((token_value >= Token::P2M0 && token_value <= Token::P2M3) ||
                              token_value == Token::ORescale)
            {
                emit_softmax_token(token, ordinal);
            }
        };
        auto emit_point = [](auto, auto, auto) { __builtin_amdgcn_sched_barrier(0); };

        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        s_wait_dscnt<0>();
    }

    template <index_t Stage,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow>
    CK_TILE_DEVICE static void RunQkScheduledStage(const BlockGemm& block_gemm,
                                                   CBlockTensor& c_block_tensor,
                                                   const ABlockTensor& a_block_tensor,
                                                   const BBlockTensor& b_block_tensor,
                                                   NextBBlockTensor& next_b_block_tensor,
                                                   const BTileWindow& b_lds_window)
    {
        auto emit_no_softmax = [](auto, auto) {};
        RunQkScheduledStageWithSoftmax<Stage>(block_gemm,
                                              c_block_tensor,
                                              a_block_tensor,
                                              b_block_tensor,
                                              next_b_block_tensor,
                                              b_lds_window,
                                              emit_no_softmax);
    }

    template <index_t WmmaOrdinal,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor>
    CK_TILE_DEVICE static void RunPvSuWmma(CBlockTensor& c_block_tensor,
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

        static_assert(BlockGemm::KIterPerWarp == 1);
        static_assert(BlockGemm::MIterPerWarp == 2);
        static_assert(BlockGemm::NIterPerWarp == 8);
        static_assert(WmmaOrdinal >= 0 && WmmaOrdinal < 16);

        constexpr index_t d_msb       = WmmaOrdinal / 4;
        constexpr index_t n           = WmmaOrdinal % 4;
        constexpr index_t m_iter      = d_msb / 2;
        constexpr index_t v_msb       = d_msb % 2;
        constexpr index_t full_n_iter = n * 2 + v_msb;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        AWarpTensor a_warp_tensor;
        a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<0, m_iter>{}, a_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

        BWarpTensor b_warp_tensor;
        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<0, full_n_iter>{}, b_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

        CWarpTensor c_warp_tensor;
        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, full_n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        c_block_tensor.set_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, full_n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    }

    template <index_t Stage,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow>
    CK_TILE_DEVICE static void RunPvScheduledStage(const BlockGemm&,
                                                   CBlockTensor& c_block_tensor,
                                                   const ABlockTensor& a_block_tensor,
                                                   const BBlockTensor& b_block_tensor,
                                                   NextBBlockTensor& next_b_block_tensor,
                                                   const BTileWindow& b_lds_window)
    {
        using Executor = BlockFmhaPipelineQRKSVSTdmD192V128ScheduleExecutor;
        using Token    = FmhaD192ScheduleToken;
        static_assert(Stage >= 0 && Stage < Executor::Schedule::kNumStages);
        static_assert(Executor::Schedule::kPvWmmasPerStage == OutputFragments::kNumFragments);
        static_assert(
            (Stage < Executor::Schedule::kNumStages - 1 && BTileWindow::NumAccessPerCoord == 16) ||
            (Stage == Executor::Schedule::kNumStages - 1 && BTileWindow::NumAccessPerCoord == 24));

        auto emit_wmma = [&](auto, auto wmma) {
            RunPvSuWmma<decltype(wmma)::value, BlockGemm>(
                c_block_tensor, a_block_tensor, b_block_tensor);
        };
        auto emit_token = [&](auto token, auto ordinal) {
            constexpr auto token_value = decltype(token)::value;
            if constexpr(token_value >= Token::VM0 && token_value <= Token::VM3)
            {
                static_assert(Stage < Executor::Schedule::kNumStages - 1);
                constexpr index_t msb =
                    static_cast<index_t>(token_value) - static_cast<index_t>(Token::VM0);
                constexpr index_t loads_per_msb = BTileWindow::NumAccessPerCoord / 4;
                constexpr index_t local_ordinal = decltype(ordinal)::value - Stage * loads_per_msb;
                static_assert(local_ordinal >= 0 && local_ordinal < loads_per_msb);
                constexpr index_t access = msb * loads_per_msb + local_ordinal;
                FmhaD192TransposeLoad::LoadAccess<access>(next_b_block_tensor, b_lds_window);
            }
            else if constexpr(token_value >= Token::KM0 && token_value <= Token::KM3)
            {
                static_assert(Stage == Executor::Schedule::kNumStages - 1);
                constexpr index_t msb =
                    static_cast<index_t>(token_value) - static_cast<index_t>(Token::KM0);
                constexpr index_t loads_per_msb = BTileWindow::NumAccessPerCoord / 4;
                constexpr index_t access        = msb * loads_per_msb + decltype(ordinal)::value;
                FmhaD192Load::LoadInstruction<access>(next_b_block_tensor, b_lds_window);
            }
        };
        auto emit_point = [](auto, auto, auto) { __builtin_amdgcn_sched_barrier(0); };

        Executor::template ExecutePvStage<Stage>(emit_wmma, emit_token, emit_point);
        s_wait_dscnt<0>();
    }

    template <index_t Stage, typename NextBBlockTensor, typename BTileWindow, typename WmmaEmitter>
    CK_TILE_DEVICE static void RunPvScheduledStageWithWmma(NextBBlockTensor& next_b_block_tensor,
                                                           const BTileWindow& b_lds_window,
                                                           WmmaEmitter& emit_wmma)
    {
        using Executor = BlockFmhaPipelineQRKSVSTdmD192V128ScheduleExecutor;
        using Token    = FmhaD192ScheduleToken;
        static_assert(Stage >= 0 && Stage < Executor::Schedule::kNumStages);
        static_assert(
            (Stage < Executor::Schedule::kNumStages - 1 && BTileWindow::NumAccessPerCoord == 16) ||
            (Stage == Executor::Schedule::kNumStages - 1 && BTileWindow::NumAccessPerCoord == 24));

        auto emit_token = [&](auto token, auto ordinal) {
            constexpr auto token_value = decltype(token)::value;
            if constexpr(token_value >= Token::VM0 && token_value <= Token::VM3)
            {
                static_assert(Stage < Executor::Schedule::kNumStages - 1);
                constexpr index_t msb =
                    static_cast<index_t>(token_value) - static_cast<index_t>(Token::VM0);
                constexpr index_t loads_per_msb = BTileWindow::NumAccessPerCoord / 4;
                constexpr index_t local_ordinal = decltype(ordinal)::value - Stage * loads_per_msb;
                static_assert(local_ordinal >= 0 && local_ordinal < loads_per_msb);
                constexpr index_t access = msb * loads_per_msb + local_ordinal;
                FmhaD192TransposeLoad::LoadAccess<access>(next_b_block_tensor, b_lds_window);
            }
            else if constexpr(token_value >= Token::KM0 && token_value <= Token::KM3)
            {
                static_assert(Stage == Executor::Schedule::kNumStages - 1);
                constexpr index_t msb =
                    static_cast<index_t>(token_value) - static_cast<index_t>(Token::KM0);
                constexpr index_t loads_per_msb = BTileWindow::NumAccessPerCoord / 4;
                constexpr index_t access        = msb * loads_per_msb + decltype(ordinal)::value;
                FmhaD192Load::LoadInstruction<access>(next_b_block_tensor, b_lds_window);
            }
        };
        auto emit_point = [](auto, auto, auto) { __builtin_amdgcn_sched_barrier(0); };

        Executor::template ExecutePvStage<Stage>(emit_wmma, emit_token, emit_point);
        s_wait_dscnt<0>();
    }

    template <typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor>
    CK_TILE_DEVICE static void RunPvSu(const BlockGemm&,
                                       CBlockTensor& c_block_tensor,
                                       const ABlockTensor& a_block_tensor,
                                       const BBlockTensor& b_block_tensor)
    {
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(BlockGemm::MakeABlockDistributionEncode())>,
                           remove_cvref_t<decltype(ABlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "D192 SU P distribution is incompatible with the PV BlockGemm");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(BlockGemm::MakeBBlockDistributionEncode())>,
                           remove_cvref_t<decltype(BBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "D192 SU V distribution is incompatible with the PV BlockGemm");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(BlockGemm::MakeCBlockDistributionEncode())>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "D192 output accumulator distribution is incompatible with the PV BlockGemm");

        static_for<0, 16, 1>{}([&](auto ordinal) {
            RunPvSuWmma<ordinal, BlockGemm>(c_block_tensor, a_block_tensor, b_block_tensor);
        });
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(
        ScoreTensor& score, RowTensor& row_max, float log2e_scale, float& delta_m0, float& delta_m1)
    {
        using Softmax = FmhaD192SplitSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        const float old_max_m0 = row_max.get_thread_buffer()[number<0>{}];
        const float old_max_m1 = row_max.get_thread_buffer()[number<1>{}];

        const auto p0_m0 = Softmax::RunPart0<0>(score, old_max_m0, log2e_scale);
        const auto p0_m1 = Softmax::RunPart0<1>(score, old_max_m0, log2e_scale);
        const auto p0_m2 = Softmax::RunPart0<2>(score, old_max_m1, log2e_scale);
        const auto p0_m3 = Softmax::RunPart0<3>(score, old_max_m1, log2e_scale);

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
        using Softmax = FmhaD192SplitSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking;

        const auto validated_max = [](float value) {
            if constexpr(kValidateMax)
            {
                return value == -numeric<float>::infinity() ? 0.0f : value;
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
            Softmax::RunPart2LocalSum<0>(score, validated_max(row_max_m0), log2e_scale);
        const float sum_m1 =
            Softmax::RunPart2LocalSum<1>(score, validated_max(row_max_m0), log2e_scale);
        const float sum_m2 =
            Softmax::RunPart2LocalSum<2>(score, validated_max(row_max_m1), log2e_scale);
        const float sum_m3 =
            Softmax::RunPart2LocalSum<3>(score, validated_max(row_max_m1), log2e_scale);

        row_sum.get_thread_buffer()[number<0>{}] =
            Softmax::UpdateRowSum(old_sum_m0, Softmax::MergeRowSum(sum_m0, sum_m1), delta_m0);
        row_sum.get_thread_buffer()[number<1>{}] =
            Softmax::UpdateRowSum(old_sum_m1, Softmax::MergeRowSum(sum_m2, sum_m3), delta_m1);

        output_scale_m0 = Softmax::Exp2(delta_m0);
        output_scale_m1 = Softmax::Exp2(delta_m1);
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor, typename OutputTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart2(ScoreTensor& score,
                                                    const RowTensor& row_max,
                                                    RowTensor& row_sum,
                                                    OutputTensor& output,
                                                    float log2e_scale,
                                                    float delta_m0,
                                                    float delta_m1)
    {
        using Mapping = typename FmhaD192SplitSoftmax::Mapping;
        static_assert(OutputTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(std::is_same_v<typename ScoreTensor::StaticTileDistribution,
                                     typename OutputTensor::StaticTileDistribution>,
                      "D192 score and output accumulators must have identical lane ownership");

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
        using Softmax = FmhaD192SplitSoftmax;
        Softmax::RescaleOutput<0>(output, output_scale_m0);
        Softmax::RescaleOutput<1>(output, output_scale_m0);
        Softmax::RescaleOutput<2>(output, output_scale_m1);
        Softmax::RescaleOutput<3>(output, output_scale_m1);
    }

    template <index_t Ordinal, typename OutputTensor>
    CK_TILE_DEVICE static void
    RunOutputRescaleToken(OutputTensor& output, float output_scale_m0, float output_scale_m1)
    {
        using Softmax = FmhaD192SplitSoftmax;
        static_assert(Ordinal >= 0 && Ordinal < 16);
        constexpr index_t output_tile = Ordinal / 4;
        constexpr index_t msb         = Ordinal % 4;
        const float output_scale      = msb < 2 ? output_scale_m0 : output_scale_m1;
        Softmax::template RescaleOutputTile<msb, output_tile>(output, output_scale);
    }

    template <typename TokenConstant, typename Ordinal, typename OutputTensor>
    CK_TILE_DEVICE static void RunPreviousTileSoftmaxToken(
        TokenConstant, Ordinal, OutputTensor& output, float output_scale_m0, float output_scale_m1)
    {
        if constexpr(TokenConstant::value == FmhaD192ScheduleToken::ORescale)
        {
            RunOutputRescaleToken<Ordinal::value>(output, output_scale_m0, output_scale_m1);
        }
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor, typename OutputTensor>
    CK_TILE_DEVICE static void RunSplitSoftmax(ScoreTensor& score,
                                               RowTensor& row_max,
                                               RowTensor& row_sum,
                                               OutputTensor& output,
                                               float log2e_scale)
    {
        float delta_m0;
        float delta_m1;
        RunSplitSoftmaxPart01<Problem>(score, row_max, log2e_scale, delta_m0, delta_m1);
        RunSplitSoftmaxPart2<Problem>(
            score, row_max, row_sum, output, log2e_scale, delta_m0, delta_m1);
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor, typename Fragments>
    CK_TILE_DEVICE static void RunSplitSoftmaxFragments(ScoreTensor& score,
                                                        RowTensor& row_max,
                                                        RowTensor& row_sum,
                                                        Fragments& output,
                                                        float log2e_scale)
    {
        float delta_m0;
        float delta_m1;
        RunSplitSoftmaxPart01<Problem>(score, row_max, log2e_scale, delta_m0, delta_m1);

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

        static_for<0, OutputFragments::kNumFragments, 1>{}([&](auto ordinal) {
            constexpr index_t d_msb = decltype(ordinal)::value / OutputFragments::kNumN;
            const float scale       = d_msb < 2 ? output_scale_m0 : output_scale_m1;
            auto& fragment          = output.at(ordinal);
            static_for<0, OutputFragments::kElementsPerFragment, 1>{}(
                [&](auto element) { fragment[decltype(element)::value] *= scale; });
        });
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedProblem()
    {
        using Shape = remove_cvref_t<typename Problem::BlockFmhaShape>;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        constexpr bool is_wave32 = get_warp_size() == 32;
#else
        constexpr bool is_wave32 = true;
#endif
        return std::is_same_v<remove_cvref_t<typename Problem::QDataType>, bf16_t> &&
               std::is_same_v<remove_cvref_t<typename Problem::KDataType>, bf16_t> &&
               std::is_same_v<remove_cvref_t<typename Problem::VDataType>, bf16_t> &&
               Shape::kM0 == 128 && Shape::kN0 == 128 && Shape::kK0 == 32 && Shape::kN1 == 128 &&
               Shape::kK1 == 32 && Shape::kQKHeaddim == 192 && Shape::kSubQKHeaddim == 192 &&
               Shape::Gemm0BlockWarps::at(number<0>{}) == 4 &&
               Shape::Gemm0BlockWarps::at(number<1>{}) == 1 &&
               Shape::Gemm0BlockWarps::at(number<2>{}) == 1 &&
               Shape::Gemm0WarpTile::at(number<0>{}) == 16 &&
               Shape::Gemm0WarpTile::at(number<1>{}) == 16 &&
               Shape::Gemm0WarpTile::at(number<2>{}) == 32 &&
               Shape::Gemm1BlockWarps::at(number<0>{}) == 4 &&
               Shape::Gemm1BlockWarps::at(number<1>{}) == 1 &&
               Shape::Gemm1BlockWarps::at(number<2>{}) == 1 &&
               Shape::Gemm1WarpTile::at(number<0>{}) == 16 &&
               Shape::Gemm1WarpTile::at(number<1>{}) == 16 &&
               Shape::Gemm1WarpTile::at(number<2>{}) == 32 && Shape::IsVLayoutRowMajor &&
               Shape::NumWarps == 4 && is_wave32;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQKReductionSteps()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        return Problem::BlockFmhaShape::kQKHeaddim / Problem::BlockFmhaShape::kK0;
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
                      "D192/V128 policy received an invalid problem");
        static_assert(LoadOnce, "D192/V128 policy requires a full-head K TDM load");
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
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramSuTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(32 % warp_num == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, 32 / warp_num>, sequence<kKPhysicalStride>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramSuTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(32 % warp_num == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, 32 / warp_num>, sequence<kVLogicalWidth>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <index_t Su,
              typename Problem,
              typename TdmConfig,
              typename LdsTileWindow,
              typename DramTileWindow>
    CK_TILE_DEVICE static void LoadKSuTdm(const TdmConfig& config,
                                          const LdsTileWindow& lds_tile_window,
                                          const DramTileWindow& dram_tile_window)
    {
        static_assert(Su >= 0 && Su < 4);
        const auto dram_origin = dram_tile_window.get_window_origin();
        auto dram_su_window    = make_tile_window(
            dram_tile_window.get_bottom_tensor_view(),
            make_tuple(number<32>{}, number<kKPhysicalStride>{}),
            make_multi_index(dram_origin.at(number<0>{}) + Su * 32, dram_origin.at(number<1>{})),
            MakeKDramSuTileDistribution<Problem>());

        const auto lds_origin = lds_tile_window.get_window_origin();
        auto lds_su_window    = make_tile_window(
            lds_tile_window.get_bottom_tensor_view(),
            make_tuple(number<32>{}, number<kKPhysicalStride>{}),
            make_multi_index(lds_origin.at(number<0>{}) + Su * 32, lds_origin.at(number<1>{})));
        load_tile_tdm(config, lds_su_window, dram_su_window);
    }

    template <index_t Su,
              typename Problem,
              typename TdmConfig,
              typename LdsTileWindow,
              typename DramTileWindow>
    CK_TILE_DEVICE static void LoadVSuTdm(const TdmConfig& config,
                                          const LdsTileWindow& lds_tile_window,
                                          const DramTileWindow& dram_tile_window)
    {
        static_assert(Su >= 0 && Su < 4);
        const auto dram_origin = dram_tile_window.get_window_origin();
        auto dram_su_window    = make_tile_window(
            dram_tile_window.get_bottom_tensor_view(),
            make_tuple(number<32>{}, number<kVLogicalWidth>{}),
            make_multi_index(dram_origin.at(number<0>{}) + Su * 32, dram_origin.at(number<1>{})),
            MakeVDramSuTileDistribution<Problem>());

        const auto lds_origin = lds_tile_window.get_window_origin();
        auto lds_su_window    = make_tile_window(
            lds_tile_window.get_bottom_tensor_view(),
            make_tuple(number<32>{}, number<kVLogicalWidth>{}),
            make_multi_index(lds_origin.at(number<0>{}) + Su * 32, lds_origin.at(number<1>{})));
        load_tile_tdm(config, lds_su_window, dram_su_window);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackK<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<kKPhysicalStride>{}),
            make_tuple(number<kKPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsReadBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackK<Problem>();
        return make_naive_tensor_descriptor(make_tuple(number<kLdsRows>{}, number<kKValidWidth>{}),
                                            make_tuple(number<kKPhysicalStride>{}, number<1>{}),
                                            number<kKPack>{},
                                            number<1>{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        static_assert(LoadOnce, "D192/V128 policy requires a full-head K LDS descriptor");
        return MakeKLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
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
                      "D192/V128 policy received an invalid problem");
        using WarpGemm = WarpGemmWmma_f32_16x16x32_bf16_bf16<true, WGAttrNumAccessEnum::Double>;

        constexpr index_t kMWarp        = 4;
        constexpr index_t kNWarp        = 1;
        constexpr index_t kNIterPerWarp = kVLogicalWidth / (kNWarp * WarpGemm::kN);
        constexpr index_t kKIterPerWarp = 32 / WarpGemm::kK;
        static_assert(kNIterPerWarp == 8 && kKIterPerWarp == 1);

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
                      "D192/V128 policy received an invalid problem");
        static_assert(LoadOnce, "D192/V128 policy requires a full-head K LDS allocation");
        return kKFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        return kVFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        return GetLdsArenaSize();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "D192/V128 policy received an invalid problem");
        return make_tuple(number<true>{}, number<kVPadAmount>{}, number<kVPadInterval>{});
    }
};

} // namespace ck_tile
