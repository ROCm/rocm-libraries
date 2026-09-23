// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/detail/fmha_dtype_traits.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_v128_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_output.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule_executor.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = FmhaTdmV128PolicyFor<Problem_>>
struct BlockFmhaPipelineQRKSVSTdmV128 : BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>
{
    using Base         = BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>;
    using Problem      = remove_cvref_t<Problem_>;
    using Policy       = remove_cvref_t<Policy_>;
    using SplitSoftmax = FmhaTdmV128SplitSoftmax;

    using QDataType           = typename Base::QDataType;
    using KDataType           = typename Base::KDataType;
    using VDataType           = typename Base::VDataType;
    using ODataType           = typename Base::ODataType;
    using SaccDataType        = typename Base::SaccDataType;
    using SMPLComputeDataType = typename Base::SMPLComputeDataType;
    using LSEDataType         = typename Base::LSEDataType;
    using PDataType           = typename Base::PDataType;
    using OaccDataType        = typename Base::OaccDataType;
    using FmhaMask            = typename Base::FmhaMask;

    using DTypeTraits = fmha::FmhaProblemTraitsT<Problem>;
    static_assert(std::is_same_v<typename DTypeTraits::QDataType, QDataType>);
    static_assert(std::is_same_v<typename DTypeTraits::KDataType, KDataType>);
    static_assert(std::is_same_v<typename DTypeTraits::VDataType, VDataType>);
    static_assert(std::is_same_v<typename DTypeTraits::ODataType, ODataType>);

    using Base::BiasEnum;
    using Base::I0;
    using Base::I1;
    using Base::kHasLogitsSoftCap;
    using Base::kHasSink;
    using Base::kHasUnevenSplits;
    using Base::kK0;
    using Base::kK1;
    using Base::kM0;
    using Base::kN0;
    using Base::kN1;
    using Base::kNWarp;
    using Base::kPadSeqLenK;
    using Base::kQKHeaddim;
    using Base::kStoreLSE;
    using Base::kSubQKHeaddim;
    using Base::MakePForGemm1;

    static_assert(Policy::template IsSupportedProblem<Problem>(),
                  "qr_tdm_v128 requires the exact BF16 D128/V128 or D192/V128 geometry");

    using Geometry = FmhaTdmV128Geometry<QDataType,
                                         KDataType,
                                         VDataType,
                                         PDataType,
                                         kQKHeaddim,
                                         Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                         Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{})>;
    static_assert(Geometry::kM == kM0 && Geometry::kN == kN0 && Geometry::kDv == kN1);

    static constexpr const char* name = "qr_tdm_v128";

    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;
    static constexpr bool kUsesFixedSegmentedLdsArena  = true;
    static constexpr bool kUsesLdsArena                = false;
    static constexpr index_t kBlockPerCu               = Policy::kBlockPerCu;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    using QKBlockGemm = remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
    using ScoreTile   = decltype(QKBlockGemm::MakeCBlockTile());
    static_assert(ScoreTile::get_thread_buffer_size() == SplitSoftmax::Mapping::kThreadBufferSize);
#endif

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(Policy::template GetSmemSize<Problem>() == Policy::GetLdsArenaSize());
        return Policy::GetLdsArenaSize();
    }

    // Prefill, double lds
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    run(const QDramBlockWindowTmp& __restrict__ q_dram_block_window_tmp,       // M0*K0 tile
        const KDramBlockWindowTmp& __restrict__ k_dram_block_window_tmp,       // N0*K0 tile
        const VDramBlockWindowTmp& __restrict__ v_dram_block_window_tmp,       // N1*K1 tile
        const BiasDramBlockWindowTmp& __restrict__ bias_dram_block_window_tmp, // M0*N0 tile
        LSEaccDramBlockWindowTmp& __restrict__ lse_acc_dram_window_tmp,        // M0*1 tile
        FmhaMask mask,
        PositionEncoding position_encoding,
        float scale_s,
        void* __restrict__ smem_ptrk0,
        void* __restrict__ smem_ptrk1,
        void* __restrict__ smem_ptrv0,
        void* __restrict__ smem_ptrv1,
        float sink_v) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Q, K, and V use TDM global-to-LDS transfers; V is consumed through
        // transposed LDS loads.
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");
        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        if constexpr(Policy::kUseOutputFragments)
        {
            using PvBlockGemm   = remove_cvref_t<decltype(gemm_1)>;
            using PvWarpGemm    = typename PvBlockGemm::WarpGemm;
            using PvCWarpTensor = typename PvWarpGemm::CWarpTensor;
            static_assert(PvBlockGemm::KIterPerWarp == 1 && PvBlockGemm::MIterPerWarp == 2 &&
                          PvBlockGemm::NIterPerWarp == 8);
            static_assert(std::is_same_v<OaccDataType, float>);
            static_assert(std::is_same_v<remove_cvref_t<typename PvCWarpTensor::DataType>, float>);
            static_assert(std::is_same_v<typename Policy::OutputFragments::Fragment, fp32x8_t>);
            static_assert(PvCWarpTensor::get_thread_buffer_size() ==
                          Policy::OutputFragments::kElementsPerFragment);
            static_assert(Policy::OutputFragments::kNumDmsb == 2 * PvBlockGemm::MIterPerWarp);
            static_assert(Policy::OutputFragments::kNumN == PvBlockGemm::NIterPerWarp / 2);
            static_assert(OaccBlockTileType::get_thread_buffer_size() ==
                          Policy::OutputFragments::Mapping::kThreadBufferSize);
            static_assert(
                std::is_same_v<typename SaccBlockTileType::StaticTileDistribution,
                               typename OaccBlockTileType::StaticTileDistribution>,
                "D192 score/output distributions must preserve the fragment reconstruction map");
        }

        auto o_acc = [&]() {
            if constexpr(Policy::kUseOutputFragments)
            {
                return Policy::OutputFragments::MakeZero();
            }
            else
            {
                return OaccBlockTileType{};
            }
        }();

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(OaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // init M, L (sink-aware)
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        if constexpr(!Policy::kUseOutputFragments)
        {
            clear_tile(o_acc);
        }
        if(__builtin_isinf_sign(sink_v) >= 0)
        {
#if CK_TILE_FMHA_FWD_FAST_EXP2
            if constexpr(kHasLogitsSoftCap)
                set_tile(m, sink_v * scale_s * C_LOG2E);
            else
                set_tile(m, sink_v * C_LOG2E);
#else
            set_tile(m, sink_v);
#endif
            set_tile(l, SMPLComputeDataType{1.0f});
        }
        else
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        const auto tile_range_result = [&mask, &q_origin]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(
                    q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
            else
            {
                auto [start, end] =
                    mask.GetTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end           = tile_range_result.get(ck_tile::number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) +
                num_sink_loop;
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse_acc =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    if(__builtin_isinf_sign(sink_v) >= 0)
                    {
                        set_tile(lse_acc, SMPLComputeDataType{sink_v * scale_s});
                    }
                    else
                    {
                        set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
                    }

                    store_tile(lse_acc_dram_window_tmp, lse_acc);
                }

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                if constexpr(Policy::kUseOutputFragments)
                {
                    return Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
                }
                else
                {
                    return o_acc;
                }
            }
        }

        // ---------------------------------------------------------------------
        // TDM configs for Q / K / V
        // pad_enable + pad_amount + pad_interval are compile-time, sourced from
        // the policy. workgroup_mask defaults to 0 (no cluster multicast).
        // V uses load_tile_tdm (single-box plain layout) -- same TDM machinery
        // as Q / K, with V dram dist switched to trivial tile-major and the
        // V LDS read view kept plain row-major (matches the write view).
        // ---------------------------------------------------------------------
        TDMConfig tdm_config_q;
        TDMConfig tdm_config_k;
        TDMConfig tdm_config_v;
        {
            constexpr auto LdsPaddingConfigQ     = Policy::template GetLdsPaddingConfigQ<Problem>();
            tdm_config_q.pad_enable              = LdsPaddingConfigQ[I0];
            tdm_config_q.pad_config.pad_amount   = LdsPaddingConfigQ[I1];
            tdm_config_q.pad_config.pad_interval = LdsPaddingConfigQ[number<2>{}];

            constexpr auto LdsPaddingConfigK     = Policy::template GetLdsPaddingConfigK<Problem>();
            tdm_config_k.pad_enable              = LdsPaddingConfigK[I0];
            tdm_config_k.pad_config.pad_amount   = LdsPaddingConfigK[I1];
            tdm_config_k.pad_config.pad_interval = LdsPaddingConfigK[number<2>{}];

            constexpr auto LdsPaddingConfigV     = Policy::template GetLdsPaddingConfigV<Problem>();
            tdm_config_v.pad_enable              = LdsPaddingConfigV[I0];
            tdm_config_v.pad_config.pad_amount   = LdsPaddingConfigV[I1];
            tdm_config_v.pad_config.pad_interval = LdsPaddingConfigV[number<2>{}];
        }

        // Q tile in LDS
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        auto q_lds_write_view = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptrk0),
            Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_read_view = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptrk0),
            Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window =
            make_tile_window(q_lds_write_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_read_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
        s_wait_tensorcnt_barrier<0>();
        auto q_tile = load_tile(q_lds_read_window);

        // K tile in LDS (sink-aware start)
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;
        const index_t physical_seqlen_k_start = logical_seqlen_k_start;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end;

        // Bias tile window (prefill path)
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             gemm_0.MakeCBlockTile().get_tile_distribution());

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());

        auto k_lds_write_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType* __restrict__>(smem_ptrk0),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType* __restrict__>(smem_ptrk0),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_write_window = make_tile_window(
            k_lds_write_view,
            Policy::template MakeKLdsBlockDescriptor<Problem, true>().get_lengths(),
            {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_read_view,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());

        // S tile in LDS
        auto s_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SaccDataType*>(reinterpret_cast<char*>(smem_ptrk0) +
                                            Policy::template GetSmemSizeK<Problem>()),
            Policy::template MakeSLdsBlockDescriptor<Problem>());
        auto s_write_lds_window = make_tile_window(
            s_lds, Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto s_read_lds_window =
            make_tile_window(s_lds,
                             Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeSRegTileDistribution<Problem>());

        // V tile in LDS (sink-aware start)
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds_write_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(smem_ptrv0)),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_read_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(smem_ptrv0)),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds_write_view,
                             Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_read_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

        constexpr bool kUseCountdownLoop = kQKHeaddim == 128 && !FmhaMask::IsMasking && !kHasSink &&
                                           BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
        bool even_buffer           = true;
        index_t i_total_loops      = 0;
        index_t remaining_loops    = num_total_loop;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);
        block_sync_lds<0>();
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);

        move_tile_window(k_dram_window, {kN0, 0});
        k_lds_write_window.set_bottom_tensor_view_data_ptr(
            static_cast<KDataType* __restrict__>(smem_ptrk1));
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        constexpr index_t k_lds_insts = k_lds_read_window.get_num_of_access();
        constexpr index_t v_lds_insts = v_lds_read_window.get_num_of_access();
        static_assert(k_lds_insts <= Geometry::kKSuLoadCount &&
                      v_lds_insts == Geometry::kVStageLoadCount);

        s_wait_tensorcnt_barrier<Policy::kKPrefetchTensorCount>();
        auto k_tile = [&]() {
            if constexpr(Policy::kUseFullHeadKSuQk)
                return null_tensor{};
            else
                return load_tile(k_lds_read_window);
        }();
        auto k_su0_tile = [&]() {
            if constexpr(Policy::kUseCustomQkStageSchedule)
            {
                auto k_lds_su0_read_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                    {0, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                k_lds_su0_read_window.set_bottom_tensor_view_data_ptr(
                    static_cast<KDataType* __restrict__>(smem_ptrk0));
                return load_tile(k_lds_su0_read_window);
            }
            else
            {
                return null_tensor{};
            }
        }();

        __builtin_amdgcn_sched_barrier(0);

        auto mainloop = [&](KDataType* __restrict__ k_lds_write_ptr,
                            KDataType* __restrict__ k_lds_read_ptr,
                            KDataType* __restrict__ v_lds_write_ptr,
                            KDataType* __restrict__ v_lds_read_ptr) {
            auto current_p_tile = make_static_distributed_tensor<PDataType>(
                Policy::template MakePRegTileDistribution<Problem>());

            const auto qk_stage0_base = [&]() {
                if constexpr(Geometry::kHeadDimQK == 128 && Policy::kUseCustomQkStageSchedule)
                {
                    auto next_su_window = make_tile_window(
                        k_lds_read_view,
                        make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                        {Geometry::kQkSuColumns, 0},
                        Policy::template MakeKSuRegTileDistribution<Problem>());
                    next_su_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                    return FmhaN128PreparedKProbe::Prepare<0>(next_su_window);
                }
                else
                {
                    return FmhaN128PreparedKProbe::Address{};
                }
            }();
            __builtin_amdgcn_sched_barrier(0);

            // Reuse the V LDS buffer only after the preceding reads have completed.
            block_sync_lds<k_lds_insts>();
            move_tile_window(v_dram_window, {kN0, 0});
            v_lds_write_window.set_bottom_tensor_view_data_ptr(v_lds_write_ptr);
            if constexpr(kUseCountdownLoop && kN0 == 128 && kN1 == 128 &&
                         std::is_same_v<VDataType, bf16_t>)
            {
                v_dram_window.tdm_load_to_lds(
                    tdm_config_v,
                    v_lds_write_window,
                    make_null_tile_window(v_dram_window.get_window_lengths()),
                    number<-1>{},
                    bool_constant<true>{});
            }
            else
            {
                load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
            }

            decltype(load_tile_transpose(v_lds_read_window)) v_tile;
            constexpr bool kStreamOutputRescale = std::is_same_v<Geometry, LegacyD192Geometry> &&
                                                  Policy::kUseOutputFragments &&
                                                  Policy::kUseSplitSoftmax;
            [[maybe_unused]] float pv_output_scale_m0 = 1.0f;
            [[maybe_unused]] float pv_output_scale_m1 = 1.0f;
            if constexpr(Policy::kUseCustomQkStageSchedule)
            {
                v_lds_read_window.set_bottom_tensor_view_data_ptr(v_lds_read_ptr);
            }

            auto run_fragment_pv_stage = [&](auto stage,
                                             const auto& a_block_tensor,
                                             const auto& b_block_tensor,
                                             auto& next_b_block_tensor,
                                             const auto& b_lds_window) {
                if constexpr(Policy::kUseOutputFragments)
                {
                    using BlockGemm   = remove_cvref_t<decltype(gemm_1)>;
                    using WarpGemm    = typename BlockGemm::WarpGemm;
                    using AWarpDstr   = typename WarpGemm::AWarpDstr;
                    using BWarpDstr   = typename WarpGemm::BWarpDstr;
                    using AWarpTensor = typename WarpGemm::AWarpTensor;
                    using BWarpTensor = typename WarpGemm::BWarpTensor;
                    using CWarpTensor = typename WarpGemm::CWarpTensor;
                    constexpr auto a_warp_y_lengths =
                        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                    constexpr auto b_warp_y_lengths =
                        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                    constexpr auto a_warp_y_index_zeros =
                        uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
                    constexpr auto b_warp_y_index_zeros =
                        uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};

                    auto rescale_fragment = [&](auto ordinal) {
                        constexpr index_t d_msb =
                            decltype(ordinal)::value / Policy::OutputFragments::kNumN;
                        const float scale = d_msb < 2 ? pv_output_scale_m0 : pv_output_scale_m1;
                        auto& fragment    = o_acc.at(ordinal);
                        static_for<0, Policy::OutputFragments::kElementsPerFragment, 1>{}(
                            [&](auto element) { fragment[decltype(element)::value] *= scale; });
                    };
                    if constexpr(kStreamOutputRescale && decltype(stage)::value == 0)
                    {
                        rescale_fragment(number<0>{});
                        rescale_fragment(number<1>{});
                    }

                    auto emit_wmma = [&](auto, auto wmma) {
                        constexpr index_t ordinal     = decltype(wmma)::value;
                        constexpr index_t d_msb       = ordinal / Policy::OutputFragments::kNumN;
                        constexpr index_t n           = ordinal % Policy::OutputFragments::kNumN;
                        constexpr index_t m_iter      = d_msb / 2;
                        constexpr index_t v_msb       = d_msb % 2;
                        constexpr index_t full_n_iter = n * 2 + v_msb;

                        AWarpTensor a_warp_tensor;
                        a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<0, m_iter>{}, a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                        BWarpTensor b_warp_tensor;
                        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<0, full_n_iter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        CWarpTensor c_warp_tensor;
                        c_warp_tensor.get_thread_buffer().template set_as<fp32x8_t>(number<0>{},
                                                                                    o_acc.at(wmma));
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                        o_acc.at(wmma) =
                            c_warp_tensor.get_thread_buffer().template get_as<fp32x8_t>(
                                number<0>{});
                        if constexpr(kStreamOutputRescale && decltype(stage)::value == 0 &&
                                     ordinal + 2 < Policy::OutputFragments::kNumFragments)
                        {
                            // Keep the next accumulator ready two PV fragments ahead.
                            __builtin_amdgcn_sched_barrier(0);
                            rescale_fragment(number<ordinal + 2>{});
                        }
                    };

                    Policy::template RunPvScheduledStageWithWmma<decltype(stage)::value>(
                        next_b_block_tensor, b_lds_window, emit_wmma);
                }
            };

            auto run_pv = [&](const auto& p_tile) {
                if constexpr(1 < k1_loops)
                {
                    static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                        move_tile_window(v_lds_read_window, {kK1, 0});
                        if constexpr(Policy::kUseCustomPvStageSchedule)
                        {
                            decltype(v_tile) v_tile_switch;
                            auto p_tile_stage = get_slice_tile(p_tile,
                                                               sequence<0, i_k1 * kK1>{},
                                                               sequence<kM0, (i_k1 + 1) * kK1>{});
                            if constexpr(Policy::kUseOutputFragments)
                            {
                                run_fragment_pv_stage(
                                    i_k1, p_tile_stage, v_tile, v_tile_switch, v_lds_read_window);
                            }
                            else
                            {
                                Policy::template RunPvScheduledStage<decltype(i_k1)::value>(
                                    gemm_1,
                                    o_acc,
                                    p_tile_stage,
                                    v_tile,
                                    v_tile_switch,
                                    v_lds_read_window);
                            }
                            v_tile = v_tile_switch;
                        }
                        else
                        {
                            auto v_tile_switch = load_tile_transpose(v_lds_read_window);
                            gemm_1(o_acc,
                                   get_slice_tile(p_tile,
                                                  sequence<0, i_k1 * kK1>{},
                                                  sequence<kM0, (i_k1 + 1) * kK1>{}),
                                   v_tile);
                            v_tile = v_tile_switch;
                        }
                    });
                    move_tile_window(v_lds_read_window, {-kK1 * (k1_loops - 1), 0});
                }

                auto p_tile_last = get_slice_tile(
                    p_tile, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, k1_loops * kK1>{});
                if constexpr(Policy::kUseCustomPvStageSchedule)
                {
                    auto k_lds_su0_read_window = make_tile_window(
                        k_lds_read_view,
                        make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                        {0, 0},
                        Policy::template MakeKSuRegTileDistribution<Problem>());
                    k_lds_su0_read_window.set_bottom_tensor_view_data_ptr(k_lds_read_ptr);
                    if constexpr(Policy::kUseOutputFragments)
                    {
                        run_fragment_pv_stage(
                            number<3>{}, p_tile_last, v_tile, k_su0_tile, k_lds_su0_read_window);
                    }
                    else
                    {
                        Policy::template RunPvScheduledStage<3>(
                            gemm_1, o_acc, p_tile_last, v_tile, k_su0_tile, k_lds_su0_read_window);
                    }
                }
                else
                {
                    gemm_1(o_acc, p_tile_last, v_tile);
                }
            };

            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C

            if constexpr(Policy::kUseFullHeadKSuQk)
            {
                constexpr auto gemm_0_su  = Policy::template GetQKBlockGemmSu<Problem>();
                auto k_lds_su_read_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                    {0, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                k_lds_su_read_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                auto s_acc_su = gemm_0_su.MakeCBlockTile();

                if constexpr(Policy::kUseCustomQkStageSchedule)
                {
                    auto k_su_tile = k_su0_tile;
                    static_for<0, Geometry::kQkStages, 1>{}([&](auto i_su) {
                        clear_tile(s_acc_su);
                        if constexpr(decltype(i_su)::value < Geometry::kQkStages - 1)
                        {
                            move_tile_window(k_lds_su_read_window, {Geometry::kQkSuColumns, 0});
                            decltype(k_su_tile) k_su_tile_next;

                            Policy::template RunQkScheduledStage<decltype(i_su)::value>(
                                gemm_0_su,
                                s_acc_su,
                                q_tile,
                                k_su_tile,
                                k_su_tile_next,
                                k_lds_su_read_window,
                                &qk_stage0_base);

                            k_su_tile = k_su_tile_next;
                        }
                        else
                        {

                            Policy::template RunQkScheduledStage<Geometry::kQkStages - 1>(
                                gemm_0_su, s_acc_su, q_tile, k_su_tile, v_tile, v_lds_read_window);
                        }
                        set_slice_tile(
                            s_acc,
                            s_acc_su,
                            sequence<0, decltype(i_su)::value * Geometry::kQkSuColumns>{},
                            sequence<kM0, (decltype(i_su)::value + 1) * Geometry::kQkSuColumns>{});
                    });
                }
                else
                {
                    static_for<0, Geometry::kQkStages, 1>{}([&](auto i_su) {
                        auto k_su_tile = load_tile(k_lds_su_read_window);
                        clear_tile(s_acc_su);
                        Policy::RunQkSu(gemm_0_su, s_acc_su, q_tile, k_su_tile);
                        set_slice_tile(
                            s_acc,
                            s_acc_su,
                            sequence<0, decltype(i_su)::value * Geometry::kQkSuColumns>{},
                            sequence<kM0, (decltype(i_su)::value + 1) * Geometry::kQkSuColumns>{});
                        move_tile_window(k_lds_su_read_window, {Geometry::kQkSuColumns, 0});
                    });
                }
            }
            else
            {
                if constexpr(1 < k0_loops)
                {
                    static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                        // loop over along the [K]ey head dimension
                        move_tile_window(k_lds_read_window, {0, kK0});
                        auto k_tile_switch = load_tile(k_lds_read_window);

                        gemm_0(s_acc,
                               get_slice_tile(q_tile,
                                              sequence<0, i_k0 * kK0>{},
                                              sequence<kM0, (i_k0 + 1) * kK0>{}),
                               k_tile);

                        k_tile = k_tile_switch;
                    });
                    // move back to the origin
                    move_tile_window(k_lds_read_window, {0, -kK0 * (k0_loops - 1)});
                }

                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       k_tile);
            }

            // STAGE 2: scale_s, add bias (prefill path, mirrors baseline)
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                const auto bias_tile = load_tile(bias_dram_window);
                tile_elementwise_inout(
                    [](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        x += type_convert<SaccDataType>(y);
#else
                        x += log2e_v<SaccDataType> * type_convert<SaccDataType>(y);
#endif
                    },
                    s_acc,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto current_k_origin = [&]() {
                    const bool in_sink = (num_sink_loop > i_total_loops);
                    if(in_sink)
                        return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                    else
                        return make_tuple(
                            kN0 * (i_total_loops - num_sink_loop) + physical_seqlen_k_start, 0);
                }();
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = current_k_origin.at(I0) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) *= scale_s;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }

            constexpr bool kDeferTensorReady =
                kQKHeaddim == 128 && !kHasSink && BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                Policy::kUseCustomQkStageSchedule && Policy::kUseOutputFragments &&
                Policy::kUseSplitSoftmax && kNWarp == 1;
            if constexpr(!kDeferTensorReady)
                s_wait_tensorcnt_barrier<Policy::kVPrefetchTensorCount>();
            if constexpr(!Policy::kUseCustomQkStageSchedule)
            {
                v_lds_read_window.set_bottom_tensor_view_data_ptr(v_lds_read_ptr);
                v_tile = load_tile_transpose(v_lds_read_window);
            }

            // Sink-aware k_origin (prefill path)
            const auto k_origin = [&]() {
                if constexpr(kUseCountdownLoop)
                {
                    // V's window already points one prefetched tile ahead.
                    return make_tuple(v_dram_window.get_window_origin().at(I0) - kN0, 0);
                }
                const bool in_sink_phase = (num_sink_loop > i_total_loops);
                if(in_sink_phase)
                    return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                else
                    return make_tuple(
                        kN0 * (i_total_loops - num_sink_loop) + physical_seqlen_k_start, 0);
            }();

            if constexpr(kUseCountdownLoop)
            {
                bool need_split_check = false;
                if constexpr(kHasUnevenSplits)
                {
                    const bool needs_tail_predicate = !Policy::kSkipExactFullTilePredicate ||
                                                      k_origin.at(I0) + kN0 > physical_seqlen_k_end;
                    need_split_check = remaining_loops == 1 && needs_tail_predicate;
                }
                bool need_mask_check = false;
                if constexpr(kPadSeqLenK)
                    need_mask_check = mask.IsEdgeTile(
                        q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});

                // Both original passes assign the same value; preserve their predicate union.
                if(__builtin_expect(need_split_check || need_mask_check, false))
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(I0) + tile_idx.at(I0);
                            const auto col = k_origin.at(I0) + tile_idx.at(I1);
                            return (need_split_check && physical_seqlen_k_end <= col) ||
                                   (need_mask_check && mask.IsOutOfBound(row, col));
                        });
                }
            }
            else
            {
                if constexpr(kHasUnevenSplits)
                {
                    const bool needs_tail_predicate = !Policy::kSkipExactFullTilePredicate ||
                                                      k_origin.at(I0) + kN0 > physical_seqlen_k_end;
                    const bool is_last_loop = kUseCountdownLoop
                                                  ? remaining_loops == 1
                                                  : i_total_loops == (num_total_loop - 1);
                    if(is_last_loop && needs_tail_predicate)
                    {
                        set_tile_if(
                            s_acc,
                            -numeric<SMPLComputeDataType>::infinity(),
                            [&, physical_seqlen_k_end_ = physical_seqlen_k_end](auto tile_idx) {
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);

                                {
                                    return physical_seqlen_k_end_ <= col;
                                }
                            });
                    }
                }

                if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                {
                    bool need_perpixel_check = mask.IsEdgeTile(
                        q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                    if(need_perpixel_check)
                    {
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                if constexpr(kHasSink)
                                    return mask.IsOutOfSinkBound(row, col);
                                else if constexpr((kQKHeaddim == 128 ||
                                                   std::is_same_v<Geometry, LegacyD192Geometry>) &&
                                                  std::is_same_v<
                                                      FmhaMask,
                                                      SimplifiedGenericAttentionMask<true>>)
                                    return mask.IsOutOfBoundUnsigned(row, col);
                                else
                                    return mask.IsOutOfBound(row, col);
                            });
                    }
                }
            }

            // Sink->normal window jump (prefill path)
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1)
                {
                    move_tile_window(k_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
                    move_tile_window(v_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
                    move_tile_window(bias_dram_window, {0, physical_seqlen_k_start - sink_seq_end});
                }
            }

            // move bias window (prefill path)
            move_tile_window(bias_dram_window, {0, kN0});

            // Gemm1
            auto s_new = [&]() {
                if constexpr(kNWarp > 1)
                {
                    auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}

                    store_tile(s_write_lds_window, s);
                    block_sync_lds();
                    return load_tile(s_read_lds_window);
                }
                else
                {
                    return cast_tile<SMPLComputeDataType>(s_acc); // S{j}
                }
            }();

            auto p_tile = [&]() {
                if constexpr(Policy::kUseSplitSoftmax)
                {
                    static_assert(kNWarp == 1);
                    static_assert(std::is_same_v<SaccDataType, float> &&
                                  std::is_same_v<SMPLComputeDataType, float>);
                    if constexpr(Policy::kUseOutputFragments)
                    {
                        if constexpr(kStreamOutputRescale)
                        {
                            float delta_m0;
                            float delta_m1;
                            Policy::template RunSplitSoftmaxPart01<Problem>(
                                s_new, m, scale_s, delta_m0, delta_m1);
                            Policy::template RunSplitSoftmaxPart2AndGetScale<Problem>(
                                s_new,
                                m,
                                l,
                                scale_s,
                                delta_m0,
                                delta_m1,
                                pv_output_scale_m0,
                                pv_output_scale_m1);
                        }
                        else
                        {
                            Policy::template RunSplitSoftmaxFragments<Problem, kDeferTensorReady>(
                                s_new, m, l, o_acc, scale_s);
                        }
                    }
                    else
                    {
                        Policy::template RunSplitSoftmax<Problem>(s_new, m, l, o_acc, scale_s);
                    }
                    return Base::template MakePForGemm1<decltype(gemm_1)>(s_new);
                }
                else
                {
                    auto m_local = block_tile_reduce<SMPLComputeDataType>(
                        s_new,
                        sequence<1>{},
                        f_max,
                        -numeric<SMPLComputeDataType>::infinity()); // m_local =
                                                                    // rowmax(S{j})
                    block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

                    static_for<0, 12, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS_READ
                    });

                    static_for<0, 4, 1>{}([&](auto i) {
                        ignore = i;
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS_READ
                    });

                    const auto m_old = m; // m{j-1}
                    tile_elementwise_inout([](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); },
                                           m,
                                           m_old,
                                           m_local); // m{j}

                    auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                        s_new.get_tile_distribution()); // Pcompute{j}

                    static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                        /// NOTICE: bias might be materialized mask including -inf values,
                        /// need consideration
                        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                     FmhaMask::IsMasking)
                        {
                            return raw_m == -numeric<SMPLComputeDataType>::infinity()
                                       ? type_convert<SMPLComputeDataType>(0.f)
                                       : raw_m;
                        }
                        else
                        {
                            return raw_m;
                        }
                    };

                    constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
                    sweep_tile_span(p_spans[I0], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        auto row_max         = scale_s * get_validated_m(m[i_idx]);
                        sweep_tile_span(p_spans[I1], [&](auto idx1) {
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);
                            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                         BiasEnum == BlockAttentionBiasEnum::ALIBI)
                            {
                                p_compute(i_j_idx) =
                                    exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                            }
                            else
                            {
                                if constexpr(kHasLogitsSoftCap)
                                {
                                    p_compute(i_j_idx) =
                                        exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                                }
                                else
                                {
                                    p_compute(i_j_idx) = exp2(scale_s * s_new[i_j_idx] - row_max);
                                }
                            }
                        });
                    });

                    auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                        p_compute,
                        sequence<1>{},
                        f_sum,
                        SMPLComputeDataType{0}); // rowsum(Pcompute{j})

                    block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

                    // l{j}, Oacc{j}
                    constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
                    sweep_tile_span(o_spans[I0], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        const auto tmp       = [&]() {
                            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                         BiasEnum == BlockAttentionBiasEnum::ALIBI)
                            {
                                return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                            }
                            else
                            {
                                if constexpr(kHasLogitsSoftCap)
                                {
                                    return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                                }
                                else
                                {
                                    auto row_max = scale_s * get_validated_m(m[i_idx]);
                                    return exp2(scale_s * m_old[i_idx] - row_max);
                                }
                            }
                        }();
                        l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                        sweep_tile_span(o_spans[I1], [&](auto idx1) {
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);

                            o_acc(i_j_idx) *= tmp;
                        });
                    });

                    return Base::template MakePForGemm1<decltype(gemm_1)>(p_compute);
                }
            }();

            current_p_tile = p_tile;

            block_sync_lds<v_lds_insts>();
            move_tile_window(k_dram_window, {kN0, 0});
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
            load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

            auto& pv_p_tile = [&]() -> auto& { return current_p_tile; }();

            run_pv(pv_p_tile);

            if constexpr(Policy::kUseFullHeadKSuQk)
            {
                s_wait_tensorcnt_barrier<Policy::kKPrefetchTensorCount>();
            }
            else
            {
                s_wait_tensorcnt_barrier<Policy::kKPrefetchTensorCount>();
                k_lds_read_window.set_bottom_tensor_view_data_ptr(k_lds_read_ptr);
                k_tile = load_tile(k_lds_read_window);

                static_for<0, 12, 1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS_READ
                });

                static_for<0, 4, 1>{}([&](auto i) {
                    ignore = i;
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS_READ
                });
            }
        }; // mainloop

        const index_t num_pipeline_loop = num_total_loop;
        do
        {
            bool is_even_loop    = kUseCountdownLoop ? even_buffer : i_total_loops % 2 == 0;
            auto k_lds_write_ptr = is_even_loop ? static_cast<KDataType* __restrict__>(smem_ptrk0)
                                                : static_cast<KDataType* __restrict__>(smem_ptrk1);
            auto k_lds_read_ptr  = is_even_loop ? static_cast<KDataType* __restrict__>(smem_ptrk1)
                                                : static_cast<KDataType* __restrict__>(smem_ptrk0);
            auto v_lds_write_ptr = is_even_loop ? static_cast<VDataType* __restrict__>(smem_ptrv1)
                                                : static_cast<VDataType* __restrict__>(smem_ptrv0);
            auto v_lds_read_ptr  = is_even_loop ? static_cast<VDataType* __restrict__>(smem_ptrv0)
                                                : static_cast<VDataType* __restrict__>(smem_ptrv1);
            mainloop(k_lds_write_ptr, k_lds_read_ptr, v_lds_write_ptr, v_lds_read_ptr);
            if constexpr(kUseCountdownLoop)
            {
                even_buffer = !even_buffer;
                --remaining_loops;
            }
            else
            {
                i_total_loops++;
            }
        } while(kUseCountdownLoop ? remaining_loops > 0 : i_total_loops < num_pipeline_loop);

        if constexpr(kStoreLSE)
        {
            // store lse acc
            auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
            sweep_tile_span(lse_acc_spans[I0], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                }
                else
                {
                    if constexpr(kHasLogitsSoftCap)
                    {
                        lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse_acc(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                    }
                }
            });

            store_tile(lse_acc_dram_window_tmp, lse_acc);
        }

        // Reconstruct the CK tensor only at the final normalization boundary.
        auto output = [&]() {
            if constexpr(Policy::kUseOutputFragments)
            {
                return Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
            }
            else
            {
                return o_acc;
            }
        }();

        constexpr auto o_spans = decltype(output)::get_distributed_spans();

        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                output(i_j_idx) *= tmp;
            });
        });

        // A partial prefetch wait can leave the terminal look-ahead TDM in flight.
        if constexpr((Policy::kKPrefetchTensorCount != 0 || Policy::kVPrefetchTensorCount != 0) &&
                     Policy::kPrefetchTailDrain)
        {
            s_wait_tensorcnt<0>();
        }

        // The final PV stage prefetches next-iteration K into VGPRs. The final loop iteration has
        // no consumer to force those DS reads to retire, so explicitly drain them before return.
        if constexpr(Policy::kPvStage3TailDsCount != 0)
        {
            s_wait_dscnt<0>();
        }

        return output;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        float sink_v,
                                        void* smem_ptrk0,
                                        void* smem_ptrk1,
                                        void* smem_ptrv0,
                                        void* smem_ptrv1) const
    {
        return run(q_dram_block_window_tmp,
                   k_dram_block_window_tmp,
                   v_dram_block_window_tmp,
                   bias_dram_block_window_tmp,
                   lse_acc_dram_window_tmp,
                   mask,
                   position_encoding,
                   scale_s,
                   smem_ptrk0,
                   smem_ptrk1,
                   smem_ptrv0,
                   smem_ptrv1,
                   sink_v);
    }
};

} // namespace ck_tile
