// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_ldsacc_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// Same algorithm as BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP, with the dK and dV
// accumulators held in LDS instead of registers.
//
// The two fp32 accumulators are kN0*headdim floats each. Kept in registers they
// are live across the whole Q loop, which on gfx1250 (wave32) costs
// kN0*headdim/kBlockSize VGPRs apiece -- 64 each at kN0=64, headdim=128. That is
// what takes the kernel from 2 waves/SIMD to 1: measured 377 VGPRs / occupancy 2
// at headdim 64 (226 TFLOPS) versus 597 VGPRs / occupancy 1 at headdim 128
// (120 TFLOPS), against a forward pipeline reaching 369 TFLOPS on the same part.
//
// Here the running sums live in LDS, which is otherwise 89% idle (36 KiB of the
// 320 KiB a gfx1250 workgroup may take), and each accumulation becomes
// load -> gemm -> store so the register tile is live only around its own gemm.
// The trade is extra LDS traffic once per Q tile; whether that pays has to be
// measured on hardware.
//
// No LDS atomics are needed: gemm_1/gemm_3 distribute C with MWarp warps
// splitting M (=kN0) disjointly and NWarp=1, so every LDS element has exactly
// one owning thread and a plain read-modify-write is race free.
template <typename Problem, typename Policy = BlockFmhaBwdPipelineLdsAccPolicy>
struct BlockFmhaBwdDQDKDVPipelineLdsAccKRKTRVR
{
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType          = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType             = remove_cvref_t<typename Problem::DDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType         = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType         = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType         = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType         = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType      = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using FmhaDropout           = remove_cvref_t<typename Problem::FmhaDropout>;
    using HotLoopScheduler      = typename Policy::template HotLoopScheduler<Problem>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr index_t kPadHeadDimQ  = Problem::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum         = Problem::BiasEnum;
    static constexpr bool kHasBiasGrad     = Problem::kHasBiasGrad;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;
    static constexpr bool kUseTrLoad       = Problem::kUseTrLoad;
    static_assert(!kUseTrLoad, "This pipeline does not use trload!");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentQGrad = 1;
    static constexpr index_t kAlignmentKGrad =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentKGrad<Problem>();
    static constexpr index_t kAlignmentVGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentVGrad<Problem>();
    static constexpr index_t kAlignmentBias = 1;

    static constexpr const char* name = "ldsacc_kr_ktr_vr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp,
              typename BiasGradDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(void* smem_ptr,
               const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
               const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
               const DDramBlockWindowTmp& d_dram_block_window_tmp,
               const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float raw_scale,
               float scale,
               float rp_undrop,
               float scale_rp_undrop,
               FmhaDropout& dropout) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<LSEDataType,
                               remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        // VGrad & KGrad accumulators, in LDS.
        //
        // These sit after every staged region rather than inside the max() over
        // phases, because they are live for the whole Q loop while K/V/Q/dO/dS
        // each die at the end of their phase. Putting them last leaves all the
        // existing staged offsets untouched.
        //
        // Both windows carry the gemm's own C distribution, so the tiles loaded
        // from them can be fed straight back into gemm_1 / gemm_3 (which assert
        // on that distribution) and handed to the epilogue unchanged.
        auto dk_acc_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<AccDataType*>(static_cast<char*>(smem_ptr) +
                                           Policy::template GetKGradAccSmemOffset<Problem>()),
            Policy::template MakeKGradAccLdsBlockDescriptor<Problem>());
        auto dv_acc_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<AccDataType*>(static_cast<char*>(smem_ptr) +
                                           Policy::template GetVGradAccSmemOffset<Problem>()),
            Policy::template MakeVGradAccLdsBlockDescriptor<Problem>());

        auto dk_acc_lds_window =
            make_tile_window(dk_acc_lds,
                             make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                             {0, 0},
                             decltype(gemm_3.MakeCBlockTile())::get_tile_distribution());
        auto dv_acc_lds_window =
            make_tile_window(dv_acc_lds,
                             make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                             {0, 0},
                             decltype(gemm_1.MakeCBlockTile())::get_tile_distribution());

        // K, HBM ->LDS ->Reg
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             k_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());

        const auto k_origin = k_dram_window.get_window_origin();
        // Early termination
        const auto [seqlen_q_start, seqlen_q_end] =
            mask.GetTileRangeAlongY(k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop =
            amd_wave_read_first_lane(integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0));

        // check early exit if no work to do.
        // __builtin_expect is load-bearing: omitting it causes incorrect AGPR allocation in
        // the dK/dV accumulation loop on some compiler versions, leading to wrong results.
        if(__builtin_expect(num_total_loop <= 0, 0))
        {
            // Nothing was accumulated, so hand back zeroed register tiles rather
            // than reading the LDS accumulators (which were never initialised).
            auto dk_zero = decltype(gemm_3.MakeCBlockTile()){};
            auto dv_zero = decltype(gemm_1.MakeCBlockTile()){};
            clear_tile(dk_zero);
            clear_tile(dv_zero);
            return make_tuple(dk_zero, dv_zero);
        }
        KDataType* k_lds_ptr =
            static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());

        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKRegBlockDescriptor<Problem>());

        auto k_reg_tensor = make_static_distributed_tensor<KDataType>(
            Policy::template MakeKRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // V, HBM ->LDS ->Reg
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             v_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeVDramTileDistribution<Problem>());

        // V has a dedicated region past the staged/accumulator blocks -- see
        // GetVSmemOffset. It used to alias K/KT at offset 0.
        VDataType* v_lds_ptr = static_cast<VDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetVSmemOffset<Problem>()));

        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // KT, read transposed straight out of the single K box
        //
        // There used to be a second LDS copy of K here, produced by shuffling
        // k_block_tile in registers, purely so gemm_4 could read K^T with a
        // plain load_tile. ds_load_tr16_b128 does that in hardware, so the
        // shuffle, the staging tile and the copy are all gone; the window below
        // points at the same box k_lds_write_window fills.
        auto kt_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

        // V is moved global->LDS by TDM, so it never lands in registers. pad is
        // disabled by the policy; workgroup_mask stays 0 (no cluster multicast).
        TDMConfig tdm_config_v;
        TDMConfig tdm_config_k;
        TDMConfig tdm_config_q;
        TDMConfig tdm_config_do;
        {
            constexpr auto LdsPaddingConfigV = Policy::template GetLdsPaddingConfigV<Problem>();
            tdm_config_v.pad_enable              = LdsPaddingConfigV[number<0>{}];
            tdm_config_v.pad_config.pad_amount   = LdsPaddingConfigV[number<1>{}];
            tdm_config_v.pad_config.pad_interval = LdsPaddingConfigV[number<2>{}];

            constexpr auto LdsPaddingConfigK = Policy::template GetLdsPaddingConfigK<Problem>();
            tdm_config_k.pad_enable              = LdsPaddingConfigK[number<0>{}];
            tdm_config_k.pad_config.pad_amount   = LdsPaddingConfigK[number<1>{}];
            tdm_config_k.pad_config.pad_interval = LdsPaddingConfigK[number<2>{}];

            constexpr auto LdsPaddingConfigQ = Policy::template GetLdsPaddingConfigQ<Problem>();
            tdm_config_q.pad_enable              = LdsPaddingConfigQ[number<0>{}];
            tdm_config_q.pad_config.pad_amount   = LdsPaddingConfigQ[number<1>{}];
            tdm_config_q.pad_config.pad_interval = LdsPaddingConfigQ[number<2>{}];

            constexpr auto LdsPaddingConfigDO = Policy::template GetLdsPaddingConfigOGrad<Problem>();
            tdm_config_do.pad_enable              = LdsPaddingConfigDO[number<0>{}];
            tdm_config_do.pad_config.pad_amount   = LdsPaddingConfigDO[number<1>{}];
            tdm_config_do.pad_config.pad_interval = LdsPaddingConfigDO[number<2>{}];
        }

        //------------------------------------------------------------------
        // Pre-Load KV: both go global->LDS by TDM, so neither lands in registers
        // on the way. K is issued first and V second, and TENSORcnt retires in
        // issue order, so waiting for "at most one outstanding" below releases K
        // while V is still in flight -- V keeps the overlap it needs, and K does
        // not have to wait behind it.
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);

        // K only: V may still be transferring.
        s_wait_tensorcnt_barrier<1>();
        k_reg_tensor = load_tile(k_lds_read_window);
        block_sync_lds();

        auto kt_reg_tensor = load_tile_transpose(kt_lds_read_window);

        // Now V as well. TDM commits on TENSORcnt, not on the LDS or
        // vector-memory counters, so block_sync_lds alone would not fence it.
        s_wait_tensorcnt_barrier<0>();

        auto v_reg_tensor = load_tile(v_lds_read_window);
        //---------------------------- Loop Load in ----------------------------//
        // Q: HBM ->Reg ->LDS
        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());

        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>()));

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakePTRegSliceBlockDescriptor<Problem>());
        // Q^T: read transposed out of the single Q box. The shuffle and the
        // second LDS copy it fed are gone -- ds_load_tr16_b128 does the
        // transpose in hardware. Q's shuffle ran once per Q-loop iteration, so
        // this removes hot-loop work, unlike K's which was once per block.
        auto qt_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQTRegSliceBlockDescriptor<Problem>());

        // dO: HBM ->Reg ->LDS
        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>()));

        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());

        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto do_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());
        // dO^T: read transposed straight out of the single dO box.
        //
        // There used to be a second LDS copy here, produced by shuffling
        // do_block_tile in registers, so gemm_1 could read dO^T with a plain
        // load_tile. ds_load_tr16_b128 does that in hardware. Unlike K, dO is
        // reloaded every Q iteration, so this shuffle was in the hot loop.
        auto dot_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kVHeaddim>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradTRegSliceBlockDescriptor<Problem>());

        // dS: Reg -> Reg -> LDS
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto ds_lds = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());

        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto ds_lds_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK4>{}),
                             ds_lds_window.get_window_origin(),
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());

        auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
        // Bias: HBM ->Reg ->Reg ->LDS
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();

        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, bias_origin.at(number<1>{})},
                             Policy::template MakeBiasTileDistribution<Problem>());

        BiasDataType* bias_lds_ptr = static_cast<BiasDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto bias_lds = make_tensor_view<address_space_enum::lds>(
            bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());

        auto bias_lds_write_window =
            make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto bias_s_lds_read_window =
            make_tile_window(bias_lds_write_window.get_bottom_tensor_view(),
                             bias_lds_write_window.get_window_lengths(),
                             bias_lds_write_window.get_window_origin(),
                             Policy::template MakeBiasSTileDistribution<decltype(gemm_0)>());

        static_assert(std::is_same_v<BiasDataType, BiasGradDataType>,
                      "BiasDataType and BiasGradDataType should be the same!");

        // LSE: HBM -> LDS ->Reg
        auto lse_dram_window = make_tile_window(
            lse_dram_block_window_tmp.get_bottom_tensor_view(),
            lse_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_start},
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        LSEDataType* lse_lds_ptr = static_cast<LSEDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>()));

        auto lse_lds = make_tensor_view<address_space_enum::lds>(
            lse_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto lse_lds_write_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0});

        auto lse_lds_read_window = make_tile_window(
            lse_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        // D: HBM ->Reg
        auto d_dram_window = make_tile_window(
            d_dram_block_window_tmp.get_bottom_tensor_view(),
            d_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_start},
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        DDataType* d_lds_ptr = static_cast<DDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>()));

        auto d_lds = make_tensor_view<address_space_enum::lds>(
            d_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto d_lds_write_window = make_tile_window(d_lds, make_tuple(number<kM0>{}), {0});

        auto d_lds_read_window = make_tile_window(
            d_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        // RandVal: HBM ->Reg
        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0), false>(
            randval_dram_block_window_tmp, seqlen_q_start);

        // BiasGrad
        // Reg ->LDS ->Reg ->HBM
        const auto dbias_origin = dbias_dram_block_window_tmp.get_window_origin();

        auto dbias_dram_window =
            make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                             dbias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, dbias_origin.at(number<1>{})}); // M/N

        auto dbias_lds_read_window =
            make_tile_window(bias_lds,
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeShuffledBiasTileDistribution<Problem>());

        // ----------------------------Loop write out------------------------------//
        auto dq_dram_window = make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dq_dram_block_window_tmp.get_window_lengths(),
                                               {seqlen_q_start, 0});

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType  = decltype(gemm_4.MakeCBlockTile());

        index_t i_total_loops = 0;
        index_t seqlen_q_step = seqlen_q_start;
        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kM0 == kK1, "kM0 should equal to kK1");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        static_assert(kM0 == kK3, "kM0 should equal to kK3");
        constexpr index_t k4_loops = kN0 / kK4;

        /*
         * Prefetch Q, LSE, dO, D
         */
        // Q and dO go global -> LDS by TDM, so nothing is prefetched into
        // registers here. Their DRAM windows are advanced only after the
        // transfer has been issued, because TDM reads at issue time whereas the
        // old load_tile read before the advance.
        auto lse_block_tile = load_tile(lse_dram_window);
        move_tile_window(lse_dram_window, {kM0});

        auto d_block_tile = load_tile(d_dram_window);
        move_tile_window(d_dram_window, {kM0});

        /*
         * Store prefetched data into LDS
         */
        block_sync_lds();
        load_tile_tdm(tdm_config_q, q_lds_window, q_dram_window);
        move_tile_window(q_dram_window, {kM0, 0});

        store_tile(lse_lds_write_window, lse_block_tile);

        load_tile_tdm(tdm_config_do, do_lds_window, do_dram_window);
        move_tile_window(do_dram_window, {kM0, 0});

        store_tile(d_lds_write_window, d_block_tile);
        // Q and dO now arrive by TDM, which commits on TENSORcnt; block_sync_lds
        // only covers the LSE/D stores that still go through dscnt.
        s_wait_tensorcnt_barrier<0>();
        block_sync_lds();

        /*
         * Prefetch LDS data into Reg to Asynchronous Data Movement and MFMA pipeline
         */

        auto q_reg_tensor  = load_tile(q_lds_read_window);
        auto lse           = load_tile(lse_lds_read_window);
        auto do_reg_tensor = load_tile(do_lds_read_window);
        auto d             = load_tile(d_lds_read_window);

        // Zero the LDS accumulators. No barrier: each element is written and
        // later read-modify-written by the same thread, so there is no sharing
        // to synchronise.
        {
            auto dk_zero = decltype(gemm_3.MakeCBlockTile()){};
            clear_tile(dk_zero);
            store_tile(dk_acc_lds_window, dk_zero);

            auto dv_zero = decltype(gemm_1.MakeCBlockTile()){};
            clear_tile(dv_zero);
            store_tile(dv_acc_lds_window, dv_zero);
        }

        __builtin_amdgcn_sched_barrier(0);
        // Hot loop
        while(i_total_loops < (num_total_loop - 1))
        {
            // STAGE 1, Q@K Gemm0
            auto s_acc = SPBlockTileType{};

            lse_block_tile = load_tile(lse_dram_window);
            move_tile_window(lse_dram_window, {kM0});

            d_block_tile = load_tile(d_dram_window);
            move_tile_window(d_dram_window, {kM0});

            s_acc = gemm_0(q_reg_tensor, k_reg_tensor);

            auto dot_reg_tensor = load_tile_transpose(dot_lds_read_window);

            HotLoopScheduler::template GemmStagedScheduler<0>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const auto bias_tile    = load_tile(bias_dram_window);
                auto shuffled_bias_tile = make_static_distributed_tensor<BiasDataType>(
                    Policy::template MakeShuffledBiasTileDistribution<Problem>());
                shuffle_tile(shuffled_bias_tile, bias_tile);
                // SGrad and Bias use the same address in LDS, finish loading ds on the previous
                // iteration to reuse LDS.
                block_sync_lds();
                store_tile(bias_lds_write_window, shuffled_bias_tile);
                block_sync_lds();
                auto bias_s_tile = load_tile(bias_s_lds_read_window);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
                        x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                    },
                    s_acc,
                    bias_s_tile);
                move_tile_window(bias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }

            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }

            static const auto get_validated_lse = [](LSEDataType raw_lse) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_lse == -numeric<LSEDataType>::infinity()
                               ? type_convert<LSEDataType>(0.f)
                               : raw_lse;
                }
                else
                {
                    return raw_lse;
                }
            };

            auto p                 = SPBlockTileType{};
            constexpr auto p_spans = decltype(p)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p(i_j_idx) = exp2(s_acc[i_j_idx] - row_lse);
                    }
                    else
                    {
                        p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);
                    }
                });
            });

            if constexpr(FmhaDropout::IsDropout)
            {
                dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                    seqlen_q_step, k_origin.at(number<0>{}), p, randval_dram_window);
            }
            const auto p_gemm = [&]() {
                if constexpr(FmhaDropout::IsDropout)
                {
                    return tile_elementwise_in(
                        [](const auto& x) { return type_convert<GemmDataType>(x > 0.f ? x : 0.f); },
                        p);
                }
                else
                {
                    return cast_tile<GemmDataType>(p);
                }
            }();

            // STAGE 3, P^T@OGrad^T Gemm1
            Policy::template PTFromGemm0CToGemm1A<Problem>(pt_reg_tensor, p_gemm);
            {
                // Running sum comes from LDS and goes straight back, so the
                // register tile is live only across its own gemm.
                auto dv_acc = load_tile(dv_acc_lds_window);
                gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);
                store_tile(dv_acc_lds_window, dv_acc);
            }

            auto qt_reg_tensor = load_tile_transpose(qt_lds_read_window);

            HotLoopScheduler::template GemmStagedScheduler<1>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 4, OGrad@V Gemm2
            auto dp_acc = SPGradBlockTileType{};

            dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);

            block_sync_lds();

            load_tile_tdm(tdm_config_q, q_lds_window, q_dram_window);
            move_tile_window(q_dram_window, {kM0, 0});

            store_tile(lse_lds_write_window, lse_block_tile);

            load_tile_tdm(tdm_config_do, do_lds_window, do_dram_window);
            move_tile_window(do_dram_window, {kM0, 0});

            store_tile(d_lds_write_window, d_block_tile);
            // same as the prologue: Q/dO are on TENSORcnt now
            s_wait_tensorcnt_barrier<0>();

            HotLoopScheduler::template GemmStagedScheduler<2>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 5, P^T(PGrad^T - D)
            auto ds                 = SPGradBlockTileType{};
            constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
            sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    bool undrop_flag       = p[i_j_idx] >= 0;
                    ds(i_j_idx)            = p[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                               ? (dp_acc[i_j_idx] - d[i_idx])
                                                               : d[i_idx]);
                });
            });

            if constexpr(kHasBiasGrad)
            {
                const auto dbias = [&]() {
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        return tile_elementwise_in(
                            [&rp_undrop](const auto& x) {
                                return type_convert<BiasGradDataType>(x * rp_undrop);
                            },
                            ds);
                    }
                    else
                    {
                        return cast_tile<BiasGradDataType>(ds);
                    }
                }();
                store_tile(bias_lds_write_window, dbias);
                block_sync_lds();
                auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
                auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                    Policy::template MakeBiasTileDistribution<Problem>());
                shuffle_tile(dbias_tile, shuffled_dbias_tile);
                store_tile(dbias_dram_window, dbias_tile);
                move_tile_window(dbias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }

            // STAGE 6, SGrad^T@Q^T Gemm3
            const auto ds_gemm = cast_tile<GemmDataType>(ds);

            Policy::template SGradTFromGemm2CToGemm3A<Problem>(dst_reg_tensor, ds_gemm);

            {
                auto dk_acc = load_tile(dk_acc_lds_window);
                gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
                store_tile(dk_acc_lds_window, dk_acc);
            }

            if constexpr(kHasBiasGrad)
            {
                // SGrad and BiasGrad use the same address in LDS.
                block_sync_lds();
            }
            store_tile(ds_lds_window, ds_gemm);

            block_sync_lds();

            auto ds_reg_tensor      = load_tile(ds_lds_read_window);
            auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
            move_tile_window(ds_lds_read_window, {0, kK4});
            q_reg_tensor = load_tile(q_lds_read_window);
            lse          = load_tile(lse_lds_read_window);

            HotLoopScheduler::template GemmStagedScheduler<3>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE7 SGrad@K^T Gemm4
            auto dq_acc = QGradBlockTileType{};
            clear_tile(dq_acc);

            static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor_next = load_tile(ds_lds_read_window);
                    move_tile_window(ds_lds_read_window, {0, kK4});
                }
                auto kt_reg_tensor_slice = get_slice_tile(kt_reg_tensor,
                                                          sequence<0, i_k4 * kK4>{},
                                                          sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);

                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
                }
            });
            move_tile_window(ds_lds_read_window, {0, -kN0});

            do_reg_tensor = load_tile(do_lds_read_window);
            d             = load_tile(d_lds_read_window);

            HotLoopScheduler::template GemmStagedScheduler<4>();

            // QGrad Scale
            if constexpr(FmhaDropout::IsDropout)
            {
                tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                       dq_acc);
            }
            else
            {
                tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            }
            if constexpr(decltype(dq_dram_window)::BottomTensorView::DstInMemOp ==
                         memory_operation_enum::set)
            {
                store_tile(dq_dram_window, dq_acc);
            }
            else
            {
                update_tile(dq_dram_window, dq_acc);
            }
            move_tile_window(dq_dram_window, {kM0, 0});

            i_total_loops += 1;
            seqlen_q_step += kM0;
        }
        __builtin_amdgcn_sched_barrier(0);

        // Tail
        auto s_acc = SPBlockTileType{};

        // STAGE 1, Q@K Gemm0
        s_acc = gemm_0(q_reg_tensor, k_reg_tensor);

        // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            const auto bias_tile    = load_tile(bias_dram_window);
            auto shuffled_bias_tile = make_static_distributed_tensor<BiasDataType>(
                Policy::template MakeShuffledBiasTileDistribution<Problem>());
            shuffle_tile(shuffled_bias_tile, bias_tile);
            // SGrad and Bias use the same address in LDS, finish loading ds in the hot loop to
            // reuse LDS.
            block_sync_lds();
            store_tile(bias_lds_write_window, shuffled_bias_tile);
            block_sync_lds();
            auto bias_s_tile = load_tile(bias_s_lds_read_window);
            tile_elementwise_inout(
                [&](auto& x, const auto& y) {
                    x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                },
                s_acc,
                bias_s_tile);
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
            sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                    const auto row         = seqlen_q_step + tile_idx.at(number<0>{});
                    const auto col         = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    s_acc(i_j_idx) *= scale;
                    position_encoding.update(s_acc(i_j_idx), row, col);
                });
            });
        }

        {
            bool need_perpixel_check = mask.IsEdgeTile(
                seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
            if(need_perpixel_check)
            {
                set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                    const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                    const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    return mask.IsOutOfBound(row, col);
                });
            }
        }

        static const auto get_validated_lse = [](LSEDataType raw_lse) {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         FmhaMask::IsMasking)
            {
                return raw_lse == -numeric<LSEDataType>::infinity() ? type_convert<LSEDataType>(0.f)
                                                                    : raw_lse;
            }
            else
            {
                return raw_lse;
            }
        };

        auto p                 = SPBlockTileType{};
        constexpr auto p_spans = decltype(p)::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    p(i_j_idx) = exp2(s_acc[i_j_idx] - row_lse);
                }
                else
                {
                    p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);
                }
            });
        });

        if constexpr(FmhaDropout::IsDropout)
        {
            dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                seqlen_q_step, k_origin.at(number<0>{}), p, randval_dram_window);
        }

        // STAGE 3, P^T@OGrad^T Gemm1
        const auto p_gemm = [&]() {
            if constexpr(FmhaDropout::IsDropout)
            {
                return tile_elementwise_in(
                    [](const auto& x) { return type_convert<GemmDataType>(x > 0.f ? x : 0.f); }, p);
            }
            else
            {
                return cast_tile<GemmDataType>(p);
            }
        }();

        Policy::template PTFromGemm0CToGemm1A<Problem, decltype(pt_reg_tensor), decltype(p_gemm)>(
            pt_reg_tensor, p_gemm);
        auto dot_reg_tensor = load_tile_transpose(dot_lds_read_window);
        {
            auto dv_acc = load_tile(dv_acc_lds_window);
            gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);
            store_tile(dv_acc_lds_window, dv_acc);
        }

        HotLoopScheduler::template GemmStagedScheduler<1>();
        __builtin_amdgcn_sched_barrier(0);

        // STAGE 4, OGrad@V Gemm2
        auto dp_acc = SPGradBlockTileType{};

        auto qt_reg_tensor = load_tile_transpose(qt_lds_read_window);

        dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);

        HotLoopScheduler::template GemmStagedScheduler<2>();
        __builtin_amdgcn_sched_barrier(0);

        // STAGE 5, P^T(PGrad^T - D)
        auto ds                 = SPGradBlockTileType{};
        constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                bool undrop_flag       = p[i_j_idx] >= 0;
                ds(i_j_idx)            = p[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                           ? (dp_acc[i_j_idx] - d[i_idx])
                                                           : d[i_idx]);
            });
        });

        if constexpr(kHasBiasGrad)
        {
            const auto dbias = [&]() {
                if constexpr(FmhaDropout::IsDropout)
                {
                    return tile_elementwise_in(
                        [&rp_undrop](const auto& x) {
                            return type_convert<BiasGradDataType>(x * rp_undrop);
                        },
                        ds);
                }
                else
                {
                    return cast_tile<BiasGradDataType>(ds);
                }
            }();
            // Finish loading bias_s to reuse LDS.
            block_sync_lds();
            store_tile(bias_lds_write_window, dbias);
            block_sync_lds();
            auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
            auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                Policy::template MakeBiasTileDistribution<Problem>());
            shuffle_tile(dbias_tile, shuffled_dbias_tile);
            store_tile(dbias_dram_window, dbias_tile);
            __builtin_amdgcn_sched_barrier(0);
        }

        // STAGE 6, SGrad^T@Q^T Gemm3
        const auto ds_gemm = cast_tile<GemmDataType>(ds);

        Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                  decltype(dst_reg_tensor),
                                                  decltype(ds_gemm)>(dst_reg_tensor, ds_gemm);

        {
            auto dk_acc = load_tile(dk_acc_lds_window);
            gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
            store_tile(dk_acc_lds_window, dk_acc);
        }

        // SGrad and Bias/BiasGrad use the same address in LDS, finish loading bias/dbias or, when
        // bias is not used, loading ds in the hot loop to reuse LDS.
        block_sync_lds();
        store_tile(ds_lds_window, ds_gemm);

        block_sync_lds();

        auto ds_reg_tensor      = load_tile(ds_lds_read_window);
        auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
        move_tile_window(ds_lds_read_window, {0, kK4});

        HotLoopScheduler::template GemmStagedScheduler<3>();
        __builtin_amdgcn_sched_barrier(0);
        // STAGE 7, SGrad@K^T Gemm4
        auto dq_acc = QGradBlockTileType{};
        clear_tile(dq_acc);

        static_for<0, k4_loops, 1>{}([&](auto i_k4) {
            if constexpr(i_k4 < k4_loops - 1)
            {
                ds_reg_tensor_next = load_tile(ds_lds_read_window);
                move_tile_window(ds_lds_read_window, {0, kK4});
            }
            auto kt_reg_tensor_slice = get_slice_tile(
                kt_reg_tensor, sequence<0, i_k4 * kK4>{}, sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});

            gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);
            if constexpr(i_k4 < k4_loops - 1)
            {
                ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
            }
        });

        HotLoopScheduler::template GemmStagedScheduler<4>();
        __builtin_amdgcn_sched_barrier(0);

        // Pull the finished accumulators out of LDS. They come back with the
        // gemm C distribution, which is what the epilogue already expects, so
        // the return type is unchanged from the register-resident pipeline.
        auto dk_acc = load_tile(dk_acc_lds_window);
        auto dv_acc = load_tile(dv_acc_lds_window);

        // Results Scale
        if constexpr(FmhaDropout::IsDropout)
        {
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dq_acc);
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dk_acc);
            tile_elementwise_inout([&rp_undrop](auto& x) { x = x * rp_undrop; }, dv_acc);
        }
        else
        {
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
        }

        if constexpr(decltype(dq_dram_window)::BottomTensorView::DstInMemOp ==
                     memory_operation_enum::set)
        {
            store_tile(dq_dram_window, dq_acc);
        }
        else
        {
            update_tile(dq_dram_window, dq_acc);
        }

        return make_tuple(dk_acc, dv_acc);
    }
};

} // namespace ck_tile
