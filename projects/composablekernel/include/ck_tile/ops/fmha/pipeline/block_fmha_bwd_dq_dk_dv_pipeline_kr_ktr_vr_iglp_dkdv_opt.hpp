// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLPDKDVOpt
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

    static_assert(kM0 == 32 && kN0 == 32 && kQKHeaddim == 128 && kVHeaddim == 128,
                  "The DK/DV-only pipeline is specialized for the D128 product path");

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

    static constexpr const char* name = "kr_ktr_vr_iglp_dkdv_opt";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // DK/DV-only pipeline:
        // - K and V reuse the same LDS base, and K^T was removed.
        // - dS stays in registers for Gemm3, so no dS LDS is needed.
        // Reserve only the LDS regions this pipeline actually addresses.
        constexpr index_t smem_size_q = Policy::template GetSmemSizeQ<Problem>();

        constexpr index_t smem_size_qt = Policy::template GetSmemSizeQT<Problem>();

        constexpr index_t smem_size_lse = Policy::template GetSmemSizeLSE<Problem>();

        constexpr index_t smem_size_k = Policy::template GetSmemSizeK<Problem>();

        constexpr index_t smem_size_v = Policy::template GetSmemSizeV<Problem>();

        constexpr index_t smem_size_do = Policy::template GetSmemSizeOGrad<Problem>();

        constexpr index_t smem_size_dot = Policy::template GetSmemSizeOGradT<Problem>();

        constexpr index_t smem_size_d = Policy::template GetSmemSizeD<Problem>();

        constexpr index_t smem_size_bias = Policy::template GetSmemSizeBias<Problem>();

        constexpr index_t smem_size_kv = max(smem_size_k, smem_size_v);

        // Raw Q and Q^T reuse the same LDS backing at disjoint times.
        constexpr index_t smem_size_qloop = smem_size_qt + smem_size_do + smem_size_dot +
                                            smem_size_lse + smem_size_d + smem_size_bias;

        // Reserve one extra M0 x N0 LDS tile so the Gemm0 C layout can
        // cross wave boundaries before Gemm1 consumes P^T.
        constexpr index_t smem_size_xwave_tile =
            sizeof(remove_cvref_t<typename Problem::GemmDataType>) *
            Policy::template MakeSGradLdsBlockDescriptor<Problem>().get_element_space_size();

        // Keep P and dS in physically separate LDS scratch regions.
        return max(smem_size_kv, smem_size_qloop + 2 * smem_size_xwave_tile);
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
        // dQ is computed by the companion Q-major kernel, so Gemm4 is omitted.

        // init VGrad & KGrad
        auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
        auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};

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
            // Note: here dk_acc&dv_acc are all cleared, return it
            return make_tuple(dk_acc, dv_acc);
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

        VDataType* v_lds_ptr =
            static_cast<VDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegBlockDescriptor<Problem>());

        // dK/dV do not need the K^T path used for dQ.
        //------------------------------------------------------------------
        // Pre-Load KV into Registers
        auto k_block_tile = load_tile(k_dram_window);
        auto v_block_tile = load_tile(v_dram_window);

        store_tile(k_lds_write_window, k_block_tile);

        block_sync_lds();
        k_reg_tensor = load_tile(k_lds_read_window);

        // K and V reuse the same LDS backing store.
        // Ensure K register load is complete before V overwrites LDS.
        block_sync_lds();

        store_tile(v_lds_write_window, v_block_tile);

        block_sync_lds();

        auto v_reg_tensor = load_tile(v_lds_read_window);
        //---------------------------- Loop Load in ----------------------------//
        // Q: HBM ->Reg ->LDS
        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());

        QDataType* q_lds_ptr =
            static_cast<QDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        // Gemm0 C -> Gemm1 A is not wave-local for the M32/N32 2x2 topology.
        // Store logical P[M,N] to shared LDS, then read the same bytes through
        // an [N,M] descriptor using Gemm1's A distribution.
        constexpr index_t p_xwave_lds_offset = Policy::template GetSmemSizeQT<Problem>() +
                                               Policy::template GetSmemSizeOGrad<Problem>() +
                                               Policy::template GetSmemSizeOGradT<Problem>() +
                                               Policy::template GetSmemSizeLSE<Problem>() +
                                               Policy::template GetSmemSizeD<Problem>() +
                                               Policy::template GetSmemSizeBias<Problem>();

        GemmDataType* p_xwave_lds_ptr = static_cast<GemmDataType*>(
            static_cast<void*>(static_cast<char*>(smem_ptr) + p_xwave_lds_offset));

        // P-ADJACENT-MPAIR:
        // P-only LDS layout for gfx12 M32/N32 BF16.
        //
        // Logical:
        //   A = M >> 1, P = M & 1, B = N >> 3, R = N & 7
        //
        // Physical element offset:
        //   A*64 + (B ^ ((A >> 1) & 3))*16 + R*2 + P
        //
        // This makes P[N,M-even:M-even+2] a naturally aligned BF16x2 pair.
        constexpr auto p_xwave_lds_desc = [&]() {
            if constexpr(kM0 == 32 && kN0 == 32 && sizeof(GemmDataType) == 2)
            {
                // Physical dimensions:
                // H,C,L,G,R,P = 2,4,2,4,8,2
                // A = H*8 + C*2 + L
                constexpr auto p_desc_0 = make_naive_tensor_descriptor(make_tuple(number<2>{},
                                                                                  number<4>{},
                                                                                  number<2>{},
                                                                                  number<4>{},
                                                                                  number<8>{},
                                                                                  number<2>{}),
                                                                       make_tuple(number<512>{},
                                                                                  number<128>{},
                                                                                  number<64>{},
                                                                                  number<16>{},
                                                                                  number<2>{},
                                                                                  number<1>{}),
                                                                       number<2>{},
                                                                       number<1>{});

                // G = B xor C.  Preserve C separately so A can be reconstructed.
                constexpr auto p_desc_xor = transform_tensor_descriptor(
                    p_desc_0,
                    make_tuple(make_pass_through_transform(number<2>{}),
                               make_xor_transform(make_tuple(number<4>{}, number<4>{})),
                               make_pass_through_transform(number<2>{}),
                               make_pass_through_transform(number<8>{}),
                               make_pass_through_transform(number<2>{})),
                    make_tuple(sequence<0>{},
                               sequence<1, 3>{},
                               sequence<2>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<0>{},
                               sequence<1, 3>{},
                               sequence<2>{},
                               sequence<4>{},
                               sequence<5>{}));

                // [H,C,L,P] -> M, [B,R] -> N
                constexpr auto p_desc = transform_tensor_descriptor(
                    p_desc_xor,
                    make_tuple(
                        make_merge_transform_v3_division_mod(
                            make_tuple(number<2>{}, number<4>{}, number<2>{}, number<2>{})),
                        make_merge_transform_v3_division_mod(make_tuple(number<4>{}, number<8>{}))),
                    make_tuple(sequence<0, 1, 2, 5>{}, sequence<3, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                static_assert(p_desc.get_element_space_size() == 32 * 32);
                return p_desc;
            }
            else
            {
                return Policy::template MakeSGradLdsBlockDescriptor<Problem>();
            }
        }();

        constexpr index_t xwave_tile_bytes =
            sizeof(GemmDataType) * p_xwave_lds_desc.get_element_space_size();

        GemmDataType* ds_xwave_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + p_xwave_lds_offset + xwave_tile_bytes));

        auto p_xwave_lds =
            make_tensor_view<address_space_enum::lds>(p_xwave_lds_ptr, p_xwave_lds_desc);

        auto p_xwave_lds_write_window =
            make_tile_window(p_xwave_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        // Keep dS in its own LDS region, but use the validated adjacent-M
        // physical layout. Gemm3 A distribution remains unchanged.
        static_assert(
            p_xwave_lds_desc.get_element_space_size() ==
            Policy::template MakeSGradLdsBlockDescriptor<Problem>().get_element_space_size());

        auto ds_xwave_lds =
            make_tensor_view<address_space_enum::lds>(ds_xwave_lds_ptr, p_xwave_lds_desc);

        auto ds_xwave_lds_write_window =
            make_tile_window(ds_xwave_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        constexpr auto p_xwave_t_lds_desc =
            transform_tensor_descriptor(p_xwave_lds_desc,
                                        make_tuple(make_pass_through_transform(number<kN0>{}),
                                                   make_pass_through_transform(number<kM0>{})),
                                        make_tuple(sequence<1>{}, sequence<0>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        // dS keeps the original SGrad LDS layout.  It used to share the
        // transpose descriptor with P only because both physical layouts
        // were identical.
        constexpr auto ds_xwave_t_lds_desc =
            transform_tensor_descriptor(p_xwave_lds_desc,
                                        make_tuple(make_pass_through_transform(number<kN0>{}),
                                                   make_pass_through_transform(number<kM0>{})),
                                        make_tuple(sequence<1>{}, sequence<0>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto pt_xwave_lds =
            make_tensor_view<address_space_enum::lds>(p_xwave_lds_ptr, p_xwave_t_lds_desc);

        auto pt_xwave_lds_read_window =
            make_tile_window(pt_xwave_lds,
                             make_tuple(number<kN0>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakePTRegSliceBlockDescriptor<Problem>());

        // Same physical LDS scratch, but reload using Gemm3 A distribution.
        auto dst_xwave_lds =
            make_tensor_view<address_space_enum::lds>(ds_xwave_lds_ptr, ds_xwave_t_lds_desc);

        auto dst_xwave_lds_read_window =
            make_tile_window(dst_xwave_lds,
                             make_tuple(number<kN0>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
        // QT: Reg -> Reg-> LDS
        auto shuffled_q_block_tile = make_static_distributed_tensor<QDataType>(
            Policy::template MakeShuffledQRegWriteBlockDescriptor<Problem>());

        QDataType* qt_lds_ptr =
            static_cast<QDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto shuffled_q_lds_write = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeShuffledQLdsWriteBlockDescriptor<Problem>());

        auto shuffled_q_lds_write_window = make_tile_window(
            shuffled_q_lds_write, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto qt_lds_read = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeQTLdsReadBlockDescriptor<Problem>());

        auto qt_lds_read_window =
            make_tile_window(qt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kM0>{}),
                             {0, 0},
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
        // dOT: Reg ->Reg ->LDS
        auto shuffled_do_block_tile = make_static_distributed_tensor<OGradDataType>(
            Policy::template MakeShuffledOGradRegWriteBlockDescriptor<Problem>());

        OGradDataType* dot_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>()));

        auto shuffled_do_lds_write = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeShuffledOGradLdsWriteBlockDescriptor<Problem>());

        auto shuffled_do_lds_write_window = make_tile_window(
            shuffled_do_lds_write, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto dot_read_lds = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsReadBlockDescriptor<Problem>());

        auto dot_lds_read_window =
            make_tile_window(dot_read_lds,
                             make_tuple(number<kVHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeOGradTRegSliceBlockDescriptor<Problem>());

        // dS stays in registers for Gemm3; no dS LDS staging is needed.

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
            Policy::template GetSmemSizeLSE<Problem>() + Policy::template GetSmemSizeD<Problem>()));

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
            Policy::template GetSmemSizeOGradT<Problem>()));

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
            Policy::template GetSmemSizeLSE<Problem>()));

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
        // dQ is handled by the companion Q-major kernel.

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());

        // Gemm3 (dS^T @ Q^T -> dK) still needs this register tile.
        auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());

        index_t i_total_loops = 0;
        index_t seqlen_q_step = seqlen_q_start;
        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kM0 == kK1, "kM0 should equal to kK1");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        static_assert(kM0 == kK3, "kM0 should equal to kK3");
        /*
         * Prefetch Q, LSE, dO, D
         */
        auto q_block_tile = load_tile(q_dram_window);
        move_tile_window(q_dram_window, {kM0, 0});
        auto lse_block_tile = load_tile(lse_dram_window);
        move_tile_window(lse_dram_window, {kM0});

        auto do_block_tile = load_tile(do_dram_window);
        move_tile_window(do_dram_window, {kM0, 0});

        /*
         * Store prefetched data into LDS
         */
        block_sync_lds();
        store_tile(q_lds_window, q_block_tile);

        store_tile(lse_lds_write_window, lse_block_tile);

        store_tile(do_lds_window, do_block_tile);
        shuffle_tile(shuffled_do_block_tile, do_block_tile);
        store_tile(shuffled_do_lds_write_window, shuffled_do_block_tile);

        block_sync_lds();

        /*
         * Prefetch LDS data into Reg to Asynchronous Data Movement and MFMA pipeline
         */

        auto q_reg_tensor = load_tile(q_lds_read_window);

        // Q-QT-ALIAS:
        // Every wave has consumed raw Q from LDS.
        // The same bytes may now become Q^T.
        block_sync_lds();

        shuffle_tile(shuffled_q_block_tile, q_block_tile);
        store_tile(shuffled_q_lds_write_window, shuffled_q_block_tile);
        // LSE-AFTER-HANDOFF: LSE uses independent LDS; keep it out of raw-Q read rendezvous.
        __builtin_amdgcn_sched_barrier(0);
        auto lse = load_tile(lse_lds_read_window);

        clear_tile(dv_acc);
        clear_tile(dk_acc);

        __builtin_amdgcn_sched_barrier(0);
        // Hot loop
        while(i_total_loops < (num_total_loop - 1))
        {
            // STAGE 1, Q@K Gemm0
            auto s_acc = SPBlockTileType{};

            s_acc = gemm_0(q_reg_tensor, k_reg_tensor);

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

            // gfx12 R2 baseline:
            // Load dOT after P/exp2, before dropout/P transform/Gemm1.
            auto dot_reg_tensor = load_tile(dot_lds_read_window);

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
            // Cross-wave P transpose through LDS.
            store_tile(p_xwave_lds_write_window, p_gemm);
            block_sync_lds();
            auto pt_xwave_reg_tensor = load_tile(pt_xwave_lds_read_window);
            gemm_1(dv_acc, pt_xwave_reg_tensor, dot_reg_tensor);

            HotLoopScheduler::template GemmStagedScheduler<1>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 4, OGrad@V Gemm2
            auto dp_acc = SPGradBlockTileType{};

            // D-PREFETCH-BEFORE-GEMM2:
            // Issue current-tile D global loads before the dO LDS load/Gemm2.
            // Keep D live until Stage5 so global latency can overlap Gemm2.
            auto d_hot_direct_window = make_tile_window(
                d_dram_block_window_tmp.get_bottom_tensor_view(),
                d_dram_block_window_tmp.get_window_lengths(),
                {seqlen_q_step},
                Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

            auto d_hot_direct = load_tile(d_hot_direct_window);
            __builtin_amdgcn_sched_barrier(0);

            // DO-JIT: leave dO in its independent LDS region until Gemm2.
            auto do_reg_tensor = load_tile(do_lds_read_window);
            __builtin_amdgcn_sched_barrier(0);
            dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);

            // Delay the next-iteration global prefetch to shorten VGPR live ranges.
            HotLoopScheduler::template GemmStagedScheduler<2>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 5, P^T(PGrad^T - D)

            // The earlier HBM->LDS->Reg D staging is intentionally omitted.
            // Current D is loaded directly immediately before Stage 5.
            // Current-tile D, reconstructed from backing view.
            auto ds                 = SPGradBlockTileType{};
            constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
            sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    bool undrop_flag       = p[i_j_idx] >= 0;
                    ds(i_j_idx)            = p[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                               ? (dp_acc[i_j_idx] - d_hot_direct[i_idx])
                                                               : d_hot_direct[i_idx]);
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

            // Cross-wave dS^T redistribution through shared LDS scratch.
            // P has already been consumed by Gemm1, so the buffer can be reused.
            block_sync_lds();
            store_tile(ds_xwave_lds_write_window, ds_gemm);
            // Ensure this wave's LDS writes are completed before
            // all waves rendezvous and begin cross-wave LDS reads.
            __builtin_amdgcn_s_waitcnt(0);
            block_sync_lds();
            auto dst_xwave_hot_reg_tensor = load_tile(dst_xwave_lds_read_window);
            // Ensure LDS reads have retired before Gemm3 consumes the fragment.
            __builtin_amdgcn_s_waitcnt(0);
            // Issue next Q after the existing broad dS read wait,
            // before QT LDS reads/Gemm3. Do not move any LDS commit or barrier.
            q_block_tile = load_tile(q_dram_window);
            move_tile_window(q_dram_window, {kM0, 0});
            // Issue next Q before QT loads/Gemm3.
            // Compiler scheduling boundary only; no runtime memory wait.
            __builtin_amdgcn_sched_barrier(0);

            // Keep Q^T in LDS until Gemm3 actually needs it.
            auto qt_reg_tensor = load_tile(qt_lds_read_window);
            // Keep the compiler from moving the Q^T load across Gemm3.
            __builtin_amdgcn_sched_barrier(0);
            gemm_3(dk_acc, dst_xwave_hot_reg_tensor, qt_reg_tensor);

            HotLoopScheduler::template GemmStagedScheduler<3>();
            __builtin_amdgcn_sched_barrier(0);

            // LSE and dO stay late; next Q alone was issued before Gemm3.

            lse_block_tile = load_tile(lse_dram_window);
            move_tile_window(lse_dram_window, {kM0});

            do_block_tile = load_tile(do_dram_window);
            move_tile_window(do_dram_window, {kM0, 0});

            // All waves have consumed the old Q^T.
            // Commit the already-prefetched next Q/LSE/dO to LDS now.
            block_sync_lds();

            store_tile(q_lds_window, q_block_tile);

            store_tile(lse_lds_write_window, lse_block_tile);

            store_tile(do_lds_window, do_block_tile);
            shuffle_tile(shuffled_do_block_tile, do_block_tile);
            store_tile(shuffled_do_lds_write_window, shuffled_do_block_tile);

            // Publish next-iteration Q/LSE/dO before register reload.
            block_sync_lds();

            // Gemm4/dQ and the original dS LDS round-trip are not needed here.
            // The next Q/LSE/dO were prefetched early and committed after Gemm3.
            q_reg_tensor = load_tile(q_lds_read_window);

            // Q-QT-ALIAS:
            // Every wave has consumed raw next-Q.
            // Reuse those bytes for next-iteration Q^T.
            block_sync_lds();

            shuffle_tile(shuffled_q_block_tile, q_block_tile);
            store_tile(shuffled_q_lds_write_window, shuffled_q_block_tile);
            // LSE-AFTER-HANDOFF: LSE uses independent LDS; keep it out of raw-Q read rendezvous.
            __builtin_amdgcn_sched_barrier(0);
            lse = load_tile(lse_lds_read_window);

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

        // Cross-wave P transpose through LDS.
        store_tile(p_xwave_lds_write_window, p_gemm);
        block_sync_lds();
        auto pt_xwave_tail_reg_tensor = load_tile(pt_xwave_lds_read_window);
        auto dot_reg_tensor           = load_tile(dot_lds_read_window);
        gemm_1(dv_acc, pt_xwave_tail_reg_tensor, dot_reg_tensor);

        HotLoopScheduler::template GemmStagedScheduler<1>();
        __builtin_amdgcn_sched_barrier(0);

        // STAGE 4, OGrad@V Gemm2
        auto dp_acc = SPGradBlockTileType{};

        auto qt_reg_tensor = load_tile(qt_lds_read_window);

        // DO-JIT: leave dO in its independent LDS region until Gemm2.
        auto do_reg_tensor = load_tile(do_lds_read_window);
        dp_acc             = gemm_2(do_reg_tensor, v_reg_tensor);

        HotLoopScheduler::template GemmStagedScheduler<2>();
        __builtin_amdgcn_sched_barrier(0);

        // STAGE 5, P^T(PGrad^T - D)

        // Read D directly from global memory immediately before Stage 5,
        // but use the same register distribution expected by the old
        // d_lds_read_window consumer.
        auto d_direct_dram_window = make_tile_window(
            d_dram_block_window_tmp.get_bottom_tensor_view(),
            d_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_step},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        auto d_direct = load_tile(d_direct_dram_window);

        // Make sure the direct global load is complete before using it.
        __builtin_amdgcn_s_waitcnt(0);

        // Consume the completed HBM D tile directly.
        auto ds                 = SPGradBlockTileType{};
        constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                bool undrop_flag       = p[i_j_idx] >= 0;
                ds(i_j_idx)            = p[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                           ? (dp_acc[i_j_idx] - d_direct[i_idx])
                                                           : d_direct[i_idx]);
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

        // Cross-wave dS^T redistribution through shared LDS scratch.
        // P has already been consumed by Gemm1, so the buffer can be reused.
        block_sync_lds();
        store_tile(ds_xwave_lds_write_window, ds_gemm);
        // Ensure this wave's LDS writes are completed before
        // all waves rendezvous and begin cross-wave LDS reads.
        __builtin_amdgcn_s_waitcnt(0);
        block_sync_lds();
        auto dst_xwave_tail_reg_tensor = load_tile(dst_xwave_lds_read_window);
        // Ensure LDS reads have retired before Gemm3 consumes the fragment.
        __builtin_amdgcn_s_waitcnt(0);
        gemm_3(dk_acc, dst_xwave_tail_reg_tensor, qt_reg_tensor);

        HotLoopScheduler::template GemmStagedScheduler<3>();
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(FmhaDropout::IsDropout)
        {
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dk_acc);
            tile_elementwise_inout([&rp_undrop](auto& x) { x = x * rp_undrop; }, dv_acc);
        }
        else
        {
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
        }

        return make_tuple(dk_acc, dv_acc);
    }
};

} // namespace ck_tile
