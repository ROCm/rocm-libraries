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
struct BlockFmhaBwdDQOnlyQMajor
{
    // gfx1201 STRUCTURAL EXPERIMENT S2-QMAJOR:
    // Q-owned DQ-only probe for BF16/D128/no-bias/no-mask/no-dropout.
    // One workgroup owns one Q tile, loops across all K/V tiles, and writes one full dQ tile.
    static constexpr bool is_qmajor_dq_pipeline = true;

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
    // D32 K^T in an N-contiguous [D][N] LDS layout (b128 reads).
    // also D64 ([D=64][N=64] K^T).
    static constexpr bool kD32DqKtB128 = kM0 == 32 && kN0 == 64 &&
                                         (kQKHeaddim == 32 || kQKHeaddim == 64) &&
                                         sizeof(typename Problem::KDataType) == 2;

    // K-split Gemm4 with dS fed straight from the TransposeC Gemm0/Gemm2 C.
    static constexpr bool kD32DqKsplit = kD32DqKtB128;

    template <typename AT,
              typename BT,
              index_t kN,
              index_t kK,
              typename BlockWarps,
              typename WarpTile>
    CK_TILE_HOST_DEVICE static constexpr auto MakeTransposedCBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<AT,
                             BT,
                             typename Problem::AccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<kM0, kN, kK>, BlockWarps, WarpTile>>;
        using WarpGemm        = WarpGemmDispatcher<AT,
                                                   BT,
                                                   typename Problem::AccDataType,
                                                   WarpTile::at(number<0>{}),
                                                   WarpTile::at(number<1>{}),
                                                   WarpTile::at(number<2>{}),
                                                   true,
                                            WarpTile::at(number<0>{}) == 16 ? false : true>;
        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<AT,
                                                                    BT,
                                                                    typename Problem::AccDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;
        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    // Row (M) distribution of a TransposeC block-gemm C tile: rows on lanes % 16, replicated over
    // the two lane halves and the N warps; M iterations as the only Y dimension.
    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEDReadTransposedC()
    {
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t L     = WG::WarpGemmAttribute::Impl::kCNLane; // 16 rows across lanes
        constexpr index_t R     = WG::WarpGemmAttribute::Impl::kCMLane; // 2 lane halves
        constexpr index_t M0    = kM0 / (MWarp * WG::kM);
        constexpr auto dstr     = make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarp, R>,
                                           tuple<sequence<M0, MWarp, L>>,
                                           tuple<sequence<1, 0>, sequence<0, 1>>,
                                           tuple<sequence<1, 0>, sequence<1, 2>>,
                                           sequence<1>,
                                           sequence<0>>{});
        static_assert(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) == kM0);
        return dstr;
    }

    // K-split dQ: one gfx12 16x16x16 WMMA per (dS block, K^T block); 8 inputs and 8 FP32
    // accumulators per lane.
    using KsWmma =
        WarpGemmAttributeWmmaImpl<WmmaTraits<gfx120_t, GemmDataType, KDataType, float, 16, 16, 16>>;
    using KsAVec = typename KsWmma::AVecType;
    using KsBVec = typename KsWmma::BVecType;
    using KsCVec = typename KsWmma::CVecType;

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

    static constexpr const char* name = "dq_qmajor_gfx12_probe";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // QMAJOR_N32_SMEM_TRIM
        // D is loaded directly from HBM in the consumer distribution,
        // so Q-major no longer reserves an LDS staging buffer for D.
        constexpr ck_tile::index_t smem_size_q    = Policy::template GetSmemSizeQ<Problem>();
        constexpr ck_tile::index_t smem_size_qt   = Policy::template GetSmemSizeQT<Problem>();
        constexpr ck_tile::index_t smem_size_lse  = Policy::template GetSmemSizeLSE<Problem>();
        constexpr ck_tile::index_t smem_size_k    = Policy::template GetSmemSizeK<Problem>();
        constexpr ck_tile::index_t smem_size_kt   = Policy::template GetSmemSizeKT<Problem>();
        constexpr ck_tile::index_t smem_size_v    = Policy::template GetSmemSizeV<Problem>();
        constexpr ck_tile::index_t smem_size_do   = Policy::template GetSmemSizeOGrad<Problem>();
        constexpr ck_tile::index_t smem_size_dot  = Policy::template GetSmemSizeOGradT<Problem>();
        constexpr ck_tile::index_t smem_size_ds   = Policy::template GetSmemSizeSGrad<Problem>();
        constexpr ck_tile::index_t smem_size_bias = Policy::template GetSmemSizeBias<Problem>();

        constexpr ck_tile::index_t smem_size_stage0_0 = smem_size_k + smem_size_kt;

        constexpr ck_tile::index_t smem_size_stage0_1 = smem_size_v;

        // LDS-PADDING-CONTROL:
        // Reserve the same stage0 footprint as V-separate,
        // but keep the original LATEKT dataflow and barriers unchanged.
        // D64-only reservation trim. K+KT and V already reuse LDS offset 0
        // sequentially under the existing barriers; no load/address/barrier
        // ordering is changed here.
        constexpr ck_tile::index_t smem_size_stage0_control =
            kQKHeaddim == 64 ? ck_tile::max(smem_size_k + smem_size_kt, smem_size_v)
                             : (smem_size_k + smem_size_kt + smem_size_v);

        // QMAJOR_COMPACT_LDS_PROBE
        // QMAJOR_DS_TAIL_FULLSIZE_CONTROL:
        // Keep tail-overlap dS address, but restore compact launch LDS size.
        constexpr ck_tile::index_t smem_size_stage1 =
            smem_size_do + smem_size_q + smem_size_lse + ck_tile::max(smem_size_bias, smem_size_ds);

        // final per-wave partial dQ reduction (f32).
        constexpr ck_tile::index_t smem_size_ksplit =
            kD32DqKsplit
                ? (kQKHeaddim == 64 ? 2 : 4) * kM0 * kQKHeaddim * ck_tile::index_t(sizeof(float))
                : 0;
        return ck_tile::max(ck_tile::max(smem_size_stage0_control, smem_size_stage1),
                            smem_size_ksplit);
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
        (void)bias_dram_block_window_tmp;
        (void)randval_dram_block_window_tmp;
        (void)dbias_dram_block_window_tmp;
        (void)position_encoding;
        (void)rp_undrop;
        (void)scale_rp_undrop;
        (void)dropout;

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

        // This first architecture proof is intentionally narrow.
        // It is selected only for the dedicated M32/N64/D128 gfx12 probe tile.
        constexpr auto gemm_0 = [&]() {
            if constexpr(kD32DqKsplit)
                return MakeTransposedCBlockGemm<QDataType,
                                                KDataType,
                                                kN0,
                                                kK0,
                                                typename BlockFmhaShape::Gemm0BlockWarps,
                                                typename BlockFmhaShape::Gemm0WarpTile>();
            else
                return Policy::template GetQKBlockGemm<Problem>();
        }();
        constexpr auto gemm_1 =
            Policy::template GetPTOGradTBlockGemm<Problem>(); // zero return type
        constexpr auto gemm_2 = [&]() {
            if constexpr(kD32DqKsplit)
                return MakeTransposedCBlockGemm<OGradDataType,
                                                VDataType,
                                                kN0,
                                                kK2,
                                                typename BlockFmhaShape::Gemm2BlockWarps,
                                                typename BlockFmhaShape::Gemm2WarpTile>();
            else
                return Policy::template GetOGradVBlockGemm<Problem>();
        }();
        constexpr auto gemm_3 =
            Policy::template GetSGradTQTBlockGemm<Problem>(); // zero return type
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType  = decltype(gemm_4.MakeCBlockTile());

        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        constexpr index_t k4_loops = kN0 / kK4;

        // -----------------------------------------------------------------
        // Q-major ownership: q window origin is already this CTA's Q tile.
        // Determine the K range that contributes to this Q tile.
        const auto q_origin   = q_dram_block_window_tmp.get_window_origin();
        const index_t q_start = q_origin.at(number<0>{});
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_start, number<kM0>{}, number<kN0>{});
        const index_t num_k_loops =
            amd_wave_read_first_lane(integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0));

        auto dq_dram_window = make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dq_dram_block_window_tmp.get_window_lengths(),
                                               dq_dram_block_window_tmp.get_window_origin());

        auto dq_acc = QGradBlockTileType{};
        clear_tile(dq_acc);

        if(__builtin_expect(num_k_loops <= 0, 0))
        {
            store_tile(dq_dram_window, cast_tile<QGradDataType>(dq_acc));
            auto dk_zero = decltype(gemm_3.MakeCBlockTile()){};
            auto dv_zero = decltype(gemm_1.MakeCBlockTile()){};
            clear_tile(dk_zero);
            clear_tile(dv_zero);
            return make_tuple(dk_zero, dv_zero);
        }

        // -----------------------------------------------------------------
        // Load Q / dO / LSE / D once and keep them in registers for all K tiles.
        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             q_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeQDramTileDistribution<Problem>());

        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeOGrad<Problem>()));
        auto q_lds           = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});
        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             do_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeOGradDramTileDistribution<Problem>());
        OGradDataType* do_lds_ptr =
            static_cast<OGradDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());
        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});
        auto do_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());

        auto lse_dram_window = make_tile_window(
            lse_dram_block_window_tmp.get_bottom_tensor_view(),
            lse_dram_block_window_tmp.get_window_lengths(),
            lse_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());
        LSEDataType* lse_lds_ptr = static_cast<LSEDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeQ<Problem>()));
        auto lse_lds             = make_tensor_view<address_space_enum::lds>(
            lse_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());
        auto lse_lds_write_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0});
        auto lse_lds_read_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0}, [&]() {
            if constexpr(kD32DqKsplit)
                return MakeLSEDReadTransposedC<decltype(gemm_0)>();
            else
                return Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>();
        }());

        auto d_dram_window = make_tile_window(
            d_dram_block_window_tmp.get_bottom_tensor_view(),
            d_dram_block_window_tmp.get_window_lengths(),
            d_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        // QMAJOR_N32_D_DIRECT_HBM_PROBE
        auto d_direct_dram_window = make_tile_window(
            d_dram_block_window_tmp.get_bottom_tensor_view(),
            d_dram_block_window_tmp.get_window_lengths(),
            d_dram_block_window_tmp.get_window_origin(),
            [&]() {
                if constexpr(kD32DqKsplit)
                    return MakeLSEDReadTransposedC<decltype(gemm_0)>();
                else
                    return Policy::template MakeLSEDLdsReadBlockDescriptor<Problem,
                                                                           decltype(gemm_0)>();
            }());
        auto q_block_tile   = load_tile(q_dram_window);
        auto do_block_tile  = load_tile(do_dram_window);
        auto lse_block_tile = load_tile(lse_dram_window);

        block_sync_lds();
        store_tile(q_lds_window, q_block_tile);
        store_tile(do_lds_window, do_block_tile);
        store_tile(lse_lds_write_window, lse_block_tile);
        block_sync_lds();

        auto q_reg_tensor  = load_tile(q_lds_read_window);
        auto do_reg_tensor = load_tile(do_lds_read_window);
        auto lse           = load_tile(lse_lds_read_window);
        // QMAJOR_DIRECT_D_LATE_LOAD
        auto d = load_tile(d_direct_dram_window);

        // -----------------------------------------------------------------
        // K/V/KT streaming windows. K and V advance across the sequence.
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem>());
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

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

        auto shuffled_k_block_tile = make_static_distributed_tensor<KDataType>(
            Policy::template MakeShuffledKRegWriteBlockDescriptor<Problem>());
        KDataType* kt_lds_ptr     = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));
        auto shuffled_k_lds_write = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeShuffledKLdsWriteBlockDescriptor<Problem>());
        auto shuffled_k_lds_write_window = make_tile_window(
            shuffled_k_lds_write, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});
        // [D=32][N=64] bf16, element offset
        //   d*64 + ((n>>3) ^ ((d>>1)&7))*8 + (n&7)
        // Physical dims A,B,C,R = 16,2,8,8 (d = A*2+B, n = C*8+R); C ^= A%8.
        constexpr auto d32_kt_base_desc = [&]() {
            constexpr auto raw = make_naive_tensor_descriptor(
                make_tuple(number<kQKHeaddim / 2>{}, number<2>{}, number<8>{}, number<8>{}),
                make_tuple(number<128>{}, number<64>{}, number<8>{}, number<1>{}),
                number<8>{},
                number<1>{});
            return transform_tensor_descriptor(
                raw,
                make_tuple(make_xor_transform(make_tuple(number<kQKHeaddim / 2>{}, number<8>{})),
                           make_pass_through_transform(number<2>{}),
                           make_pass_through_transform(number<8>{})),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));
        }();
        // Read view [D, N] (Gemm4 B operand), write view [N, D] (DRAM tile order).
        constexpr auto d32_kt_read_desc = transform_tensor_descriptor(
            d32_kt_base_desc,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<kQKHeaddim / 2>{}, number<2>{})),
                       make_merge_transform_v3_division_mod(make_tuple(number<8>{}, number<8>{}))),
            make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        constexpr auto d32_kt_write_desc = transform_tensor_descriptor(
            d32_kt_base_desc,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<kQKHeaddim / 2>{}, number<2>{})),
                       make_merge_transform_v3_division_mod(make_tuple(number<8>{}, number<8>{}))),
            make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));
        static_assert(d32_kt_read_desc.get_element_space_size() == kQKHeaddim * 64);
        if constexpr(kD32DqKtB128)
            static_assert(Policy::template GetSmemSizeKT<Problem>() >= kQKHeaddim * 64 * 2);
        auto kt_lds_read = [&]() {
            if constexpr(kD32DqKtB128)
                return make_tensor_view<address_space_enum::lds>(kt_lds_ptr, d32_kt_read_desc);
            else
                return make_tensor_view<address_space_enum::lds>(
                    kt_lds_ptr, Policy::template MakeKTLdsReadBlockDescriptor<Problem>());
        }();
        auto d32_kt_lds_write_window = [&]() {
            if constexpr(kD32DqKtB128)
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(kt_lds_ptr, d32_kt_write_desc),
                    make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                    {0, 0});
            else
                return 0;
        }();
        auto kt_lds_read_window =
            make_tile_window(kt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

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

        // dS staging for Gemm4.
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>()));
        auto ds_lds              = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());
        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
        auto ds_lds_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK4>{}),
                             ds_lds_window.get_window_origin(),
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());

        // per-wave partial dQ [mb][db] over this wave's keys.
        KsCVec ks_dq[2][kQKHeaddim / 16];
        if constexpr(kD32DqKsplit)
            for(int i = 0; i < 2; ++i)
                for(int j = 0; j < kQKHeaddim / 16; ++j)
                    ks_dq[i][j] = KsCVec{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        index_t k_step = seqlen_k_start;
        for(index_t ik = 0; ik < num_k_loops; ++ik)
        {
            // K/V HBM -> LDS, make K and K^T register views, then V register view.
            auto k_block_tile = load_tile(k_dram_window);
            auto v_block_tile = load_tile(v_dram_window);

            block_sync_lds();
            store_tile(k_lds_write_window, k_block_tile);
            if constexpr(kD32DqKtB128)
                store_tile(d32_kt_lds_write_window, k_block_tile);
            else
            {
                shuffle_tile(shuffled_k_block_tile, k_block_tile);
                store_tile(shuffled_k_lds_write_window, shuffled_k_block_tile);
            }
            block_sync_lds();
            // QMAJOR_LATE_KT_LOAD_PROBE:
            // Only normal K must complete before V overwrites the base LDS region.
            k_reg_tensor = load_tile(k_lds_read_window);

            block_sync_lds();
            store_tile(v_lds_write_window, v_block_tile);
            block_sync_lds();

            auto v_reg_tensor = load_tile(v_lds_read_window);

            // QMAJOR_LATE_KT_LOAD_PROBE:
            // KT lives in the non-overlapping +8192 LDS region.
            auto kt_reg_tensor = [&]() {
                if constexpr(kD32DqKsplit)
                    return 0; // K^T is read per wave as raw b128 fragments below
                else
                    return load_tile(kt_lds_read_window);
            }();

            // QK -> P
            auto s_acc = SPBlockTileType{};
            s_acc      = gemm_0(q_reg_tensor, k_reg_tensor);
            HotLoopScheduler::template GemmStagedScheduler<0>();

            // Q-major causal/mask support:
            // GetTileRangeAlongX prunes whole K tiles, but an edge/diagonal
            // tile still needs elementwise masking before softmax recovery.
            if constexpr(FmhaMask::IsMasking)
            {
                const bool need_perpixel_check =
                    mask.IsEdgeTile(q_start, k_step, number<kM0>{}, number<kN0>{});

                if(need_perpixel_check)
                {
                    set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = q_start + tile_idx.at(number<0>{});
                        const auto col = k_step + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }

            auto p                 = SPBlockTileType{};
            constexpr auto p_spans = decltype(p)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                // a row with no unmasked key has lse == -inf.
                // Use 0 (as the DKDV pipeline does) so its masked p is 0, not NaN.
                const auto raw_lse = lse[i_idx];
                auto row_lse       = log2e_v<LSEDataType> * [&]() {
                    if constexpr(FmhaMask::IsMasking ||
                                 BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    {
                        return raw_lse == -numeric<LSEDataType>::infinity()
                                         ? type_convert<LSEDataType>(0.f)
                                         : raw_lse;
                    }
                    else
                    {
                        return raw_lse;
                    }
                }();
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p(i_j_idx)             = exp2(scale * s_acc[i_j_idx] - row_lse);
                });
            });

            // dP = dO @ V ; dS = P * (dP - D)
            auto dp_acc = SPGradBlockTileType{};
            dp_acc      = gemm_2(do_reg_tensor, v_reg_tensor);
            HotLoopScheduler::template GemmStagedScheduler<2>();

            auto ds                 = SPGradBlockTileType{};
            constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
            sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ds(i_j_idx) =
                        p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]); // QMAJOR_N32_DIRECT_D_FIX
                });
            });

            const auto ds_gemm = cast_tile<GemmDataType>(ds);
            if constexpr(kD32DqKsplit)
            {
                // A = this wave's dS (TransposeC C: row lane%16, keys 8*(lane/16)+j), K = 16;
                // B = K^T[d][wave keys] from the [D=32][N=64] layout
                //   d*64 + ((n>>3) ^ ((d>>1)&7))*8 + (n&7).
                static_assert(remove_cvref_t<decltype(ds_gemm)>::get_thread_buffer_size() == 16);
                const index_t wv = get_warp_id(), ln = get_lane_id();
                const auto& dsb = ds_gemm.get_thread_buffer();
                KsAVec ka[2];
                for(int mb = 0; mb < 2; ++mb)
                    for(int j = 0; j < 8; ++j)
                        ka[mb][j] = dsb[mb * 8 + j];
                const index_t n = 16 * wv + 8 * (ln / 16);
                for(int db = 0; db < kQKHeaddim / 16; ++db)
                {
                    const index_t dd = 16 * db + ln % 16;
                    const KsBVec kb  = *reinterpret_cast<const KsBVec*>(
                        kt_lds_ptr + dd * 64 + (((n >> 3) ^ ((dd >> 1) & 7)) << 3));
                    for(int mb = 0; mb < 2; ++mb)
                        KsWmma{}(ks_dq[mb][db], ka[mb], kb);
                }
            }
            else
            {
                block_sync_lds();
                store_tile(ds_lds_window, ds_gemm);
                block_sync_lds();

                auto ds_reg_tensor      = load_tile(ds_lds_read_window);
                auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
                move_tile_window(ds_lds_read_window, {0, kK4});

                static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                    if constexpr(i_k4 < k4_loops - 1)
                    {
                        ds_reg_tensor_next = load_tile(ds_lds_read_window);
                        move_tile_window(ds_lds_read_window, {0, kK4});
                        if constexpr(kM0 == 32 && kN0 == 64 && kQKHeaddim == 64)
                            __builtin_amdgcn_sched_barrier(0);
                    }
                    auto kt_reg_tensor_slice =
                        get_slice_tile(kt_reg_tensor,
                                       sequence<0, i_k4 * kK4>{},
                                       sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                    gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);
                    if constexpr(i_k4 < k4_loops - 1)
                        ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
                });
                move_tile_window(ds_lds_read_window, {0, -kN0});
                HotLoopScheduler::template GemmStagedScheduler<4>();
            }
            move_tile_window(k_dram_window, {kN0, 0});
            move_tile_window(v_dram_window, {kN0, 0});
            k_step += kN0;
        }

        if constexpr(kD32DqKsplit)
        {
            // Sum the four per-wave partials once (fixed order w = 0..3) into the Gemm4 C
            // distribution through LDS.
            block_sync_lds();
            float* red       = static_cast<float*>(smem_ptr);
            const index_t wv = get_warp_id(), ln = get_lane_id();
            constexpr int kRedParts = kQKHeaddim == 64 ? 1 : 4;
            if constexpr(kQKHeaddim == 64)
            {
                // pairwise in 16 KB; final sum (w0+w2)+(w1+w3) in slot 0.
                auto ridx = [&](int slot, int mb, int db, int j) {
                    return (slot * kM0 + 16 * mb + 8 * (ln / 16) + j) * kQKHeaddim + 16 * db +
                           ln % 16;
                };
                auto put = [&](int slot) {
                    for(int mb = 0; mb < 2; ++mb)
                        for(int db = 0; db < kQKHeaddim / 16; ++db)
                            for(int j = 0; j < 8; ++j)
                                red[ridx(slot, mb, db, j)] = ks_dq[mb][db][j];
                };
                auto add = [&](int slot) {
                    for(int mb = 0; mb < 2; ++mb)
                        for(int db = 0; db < kQKHeaddim / 16; ++db)
                            for(int j = 0; j < 8; ++j)
                                ks_dq[mb][db][j] += red[ridx(slot, mb, db, j)];
                };
                if(wv >= 2)
                    put(wv - 2);
                block_sync_lds();
                if(wv < 2)
                    add(wv);
                block_sync_lds();
                if(wv == 1)
                    put(0);
                block_sync_lds();
                if(wv == 0)
                {
                    add(0);
                    put(0);
                }
            }
            else
            {
                for(int mb = 0; mb < 2; ++mb)
                    for(int db = 0; db < 2; ++db)
                        for(int j = 0; j < 8; ++j)
                            red[(wv * kM0 + 16 * mb + 8 * (ln / 16) + j) * kQKHeaddim + 16 * db +
                                ln % 16] = ks_dq[mb][db][j];
            }
            block_sync_lds();
            constexpr auto red_desc =
                make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                             make_tuple(number<kQKHeaddim>{}, number<1>{}),
                                             number<1>{},
                                             number<1>{});
            static_for<0, kRedParts, 1>{}([&](auto w_) {
                auto red_view = make_tensor_view<address_space_enum::lds>(
                    red + w_.value * kM0 * kQKHeaddim, red_desc);
                auto part =
                    load_tile(make_tile_window(red_view,
                                               make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                               {0, 0},
                                               dq_acc.get_tile_distribution()));
                static_assert(remove_cvref_t<decltype(part)>::get_thread_buffer_size() ==
                              remove_cvref_t<decltype(dq_acc)>::get_thread_buffer_size());
                for(int i = 0; i < dq_acc.get_thread_buffer_size(); ++i)
                    dq_acc.get_thread_buffer()(i) += part.get_thread_buffer()[i];
            });
        }
        // Scale once after the whole K reduction, then write this Q tile once.
        tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
        if constexpr(decltype(dq_dram_window)::BottomTensorView::DstInMemOp ==
                     memory_operation_enum::set)
            store_tile(dq_dram_window, cast_tile<QGradDataType>(dq_acc));
        else
            update_tile(dq_dram_window, cast_tile<QGradDataType>(dq_acc));

        auto dk_zero = decltype(gemm_3.MakeCBlockTile()){};
        auto dv_zero = decltype(gemm_1.MakeCBlockTile()){};
        clear_tile(dk_zero);
        clear_tile(dv_zero);
        return make_tuple(dk_zero, dv_zero);
    }
};

// Marker trait consumed by FmhaBwdDQDKDVKernel for Q-owned direct scheduling.
template <typename, typename = void>
struct fmha_bwd_qmajor_dq_pipeline : std::false_type
{
};

template <typename T>
struct fmha_bwd_qmajor_dq_pipeline<T, std::void_t<decltype(T::is_qmajor_dq_pipeline)>>
    : std::bool_constant<T::is_qmajor_dq_pipeline>
{
};

} // namespace ck_tile
