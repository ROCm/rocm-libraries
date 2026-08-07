// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/cast_tile_mx.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS, targeting gfx1250
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmDefaultPolicy>
struct BlockFmhaPipelineQRKSVSTdm
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);
    static constexpr bool kKLoadOnce = BlockFmhaShape::kM0 > 64;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;
    static constexpr index_t kNWarp        = BlockFmhaShape::Gemm0BlockWarps::at(I1);
    static constexpr index_t kNXdl         = BlockFmhaShape::Gemm0WarpTile::at(I1);

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ =
        Problem::kPadHeadDimQ; // support multiple of vector(like 8x)
    static constexpr bool kPadHeadDimV =
        Problem::kPadHeadDimV; // support multiple of vector(like 8x)

    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr bool kHasDropout       = Problem::kHasDropout;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr auto QScaleEnum        = Problem::QScaleEnum;
    static constexpr bool kBlockScale = QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE;
    // Granularities that take the descale arguments on the run() overload below,
    // and whose P is dynamically quantized against gemm1's A-side scale operand.
    static constexpr bool kQuantized = QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR ||
                                       QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD ||
                                       kBlockScale;
    static constexpr bool kStoreLSE        = Problem::kStoreLSE;
    static constexpr bool kHasUnevenSplits = true;
    static constexpr bool kHasSink         = Problem::kHasSink;

    // Only gemm1 runs the hardware scaled MMA, and only 8-bit warp tiles with
    // K=128 have one (v_wmma_scale_f32_16x16x128_f8f6f4). No warp gemm trait
    // reports that, hence the tile-K comparison.
    static constexpr bool kHwGemm1Scale = is_any_of<VDataType, fp8_t, bf8_t>::value &&
                                          BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 128;

    // Reduction-axis elements covered by one byte of the four-byte scale operand.
    static constexpr index_t kScaleBytes        = 4;
    static constexpr index_t kGemm1KPerByte     = kK1 / kScaleBytes;
    static constexpr index_t kPScaleGranularity = kGemm1KPerByte;
    // Keys spanned by one warp N iteration of gemm0, and how many tile kN0.
    static constexpr index_t kGemm0KVPerNIter = kNXdl * kNWarp;
    static constexpr index_t kGemm0NIters     = kN0 / kGemm0KVPerNIter;

    // Finest KV extent that has to share a single descale: a scale block
    // straddling either unit would have part of itself silently ignored (gemm1)
    // or applied to the wrong columns (gemm0). fmha_fwd_kernel.hpp checks
    // block_scale_size_kv against this.
    static constexpr index_t kKVScaleAlign =
        kHwGemm1Scale ? (kGemm1KPerByte < kGemm0KVPerNIter ? kGemm0KVPerNIter : kGemm1KPerByte)
                      : kN0;
    static_assert(!kHwGemm1Scale || (kKVScaleAlign % kGemm1KPerByte == 0 &&
                                     kKVScaleAlign % kGemm0KVPerNIter == 0),
                  "qr_tdm pipeline: the two KV scale resolutions must nest");

    // Replicate one E8M0 code across all four K sub-block slots. Packed4Scale's
    // variadic constructor cannot be used here: it assigns its *last* argument
    // to byte 0.
    CK_TILE_HOST_DEVICE static int32_t to_e8m0_scale_broadcast(e8m0_t d)
    {
        Packed4Scale_E8M0 packed;
        packed.data() = 0;
        static_for<0, kScaleBytes, 1>{}([&](auto i) { packed.pack_scale(d, i); });
        return static_cast<int32_t>(packed.data());
    }

    // Neutral scale operand. A zeroed one is *not* neutral: E8M0 0x00 decodes
    // to 2^-127.
    CK_TILE_HOST_DEVICE static int32_t e8m0_one()
    {
        Packed4Scale_E8M0 packed(1.0f, 1.0f, 1.0f, 1.0f);
        return static_cast<int32_t>(packed.data());
    }

    // From an SGPR the hardware reads only bits[7:0] of the scale operand and
    // broadcasts them to all four K sub-blocks, silently dropping the other
    // three bytes. This operand carries a different exponent per sub-block, so
    // it must sit in a VGPR even though every lane holds identical bits.
    CK_TILE_DEVICE static int32_t pin_to_vgpr(int32_t v)
    {
        asm volatile("" : "+v"(v));
        return v;
    }

    // gemm1's V-side operand: one E8M0 code per K sub-block, i.e. per
    // kGemm1KPerByte consecutive keys starting at kv_base. kv_last is the last
    // key that exists.
    CK_TILE_DEVICE static int32_t pack_v_scale(const float* v_descale_ptr,
                                               index_t kv_base,
                                               index_t kv_last,
                                               index_t block_scale_size_kv)
    {
        Packed4Scale_E8M0 packed;
        packed.data() = 0;
        static_for<0, kScaleBytes, 1>{}([&](auto i) {
            // Sub-blocks past the end of the sequence still load, so clamp the index.
            const index_t kv = min(kv_base + i * kGemm1KPerByte, kv_last);
            packed.pack_scale(e8m0_t(cast_pointer_to_constant_address_space(
                                  v_descale_ptr)[kv / block_scale_size_kv]),
                              i);
        });
        return pin_to_vgpr(static_cast<int32_t>(packed.data()));
    }

    template <typename Gemm1>
    CK_TILE_DEVICE static auto make_gemm1_scale([[maybe_unused]] int32_t scale)
    {
        if constexpr(kHwGemm1Scale)
        {
            return [scale](auto, auto) { return scale; };
        }
        else
        {
            return typename remove_cvref_t<Gemm1>::no_scale{};
        }
    }

    static_assert((CK_TILE_FMHA_FWD_FAST_EXP2 &&
                   (kHasLogitsSoftCap && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                    !kHasLogitsSoftCap)) ||
                  (!CK_TILE_FMHA_FWD_FAST_EXP2 && !kHasLogitsSoftCap));

    static_assert(!kHasDropout, "qr_tdm pipeline does not yet support dropout");

    // MX and KV_BLOCKSCALE have no code path here: they would compile and run
    // with every descale silently dropped.
    static_assert(QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE,
                  "qr_tdm pipeline: unsupported quantization granularity");

    // The descale index assumes a single-phase KV walk; sink makes it two-phase,
    // so the index would drift away from the tile being consumed.
    static_assert(!(kBlockScale && kHasSink),
                  "qr_tdm pipeline: BLOCKSCALE + sink would read the wrong descale block");

    static_assert(!kBlockScale || kHwGemm1Scale,
                  "qr_tdm pipeline: BLOCKSCALE requires the scaled MMA in gemm1");

    // One gemm1 call gets one packed V operand, so it must be exactly one
    // warp-GEMM K iteration or the second would reuse the first one's bytes.
    static_assert(!kHwGemm1Scale || kK1 == BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                  "qr_tdm pipeline: the scaled gemm1 assumes kK1 is one warp-GEMM K step");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kAlignmentO = Policy::template GetAlignmentO<Problem>();
    // Required by fmha_fwd_kernel.hpp's qr_async_trload-style branch.
    static constexpr index_t kAlignmentOacc = Policy::template GetAlignmentO<Problem>();

    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim <= 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS || kM0 >= 256)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            }
        }
    }();

    static constexpr const char* name = "qr_tdm";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    // Re-pack gemm_0 C into gemm_1 A: C is M-outer (MIter,KIter), A is K-outer.
    // Lanes already align (NWarp==1), so this is an in-thread block reorder;
    // identity when MIterPerWarp==1. On the quantized granularities P is also
    // converted to fp8 with a per-kPScaleGranularity E8M0 scale, returned in
    // p_scale for gemm1's A-side scale operand.
    template <typename Gemm1, typename PComputeTensor>
    CK_TILE_DEVICE static auto MakePForGemm1(const PComputeTensor& p_compute, int32_t& p_scale)
    {
        constexpr index_t kPMI = Gemm1::MIterPerWarp;
        constexpr index_t kPKI = kN0 / Gemm1::WarpGemm::kK;

        auto p_tile = make_static_distributed_tensor<PDataType>(
            Policy::template MakePRegTileDistribution<Problem>());

        if constexpr(kQuantized)
        {
            // cast_tile_mx_wmma() cuts the thread buffer into K sub-blocks in
            // order, so it must see P already in gemm1's A order; a permuting
            // repack would attach every code to the wrong sub-block.
            static_assert(kPMI * kPKI == 1,
                          "qr_tdm pipeline: the dynamic P scale needs the (MIter,KIter) repack "
                          "to be the identity");
            using WG = typename Gemm1::WarpGemm;
            const auto packed =
                cast_tile_mx_wmma<kPScaleGranularity,
                                  WG::WarpGemmAttribute::Impl::kAMLane,
                                  WG::WarpGemmAttribute::Impl::kABKLane>(p_tile, p_compute);
            static_assert(packed.size() == 1);
            p_scale = packed[I0];
        }
        else
        {
            const auto p_src        = cast_tile<PDataType>(p_compute);
            constexpr index_t kPBuf = decltype(p_src)::get_thread_buffer_size();
            constexpr index_t kPSub = kPBuf / (kPMI * kPKI);
            using p_bulk_t          = array<PDataType, kPSub>;
            static_for<0, kPMI, 1>{}([&](auto mi) {
                static_for<0, kPKI, 1>{}([&](auto ki) {
                    p_tile.get_thread_buffer().template set_as<p_bulk_t>(
                        number<ki * kPMI + mi>{},
                        p_src.get_thread_buffer().template get_as<p_bulk_t>(
                            number<mi * kPKI + ki>{}));
                });
            });
            p_scale = e8m0_one();
        }
        return p_tile;
    }

    // Decode (single shared-memory buffer)
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    run(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
        const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
        const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,        // M0*1 tile
        FmhaMask mask,
        PositionEncoding position_encoding,
        float scale_s,
        void* smem_ptr,
        float sink_v,
        const float* k_descale_ptr,
        const float* v_descale_ptr,
        index_t block_scale_size_kv,
        e8m0_t v_descale) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

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

        auto o_acc = OaccBlockTileType{};

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(o_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // init M, L (sink-aware: when sink_v is finite, pre-seed m/l)
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        clear_tile(o_acc);
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

        // Sink-aware tile range: GetSinkTileRangeAlongX returns
        // (sink_seq_end, seqlen_k_start, seqlen_k_end). For non-sink,
        // sink_seq_end is always 0.
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
                return o_acc;
            }
        }

        // TDM configs for Q / K / V
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

        // Writer (TDM) and reader share a plain row-major desc: a TDM box-major
        // write cannot produce an XOR'd layout, so no swizzle here.
        auto q_lds_write_view = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptr), Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_read_view = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptr), Policy::template MakeQLdsBlockDescriptor<Problem>());

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

        // K tile in LDS
        // For sink: kv_load_start is 0 when there are sink tokens (we start
        // reading from seq position 0), otherwise seqlen_k_start.
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;
        const index_t physical_seqlen_k_start = logical_seqlen_k_start;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end;

        // Bias tile window (ELEMENTWISE_BIAS or ALIBI -- null window for NO_BIAS)
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             gemm_0.MakeCBlockTile().get_tile_distribution());

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem>());

        // Plain row-major desc, see the Q window above.
        auto k_lds_write_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType*>(smem_ptr), Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType*>(smem_ptr), Policy::template MakeKLdsBlockDescriptor<Problem>());

        auto k_lds_write_window =
            make_tile_window(k_lds_write_view,
                             Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});
        auto k_lds_read_window =
            make_tile_window(k_lds_read_view,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());

        // S tile in LDS
        auto s_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SaccDataType*>(reinterpret_cast<char*>(smem_ptr) +
                                            Policy::template GetSmemSizeK<Problem>()),
            Policy::template MakeSLdsBlockDescriptor<Problem>());
        auto s_write_lds_window = make_tile_window(
            s_lds, Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto s_read_lds_window =
            make_tile_window(s_lds,
                             Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeSRegTileDistribution<Problem>());

        // V tile in LDS
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds_write_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_ptr) +
                                         Policy::template GetSmemSizeK<Problem>() +
                                         Policy::template GetSmemSizeS<Problem>()),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        // V LDS read view shares the plain row-major desc of the write view,
        // which is what the TDM box-major writer produces.
        auto v_lds_read_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_ptr) +
                                         Policy::template GetSmemSizeK<Problem>() +
                                         Policy::template GetSmemSizeS<Problem>()),
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

        s_wait_tensorcnt_barrier<0>();
        auto q_tile = load_tile(q_lds_read_window);

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);

        // gemm1's scale operand where v_descale is constant over the whole KV
        // loop; BLOCKSCALE replaces it per tile below.
        [[maybe_unused]] const int32_t const_v_scale =
            kHwGemm1Scale ? to_e8m0_scale_broadcast(v_descale) : e8m0_one();

        block_sync_lds();
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        do
        {
            // First key of this KV tile; K and V descales are both indexed off it.
            [[maybe_unused]] const index_t kv_tile_start = kv_load_start + i_total_loops * kN0;
            [[maybe_unused]] const index_t kv_last       = physical_seqlen_k_end - 1;

            block_sync_lds();
            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window); // prefetch load v tile

            // move V tile windows
            move_tile_window(v_dram_window, {kN0, 0});

            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C

            if constexpr(1 < k0_loops)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    s_wait_tensorcnt_barrier<0>();

                    auto k_tile = load_tile(k_lds_read_window);

                    gemm_0(s_acc,
                           get_slice_tile(q_tile,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kM0, (i_k0 + 1) * kK0>{}),
                           k_tile);

                    // loop over along the [K]ey head dimension
                    move_tile_window(k_dram_window, {0, kK0});
                    block_sync_lds();
                    load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
                });
                // move back to the origin
                move_tile_window(k_dram_window, {0, -kK0 * (k0_loops - 1)});
            }

            s_wait_tensorcnt_barrier<0>();

            auto k_tile = load_tile(k_lds_read_window);

            gemm_0(s_acc,
                   get_slice_tile(q_tile,
                                  sequence<0, (k0_loops - 1) * kK0>{},
                                  sequence<kM0, k0_loops * kK0>{}),
                   k_tile);

            // k_descale varies along gemm0's N axis, not its reduction axis, so
            // it is applied in fp32 here, before the row max. One warp N
            // iteration shares one table entry, which keeps the index and the
            // multiply operand wave-uniform.
            if constexpr(kBlockScale)
            {
                // Without the barrier the scheduler sinks gemm1's WMMAs into
                // this sweep and the longer s_acc/o_acc live ranges cost enough
                // VGPRs to drop a wave per SIMD.
                __builtin_amdgcn_sched_barrier(0);
                constexpr auto ks_spans = decltype(s_acc)::get_distributed_spans();
                static_assert(remove_cvref_t<decltype(ks_spans[number<1>{}])>::Impl::at(0) ==
                                  kGemm0NIters,
                              "qr_tdm pipeline: s_acc's N span must lead with gemm0's warp N "
                              "iteration");
                sweep_tile_span(ks_spans[number<1>{}], [&](auto idx1) {
                    constexpr index_t n_iter = idx1.impl_.at(number<0>{});
                    const index_t kv      = min(kv_tile_start + n_iter * kGemm0KVPerNIter, kv_last);
                    const float k_descale = cast_pointer_to_constant_address_space(
                        k_descale_ptr)[kv / block_scale_size_kv];
                    sweep_tile_span(ks_spans[number<0>{}], [&](auto idx0) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) *= k_descale;
                    });
                });
            }

            // STAGE 2: scale_s, add bias
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                // Pre-scale s_acc by scale_s before adding bias
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

            // Sink-aware k_origin: in sink phase, tiles start at 0;
            // in normal phase, tiles start at physical_seqlen_k_start.
            const auto k_origin = [&]() {
                const bool in_sink_phase = (num_sink_loop > i_total_loops);
                if(in_sink_phase)
                    return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                else
                    return make_tuple(
                        kN0 * (i_total_loops - num_sink_loop) + physical_seqlen_k_start, 0);
            }();

            if constexpr(kHasUnevenSplits)
            {
                if(i_total_loops == (num_total_loop - 1))
                {
                    set_tile_if(s_acc,
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
                bool need_perpixel_check =
                    mask.IsEdgeTile(q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(I0) + tile_idx.at(I0);
                            const auto col = k_origin.at(I0) + tile_idx.at(I1);
                            if constexpr(kHasSink)
                                return mask.IsOutOfSinkBound(row, col);
                            else
                                return mask.IsOutOfBound(row, col);
                        });
                }
            }

            // Sink->normal window jump: at the boundary between sink region
            // and normal region, jump K/V/bias dram windows forward.
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1)
                {
                    move_tile_window(k_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
                    move_tile_window(v_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
                    move_tile_window(bias_dram_window, {0, physical_seqlen_k_start - sink_seq_end});
                }
            }

            // move K and bias tile windows after current status checked
            // prefetch next-tile along [K]ey sequence length dimension
            move_tile_window(k_dram_window, {kN0, 0});
            move_tile_window(bias_dram_window, {0, kN0});

            block_sync_lds();
            load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

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

            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s_new,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s_new.get_tile_distribution()); // Pcompute{j}

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                /// NOTICE: bias might be materialized mask including -inf values, need
                /// consideration
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
                        p_compute(i_j_idx) = exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            p_compute(i_j_idx) = exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s_new[i_j_idx] - row_max);
                        }
                    }
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

            int32_t p_scale = 0;
            auto p_tile     = MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale);

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

            // V TDM write must fully commit before ds_load_tr reads it.
            s_wait_tensorcnt_barrier<0>();

            auto v_tile = load_tile_transpose(v_lds_read_window);

            const auto p_scale_arg = make_gemm1_scale<decltype(gemm_1)>(p_scale);

            // gemm1's V-side operand for the K range one call reduces over; the
            // coarser granularities reuse the constant.
            auto v_scale = [&](auto i_k1) {
                if constexpr(kBlockScale)
                {
                    return make_gemm1_scale<decltype(gemm_1)>(pack_v_scale(
                        v_descale_ptr, kv_tile_start + i_k1 * kK1, kv_last, block_scale_size_kv));
                }
                else
                {
                    ignore = i_k1;
                    return make_gemm1_scale<decltype(gemm_1)>(const_v_scale);
                }
            };

            if constexpr(1 < k1_loops)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    gemm_1(o_acc,
                           get_slice_tile(p_tile,
                                          sequence<0, i_k1 * kK1>{},
                                          sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_tile,
                           p_scale_arg,
                           v_scale(i_k1));

                    // loop over along the [V]alue Sequence length
                    move_tile_window(v_lds_read_window, {kK1, 0});
                    v_tile = load_tile_transpose(v_lds_read_window);
                });
                // move back to the origin
                move_tile_window(v_lds_read_window, {-kK1 * (k1_loops - 1), 0});
            }

            gemm_1(o_acc,
                   get_slice_tile(p_tile,
                                  sequence<0, (k1_loops - 1) * kK1>{},
                                  sequence<kM0, k1_loops * kK1>{}),
                   v_tile,
                   p_scale_arg,
                   v_scale(number<k1_loops - 1>{}));

        } while(++i_total_loops < num_total_loop);

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

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            auto tmp             = [&]() {
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
                o_acc(i_j_idx) *= tmp;
            });
        });

        return o_acc;
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
        float sink_v,
        const float* k_descale_ptr,
        const float* v_descale_ptr,
        index_t block_scale_size_kv,
        e8m0_t v_descale) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

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

        auto o_acc = OaccBlockTileType{};

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(o_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // init M, L (sink-aware)
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        clear_tile(o_acc);
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
                return o_acc;
            }
        }

        // TDM configs for Q / K / V
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

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);

        // gemm1's scale operand where v_descale is constant over the whole KV
        // loop; BLOCKSCALE replaces it per tile below.
        [[maybe_unused]] const int32_t const_v_scale =
            kHwGemm1Scale ? to_e8m0_scale_broadcast(v_descale) : e8m0_one();
        block_sync_lds<0>();
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);

        move_tile_window(k_dram_window, {kN0, 0});
        k_lds_write_window.set_bottom_tensor_view_data_ptr(
            static_cast<KDataType* __restrict__>(smem_ptrk1));
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        constexpr index_t k_lds_insts = k_lds_read_window.get_num_of_access();
        constexpr index_t v_lds_insts = v_lds_read_window.get_num_of_access();

        s_wait_tensorcnt_barrier<0>();
        auto k_tile = load_tile(k_lds_read_window);

        __builtin_amdgcn_sched_barrier(0);

        auto mainloop = [&](KDataType* __restrict__ k_lds_write_ptr,
                            KDataType* __restrict__ k_lds_read_ptr,
                            KDataType* __restrict__ v_lds_write_ptr,
                            KDataType* __restrict__ v_lds_read_ptr) {
            // First key of this KV tile; K and V descales are both indexed off it.
            [[maybe_unused]] const index_t kv_tile_start = kv_load_start + i_total_loops * kN0;
            [[maybe_unused]] const index_t kv_last       = physical_seqlen_k_end - 1;

            // move V tile windows
            block_sync_lds<k_lds_insts>();
            move_tile_window(v_dram_window, {kN0, 0});
            v_lds_write_window.set_bottom_tensor_view_data_ptr(v_lds_write_ptr);
            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);

            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C

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

            // k_descale varies along gemm0's N axis, not its reduction axis, so
            // it is applied in fp32 here, before the row max. One warp N
            // iteration shares one table entry, which keeps the index and the
            // multiply operand wave-uniform.
            if constexpr(kBlockScale)
            {
                // Without the barrier the scheduler sinks gemm1's WMMAs into
                // this sweep and the longer s_acc/o_acc live ranges cost enough
                // VGPRs to drop a wave per SIMD.
                __builtin_amdgcn_sched_barrier(0);
                constexpr auto ks_spans = decltype(s_acc)::get_distributed_spans();
                static_assert(remove_cvref_t<decltype(ks_spans[number<1>{}])>::Impl::at(0) ==
                                  kGemm0NIters,
                              "qr_tdm pipeline: s_acc's N span must lead with gemm0's warp N "
                              "iteration");
                sweep_tile_span(ks_spans[number<1>{}], [&](auto idx1) {
                    constexpr index_t n_iter = idx1.impl_.at(number<0>{});
                    const index_t kv      = min(kv_tile_start + n_iter * kGemm0KVPerNIter, kv_last);
                    const float k_descale = cast_pointer_to_constant_address_space(
                        k_descale_ptr)[kv / block_scale_size_kv];
                    sweep_tile_span(ks_spans[number<0>{}], [&](auto idx0) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) *= k_descale;
                    });
                });
            }

            // STAGE 2: scale_s, add bias
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

            s_wait_tensorcnt_barrier<0>();
            v_lds_read_window.set_bottom_tensor_view_data_ptr(v_lds_read_ptr);
            auto v_tile = load_tile_transpose(v_lds_read_window);

            // Sink-aware k_origin (prefill path)
            const auto k_origin = [&]() {
                const bool in_sink_phase = (num_sink_loop > i_total_loops);
                if(in_sink_phase)
                    return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                else
                    return make_tuple(
                        kN0 * (i_total_loops - num_sink_loop) + physical_seqlen_k_start, 0);
            }();

            if constexpr(kHasUnevenSplits)
            {
                if(i_total_loops == (num_total_loop - 1))
                {
                    set_tile_if(s_acc,
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
                bool need_perpixel_check =
                    mask.IsEdgeTile(q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(I0) + tile_idx.at(I0);
                            const auto col = k_origin.at(I0) + tile_idx.at(I1);
                            if constexpr(kHasSink)
                                return mask.IsOutOfSinkBound(row, col);
                            else
                                return mask.IsOutOfBound(row, col);
                        });
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

            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s_new,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
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
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s_new.get_tile_distribution()); // Pcompute{j}

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                /// NOTICE: bias might be materialized mask including -inf values, need
                /// consideration
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
                        p_compute(i_j_idx) = exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            p_compute(i_j_idx) = exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s_new[i_j_idx] - row_max);
                        }
                    }
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

            int32_t p_scale = 0;
            auto p_tile     = MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale);

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

            block_sync_lds<v_lds_insts>();
            move_tile_window(k_dram_window, {kN0, 0});
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
            load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

            const auto p_scale_arg = make_gemm1_scale<decltype(gemm_1)>(p_scale);

            // gemm1's V-side operand for the K range one call reduces over; the
            // coarser granularities reuse the constant.
            auto v_scale = [&](auto i_k1) {
                if constexpr(kBlockScale)
                {
                    return make_gemm1_scale<decltype(gemm_1)>(pack_v_scale(
                        v_descale_ptr, kv_tile_start + i_k1 * kK1, kv_last, block_scale_size_kv));
                }
                else
                {
                    ignore = i_k1;
                    return make_gemm1_scale<decltype(gemm_1)>(const_v_scale);
                }
            };

            if constexpr(1 < k1_loops)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    // loop over along the [V]alue Sequence length
                    move_tile_window(v_lds_read_window, {kK1, 0});
                    auto v_tile_switch = load_tile_transpose(v_lds_read_window);

                    gemm_1(o_acc,
                           get_slice_tile(p_tile,
                                          sequence<0, i_k1 * kK1>{},
                                          sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_tile,
                           p_scale_arg,
                           v_scale(i_k1));

                    v_tile = v_tile_switch;
                });
                // move back to the origin
                move_tile_window(v_lds_read_window, {-kK1 * (k1_loops - 1), 0});
            }

            gemm_1(o_acc,
                   get_slice_tile(p_tile,
                                  sequence<0, (k1_loops - 1) * kK1>{},
                                  sequence<kM0, k1_loops * kK1>{}),
                   v_tile,
                   p_scale_arg,
                   v_scale(number<k1_loops - 1>{}));

            s_wait_tensorcnt_barrier<0>();
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
        }; // mainloop

        do
        {
            bool is_even_loop    = i_total_loops % 2 == 0;
            auto k_lds_write_ptr = is_even_loop ? static_cast<KDataType* __restrict__>(smem_ptrk0)
                                                : static_cast<KDataType* __restrict__>(smem_ptrk1);
            auto k_lds_read_ptr  = is_even_loop ? static_cast<KDataType* __restrict__>(smem_ptrk1)
                                                : static_cast<KDataType* __restrict__>(smem_ptrk0);
            auto v_lds_write_ptr = is_even_loop ? static_cast<VDataType* __restrict__>(smem_ptrv1)
                                                : static_cast<VDataType* __restrict__>(smem_ptrv0);
            auto v_lds_read_ptr  = is_even_loop ? static_cast<VDataType* __restrict__>(smem_ptrv0)
                                                : static_cast<VDataType* __restrict__>(smem_ptrv1);
            mainloop(k_lds_write_ptr, k_lds_read_ptr, v_lds_write_ptr, v_lds_read_ptr);
            i_total_loops++;
        } while(i_total_loops < num_total_loop);

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

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            auto tmp             = [&]() {
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
                o_acc(i_j_idx) *= tmp;
            });
        });

        return o_acc;
    }

    // Two overloads rather than defaulted descale arguments: a granularity that
    // needs them can then never be invoked without them, and vice versa.
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
                                        void* smem_ptr,
                                        float sink_v) const
    {
        static_assert(!kQuantized, "qr_tdm pipeline: this granularity needs the descale arguments");
        return run(q_dram_block_window_tmp,
                   k_dram_block_window_tmp,
                   v_dram_block_window_tmp,
                   bias_dram_block_window_tmp,
                   lse_acc_dram_window_tmp,
                   mask,
                   position_encoding,
                   scale_s,
                   smem_ptr,
                   sink_v,
                   nullptr,
                   nullptr,
                   1,
                   e8m0_t(1.0f));
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
                                        void* smem_ptr,
                                        float sink_v,
                                        const float* k_descale_ptr,
                                        const float* v_descale_ptr,
                                        index_t block_scale_size_kv,
                                        e8m0_t v_descale) const
    {
        static_assert(kQuantized,
                      "qr_tdm pipeline: this granularity ignores the descale arguments");
        return run(q_dram_block_window_tmp,
                   k_dram_block_window_tmp,
                   v_dram_block_window_tmp,
                   bias_dram_block_window_tmp,
                   lse_acc_dram_window_tmp,
                   mask,
                   position_encoding,
                   scale_s,
                   smem_ptr,
                   sink_v,
                   k_descale_ptr,
                   v_descale_ptr,
                   block_scale_size_kv,
                   v_descale);
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
        static_assert(!kQuantized, "qr_tdm pipeline: this granularity needs the descale arguments");
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
                   sink_v,
                   nullptr,
                   nullptr,
                   1,
                   e8m0_t(1.0f));
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
                                        void* smem_ptrv1,
                                        const float* k_descale_ptr,
                                        const float* v_descale_ptr,
                                        index_t block_scale_size_kv,
                                        e8m0_t v_descale) const
    {
        static_assert(kQuantized,
                      "qr_tdm pipeline: this granularity ignores the descale arguments");
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
                   sink_v,
                   k_descale_ptr,
                   v_descale_ptr,
                   block_scale_size_kv,
                   v_descale);
    }
};

} // namespace ck_tile
