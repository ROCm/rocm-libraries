// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/sageattention/block/block_sageattention_quant_scale_enum.hpp"
#include "ck_tile/ops/sageattention/pipeline/block_sageattn_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"

namespace ck_tile {

// Quantized sparse attention pipeline (sparge_sage): SageAttention async pipeline grafted with
// the sparge delta-encoded LUT block-skipping. descale-follow-LUT: k descale indexes the real
// absolute position of the LUT-selected block (k_abs_pos), not the contiguous assumption.
template <typename Problem_, typename Policy_ = BlockSageAttentionPipelineQRKSVSAsyncDefaultPolicy>
struct BlockFmhaPipelineQRKSVSAsyncSpargeSage
{
    using Problem             = remove_cvref_t<Problem_>;
    using Policy              = remove_cvref_t<Policy_>;
    using QDataType           = remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using PDataType           = remove_cvref_t<typename Problem::PDataType>;
    static_assert(std::is_same_v<PDataType, VDataType>,
                  "SpargeSage pipeline requires PDataType == VDataType for the PV gemm");
    static_assert(std::is_same_v<QDataType, half_t> || std::is_same_v<QDataType, bf16_t> ||
                      std::is_same_v<PDataType, fp8_t>,
                  "SpargeSage pipeline requires PDataType = fp8_t when Q/K are quantized "
                  "(or half/bf16 Q for the unquantized SageAttn path)");
    static_assert(std::is_same_v<QDataType, half_t> || std::is_same_v<QDataType, bf16_t> ||
                      std::is_same_v<VDataType, fp8_t>,
                  "SpargeSage pipeline requires VDataType = fp8_t when Q/K are quantized "
                  "(or half/bf16 Q for the unquantized SageAttn path)");
    using OaccDataType     = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType        = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant = remove_cvref_t<typename Problem::AttentionVariant>;
    using AttnMask         = remove_cvref_t<typename Problem::AttnMask>;

    using BlockSageAttnShape         = remove_cvref_t<typename Problem::BlockSageAttnShape>;
    using VLayout                    = remove_cvref_t<typename BlockSageAttnShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kNumWarps = kBlockSize / get_warp_size();

    static constexpr index_t kM0           = BlockSageAttnShape::kM0;
    static constexpr index_t kN0           = BlockSageAttnShape::kN0;
    static constexpr index_t kK0           = BlockSageAttnShape::kK0;
    static constexpr index_t kN1           = BlockSageAttnShape::kN1;
    static constexpr index_t kK1           = BlockSageAttnShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockSageAttnShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockSageAttnShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static_assert(Problem::kPadSeqLenQ == true && Problem::kPadHeadDimQ == true &&
                  Problem::kPadHeadDimV == true);
    static constexpr bool kPadSeqLenQ  = true;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = true;
    static constexpr bool kPadHeadDimV = true;
    static constexpr auto QScaleEnum   = Problem::QScaleEnum;

    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();
    static constexpr index_t kAlignmentO = Policy::template GetAlignmentO<Problem>();

    // FP8 softmax shift: exp2(s - m - shift) maps softmax into representable FP8 range.
    // OCP E4M3 max exp 8 -> shift 8; FNUZ E4M3 max exp 7 -> shift 7.
    static constexpr float OCP_FP8_SHIFT  = 8.0f;
    static constexpr float FNUZ_FP8_SHIFT = 7.0f;

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
                return 2;
            }
            else if constexpr(kQKHeaddim <= 192)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            };
        }
    }();

    static constexpr const char* name = "qr_async_sparge_sage";

    // Tail kNumWarps floats reserved for the pv-skip cross-warp predicate reduction.
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>() +
               static_cast<ck_tile::index_t>(kNumWarps * sizeof(float));
    }

    template <BlockAttentionBiasEnum BiasEnum = BlockAttentionBiasEnum::NO_BIAS,
              typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& /*k_element_func*/,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               // ELEMENTWISE_BIAS: [M0, N0] bias window; dummy for NO_BIAS / ALIBI.
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const int* kv_block_idx_ptr,
               int kv_blocks,
               AttnMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               [[maybe_unused]] const float* q_descale_ptr = nullptr,
               const float* k_descale_ptr                  = nullptr,
               const float* v_descale_ptr                  = nullptr,
               [[maybe_unused]] float q_descale_value      = 1.0f,
               float pvthreshd                             = 0.0f,
               const void* pvthreshd_per_head              = nullptr,
               float logits_soft_cap                       = 0.0f) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto LdsSeq = Policy::template GetLdsBufferSequence<Problem>();

        // pv-skip cross-warp reduction scratch (tail reserved in GetSmemSize).
        float* const skip_scratch =
            reinterpret_cast<float*>(reinterpret_cast<char*>(smem_ptr) +
                                     Policy::template GetSmemSize<Problem>());

        // K tile in LDS
        auto k_lds_ptr   = reinterpret_cast<KDataType*>(smem_ptr);
        auto k_lds_store = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(
                        k_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf)),
                    Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf).get_lengths(),
                    {0, 0, 0});
            },
            number<Policy::NumKVLdsBuffers>{});

        auto k_lds_Load_view = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<Problem>());

        auto k_lds_load =
            make_tile_window(k_lds_Load_view,
                             Policy::template MakeKLdsLoadBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());
        q_dram_window.init_raw();

        // K async copy is inline asm, so Q must use inline-asm load too.
        // rocm-6.2+: manually clearing q miscompiles; the distributed-tensor ctor clears it.
        auto q = decltype(load_tile(q_dram_window)){};
        load_tile_raw(q, q_dram_window);
        __builtin_amdgcn_sched_barrier(0);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());

        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using SBlockTileType =
            std::conditional_t<std::is_same_v<typename SaccBlockTileType::DataType, SaccDataType>,
                               SaccBlockTileType,
                               decltype(cast_tile<SaccDataType>(SaccBlockTileType{}))>;

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }
        __builtin_amdgcn_sched_barrier(0);
        const auto q_origin = q_dram_window.get_window_origin();

        // sparge LUT graft: K/V windows traverse LUT-selected K-blocks (LUT entries index
        // K-blocks; multiply by kN0). k_abs_pos tracks the selected block's absolute K position
        // for descale-follow-LUT.
        const int seqlen_k_start = kv_block_idx_ptr[0] * kN0;
        const auto num_total_loop = kv_blocks;
        index_t k_abs_pos = seqlen_k_start;

        if constexpr(AttnMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                // rocm-6.1: fully-masked tile must fence(0) or compute is corrupted.
                buffer_load_fence(0);
                return o_acc;
            }
            __builtin_amdgcn_sched_barrier(0);
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0});

        auto k_dram_window = make_tile_window(
            k_dram_block_window.get_bottom_tensor_view(),
            k_dram_block_window.get_window_lengths(),
            k_dram_block_window.get_window_origin(),
            Policy::template MakeKDramTileDistribution<Problem>());
        k_dram_window.init_raw();
        constexpr auto k_oob_ck = bool_constant<true>{};
        constexpr auto k_pre_np = bool_constant<false>{};

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start},
                             Policy::template MakeVDramTileDistribution<Problem>());

        // ELEMENTWISE_BIAS: load the bias tile into s_acc's distribution so the add is
        // index-aligned regardless of the int8-MFMA SwizzleB/TransposedC C layout. Origin
        // follows the LUT-selected K block.
        auto bias_dram_window = [&]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                return make_tile_window(
                    bias_dram_block_window_tmp.get_bottom_tensor_view(),
                    bias_dram_block_window_tmp.get_window_lengths(),
                    {q_origin.at(number<0>{}), seqlen_k_start},
                    SaccBlockTileType{}.get_tile_distribution());
            }
            else
            {
                return bias_dram_block_window_tmp;
            }
        }();

        async_load_tile_raw(
            k_lds_store(LdsSeq.at(number<0>{})), k_dram_window, number<-1>{}, k_oob_ck, k_pre_np);
        move_tile_window(k_dram_window, {0, kK0});
        __builtin_amdgcn_sched_barrier(0);

        buffer_load_fence(k_dram_window.get_num_of_access(), q.get_thread_buffer());
        (void)q_element_func; // rocm-6.x: applying q element func spills to scratch on hdim=64/32

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);
        constexpr index_t kGemm0MPerWarp = BlockSageAttnShape::Gemm0WarpTile::at(number<0>{});
        static_assert(kGemm0MPerWarp == 32);
        constexpr index_t kWarpSz = get_warp_size();

        // pv-skip: skip a block when every Q-row's block-peak is > pvthreshd below the running
        // max. m_local is taken after bias, so the predicate is valid for NO_BIAS/ALIBI/ELEMENTWISE.
        // On skip m == m_old (rescale 1, l unchanged) and p_compute is zeroed; V-LDS stores /
        // window moves / k_abs_pos stay unconditional to keep descale aligned.
        // soft-cap disables pv-skip (raw-QK units no longer match; cap already bounds the range).
        const float pvthreshd_eff =
            pvthreshd_per_head
                ? reinterpret_cast<const float*>(pvthreshd_per_head)[block_indices.qo_head_idx]
                : pvthreshd;
        const bool stage2_enabled  = (pvthreshd_eff > 0.0f) && (logits_soft_cap <= 0.0f);
        const float skip_threshold = -pvthreshd_eff;

        auto compute_skip_flag = [&](const auto& m_local_t, const auto& m_ij_t) -> bool {
            if(!stage2_enabled)
                return false;
            float lane_max_diff = -ck_tile::numeric<float>::infinity();
            constexpr auto m_spans = remove_cvref_t<decltype(m_ij_t)>::get_distributed_spans();
            sweep_tile_span(m_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const float diff     = static_cast<float>(m_local_t[i_idx]) -
                                       static_cast<float>(m_ij_t[i_idx]);
                if(diff > lane_max_diff)
                    lane_max_diff = diff;
            });
            for(index_t o = get_warp_size() / 2; o > 0; o /= 2) // intra-warp max
            {
                const float v_remote = warp_shuffle_down(lane_max_diff, static_cast<uint32_t>(o));
                if(v_remote > lane_max_diff)
                    lane_max_diff = v_remote;
            }
            const index_t lane_id = ck_tile::get_lane_id();
            const index_t warp_id = ck_tile::get_warp_id();
            if(lane_id == 0)
                skip_scratch[warp_id] = lane_max_diff;
            block_sync_lds();
            float block_max_diff = -ck_tile::numeric<float>::infinity();
#pragma unroll
            for(index_t w = 0; w < kNumWarps; ++w)
            {
                const float v = skip_scratch[w];
                if(v > block_max_diff)
                    block_max_diff = v;
            }
            return block_max_diff < skip_threshold;
        };
        // which half of the warp (used for PERTHREAD K-scale indexing)
        index_t sub_warp_idx = (threadIdx.x % kWarpSz) / kGemm0MPerWarp;
        do
        {
            float k_descale = 1.0f;
            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE)
            {
                // descale-follow-LUT: index by the real absolute position of this LUT-selected block
                const index_t kv_idx = k_abs_pos / Problem::kBlockScaleSizeK;
                k_descale            = k_descale_ptr[kv_idx];
            }
            else if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
            {
                // PERTENSOR: single per-(b,h) global K scale (no descale-follow-LUT).
                k_descale = k_descale_ptr[0];
            }
            constexpr index_t kNumKScalesPW =
                QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP
                    ? kN0 / Problem::kBlockScaleSizeK
                    : 1;
            constexpr index_t kNumKScalesPT =
                QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD
                    ? kN0 / Problem::kBlockScaleSizeK / 2
                    : 1;
            float k_scales_perwarp[kNumKScalesPW > 0 ? kNumKScalesPW : 1] = {};
            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP)
            {
                // descale-follow-LUT: index by the real absolute position of this LUT-selected block
                const index_t kv_idx = k_abs_pos / Problem::kBlockScaleSizeK;
#pragma unroll
                for(index_t i = 0; i < kNumKScalesPW; i++)
                    k_scales_perwarp[i] = k_descale_ptr[kv_idx + i];
            }
            float k_scales_reg[kNumKScalesPT > 0 ? kNumKScalesPT : 1] = {};
            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
            {
                // descale-follow-LUT: index by the real absolute position of this LUT-selected block
                const index_t k_global_start    = k_abs_pos;
                const index_t k_scale_start_idx = k_global_start / Problem::kBlockScaleSizeK;
#pragma unroll
                for(index_t i = 0; i < kNumKScalesPT; i++)
                    k_scales_reg[i] = k_descale_ptr[k_scale_start_idx + 2 * i + sub_warp_idx];
            }

            // STAGE 1, QK gemm
            auto s_acc_gemm = SaccBlockTileType{};
            clear_tile(s_acc_gemm);
            if constexpr(k0_loops > 1)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    async_load_tile_raw(k_lds_store(number<LdsSeq.at(number<i_k0 + 1>{})>{}),
                                        k_dram_window,
                                        number<-1>{},
                                        k_oob_ck,
                                        k_pre_np);
                    if constexpr(i_k0 < k0_loops - 1)
                        move_tile_window(k_dram_window, {0, kK0});

                    async_load_fence(k_dram_window.get_num_of_access());
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    gemm_0(s_acc_gemm,
                           get_slice_tile(
                               q, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{}),
                           get_slice_tile(k_lds_load,
                                          sequence<(LdsSeq.at(number<i_k0>{})) * kN0, 0>{},
                                          sequence<(LdsSeq.at(number<i_k0>{}) + 1) * kN0, kK0>{}));
                });
            }

            // Compiler workaround: at k0_loops<=2 the fence/barrier below gets scheduled
            // inside the 1st loop without this barrier.
            if constexpr(k0_loops <= 2)
                __builtin_amdgcn_sched_barrier(0);

            // WG-uniform LUT delta; readfirstlane pins to SGPR for scalar tile-window math.
            // Guard the read: on the last iteration kv_block_idx_ptr[i_total_loops + 1] would
            // index one past the per-row LUT (OOB on the final row); block_idx is unused there.
            int block_idx =
                (i_total_loops + 1 < num_total_loop)
                    ? __builtin_amdgcn_readfirstlane(kv_block_idx_ptr[i_total_loops + 1])
                    : 0;

            async_load_fence();
            __builtin_amdgcn_s_barrier();

            auto v_buf = load_tile(v_dram_window, number<-1>{}, bool_constant<false>{});
            __builtin_amdgcn_sched_barrier(0);
            { // tail
                gemm_0(
                    s_acc_gemm,
                    get_slice_tile(
                        q, sequence<0, (k0_loops - 1) * kK0>{}, sequence<kM0, k0_loops * kK0>{}),
                    get_slice_tile(k_lds_load,
                                   sequence<(LdsSeq.at(number<k0_loops - 1>{})) * kN0, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops - 1>{}) + 1) * kN0, kK0>{}));
            }
            __builtin_amdgcn_sched_barrier(1);

            auto s_acc = [&]() {
                using GemmDataType = typename decltype(s_acc_gemm)::DataType;
                if constexpr(std::is_same_v<GemmDataType, SaccDataType>)
                {
                    return s_acc_gemm;
                }
                else
                {
                    return cast_tile<SaccDataType>(s_acc_gemm);
                }
            }();

            if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD)
            {
                // PERTHREAD (kBlockScaleSizeK=16): SwizzleB+TransposedC MFMA 32x32x16 gives each
                // thread 16 consecutive K elements, so col_offset>>4 maps to the K scale index.
                static_assert(Problem::kBlockScaleSizeK == 16,
                              "PERTHREAD: kBlockScaleSizeK must be 16");

                using BlockGemm0 = remove_cvref_t<decltype(gemm_0)>;
                constexpr auto WarpGemmCfg =
                    BlockGemm0::Policy::template GetWarpGemmMWarpNWarp<Problem>();
                using WarpGemm0Type = remove_cvref_t<decltype(WarpGemmCfg.template at<0>())>;
                using ExpectedWarpGemmI8 =
                    WarpGemmMfmaI8I8I32M32N32K32SwizzleBTransposedCDistribution<4>;
                using ExpectedWarpGemmFp8 =
                    WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<4>;
                static_assert(
                    std::is_same_v<WarpGemm0Type, ExpectedWarpGemmI8> ||
                        std::is_same_v<WarpGemm0Type, ExpectedWarpGemmFp8>,
                    "PERTHREAD requires "
                    "WarpGemmMfma[I8I8I32|Fp8Fp8F32]M32N32K32SwizzleBTransposedCDistribution for "
                    "16 consecutive K elements");

                constexpr auto s_acc_spans               = decltype(s_acc)::get_distributed_spans();
                float combined_scales_reg[kNumKScalesPT] = {};
#pragma unroll
                for(index_t i = 0; i < kNumKScalesPT; i++)
                    combined_scales_reg[i] = q_descale_value * k_scales_reg[i];
                sweep_tile_span(s_acc_spans[number<0>{}], [&](auto idx0) {
                    index_t col_offset = 0;
                    sweep_tile_span(s_acc_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        // >>4: 16 K elements per scale group (kBlockScaleSizeK=16)
                        const index_t scale_idx = col_offset >> 4;
                        s_acc(i_j_idx) *= combined_scales_reg[scale_idx];
                        col_offset++;
                    });
                });
            }
            else if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP)
            {
                // PERWARP (kBlockScaleSizeK=64): SwizzleB+TransposedC interleaves thread_i and
                // thread_(i+32) over K, so every 32 idx1 steps span 64 K elements (col_offset>>5).
                using BlockGemm0 = remove_cvref_t<decltype(gemm_0)>;
                constexpr auto WarpGemmCfg =
                    BlockGemm0::Policy::template GetWarpGemmMWarpNWarp<Problem>();
                using WarpGemm0Type = remove_cvref_t<decltype(WarpGemmCfg.template at<0>())>;
                using ExpectedWarpGemmI8 =
                    WarpGemmMfmaI8I8I32M32N32K32SwizzleBTransposedCDistribution<4>;
                using ExpectedWarpGemmFp8 =
                    WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<4>;
                static_assert(
                    std::is_same_v<WarpGemm0Type, ExpectedWarpGemmI8> ||
                        std::is_same_v<WarpGemm0Type, ExpectedWarpGemmFp8>,
                    "PERWARP requires "
                    "WarpGemmMfma[I8I8I32|Fp8Fp8F32]M32N32K32SwizzleBTransposedCDistribution for "
                    "correct K element grouping");

                constexpr auto s_acc_spans               = decltype(s_acc)::get_distributed_spans();
                float combined_scales_reg[kNumKScalesPW] = {};
#pragma unroll
                for(index_t i = 0; i < kNumKScalesPW; i++)
                    combined_scales_reg[i] = q_descale_value * k_scales_perwarp[i];
                sweep_tile_span(s_acc_spans[number<0>{}], [&](auto idx0) {
                    index_t col_offset = 0;
                    sweep_tile_span(s_acc_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        // >>5: 64 K elements per scale group (kBlockScaleSizeK=64)
                        const index_t scale_idx = col_offset >> 5;
                        s_acc(i_j_idx) *= combined_scales_reg[scale_idx];
                        col_offset++;
                    });
                });
            }
            else if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                              QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
            {
                // BLOCKSCALE/PERTENSOR: scalar dequant = q_descale_value * k_descale (k_descale
                // is per-block descale-follow-LUT for BLOCKSCALE, k_descale_ptr[0] for PERTENSOR).
                const float qk_descale = q_descale_value * k_descale;
                s_acc = tile_elementwise_in(scales<float>(qk_descale), s_acc);
            }
            else
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
            }

            // Gemma soft-cap on descaled s_acc before masking (NO_BIAS only): with raw scale
            // sc = scale_s/log2e, s' = (cap/sc)*tanh(sc/cap*s); exp2(scale_s*s') then restores
            // exp(cap*tanh(scale*s/cap)). cap == 0 disables.
            if(logits_soft_cap > 0.0f)
            {
                const float sc        = (scale_s != 0.0f)
                                            ? scale_s / ck_tile::log2e_v<float>
                                            : 1.0f;
                const float scc       = sc / logits_soft_cap;
                const float cap_div_sc = logits_soft_cap / sc;
                tile_elementwise_inout(
                    [&](auto& x) {
                        x = type_convert<SaccDataType>(
                            cap_div_sc * tanh_fast<float>(scc * type_convert<float>(x)));
                    },
                    s_acc);
            }

            // ALIBI: add slope*pos to descaled s_acc; kernel pre-divides slope by scale_s so
            // exp2(scale_s*...) restores slope*log2e*pos. col follows the LUT-selected K block.
            if constexpr(!std::is_same_v<remove_cvref_t<PositionEncoding>,
                                         EmptyPositionEncoding<SaccDataType>>)
            {
                const auto k_origin    = k_dram_block_window.get_window_origin();
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }

            // ELEMENTWISE_BIAS: add bias*log2e/scale_s so exp2(scale_s*...) restores bias*log2e.
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const auto bias_tile  = load_tile(bias_dram_window);
                const float bias_gain = (scale_s != 0.0f)
                                            ? (ck_tile::log2e_v<float> / scale_s)
                                            : 0.0f;
                tile_elementwise_inout(
                    [&bias_gain](auto& x, const auto& y) {
                        x += bias_gain * type_convert<SaccDataType>(y);
                    },
                    s_acc,
                    bias_tile);
            }
            // STAGE 2, scale_s, mask, softmax
            if constexpr(kPadSeqLenK || AttnMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});

                if(need_perpixel_check)
                {
                    auto apply_mask = [&](auto&& mask_func) {
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row =
                                    q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                const auto col =
                                    k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                                return !mask_func(variant_params,
                                                  block_indices.batch_idx,
                                                  row,
                                                  col,
                                                  block_indices.qo_head_idx,
                                                  block_indices.kv_head_idx);
                            });
                    };

                    apply_mask([&](auto&&... args) {
                        return variant.LogitsMask(std::forward<decltype(args)>(args)...);
                    });
                }
            }

            const auto s = cast_tile<SMPLComputeDataType>(s_acc);
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity());
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m;
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);

            // pv-skip decided on RAW (descaled) QK m_local vs running m, before the FP8 shift/exp2
            // (same raw-QK basis as sparge + host ref). See pvthreshd_eff comment above.
            const bool skip_block = compute_skip_flag(m_local, m);

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution());

            __builtin_amdgcn_sched_barrier(0x7F);
            // K tail and V share this LDS buffer: barrier so gemm_0's K reads finish before V store.
            if constexpr(LdsSeq.at(number<k0_loops - 1>{}) == LdsSeq.at(number<k0_loops>{}))
            {
                __builtin_amdgcn_s_barrier();
            }
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_buf);

                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});

                store_tile(
                    v_lds_window_tmp,
                    tile_elementwise_in(v_element_func, v_shuffle_tmp));
            }
            else
            {
                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});
                store_tile(v_lds_window_tmp,
                           tile_elementwise_in(v_element_func, v_buf));
            }

            if constexpr(k1_loops > 1)
            {
                // Compiler workaround: moving the window right after load_tile spills to scratch.
                move_tile_window(
                    v_dram_window,
                    {0, kK1});
                v_buf = load_tile(
                    v_dram_window, number<-1>{}, bool_constant<false>{});
            }
            __builtin_amdgcn_sched_barrier(0);

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                if constexpr(AttnMask::IsMasking)
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
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                // precompute row_max = scale_s*m - shift so exp2(scale_s*s - row_max) folds the shift.
                auto validated_m = get_validated_m(m[i_idx]);
                auto row_max     = scale_s * validated_m;
                if constexpr(QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                             QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                             QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD ||
                             QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
                {
#if CK_TILE_USE_OCP_FP8
                    validated_m -= OCP_FP8_SHIFT;
                    row_max -= OCP_FP8_SHIFT;
#else
                    validated_m -= FNUZ_FP8_SHIFT;
                    row_max -= FNUZ_FP8_SHIFT;
#endif
                }
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // pv-skip: zero P so rowsum_p == 0 and the PV gemm contributes nothing.
                    p_compute(i_j_idx) =
                        skip_block ? SMPLComputeDataType{0} : exp2(scale_s * s[i_j_idx] - row_max);
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto m_new = get_validated_m(m[i_idx]);
                auto row_max     = scale_s * m_new;
                const auto tmp   = exp2(scale_s * m_old[i_idx] - row_max);
                l(i_idx) = tmp * l(i_idx) + rowsum_p[i_idx];
                sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    o_acc(i_j_idx) *= tmp;
                });
            });

            const auto p = [&]() {
#if CK_TILE_FMHA_FLOAT_TO_FLOAT16_RTN
                // cast_tile_pkrtz uses cvt_pkrtz (round-to-zero) -> fp32->fp16 precision loss.
                return cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, p_compute));
#else
                if constexpr(std::is_same_v<PDataType, fp16_t>)
                    return impl::cast_tile_pkrtz_fp16_fp32<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
                else
                    return cast_tile<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
#endif
            }();

            // STAGE 3, KV gemm
            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    if constexpr(i_k1 != 0 && i_k1 < k1_loops - 1)
                    {
                        v_buf = load_tile(
                            v_dram_window, number<-1>{}, bool_constant<false>{});
                    }
                    block_sync_lds();
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                           get_slice_tile(
                               v_lds_window,
                               sequence<(LdsSeq.at(number<k0_loops + i_k1>{})) * kN1, 0>{},
                               sequence<(LdsSeq.at(number<k0_loops + i_k1>{}) + 1) * kN1, kK1>{}));

                    if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_tile(v_shuffle_tmp, v_buf);
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp));
                    }
                    else
                    {
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func, v_buf));
                    }
                    if constexpr(i_k1 < k1_loops - 1)
                        move_tile_window(v_dram_window, {0, kK1});
                });
            }
            i_total_loops++;
            if(i_total_loops < num_total_loop)
            {
                // Window advance by the LUT block delta. V already advanced kN0 in this loop, so it
                // needs an extra (block_idx-1)*kN0; K needs block_idx*kN0. k_abs_pos drives k descale.
                k_abs_pos += kN0 * block_idx;

                // bias follows K's LUT delta; otherwise non-contiguous LUT blocks misalign bias.
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    move_tile_window(bias_dram_window, {0, kN0 * block_idx});
                move_tile_window(v_dram_window, {0, kN0 * (block_idx - 1)});
                move_tile_window(k_dram_block_window, {kN0 * block_idx, 0});

                k_dram_window.set_window_origin(k_dram_block_window.get_window_origin());

                if constexpr(k1_loops >= 2 &&
                             LdsSeq.at(number<0>{}) == LdsSeq.at(number<k0_loops + k1_loops - 2>{}))
                    __builtin_amdgcn_s_barrier();
                async_load_tile_raw(k_lds_store(LdsSeq.at(number<0>{})),
                                    k_dram_window,
                                    number<-1>{},
                                    k_oob_ck,
                                    k_pre_np);
                move_tile_window(k_dram_window, {0, kK0});
            }
            // tail
            {
                block_sync_lds();
                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                    get_slice_tile(
                        v_lds_window,
                        sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{})) * kN1, 0>{},
                        sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{}) + 1) * kN1, kK1>{}));
            }

        } while(i_total_loops < num_total_loop);

        // Per-channel v_descale (quantized modes), after the loop and before normalization.
        if constexpr(Problem::QScaleEnum == ck_tile::BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                     Problem::QScaleEnum == ck_tile::BlockSageAttentionQuantScaleEnum::PERWARP ||
                     Problem::QScaleEnum == ck_tile::BlockSageAttentionQuantScaleEnum::PERTHREAD ||
                     Problem::QScaleEnum == ck_tile::BlockSageAttentionQuantScaleEnum::PERTENSOR)
        {
            // Barrier so the last gemm_1's V LDS reads finish before reusing K/V LDS space.
            block_sync_lds();

            // Stage per-channel v_descale [hdim_v] into the now-free K/V LDS space.
            auto v_descale_lds = reinterpret_cast<float*>(smem_ptr);

            const index_t num_threads = kBlockSize;
            for(index_t i = threadIdx.x; i < kN1; i += num_threads)
            {
                v_descale_lds[i] = v_descale_ptr[i];
            }
            block_sync_lds();

            // channel_idx uses tile_idx.at(1) without an i_n1 offset, so this assumes a single
            // N1 tile spans hdim_v (hdim_v <= kN1).
            static_assert(kN1 >= kQKHeaddim,
                          "sage V per-channel descale assumes a single N1 tile "
                          "(hdim_v <= kN1)");
            constexpr auto o_tmp_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_tmp_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(o_tmp_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        o_acc.get_tile_distribution(), i_j_idx);
                    const index_t channel_idx = tile_idx.at(number<1>{});
                    const float v_scale       = v_descale_lds[channel_idx];
                    o_acc(i_j_idx) *= v_scale;
                });
            });
        }

        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(AttnMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <BlockAttentionBiasEnum BiasEnum = BlockAttentionBiasEnum::NO_BIAS,
              typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const int* kv_block_idx_ptr,
               int kv_blocks,
               AttnMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               const float* q_descale_ptr             = nullptr,
               const float* k_descale_ptr             = nullptr,
               const float* v_descale_ptr             = nullptr,
               [[maybe_unused]] float q_descale_value = 1.0f,
               float pvthreshd                        = 0.0f,
               const void* pvthreshd_per_head         = nullptr,
               float logits_soft_cap                  = 0.0f) const
    {
        return operator()<BiasEnum>(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          bias_dram_block_window_tmp,
                          kv_block_idx_ptr,
                          kv_blocks,
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          q_descale_ptr,
                          k_descale_ptr,
                          v_descale_ptr,
                          q_descale_value,
                          pvthreshd,
                          pvthreshd_per_head,
                          logits_soft_cap);
    }
};

} // namespace ck_tile
