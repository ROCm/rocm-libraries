// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/sageattention/block/block_sageattention_quant_scale_enum.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/block_fmha_pipeline_qr_ks_vs_async_sparge_sage.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/block_sparge_mask_pipelines.hpp"
#include "ck_tile/ops/sparse_attn/kernel/fmha_fwd_sparge_kernel.hpp"

#include <cassert>
#include <type_traits>

// sparge_sage attention kernel: quantized sparse attention. Reads quantized Q/K (INT8/FP8 from the
// fused preprocess) + per-(b,h) descales (PERWARP / PERTHREAD / BLOCKSCALE / PERTENSOR) + fp8 V
// (+ per-channel v_descale), traverses the LUT-selected K-blocks, applies SageAttention
// descale-follow-LUT. Batch + group, causal / sliding-window mask, ALIBI / elementwise bias,
// hdim128. Preprocess + mask-prediction kernels are instantiated separately in codegen.

namespace ck_tile {

template <typename SagePipeline_,
          typename EpiloguePipeline_,
          BlockAttentionBiasEnum BiasEnum_ = BlockAttentionBiasEnum::NO_BIAS>
struct FmhaFwdSpargeSageKernel
{
    using SagePipeline     = remove_cvref_t<SagePipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using Problem          = remove_cvref_t<typename SagePipeline::Problem>;

    // Bias added to the descaled fp32 s_acc; elementwise tiles load into the gemm0 C (s_acc)
    // distribution so the int8-MFMA SwizzleB/TransposedC layout aligns by construction.
    static constexpr auto BiasEnum = BiasEnum_;
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                      BiasEnum == BlockAttentionBiasEnum::ALIBI ||
                      BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                  "sparge_sage: only NO_BIAS / ALIBI / ELEMENTWISE_BIAS supported.");

    static constexpr index_t kBlockSize  = SagePipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = SagePipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using QDataType    = remove_cvref_t<typename SagePipeline::QDataType>; // int8
    using KDataType    = remove_cvref_t<typename SagePipeline::KDataType>; // int8
    using VDataType    = remove_cvref_t<typename SagePipeline::VDataType>; // fp8
    using ODataType    = remove_cvref_t<typename SagePipeline::ODataType>;
    using SaccDataType = remove_cvref_t<typename SagePipeline::SaccDataType>;

    using VLayout = remove_cvref_t<typename SagePipeline::VLayout>;

    static constexpr bool kIsGroupMode = SagePipeline::kIsGroupMode;

    static constexpr bool kPadSeqLenQ  = SagePipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = SagePipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = SagePipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = SagePipeline::kPadHeadDimV;
    static constexpr auto QScaleEnum   = Problem::QScaleEnum;
    static_assert(QScaleEnum == BlockSageAttentionQuantScaleEnum::PERWARP ||
                      QScaleEnum == BlockSageAttentionQuantScaleEnum::BLOCKSCALE ||
                      QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTHREAD ||
                      QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR,
                  "sparge_sage: PERWARP|BLOCKSCALE|PERTHREAD|PERTENSOR only");

    using AttentionVariant         = remove_cvref_t<typename SagePipeline::AttentionVariant>;
    using AttnMask                 = remove_cvref_t<typename SagePipeline::AttnMask>;
    static constexpr bool kHasMask = AttnMask::IsMasking;

    static constexpr bool kUseAsyncCopy = SagePipeline::Policy::AsyncCopy;

    struct Kargs
    {
        const void* q_ptr; // int8 (from workspace)
        const void* k_ptr; // int8
        const void* v_ptr; // fp8
        void* o_ptr;

        const void* lut_ptr;             // delta-encoded LUT (int32)
        const void* valid_block_num_ptr; // int32

        const float* q_descale_ptr; // [batch, nhead_q, num_block_scale_q]
        const float* k_descale_ptr; // [batch, nhead_k, num_block_scale_k]
        const float* v_descale_ptr; // [batch, nhead_k, hdim_v] (per-channel)

        index_t seqlen_q;
        index_t seqlen_k;
        index_t hdim_q;
        index_t hdim_v;

        index_t num_head_q;
        index_t nhead_ratio_qk;
        float scale_s;

        index_t stride_q;
        index_t stride_k;
        index_t stride_v;
        index_t stride_o;
        index_t nhead_stride_q;
        index_t nhead_stride_k;
        index_t nhead_stride_v;
        index_t nhead_stride_o;
        index_t batch_stride_q;
        index_t batch_stride_k;
        index_t batch_stride_v;
        index_t batch_stride_o;

        // descale strides; layout [batch, nhead, num_block_scale].
        index_t nhead_stride_q_descale;
        index_t nhead_stride_k_descale;
        index_t nhead_stride_v_descale;
        index_t batch_stride_q_descale;
        index_t batch_stride_k_descale;
        index_t batch_stride_v_descale;
        index_t block_scale_size_q;
        index_t block_scale_size_k;

        // Generic mask window (no-mask: left=-1, right=-1, mask_type=0).
        index_t window_size_left;
        index_t window_size_right;
        GenericAttentionMaskEnum mask_type;

        // ALIBI: slope array; stride_bias=0 -> slope[i_nhead] (shared), else
        // slope[b*stride_bias+h]. ELEMENTWISE_BIAS: dense [.., Sq, Sk]; stride_bias = row stride,
        // nhead/batch select (b,h).
        const void* bias_ptr      = nullptr;
        index_t stride_bias       = 0;
        index_t nhead_stride_bias = 0;
        index_t batch_stride_bias = 0;

        // Group / varlen (batch leaves all nullptr / 0). seqstart_*_ptr: per-batch token starts.
        // seqstart_q_block_ptr / mask_batch_offset_ptr: packed VBN/LUT offsets; q/k descale use the
        // same block-packed scheme (block_id * scales_per_block). quant Q/K and descale nhead
        // strides carry packed totals (host: total_tokens*hdim and total_*_scale).
        const int32_t* seqstart_q_ptr        = nullptr;
        const int32_t* seqstart_k_ptr        = nullptr;
        const int32_t* seqlen_q_ptr          = nullptr;
        const int32_t* seqlen_k_ptr          = nullptr;
        const int32_t* seqstart_q_block_ptr  = nullptr;
        const int32_t* seqstart_k_block_ptr  = nullptr;
        const int32_t* mask_batch_offset_ptr = nullptr;
        index_t batch                        = 0;

        // pv-skip: runtime PV-norm block skip (log2 units; 0 = disabled). per_head ptr (length
        // nhead_q, nullable) overrides the scalar per Q-head.
        float pvthreshd                    = 0.0f;
        const void* pvthreshd_per_head_ptr = nullptr;

        float logits_soft_cap = 0.0f; // 0 = disabled
    };

    CK_TILE_HOST static Kargs MakeKargs(const void* q_ptr,
                                        const void* k_ptr,
                                        const void* v_ptr,
                                        void* o_ptr,
                                        const void* lut_ptr,
                                        const void* valid_block_num_ptr,
                                        const void* q_descale_ptr,
                                        const void* k_descale_ptr,
                                        const void* v_descale_ptr,
                                        index_t seqlen_q,
                                        index_t seqlen_k,
                                        index_t hdim_q,
                                        index_t hdim_v,
                                        index_t num_head_q,
                                        index_t nhead_ratio_qk,
                                        float scale_s,
                                        index_t stride_q,
                                        index_t stride_k,
                                        index_t stride_v,
                                        index_t stride_o,
                                        index_t nhead_stride_q,
                                        index_t nhead_stride_k,
                                        index_t nhead_stride_v,
                                        index_t nhead_stride_o,
                                        index_t batch_stride_q,
                                        index_t batch_stride_k,
                                        index_t batch_stride_v,
                                        index_t batch_stride_o,
                                        index_t nhead_stride_q_descale,
                                        index_t nhead_stride_k_descale,
                                        index_t nhead_stride_v_descale,
                                        index_t batch_stride_q_descale,
                                        index_t batch_stride_k_descale,
                                        index_t batch_stride_v_descale,
                                        index_t block_scale_size_q,
                                        index_t block_scale_size_k,
                                        index_t window_size_left  = -1,
                                        index_t window_size_right = -1,
                                        index_t mask_type         = 0,
                                        const void* bias_ptr      = nullptr,
                                        index_t stride_bias       = 0,
                                        index_t nhead_stride_bias = 0,
                                        index_t batch_stride_bias = 0)
    {
        Kargs kargs;
        kargs.q_ptr               = q_ptr;
        kargs.k_ptr               = k_ptr;
        kargs.v_ptr               = v_ptr;
        kargs.o_ptr               = o_ptr;
        kargs.lut_ptr             = lut_ptr;
        kargs.valid_block_num_ptr = valid_block_num_ptr;
        kargs.q_descale_ptr       = reinterpret_cast<const float*>(q_descale_ptr);
        kargs.k_descale_ptr       = reinterpret_cast<const float*>(k_descale_ptr);
        kargs.v_descale_ptr       = reinterpret_cast<const float*>(v_descale_ptr);
        kargs.seqlen_q            = seqlen_q;
        kargs.seqlen_k            = seqlen_k;
        kargs.hdim_q              = hdim_q;
        kargs.hdim_v              = hdim_v;
        kargs.num_head_q          = num_head_q;
        kargs.nhead_ratio_qk      = nhead_ratio_qk;
#if CK_TILE_FMHA_FWD_FAST_EXP2
        kargs.scale_s = static_cast<float>(scale_s * ck_tile::log2e_v<>);
#else
        kargs.scale_s = scale_s;
#endif
        kargs.stride_q               = stride_q;
        kargs.stride_k               = stride_k;
        kargs.stride_v               = stride_v;
        kargs.stride_o               = stride_o;
        kargs.nhead_stride_q         = nhead_stride_q;
        kargs.nhead_stride_k         = nhead_stride_k;
        kargs.nhead_stride_v         = nhead_stride_v;
        kargs.nhead_stride_o         = nhead_stride_o;
        kargs.batch_stride_q         = batch_stride_q;
        kargs.batch_stride_k         = batch_stride_k;
        kargs.batch_stride_v         = batch_stride_v;
        kargs.batch_stride_o         = batch_stride_o;
        kargs.nhead_stride_q_descale = nhead_stride_q_descale;
        kargs.nhead_stride_k_descale = nhead_stride_k_descale;
        kargs.nhead_stride_v_descale = nhead_stride_v_descale;
        kargs.batch_stride_q_descale = batch_stride_q_descale;
        kargs.batch_stride_k_descale = batch_stride_k_descale;
        kargs.batch_stride_v_descale = batch_stride_v_descale;
        kargs.block_scale_size_q     = block_scale_size_q;
        kargs.block_scale_size_k     = block_scale_size_k;
        kargs.window_size_left       = window_size_left;
        kargs.window_size_right      = window_size_right;
        kargs.mask_type              = static_cast<GenericAttentionMaskEnum>(mask_type);
        kargs.bias_ptr               = bias_ptr;
        kargs.stride_bias            = stride_bias;
        kargs.nhead_stride_bias      = nhead_stride_bias;
        kargs.batch_stride_bias      = batch_stride_bias;
        return kargs;
    }

    // Group-mode kargs: batch kargs plus packed/seqstart tables. seqlen_q/seqlen_k read per-batch
    // on device; int8/descale nhead strides carry packed totals.
    CK_TILE_HOST static Kargs MakeKargsGroup(const void* q_ptr,
                                             const void* k_ptr,
                                             const void* v_ptr,
                                             void* o_ptr,
                                             const void* lut_ptr,
                                             const void* valid_block_num_ptr,
                                             const void* q_descale_ptr,
                                             const void* k_descale_ptr,
                                             const void* v_descale_ptr,
                                             index_t hdim_q,
                                             index_t hdim_v,
                                             index_t num_head_q,
                                             index_t nhead_ratio_qk,
                                             float scale_s,
                                             index_t stride_q,
                                             index_t stride_k,
                                             index_t stride_v,
                                             index_t stride_o,
                                             index_t nhead_stride_q,
                                             index_t nhead_stride_k,
                                             index_t nhead_stride_v,
                                             index_t nhead_stride_o,
                                             index_t nhead_stride_q_descale,
                                             index_t nhead_stride_k_descale,
                                             index_t nhead_stride_v_descale,
                                             index_t batch_stride_v_descale,
                                             index_t block_scale_size_q,
                                             index_t block_scale_size_k,
                                             const int32_t* seqstart_q_ptr,
                                             const int32_t* seqstart_k_ptr,
                                             const int32_t* seqlen_q_ptr,
                                             const int32_t* seqlen_k_ptr,
                                             const int32_t* seqstart_q_block_ptr,
                                             const int32_t* seqstart_k_block_ptr,
                                             const int32_t* mask_batch_offset_ptr,
                                             index_t batch,
                                             index_t window_size_left  = -1,
                                             index_t window_size_right = -1,
                                             index_t mask_type         = 0,
                                             const void* bias_ptr      = nullptr,
                                             index_t stride_bias       = 0,
                                             index_t nhead_stride_bias = 0,
                                             index_t batch_stride_bias = 0)
    {
        Kargs kargs                 = MakeKargs(q_ptr,
                                k_ptr,
                                v_ptr,
                                o_ptr,
                                lut_ptr,
                                valid_block_num_ptr,
                                q_descale_ptr,
                                k_descale_ptr,
                                v_descale_ptr,
                                /*seqlen_q*/ 0,
                                /*seqlen_k*/ 0,
                                hdim_q,
                                hdim_v,
                                num_head_q,
                                nhead_ratio_qk,
                                scale_s,
                                stride_q,
                                stride_k,
                                stride_v,
                                stride_o,
                                nhead_stride_q,
                                nhead_stride_k,
                                nhead_stride_v,
                                nhead_stride_o,
                                /*batch_stride_q*/ 0,
                                /*batch_stride_k*/ 0,
                                /*batch_stride_v*/ 0,
                                /*batch_stride_o*/ 0,
                                nhead_stride_q_descale,
                                nhead_stride_k_descale,
                                nhead_stride_v_descale,
                                /*batch_stride_q_descale*/ 0,
                                /*batch_stride_k_descale*/ 0,
                                batch_stride_v_descale,
                                block_scale_size_q,
                                block_scale_size_k,
                                window_size_left,
                                window_size_right,
                                mask_type,
                                bias_ptr,
                                stride_bias,
                                nhead_stride_bias,
                                batch_stride_bias);
        kargs.seqstart_q_ptr        = seqstart_q_ptr;
        kargs.seqstart_k_ptr        = seqstart_k_ptr;
        kargs.seqlen_q_ptr          = seqlen_q_ptr;
        kargs.seqlen_k_ptr          = seqlen_k_ptr;
        kargs.seqstart_q_block_ptr  = seqstart_q_block_ptr;
        kargs.seqstart_k_block_ptr  = seqstart_k_block_ptr;
        kargs.mask_batch_offset_ptr = mask_batch_offset_ptr;
        kargs.batch                 = batch;
        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(index_t batch_size_, index_t nhead_, index_t seqlen_q_, index_t hdim_v_)
    {
        // Axis order matches the non-quant sparge kernel (nhead=x, batch=y, block=z) so the two
        // sibling kernels that share the preprocess/mask-prediction use one grid convention.
        return dim3(nhead_,
                    batch_size_,
                    integer_divide_ceil(seqlen_q_, SagePipeline::kM0) *
                        integer_divide_ceil(hdim_v_, SagePipeline::kN1));
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        // Masked M-tile reversal below and the pipeline's per-channel V descale both assume a
        // single N1 tile spans hdim_v (num_tile_n1 == 1), i.e. hdim_v <= kN1.
        static_assert(SagePipeline::kN1 >= SagePipeline::kQKHeaddim,
                      "sage masked M-tile reversal assumes a single N1 tile "
                      "(hdim_v <= kN1)");
        const index_t num_tile_n1 = integer_divide_ceil(kargs.hdim_v, SagePipeline::kN1);
        const index_t i_block     = blockIdx.z;
        const index_t i_nhead     = blockIdx.x;
        const index_t i_batch     = blockIdx.y;
        const index_t i_tile_m    = i_block / num_tile_n1;
        const index_t i_tile_n    = i_block - i_tile_m * num_tile_n1;
        if constexpr(kHasMask)
        {
            // Reverse M tile so masked (top) rows run last (assumes num_tile_n1 == 1).
            return make_tuple(gridDim.z - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
        }
        else
        {
            return make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
        }
    }

    CK_TILE_HOST static dim3 BlockSize()
    {
        if(is_wave32())
            return dim3(kBlockSize / 2);
        return dim3(kBlockSize);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(SagePipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct BlockIndices
    {
        index_t batch_idx;
        index_t qo_head_idx;
        index_t kv_head_idx;
    };

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        __shared__ char smem_ptr[GetSmemSize()];

        const auto tile_index  = GetTileIndex(kargs);
        const index_t i_tile_m = tile_index.at(number<0>{});
        const index_t i_tile_n = tile_index.at(number<1>{});
        const index_t i_nhead  = tile_index.at(number<2>{});
        const index_t i_batch  = tile_index.at(number<3>{});
        const index_t i_m0     = __builtin_amdgcn_readfirstlane(i_tile_m * SagePipeline::kM0);
        const index_t i_n1     = __builtin_amdgcn_readfirstlane(i_tile_n * SagePipeline::kN1);

        const index_t i_nhead_k = i_nhead / kargs.nhead_ratio_qk;

        long_index_t batch_offset_q;
        long_index_t batch_offset_k;
        long_index_t batch_offset_v;
        long_index_t batch_offset_o;
        index_t seqlen_q_actual;
        index_t seqlen_k_actual;
        // Group: token starts from seqstart; per-batch scale starts use the same
        // batch-outer/head-mid packed block scheme as means/LUT.
        long_index_t q_scale_block_start = 0; // packed scale-block start for (batch,head) in q
        long_index_t k_scale_block_start = 0; // packed scale-block start for (batch,kv_head) in k
        if constexpr(kIsGroupMode)
        {
            const long_index_t qstart = static_cast<long_index_t>(kargs.seqstart_q_ptr[i_batch]);
            const long_index_t kstart = static_cast<long_index_t>(kargs.seqstart_k_ptr[i_batch]);
            batch_offset_q            = qstart * kargs.stride_q;
            batch_offset_k            = kstart * kargs.stride_k;
            batch_offset_v            = kstart * kargs.stride_v;
            batch_offset_o            = qstart * kargs.stride_o;

            seqlen_q_actual =
                kargs.seqlen_q_ptr != nullptr
                    ? kargs.seqlen_q_ptr[i_batch]
                    : (kargs.seqstart_q_ptr[i_batch + 1] - kargs.seqstart_q_ptr[i_batch]);
            seqlen_k_actual =
                kargs.seqlen_k_ptr != nullptr
                    ? kargs.seqlen_k_ptr[i_batch]
                    : (kargs.seqstart_k_ptr[i_batch + 1] - kargs.seqstart_k_ptr[i_batch]);

            if(static_cast<index_t>(i_tile_m * SagePipeline::kM0) >= seqlen_q_actual)
                return;

            // Packed scale-block start = (bstart*nhead + head*blocks_b) * scales_per_blk.
            const index_t scales_per_blk_q = SagePipeline::kM0 / kargs.block_scale_size_q;
            const index_t scales_per_blk_k = SagePipeline::kN0 / kargs.block_scale_size_k;
            const index_t q_bstart =
                __builtin_amdgcn_readfirstlane(kargs.seqstart_q_block_ptr[i_batch]);
            const index_t q_blocks_b =
                __builtin_amdgcn_readfirstlane(kargs.seqstart_q_block_ptr[i_batch + 1] - q_bstart);
            const index_t k_bstart =
                __builtin_amdgcn_readfirstlane(kargs.seqstart_k_block_ptr[i_batch]);
            const index_t k_blocks_b =
                __builtin_amdgcn_readfirstlane(kargs.seqstart_k_block_ptr[i_batch + 1] - k_bstart);
            q_scale_block_start = (static_cast<long_index_t>(q_bstart) * kargs.num_head_q +
                                   static_cast<long_index_t>(i_nhead) * q_blocks_b) *
                                  scales_per_blk_q;
            const index_t nhead_k = kargs.num_head_q / kargs.nhead_ratio_qk;
            k_scale_block_start   = (static_cast<long_index_t>(k_bstart) * nhead_k +
                                   static_cast<long_index_t>(i_nhead_k) * k_blocks_b) *
                                  scales_per_blk_k;
        }
        else
        {
            batch_offset_q  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_o  = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
            seqlen_q_actual = kargs.seqlen_q;
            seqlen_k_actual = kargs.seqlen_k;
        }

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr = reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                                 static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k +
                                 batch_offset_k;
        const VDataType* v_ptr = reinterpret_cast<const VDataType*>(kargs.v_ptr) +
                                 static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_v +
                                 batch_offset_v;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // LUT / VBN. Batch: rectangular. Group: batch-outer/head-mid packed (lut X_b = q_b*k_b via
        // mask_batch_offset_ptr, vbn X_b = q_b via seqstart_q_block_ptr); new_index = Xstart_b*Hq +
        // head*X_b + local.
        const int* lut_row = [&]() -> const int* {
            const auto* base = reinterpret_cast<const int*>(kargs.lut_ptr);
            if constexpr(kIsGroupMode)
            {
                const long_index_t lut_xstart_b =
                    __builtin_amdgcn_readfirstlane(kargs.mask_batch_offset_ptr[i_batch]);
                const long_index_t lut_x_b =
                    __builtin_amdgcn_readfirstlane(kargs.mask_batch_offset_ptr[i_batch + 1]) -
                    lut_xstart_b;
                const index_t k_blocks_b = integer_divide_ceil(seqlen_k_actual, SagePipeline::kN0);
                return base + lut_xstart_b * kargs.num_head_q +
                       static_cast<long_index_t>(i_nhead) * lut_x_b +
                       static_cast<long_index_t>(i_tile_m) * k_blocks_b;
            }
            else
            {
                const index_t num_q_blocks = integer_divide_ceil(kargs.seqlen_q, SagePipeline::kM0);
                const index_t num_k_blocks = integer_divide_ceil(kargs.seqlen_k, SagePipeline::kN0);
                return base + ((static_cast<long_index_t>(i_batch) * kargs.num_head_q + i_nhead) *
                                   num_q_blocks +
                               i_tile_m) *
                                  num_k_blocks;
            }
        }();
        const int valid_block_num_value = [&]() -> int {
            const auto* base = reinterpret_cast<const int*>(kargs.valid_block_num_ptr);
            if constexpr(kIsGroupMode)
            {
                const long_index_t vbn_xstart_b =
                    __builtin_amdgcn_readfirstlane(kargs.seqstart_q_block_ptr[i_batch]);
                const long_index_t vbn_x_b =
                    __builtin_amdgcn_readfirstlane(kargs.seqstart_q_block_ptr[i_batch + 1]) -
                    vbn_xstart_b;
                return base[vbn_xstart_b * kargs.num_head_q +
                            static_cast<long_index_t>(i_nhead) * vbn_x_b +
                            static_cast<long_index_t>(i_tile_m)];
            }
            else
            {
                const index_t num_q_blocks = integer_divide_ceil(kargs.seqlen_q, SagePipeline::kM0);
                return base[(static_cast<long_index_t>(i_batch) * kargs.num_head_q + i_nhead) *
                                num_q_blocks +
                            i_tile_m];
            }
        }();

        // descale pointers. Batch: offset to (batch, head). Group: q/k use the packed scale-block
        // start (indexed by row/abs-pos / tps); v descale stays per-channel (per batch,head_k).
        // PERTENSOR: one global scale per (batch,head) laid as [batch, nhead, 1] in both modes, so
        // the per-(b,h) offset is identical (q_scale_block_start applies only to packed schemes).
        const float* q_descale_ptr =
            (QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
                ? kargs.q_descale_ptr +
                      static_cast<long_index_t>(i_batch) * kargs.batch_stride_q_descale +
                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q_descale
            : kIsGroupMode ? kargs.q_descale_ptr + q_scale_block_start
                           : kargs.q_descale_ptr +
                                 static_cast<long_index_t>(i_batch) * kargs.batch_stride_q_descale +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q_descale;
        const float* k_descale_ptr =
            (QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
                ? kargs.k_descale_ptr +
                      static_cast<long_index_t>(i_batch) * kargs.batch_stride_k_descale +
                      static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k_descale
            : kIsGroupMode
                ? kargs.k_descale_ptr + k_scale_block_start
                : kargs.k_descale_ptr +
                      static_cast<long_index_t>(i_batch) * kargs.batch_stride_k_descale +
                      static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k_descale;
        const float* v_descale_ptr =
            kargs.v_descale_ptr +
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_v_descale +
            static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_v_descale;

        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(seqlen_q_actual, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<SagePipeline::kAlignmentQ>{},
                number<1>{});
            return pad_tensor_view(
                q_dram_naive,
                make_tuple(number<SagePipeline::kM0>{}, number<SagePipeline::kSubQKHeaddim>{}),
                sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(seqlen_k_actual, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<SagePipeline::kAlignmentK>{},
                number<1>{});
            constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<SagePipeline::kN0>{}, number<SagePipeline::kK0>{}),
                sequence<kPadSeqLenK_, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(seqlen_k_actual, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<SagePipeline::kAlignmentV>{},
                    number<1>{});
                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                     make_pass_through_transform(seqlen_k_actual)),
                                          make_tuple(sequence<1>{}, sequence<0>{}),
                                          make_tuple(sequence<0>{}, sequence<1>{}));
                constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
                return pad_tensor_view(
                    v_dram_transposed,
                    make_tuple(number<SagePipeline::kN1>{}, number<SagePipeline::kK1>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK_>{});
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, seqlen_k_actual),
                    make_tuple(kargs.stride_v, 1),
                    number<SagePipeline::kAlignmentV>{},
                    number<1>{});
                constexpr bool kPadHeadDimV_ = kUseAsyncCopy ? kPadHeadDimV : false;
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<SagePipeline::kN1>{}, number<SagePipeline::kK1>{}),
                    sequence<kPadHeadDimV_, kPadSeqLenK>{});
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<SagePipeline::kM0>{}, number<SagePipeline::kSubQKHeaddim>{}),
            {i_m0, 0});
        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<SagePipeline::kN0>{}, number<SagePipeline::kK0>{}), {0, 0});
        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<SagePipeline::kN1>{}, number<SagePipeline::kK1>{}),
                             {i_n1, 0});

        // ELEMENTWISE_BIAS: dense [.., Sq, Sk] for this (batch, head); pipeline loads [M0,N0] tiles
        // into the gemm0 C (s_acc) distribution and advances by the LUT delta. NO_BIAS / ALIBI get
        // a 1x1 dummy window.
        auto bias_dram_window = [&]() {
            using BiasDataType = SaccDataType;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const auto* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias +
                    static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_bias;
                const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    bias_ptr,
                    make_tuple(seqlen_q_actual, seqlen_k_actual),
                    make_tuple(kargs.stride_bias, 1),
                    number<1>{},
                    number<1>{});
                const auto bias_dram = pad_tensor_view(
                    bias_dram_naive,
                    make_tuple(number<SagePipeline::kM0>{}, number<SagePipeline::kN0>{}),
                    sequence<kPadSeqLenQ, kPadSeqLenK>{});
                return make_tile_window(
                    bias_dram,
                    make_tuple(number<SagePipeline::kM0>{}, number<SagePipeline::kN0>{}),
                    {i_m0, 0});
            }
            else
            {
                const BiasDataType* null_bias = static_cast<const BiasDataType*>(nullptr);
                const auto dummy_naive        = make_naive_tensor_view<address_space_enum::global>(
                    null_bias, make_tuple(1, 1), make_tuple(1, 1), number<1>{}, number<1>{});
                const auto dummy = pad_tensor_view(
                    dummy_naive,
                    make_tuple(number<SagePipeline::kM0>{}, number<SagePipeline::kN0>{}),
                    sequence<true, true>{});
                return make_tile_window(
                    dummy,
                    make_tuple(number<SagePipeline::kM0>{}, number<SagePipeline::kN0>{}),
                    {0, 0});
            }
        }();

        AttnMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<AttnMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    seqlen_q_actual,
                    seqlen_k_actual,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return AttnMask{seqlen_q_actual, seqlen_k_actual};
        }();
        // ALIBI: pipeline adds slope_eff*pos to the descaled s_acc, then softmax multiplies by
        // kargs.scale_s (= scale*log2e under FAST_EXP2) inside exp2. To reproduce the reference
        // exp(scale*s + slope*pos), set slope_eff = slope*log2e / scale_s = slope / scale.
        // Slope addressing matches sparge: stride_bias=0 -> slope[i_nhead]; else slope[b*stride+h].
        auto position_encoding = [&]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto* slope_arr = reinterpret_cast<const SaccDataType*>(kargs.bias_ptr);
                const long_index_t off =
                    static_cast<long_index_t>(i_batch) * kargs.stride_bias + i_nhead;
                SaccDataType slope = slope_arr ? slope_arr[off] : SaccDataType{0};
                const float scale_eff =
                    kargs.scale_s != 0.0f ? (ck_tile::log2e_v<float> / kargs.scale_s) : 0.0f;
                slope = static_cast<SaccDataType>(static_cast<float>(slope) * scale_eff);
                return ck_tile::make_alibi_from_lr_mask<SaccDataType, true>(
                    slope,
                    kHasMask ? kargs.window_size_left : -1,
                    kHasMask ? kargs.window_size_right : 0,
                    seqlen_q_actual,
                    seqlen_k_actual,
                    kHasMask ? kargs.mask_type : GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            }
            else
            {
                return ck_tile::EmptyPositionEncoding<SaccDataType>{};
            }
        }();

        AttentionVariant variant;
        StandardAttentionParams<AttnMask> variant_params{mask, kargs.scale_s};

        // PERWARP q_descale: one scalar per warp (Gemm0 MPerWarp == kBlockScaleSizeQ).
        using SageShape                  = typename SagePipeline::BlockSageAttnShape;
        constexpr index_t kWarpSize      = get_warp_size();
        constexpr index_t kGemm0MPerWarp = SageShape::Gemm0WarpTile::at(number<0>{});
        constexpr index_t kBlockSq       = Problem::kBlockScaleSizeQ;
        const index_t wave_id            = __builtin_amdgcn_readfirstlane(threadIdx.x / kWarpSize);
        const index_t q_row_raw = i_m0 + wave_id * kGemm0MPerWarp + threadIdx.x % kGemm0MPerWarp;
        const index_t q_scale_idx_raw = integer_divide_floor(q_row_raw, kBlockSq);
        const index_t max_q_scale_idx =
            seqlen_q_actual > 0 ? integer_divide_ceil(seqlen_q_actual, kBlockSq) - 1 : 0;
        const index_t q_scale_idx =
            q_scale_idx_raw < max_q_scale_idx ? q_scale_idx_raw : max_q_scale_idx;
        // PERTENSOR: single per-(b,h) scalar; other modes index by scale-block.
        const float q_descale = (QScaleEnum == BlockSageAttentionQuantScaleEnum::PERTENSOR)
                                    ? q_descale_ptr[0]
                                    : q_descale_ptr[q_scale_idx];

        BlockIndices block_indices{i_batch, i_nhead, i_nhead_k};

        auto o_acc_tile = SagePipeline{}.template operator()<BiasEnum>(q_dram_window,
                                                                       k_dram_window,
                                                                       v_dram_window,
                                                                       bias_dram_window,
                                                                       lut_row,
                                                                       valid_block_num_value,
                                                                       mask,
                                                                       position_encoding,
                                                                       kargs.scale_s,
                                                                       variant,
                                                                       variant_params,
                                                                       block_indices,
                                                                       smem_ptr,
                                                                       nullptr,
                                                                       k_descale_ptr,
                                                                       v_descale_ptr,
                                                                       q_descale,
                                                                       kargs.pvthreshd,
                                                                       kargs.pvthreshd_per_head_ptr,
                                                                       kargs.logits_soft_cap);

        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(seqlen_q_actual, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<SagePipeline::kAlignmentO>{},
                number<1>{});
            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<SagePipeline::kM0>{}, number<SagePipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();
        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<SagePipeline::kM0>{}, number<SagePipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
