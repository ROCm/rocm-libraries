// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/block_fmha_pipeline_qr_ks_vs_async_sparge.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/block_sparge_mask_pipelines.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

// SpargeAttention pipeline: preprocess (K/Q means + sims) -> mask prediction -> attention.
// hp.smooth_k is intentionally a no-op: softmax max-subtract absorbs the
// constant -dot(q_means, km) shift, so the correction step is unnecessary.

namespace ck_tile {

struct SpargePreprocessKargs
{
    const void* k_data;
    const void* q_data;

    // Batch: [B,H,num_blocks,D] / [B,H,num_blocks]. Group: [H,total_blocks,D] / [H,total_blocks].
    float* k_means;
    float* k_sim;   // nullable when simthreshold <= 0
    float* q_means; // nullptr skips Q means
    float* q_sim;

    index_t batch;
    index_t nhead_k;
    index_t nhead_q;
    index_t seqlen_k;
    index_t seqlen_q;
    index_t hdim;
    index_t block_size;
    index_t num_k_blocks;       // batch: per-sequence; group: max across batches
    index_t num_q_blocks;
    index_t k_total_wg;         // batch: B*Hk*num_k_blocks; group: Hk*total_k_blocks

    index_t k_batch_stride;
    index_t k_head_stride;
    index_t k_seq_stride;
    index_t q_batch_stride;
    index_t q_head_stride;
    index_t q_seq_stride;

    float simthreshold;
    const float* km_ptr;  // K smoothing km[B,Hk,D]; nullptr disables.

    // Group / varlen mode (batch leaves all nullptr).
    const int32_t* seqstart_q_ptr        = nullptr;
    const int32_t* seqstart_k_ptr        = nullptr;
    const int32_t* seqlen_q_ptr          = nullptr;
    const int32_t* seqlen_k_ptr          = nullptr;
    const int32_t* seqstart_q_block_ptr  = nullptr;
    const int32_t* seqstart_k_block_ptr  = nullptr;
    index_t        total_q_blocks        = 0;
    index_t        total_k_blocks        = 0;
};

// Grid: K blocks first (k_total_wg), then Q blocks. K branch optionally subtracts global
// K-mean (km_ptr, fused). Group mode is head-major packed; binary-search seqstart_*_block_ptr
// recovers (b, block_in_batch).
template <typename InputType_, bool kIsGroupMode_ = false>
struct FmhaFwdSpargePreprocessKernel
{
    using InputType = remove_cvref_t<InputType_>;
    using Pipeline  = BlockSpargePreprocessPipeline<InputType>;

    static constexpr bool kIsGroupMode = kIsGroupMode_;

    static constexpr index_t kBlockSize  = Pipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = 4;

    CK_TILE_HOST static dim3 GridSize(const SpargePreprocessKargs& kargs)
    {
        uint32_t total = static_cast<uint32_t>(kargs.k_total_wg);
        if(kargs.q_means != nullptr)
        {
            if constexpr(kIsGroupMode)
                total += static_cast<uint32_t>(kargs.nhead_q * kargs.total_q_blocks);
            else
                total += static_cast<uint32_t>(kargs.batch * kargs.nhead_q * kargs.num_q_blocks);
        }
        return dim3(total);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static std::size_t GetSmemSize(index_t hdim)
    {
        return static_cast<std::size_t>(Pipeline::GetSmemSize(hdim));
    }

    // Binary search: largest b with seqstart_block_ptr[b] <= g_block.
    // readfirstlane keeps the WG-uniform result in SGPR.
    CK_TILE_DEVICE static index_t
    block_to_batch(const int32_t* seqstart_block_ptr, index_t batch, index_t g_block)
    {
        index_t lo = 0;
        index_t hi = batch;
        while(lo + 1 < hi)
        {
            const index_t mid = (lo + hi) / 2;
            if(g_block < seqstart_block_ptr[mid])
                hi = mid;
            else
                lo = mid;
        }
        return __builtin_amdgcn_readfirstlane(lo);
    }

    CK_TILE_DEVICE void operator()(SpargePreprocessKargs kargs) const
    {
        const index_t gid = static_cast<index_t>(blockIdx.x);
        extern __shared__ char smem_raw[];

        if(gid < kargs.k_total_wg)
        {
            index_t      head_id, batch_id, block_id;
            long_index_t k_batch_offset;
            index_t      seqlen_k_actual;
            long_index_t mean_out_off;
            long_index_t sim_out_off;
            long_index_t km_bh_off;

            if constexpr(kIsGroupMode)
            {
                const index_t g_block = __builtin_amdgcn_readfirstlane(gid % kargs.total_k_blocks);
                head_id               = __builtin_amdgcn_readfirstlane(gid / kargs.total_k_blocks);
                batch_id              = block_to_batch(
                    kargs.seqstart_k_block_ptr, kargs.batch, g_block);
                const index_t kstart_b = __builtin_amdgcn_readfirstlane(
                    kargs.seqstart_k_block_ptr[batch_id]);
                block_id              = g_block - kstart_b;

                const long_index_t kstart =
                    static_cast<long_index_t>(kargs.seqstart_k_ptr[batch_id]);
                k_batch_offset  = kstart * kargs.k_seq_stride;
                seqlen_k_actual = __builtin_amdgcn_readfirstlane(
                    kargs.seqlen_k_ptr != nullptr
                        ? kargs.seqlen_k_ptr[batch_id]
                        : (kargs.seqstart_k_ptr[batch_id + 1] - kargs.seqstart_k_ptr[batch_id]));

                mean_out_off =
                    (static_cast<long_index_t>(head_id) * kargs.total_k_blocks + g_block) *
                    kargs.hdim;
                sim_out_off =
                    static_cast<long_index_t>(head_id) * kargs.total_k_blocks + g_block;
                km_bh_off =
                    static_cast<long_index_t>(batch_id) * kargs.nhead_k + head_id;
            }
            else
            {
                block_id = gid % kargs.num_k_blocks;
                head_id  = (gid / kargs.num_k_blocks) % kargs.nhead_k;
                batch_id = gid / (kargs.nhead_k * kargs.num_k_blocks);

                k_batch_offset  = static_cast<long_index_t>(batch_id) * kargs.k_batch_stride;
                seqlen_k_actual = kargs.seqlen_k;

                const long_index_t bh =
                    static_cast<long_index_t>(batch_id) * kargs.nhead_k + head_id;
                mean_out_off = bh * static_cast<long_index_t>(kargs.num_k_blocks) * kargs.hdim +
                               static_cast<long_index_t>(block_id) * kargs.hdim;
                sim_out_off  = bh * kargs.num_k_blocks + block_id;
                km_bh_off    = bh;
            }

            const auto* slice =
                reinterpret_cast<const InputType*>(kargs.k_data) +
                k_batch_offset +
                static_cast<long_index_t>(head_id) * kargs.k_head_stride;
            float* mean_out = kargs.k_means + mean_out_off;
            float* sim_out  = (kargs.k_sim != nullptr) ? kargs.k_sim + sim_out_off : nullptr;

            typename Pipeline::Params pp;
            pp.seqlen        = seqlen_k_actual;
            pp.hdim          = kargs.hdim;
            pp.block_size    = kargs.block_size;
            pp.block_id      = block_id;
            pp.stride_seq    = kargs.k_seq_stride;
            pp.simthreshold = kargs.simthreshold;
            pp.km_ptr        = kargs.km_ptr
                ? kargs.km_ptr + km_bh_off * static_cast<long_index_t>(kargs.hdim) : nullptr;

            Pipeline{}(slice, mean_out, sim_out, pp, smem_raw);
        }
        else
        {
            const index_t q_gid = gid - kargs.k_total_wg;
            index_t      head_id, batch_id, block_id;
            long_index_t q_batch_offset;
            index_t      seqlen_q_actual;
            long_index_t mean_out_off;
            long_index_t sim_out_off;

            if constexpr(kIsGroupMode)
            {
                const index_t g_block = __builtin_amdgcn_readfirstlane(q_gid % kargs.total_q_blocks);
                head_id               = __builtin_amdgcn_readfirstlane(q_gid / kargs.total_q_blocks);
                batch_id              = block_to_batch(
                    kargs.seqstart_q_block_ptr, kargs.batch, g_block);
                const index_t qstart_b = __builtin_amdgcn_readfirstlane(
                    kargs.seqstart_q_block_ptr[batch_id]);
                block_id              = g_block - qstart_b;

                const long_index_t qstart =
                    static_cast<long_index_t>(kargs.seqstart_q_ptr[batch_id]);
                q_batch_offset  = qstart * kargs.q_seq_stride;
                seqlen_q_actual = __builtin_amdgcn_readfirstlane(
                    kargs.seqlen_q_ptr != nullptr
                        ? kargs.seqlen_q_ptr[batch_id]
                        : (kargs.seqstart_q_ptr[batch_id + 1] - kargs.seqstart_q_ptr[batch_id]));

                mean_out_off =
                    (static_cast<long_index_t>(head_id) * kargs.total_q_blocks + g_block) *
                    kargs.hdim;
                sim_out_off =
                    static_cast<long_index_t>(head_id) * kargs.total_q_blocks + g_block;
            }
            else
            {
                block_id = q_gid % kargs.num_q_blocks;
                head_id  = (q_gid / kargs.num_q_blocks) % kargs.nhead_q;
                batch_id = q_gid / (kargs.nhead_q * kargs.num_q_blocks);

                q_batch_offset  = static_cast<long_index_t>(batch_id) * kargs.q_batch_stride;
                seqlen_q_actual = kargs.seqlen_q;

                const long_index_t bh =
                    static_cast<long_index_t>(batch_id) * kargs.nhead_q + head_id;
                mean_out_off = bh * static_cast<long_index_t>(kargs.num_q_blocks) * kargs.hdim +
                               static_cast<long_index_t>(block_id) * kargs.hdim;
                sim_out_off  = bh * kargs.num_q_blocks + block_id;
            }

            const auto* slice =
                reinterpret_cast<const InputType*>(kargs.q_data) +
                q_batch_offset +
                static_cast<long_index_t>(head_id) * kargs.q_head_stride;
            float* mean_out = kargs.q_means + mean_out_off;
            float* sim_out  = (kargs.q_sim != nullptr) ? kargs.q_sim + sim_out_off : nullptr;

            typename Pipeline::Params pp;
            pp.seqlen        = seqlen_q_actual;
            pp.hdim          = kargs.hdim;
            pp.block_size    = kargs.block_size;
            pp.block_id      = block_id;
            pp.stride_seq    = kargs.q_seq_stride;
            pp.simthreshold = kargs.simthreshold;
            pp.km_ptr        = nullptr;

            Pipeline{}(slice, mean_out, sim_out, pp, smem_raw);
        }
    }
};

struct SpargeMaskPredictionKargs
{
    const float* k_means;
    const float* q_means;
    const float* k_sim;            // nullable when simthreshold <= 0
    const float* q_sim;
    int32_t* lut_out;              // batch: [B,Hq,nq,nk]; group: packed by lut_batch_offset_ptr
    int32_t* valid_block_num_out;  // batch: [B,Hq,nq]; group: [Hq,total_q_blocks]

    index_t batch;
    index_t nhead_q;
    index_t nhead_k;
    index_t nhead_ratio_qk;
    index_t num_q_blocks;
    index_t num_k_blocks;
    index_t hdim;
    float cdfthreshd;
    float topk;
    float simthreshold;
    const float* cdfthreshd_per_head;
    const float* topk_per_head;
    const float* simthreshold_per_head;
    index_t causal_type;
    bool attention_sink;
    index_t seqlen_q;
    index_t seqlen_k;
    index_t block_size;
    index_t window_left;
    index_t window_right;

    // Group / varlen mode (batch leaves all nullptr).
    const int32_t* seqstart_q_ptr        = nullptr;
    const int32_t* seqstart_k_ptr        = nullptr;
    const int32_t* seqlen_q_ptr          = nullptr;
    const int32_t* seqlen_k_ptr          = nullptr;
    const int32_t* seqstart_q_block_ptr  = nullptr;
    const int32_t* seqstart_k_block_ptr  = nullptr;
    const int32_t* lut_batch_offset_ptr  = nullptr;
    index_t        total_q_blocks        = 0;
    index_t        total_k_blocks        = 0;
};

template <bool kIsGroupMode_ = false>
struct FmhaFwdSpargeMaskPredictionKernel
{
    using Pipeline = BlockSpargeMaskPredictionPipeline;

    static constexpr bool kIsGroupMode = kIsGroupMode_;

    static constexpr index_t kBlockSize  = Pipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = 4;

    CK_TILE_HOST static dim3 GridSize(const SpargeMaskPredictionKargs& kargs)
    {
        assert(kargs.num_k_blocks <= Pipeline::kMaxKBlocksPow2 &&
               "num_k_blocks exceeds sort capacity (kMaxKBlocksPow2)");
        if constexpr(kIsGroupMode)
            return dim3(static_cast<uint32_t>(kargs.nhead_q * kargs.total_q_blocks));
        else
            return dim3(static_cast<uint32_t>(kargs.batch * kargs.nhead_q * kargs.num_q_blocks));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static std::size_t GetSmemSize(index_t hdim, index_t num_k_blocks)
    {
        return static_cast<std::size_t>(Pipeline::GetSmemSize(hdim, num_k_blocks));
    }

    CK_TILE_DEVICE static index_t
    block_to_batch(const int32_t* seqstart_block_ptr, index_t batch, index_t g_block)
    {
        index_t lo = 0;
        index_t hi = batch;
        while(lo + 1 < hi)
        {
            const index_t mid = (lo + hi) / 2;
            if(g_block < seqstart_block_ptr[mid])
                hi = mid;
            else
                lo = mid;
        }
        return __builtin_amdgcn_readfirstlane(lo);
    }

    CK_TILE_DEVICE void operator()(SpargeMaskPredictionKargs kargs) const
    {
        extern __shared__ char smem_raw[];

        if constexpr(kIsGroupMode)
        {
            const index_t gid       = static_cast<index_t>(blockIdx.x);
            const index_t g_q_block = __builtin_amdgcn_readfirstlane(gid % kargs.total_q_blocks);
            const index_t head      = __builtin_amdgcn_readfirstlane(gid / kargs.total_q_blocks);
            const index_t kv_head   = __builtin_amdgcn_readfirstlane(head / kargs.nhead_ratio_qk);
            const index_t b         = block_to_batch(
                kargs.seqstart_q_block_ptr, kargs.batch, g_q_block);
            const index_t qstart_b   = __builtin_amdgcn_readfirstlane(
                kargs.seqstart_q_block_ptr[b]);
            const index_t kstart_b   = __builtin_amdgcn_readfirstlane(
                kargs.seqstart_k_block_ptr[b]);
            const index_t q_block_in_b = g_q_block - qstart_b;

            const index_t seqlen_q_b = __builtin_amdgcn_readfirstlane(
                kargs.seqlen_q_ptr
                    ? kargs.seqlen_q_ptr[b]
                    : (kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]));
            const index_t seqlen_k_b = __builtin_amdgcn_readfirstlane(
                kargs.seqlen_k_ptr
                    ? kargs.seqlen_k_ptr[b]
                    : (kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]));
            const index_t num_q_blocks_b =
                ck_tile::integer_divide_ceil(seqlen_q_b, kargs.block_size);
            const index_t num_k_blocks_b =
                ck_tile::integer_divide_ceil(seqlen_k_b, kargs.block_size);

            // Pre-slice base pointers to (head, batch's q-blocks); pipeline indexes
            // (b*nhead+head)*num_blocks*D so pass b=0, nhead=1 with block_id=q_block_in_b.
            const long_index_t q_means_head_off =
                static_cast<long_index_t>(head) * kargs.total_q_blocks * kargs.hdim;
            const float* q_means_view =
                kargs.q_means + q_means_head_off +
                static_cast<long_index_t>(qstart_b) * kargs.hdim;
            const float* q_sim_view = (kargs.q_sim != nullptr)
                ? kargs.q_sim +
                    static_cast<long_index_t>(head) * kargs.total_q_blocks + qstart_b
                : nullptr;

            const long_index_t k_means_head_off =
                static_cast<long_index_t>(kv_head) * kargs.total_k_blocks * kargs.hdim;
            const float* k_means_view =
                kargs.k_means + k_means_head_off +
                static_cast<long_index_t>(kstart_b) * kargs.hdim;
            const float* k_sim_view = (kargs.k_sim != nullptr)
                ? kargs.k_sim +
                    static_cast<long_index_t>(kv_head) * kargs.total_k_blocks + kstart_b
                : nullptr;

            const long_index_t per_head_lut_size =
                static_cast<long_index_t>(kargs.lut_batch_offset_ptr[kargs.batch]);
            const long_index_t lut_b_off =
                static_cast<long_index_t>(kargs.lut_batch_offset_ptr[b]);
            int32_t* lut_row =
                kargs.lut_out +
                static_cast<long_index_t>(head) * per_head_lut_size +
                lut_b_off +
                static_cast<long_index_t>(q_block_in_b) * num_k_blocks_b;
            int32_t* vbn_ptr =
                kargs.valid_block_num_out +
                static_cast<long_index_t>(head) * kargs.total_q_blocks +
                qstart_b + q_block_in_b;

            const float head_cdfthreshd   = kargs.cdfthreshd_per_head
                ? kargs.cdfthreshd_per_head[head] : kargs.cdfthreshd;
            const float head_topk          = kargs.topk_per_head
                ? kargs.topk_per_head[head] : kargs.topk;
            const float head_simthreshold = kargs.simthreshold_per_head
                ? kargs.simthreshold_per_head[head] : kargs.simthreshold;

            Pipeline::MaskRunArgs args;
            args.k_means      = k_means_view;
            args.q_means      = q_means_view;
            args.k_sim        = k_sim_view;
            args.q_sim        = q_sim_view;
            args.lut_row      = lut_row;
            args.vbn_ptr      = vbn_ptr;
            args.b            = 0;
            args.head         = 0;
            args.kv_head      = 0;
            args.q_block      = q_block_in_b;
            args.nhead_q      = 1;
            args.nhead_k      = 1;
            args.num_q_blocks = num_q_blocks_b;
            args.num_k_blocks = num_k_blocks_b;
            args.hdim         = kargs.hdim;
            args.head_cdfthreshd   = head_cdfthreshd;
            args.head_topk          = head_topk;
            args.head_simthreshold = head_simthreshold;
            args.causal_type    = kargs.causal_type;
            args.attention_sink = kargs.attention_sink;
            args.seqlen_q       = seqlen_q_b;
            args.seqlen_k       = seqlen_k_b;
            args.block_size     = kargs.block_size;
            args.window_left    = kargs.window_left;
            args.window_right   = kargs.window_right;
            Pipeline{}.run_with_indices(args, smem_raw);
        }
        else
        {
            Pipeline{}(kargs.k_means,
                       kargs.q_means,
                       kargs.k_sim,
                       kargs.q_sim,
                       kargs.lut_out,
                       kargs.valid_block_num_out,
                       kargs.nhead_q,
                       kargs.nhead_k,
                       kargs.nhead_ratio_qk,
                       kargs.num_q_blocks,
                       kargs.num_k_blocks,
                       kargs.hdim,
                       kargs.cdfthreshd,
                       kargs.topk,
                       kargs.simthreshold,
                       kargs.cdfthreshd_per_head,
                       kargs.topk_per_head,
                       kargs.simthreshold_per_head,
                       kargs.causal_type,
                       kargs.attention_sink,
                       kargs.seqlen_q,
                       kargs.seqlen_k,
                       kargs.block_size,
                       kargs.window_left,
                       kargs.window_right,
                       smem_raw);
        }
    }
};

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdSpargeKernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using RandValOutputDataType =
        ck_tile::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    using PreprocessKernel        = FmhaFwdSpargePreprocessKernel<QDataType, FmhaPipeline::kIsGroupMode>;
    using MaskPredictionKernel    = FmhaFwdSpargeMaskPredictionKernel<FmhaPipeline::kIsGroupMode>;

    using VLayout = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = FmhaPipeline::BiasEnum;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;
    static constexpr bool kHasDropout       = FmhaPipeline::kHasDropout;
    static constexpr bool kDoFp8StaticQuant =
        (FmhaPipeline::Problem::QScaleEnum != ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE);
    static constexpr bool kIsGroupMode = FmhaPipeline::kIsGroupMode;
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                      BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                      BiasEnum == BlockAttentionBiasEnum::ALIBI,
                  "Sparge: only NO_BIAS / ELEMENTWISE_BIAS / ALIBI supported.");
    static_assert(!kStoreLSE, "Sparge sparse attention does not support LSE output.");
    static_assert(!kHasDropout, "Sparge sparse attention does not support dropout.");
    static_assert(!kHasLogitsSoftCap || BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "Sparge: logits soft-cap requires NO_BIAS.");
    static_assert(!kDoFp8StaticQuant,
                  "Sparge sparse attention does not support FP8 static quantization yet.");

    using AttentionVariant = ck_tile::remove_cvref_t<typename FmhaPipeline::AttentionVariant>;
    using FmhaMask         = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    static constexpr bool kUseAsyncCopy = FmhaPipeline::Policy::AsyncCopy;

    template <ck_tile::index_t I>
    struct FmhaFwdEmptyKargs
    {
    };

    struct FmhaFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_ptr;

        const void* lut_ptr;             // delta-encoded
        const void* valid_block_num_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        ck_tile::index_t nhead_ratio_qk;
        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;
    };

    struct FmhaFwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    static constexpr bool kHasBias = (BiasEnum != BlockAttentionBiasEnum::NO_BIAS);
    struct FmhaFwdBiasKargs
    {
        const void* bias_ptr;
        ck_tile::index_t stride_bias;
        ck_tile::index_t nhead_stride_bias;
        ck_tile::index_t batch_stride_bias;
    };
    struct FmhaFwdLogitsSoftCapKargs
    {
        float logits_soft_cap;
    };

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kHasBias, FmhaFwdBiasKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<3>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;

        float pvthreshd;
        const void* pvthreshd_per_head_ptr;
    };

    struct FmhaFwdGroupModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kHasBias, FmhaFwdBiasKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<3>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_q_ptr;
        const int32_t* seqlen_k_ptr;
        const int32_t* seqstart_q_block_ptr;
        const int32_t* lut_batch_offset_ptr;
        ck_tile::index_t batch;

        float pvthreshd;
        const void* pvthreshd_per_head_ptr;
    };

    using Kargs =
        std::conditional_t<kIsGroupMode, FmhaFwdGroupModeKargs, FmhaFwdBatchModeKargs>;

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* o_ptr,
              const void* lut_ptr,
              const void* valid_block_num_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              float pvthreshd                = 0.0f,
              const void* pvthreshd_per_head = nullptr,
              const void* bias_ptr              = nullptr,
              ck_tile::index_t stride_bias       = 0,
              ck_tile::index_t nhead_stride_bias = 0,
              ck_tile::index_t batch_stride_bias = 0,
              float logits_soft_cap             = 0.0f)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     lut_ptr,
                     valid_block_num_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o},
                    {},
                    {},
                    {},
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o,
                    pvthreshd,
                    pvthreshd_per_head};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.logits_soft_cap = logits_soft_cap;
        }
        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* o_ptr,
              const void* lut_ptr,
              const void* valid_block_num_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_o,
              const int32_t* seqstart_q_ptr,
              const int32_t* seqstart_k_ptr,
              const int32_t* seqlen_q_ptr,
              const int32_t* seqlen_k_ptr,
              const int32_t* seqstart_q_block_ptr,
              const int32_t* lut_batch_offset_ptr,
              ck_tile::index_t batch,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              float pvthreshd                = 0.0f,
              const void* pvthreshd_per_head = nullptr,
              const void* bias_ptr               = nullptr,
              ck_tile::index_t stride_bias       = 0,
              ck_tile::index_t nhead_stride_bias = 0,
              ck_tile::index_t batch_stride_bias = 0,
              float logits_soft_cap              = 0.0f)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     lut_ptr,
                     valid_block_num_ptr,
                     /*seqlen_q*/ 0,
                     /*seqlen_k*/ 0,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o},
                    {},
                    {},
                    {},
                    seqstart_q_ptr,
                    seqstart_k_ptr,
                    seqlen_q_ptr,
                    seqlen_k_ptr,
                    seqstart_q_block_ptr,
                    lut_batch_offset_ptr,
                    batch,
                    pvthreshd,
                    pvthreshd_per_head};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.logits_soft_cap = logits_soft_cap;
        }
        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_)
    {
        return dim3(nhead_,
                    batch_size_,
                    ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                        ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1));
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

        const index_t i_block = blockIdx.z;
        const index_t i_nhead = blockIdx.x;
        const index_t i_batch = blockIdx.y;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        if constexpr(kHasMask)
        {
            // assumes num_tile_n1 == 1
            return ck_tile::make_tuple(gridDim.z - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
        }
        else
        {
            return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
        }
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {

        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q = 0;
        long_index_t batch_offset_k = 0;
        long_index_t batch_offset_v = 0;
        long_index_t batch_offset_o = 0;

        index_t seqlen_q_actual = 0;
        index_t seqlen_k_actual = 0;

        if constexpr(kIsGroupMode)
        {
            const long_index_t qstart =
                static_cast<long_index_t>(kargs.seqstart_q_ptr[i_batch]);
            const long_index_t kstart =
                static_cast<long_index_t>(kargs.seqstart_k_ptr[i_batch]);
            batch_offset_q = qstart * kargs.stride_q;
            batch_offset_k = kstart * kargs.stride_k;
            batch_offset_v = kstart * kargs.stride_v;
            batch_offset_o = qstart * kargs.stride_o;

            seqlen_q_actual =
                kargs.seqlen_q_ptr != nullptr
                    ? kargs.seqlen_q_ptr[i_batch]
                    : (kargs.seqstart_q_ptr[i_batch + 1] - kargs.seqstart_q_ptr[i_batch]);
            seqlen_k_actual =
                kargs.seqlen_k_ptr != nullptr
                    ? kargs.seqlen_k_ptr[i_batch]
                    : (kargs.seqstart_k_ptr[i_batch + 1] - kargs.seqstart_k_ptr[i_batch]);

            if(static_cast<index_t>(i_tile_m * FmhaPipeline::kM0) >= seqlen_q_actual)
                return;
        }
        else
        {
            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
            seqlen_q_actual = kargs.seqlen_q;
            seqlen_k_actual = kargs.seqlen_k;
        }

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;

        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // LUT / VBN: batch is rectangular; group is packed by lut_batch_offset_ptr / seqstart_q_block_ptr.
        const long_index_t lut_per_head_size = [&]() -> long_index_t {
            if constexpr(kIsGroupMode)
                return __builtin_amdgcn_readfirstlane(
                    kargs.lut_batch_offset_ptr[kargs.batch]);
            return 0;
        }();
        const long_index_t vbn_per_head_size = [&]() -> long_index_t {
            if constexpr(kIsGroupMode)
                return __builtin_amdgcn_readfirstlane(
                    kargs.seqstart_q_block_ptr[kargs.batch]);
            return 0;
        }();
        const long_index_t lut_b_off = [&]() -> long_index_t {
            if constexpr(kIsGroupMode)
                return __builtin_amdgcn_readfirstlane(
                    kargs.lut_batch_offset_ptr[i_batch]);
            return 0;
        }();
        const long_index_t vbn_b_off = [&]() -> long_index_t {
            if constexpr(kIsGroupMode)
                return __builtin_amdgcn_readfirstlane(
                    kargs.seqstart_q_block_ptr[i_batch]);
            return 0;
        }();
        const int* lut_row = [&]() -> const int* {
            const auto* base = reinterpret_cast<const int*>(kargs.lut_ptr);
            if constexpr(kIsGroupMode)
            {
                const index_t k_blocks_b =
                    ck_tile::integer_divide_ceil(seqlen_k_actual, FmhaPipeline::kN0);
                return base +
                       static_cast<long_index_t>(i_nhead) * lut_per_head_size +
                       lut_b_off +
                       static_cast<long_index_t>(i_tile_m) * k_blocks_b;
            }
            else
            {
                const index_t num_q_blocks =
                    ck_tile::integer_divide_ceil(kargs.seqlen_q, FmhaPipeline::kM0);
                const index_t num_k_blocks =
                    ck_tile::integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0);
                const long_index_t lut_batch_head_offset =
                    (static_cast<long_index_t>(i_batch) * kargs.num_head_q + i_nhead) *
                    num_q_blocks;
                return base + (lut_batch_head_offset + i_tile_m) * num_k_blocks;
            }
        }();
        const int valid_block_num_value = [&]() -> int {
            const auto* base = reinterpret_cast<const int*>(kargs.valid_block_num_ptr);
            if constexpr(kIsGroupMode)
            {
                return base[
                    static_cast<long_index_t>(i_nhead) * vbn_per_head_size +
                    vbn_b_off + i_tile_m];
            }
            else
            {
                const index_t num_q_blocks =
                    ck_tile::integer_divide_ceil(kargs.seqlen_q, FmhaPipeline::kM0);
                return base[
                    (static_cast<long_index_t>(i_batch) * kargs.num_head_q + i_nhead) *
                        num_q_blocks +
                    i_tile_m];
            }
        }();

        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(seqlen_q_actual, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<FmhaPipeline::kAlignmentQ>{},
                number<1>{});
            if constexpr(FmhaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(seqlen_k_actual, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK_, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(seqlen_k_actual, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
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
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK_>{});
            }
            else
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, seqlen_k_actual),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                constexpr bool kPadHeadDimV_ = kUseAsyncCopy ? kPadHeadDimV : false;
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<kPadHeadDimV_, kPadSeqLenK>{});
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(number<FmhaPipeline::kM0>{},
                                      number<FmhaPipeline::kSubQKHeaddim>{});
                else
                    return make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                             {i_n1, 0});

        auto bias_dram_window = [&]() {
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
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{}),
                    sequence<kPadSeqLenQ, kPadSeqLenK>{});
                return make_tile_window(
                    bias_dram,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{}),
                    {i_m0, 0});
            }
            else
            {
                const BiasDataType* null_bias = static_cast<const BiasDataType*>(nullptr);
                const auto dummy_naive = make_naive_tensor_view<address_space_enum::global>(
                    null_bias,
                    make_tuple(1, 1), make_tuple(1, 1), number<1>{}, number<1>{});
                const auto dummy = pad_tensor_view(
                    dummy_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{}),
                    sequence<true, true>{});
                return make_tile_window(
                    dummy,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{}),
                    {0, 0});
            }
        }();

        FmhaMask mask_obj = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    seqlen_q_actual,
                    seqlen_k_actual,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{seqlen_q_actual, seqlen_k_actual};
        }();

        // ALIBI slope: stride_bias=0 -> slope[i_nhead] (shared); stride_bias=nhead -> slope[b*nhead+h].
        auto position_encoding = [&]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto* slope_arr = reinterpret_cast<const SaccDataType*>(kargs.bias_ptr);
                const long_index_t off =
                    static_cast<long_index_t>(i_batch) * kargs.stride_bias + i_nhead;
                SaccDataType slope = slope_arr ? slope_arr[off] : SaccDataType{0};
                // FAST_EXP2: pre-scale slope by log2e so ALIBI matches the
                // already-scaled QK term (scale_s folded log2e at MakeKargs).
#if CK_TILE_FMHA_FWD_FAST_EXP2
                slope = slope * ck_tile::log2e_v<SaccDataType>;
#endif
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
        const auto variant_params = [&]() {
            if constexpr(kHasLogitsSoftCap)
                return ck_tile::LogitsSoftCapParams<FmhaMask, CK_TILE_FMHA_FWD_FAST_EXP2>{
                    mask_obj, kargs.scale_s, kargs.logits_soft_cap};
            else
                return ck_tile::StandardAttentionParams<FmhaMask>{mask_obj, kargs.scale_s};
        }();

        BlockIndices block_indices{i_batch, i_nhead, i_nhead / kargs.nhead_ratio_qk};

        auto o_acc_tile = [&]() {
            __shared__ char smem_ptr[FmhaPipeline::GetSmemSize()];
            return FmhaPipeline{}(q_dram_window,
                                  k_dram_window,
                                  v_dram_window,
                                  bias_dram_window,
                                  position_encoding,
                                  lut_row,
                                  valid_block_num_value,
                                  mask_obj,
                                  kargs.scale_s,
                                  variant,
                                  variant_params,
                                  block_indices,
                                  smem_ptr,
                                  kargs.pvthreshd,
                                  kargs.pvthreshd_per_head_ptr);
        }();

        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(seqlen_q_actual, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<FmhaPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
