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
// smooth_k: km only affects quant (sparge_sage), where K is centered by its per-channel global
// mean before int8/fp8 quant (official SpargeAttn) to shrink quant error. The implied -q@km^T is a
// per-row constant softmax absorbs, so no attention/dequant fixup is needed and selection is
// unchanged. Non-quant path keeps km_ptr nullptr; Q is never centered.

namespace ck_tile {

// Per-side preprocess kargs (one tensor's means + sims). K and Q each carry only the fields
// they use; GridSize() exposes work-group counts so the caller never recomputes them.
struct SpargePreprocessOneSideKargs
{
    const void* data;  // K or Q input

    // Batch: [B,H,num_blocks,D] / [B,H,num_blocks]. Group: batch-outer/head-mid packed,
    // per-batch block [H, blocks_b] (means with trailing D), concatenated across batch.
    float* means;
    float* sim;  // nullable when simthreshold <= 0

    index_t batch;
    index_t nhead;
    index_t seqlen;     // batch mode: per-sequence length
    index_t hdim;
    index_t block_size;
    index_t num_blocks;  // batch: per-sequence; group: max across batches

    index_t batch_stride;
    index_t head_stride;
    index_t seq_stride;

    float simthreshold;
    const float* km_ptr;  // K smoothing km[B,H,D]; nullptr disables (Q side: always nullptr).

    // Group / varlen mode (batch leaves all nullptr / 0).
    const int32_t* seqstart_ptr        = nullptr;
    const int32_t* seqlen_ptr          = nullptr;
    const int32_t* seqstart_block_ptr  = nullptr;
    index_t        total_blocks        = 0;

    // Quant outputs (sparge_sage): int8 buffer matches the raw input token layout (hdim stride 1,
    // same batch/head/seq strides); scale_out is [batch, nhead, num_block_scale],
    // num_block_scale = ceil(seqlen / tokens_per_scale).
    int8_t*  quant_out        = nullptr;
    float*   scale_out        = nullptr;
    index_t  tokens_per_scale = 0;       // PERWARP: Q=32, K=64
    index_t  num_block_scale  = 0;       // ceil(seqlen / tokens_per_scale) (batch mode)
};

// Fused K+Q preprocess: one launch covers both sides. The grid x axis concatenates the K
// block-range [0, k_xblocks) and the Q range [k_xblocks, ...); y = max(nhead_k, nhead_q) with the
// smaller side guarded. Outputs land at the same offsets as the separate kernels.
struct SpargePreprocessFusedKargs
{
    SpargePreprocessOneSideKargs k;
    SpargePreprocessOneSideKargs q;
};

// Binary search: largest b with seqstart_block_ptr[b] <= g_block. readfirstlane -> SGPR.
CK_TILE_DEVICE index_t
sparge_block_to_batch(const int32_t* seqstart_block_ptr, index_t batch, index_t g_block)
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

// Single-side block-mean / block-sim preprocess (optionally subtracting km). Batch: 3D grid
// (x=block, y=head, z=batch). Group: x = total_blocks, binary search over seqstart_block_ptr
// recovers (batch, block_in_batch); y = head.
template <typename InputType_,
          bool kIsGroupMode_                              = false,
          index_t kBlockSize_                             = 256,
          BlockSageAttentionQuantScaleEnum QScale_        = BlockSageAttentionQuantScaleEnum::NO_SCALE,
          typename QuantType_                             = int8_t>
struct FmhaFwdSpargePreprocessOneSideKernel
{
    using InputType = remove_cvref_t<InputType_>;
    using QuantType = remove_cvref_t<QuantType_>;
    using Pipeline  = BlockSpargePreprocessPipeline<InputType, kBlockSize_, QScale_, QuantType>;

    static constexpr bool kIsGroupMode = kIsGroupMode_;
    static constexpr BlockSageAttentionQuantScaleEnum QScale = QScale_;
    static constexpr bool kDoQuant = (QScale_ != BlockSageAttentionQuantScaleEnum::NO_SCALE);

    static constexpr index_t kBlockSize  = Pipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = 4;

    CK_TILE_HOST static dim3 GridSize(const SpargePreprocessOneSideKargs& kargs)
    {
        if constexpr(kIsGroupMode)
            return dim3(static_cast<uint32_t>(kargs.total_blocks),
                        static_cast<uint32_t>(kargs.nhead),
                        1);
        else
            return dim3(static_cast<uint32_t>(kargs.num_blocks),
                        static_cast<uint32_t>(kargs.nhead),
                        static_cast<uint32_t>(kargs.batch));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static std::size_t GetSmemSize(index_t hdim)
    {
        return static_cast<std::size_t>(Pipeline::GetSmemSize(hdim));
    }

    CK_TILE_DEVICE void operator()(SpargePreprocessOneSideKargs kargs) const
    {
        extern __shared__ char smem_raw[];
        run_side(kargs, blockIdx.x, blockIdx.y, blockIdx.z, smem_raw);
    }

    // Body extracted so the fused K+Q kernel can re-dispatch with synthetic grid coords (gx/gy/gz).
    CK_TILE_DEVICE static void run_side(const SpargePreprocessOneSideKargs& kargs,
                                        index_t gx, index_t gy, index_t gz, char* smem_raw)
    {
        index_t      head_id, batch_id, block_id;
        long_index_t batch_offset;
        index_t      seqlen_actual;
        long_index_t mean_out_off;
        long_index_t sim_out_off;
        long_index_t km_bh_off;
        // Group: packed scale-block index (= packed_idx, scaled by scales_per_block below).
        long_index_t scale_packed_block = 0;

        if constexpr(kIsGroupMode)
        {
            const index_t g_block = __builtin_amdgcn_readfirstlane(gx);
            head_id               = __builtin_amdgcn_readfirstlane(gy);
            batch_id              = sparge_block_to_batch(
                kargs.seqstart_block_ptr, kargs.batch, g_block);
            const index_t start_b = __builtin_amdgcn_readfirstlane(
                kargs.seqstart_block_ptr[batch_id]);
            block_id = g_block - start_b;
            // Batch-outer/head-mid packed: per-batch [H, X_b] (X_b = blocks of this batch),
            // concatenated across batch. packed_idx = Xstart_b*H + head*X_b + local.
            const index_t x_b = __builtin_amdgcn_readfirstlane(
                kargs.seqstart_block_ptr[batch_id + 1] - start_b);

            const long_index_t start =
                static_cast<long_index_t>(kargs.seqstart_ptr[batch_id]);
            batch_offset  = start * kargs.seq_stride;
            seqlen_actual = __builtin_amdgcn_readfirstlane(
                kargs.seqlen_ptr != nullptr
                    ? kargs.seqlen_ptr[batch_id]
                    : (kargs.seqstart_ptr[batch_id + 1] - kargs.seqstart_ptr[batch_id]));

            const long_index_t packed_idx =
                static_cast<long_index_t>(start_b) * kargs.nhead +
                static_cast<long_index_t>(head_id) * x_b + block_id;
            mean_out_off = packed_idx * kargs.hdim;
            sim_out_off  = packed_idx;
            km_bh_off =
                static_cast<long_index_t>(batch_id) * kargs.nhead + head_id;
            scale_packed_block = packed_idx;
        }
        else
        {
            block_id = static_cast<index_t>(gx);
            head_id  = static_cast<index_t>(gy);
            batch_id = static_cast<index_t>(gz);

            batch_offset  = static_cast<long_index_t>(batch_id) * kargs.batch_stride;
            seqlen_actual = kargs.seqlen;

            const long_index_t bh =
                static_cast<long_index_t>(batch_id) * kargs.nhead + head_id;
            mean_out_off = bh * static_cast<long_index_t>(kargs.num_blocks) * kargs.hdim +
                           static_cast<long_index_t>(block_id) * kargs.hdim;
            sim_out_off  = bh * kargs.num_blocks + block_id;
            km_bh_off    = bh;
        }

        const auto* slice =
            reinterpret_cast<const InputType*>(kargs.data) +
            batch_offset +
            static_cast<long_index_t>(head_id) * kargs.head_stride;
        float* mean_out = kargs.means + mean_out_off;
        float* sim_out  = (kargs.sim != nullptr) ? kargs.sim + sim_out_off : nullptr;

        typename Pipeline::Params pp;
        pp.seqlen       = seqlen_actual;
        pp.hdim         = kargs.hdim;
        pp.block_size   = kargs.block_size;
        pp.block_id     = block_id;
        pp.stride_seq   = kargs.seq_stride;
        pp.simthreshold = kargs.simthreshold;
        pp.km_ptr       = kargs.km_ptr
            ? kargs.km_ptr + km_bh_off * static_cast<long_index_t>(kargs.hdim) : nullptr;

        // Quant outputs (sparge_sage). int8 shares the raw input token layout; the pipeline
        // re-applies the block_id*block_size window. Scales: batch [batch, nhead, num_block_scale];
        // group packed by the same batch-outer/head-mid block index * scales_per_block.
        if constexpr(kDoQuant)
        {
            const index_t tps = kargs.tokens_per_scale;
            pp.quant_out       = nullptr;
            pp.scale_out       = nullptr;
            pp.tokens_per_scale = tps;
            pp.quant_stride_seq = kargs.seq_stride;
            if(kargs.quant_out != nullptr && tps > 0)
            {
                // quant origin = (batch, head) seq start; pipeline re-applies the block window.
                pp.quant_out = reinterpret_cast<QuantType*>(kargs.quant_out) + batch_offset +
                              static_cast<long_index_t>(head_id) * kargs.head_stride;
                const index_t scales_per_block = kargs.block_size / tps;
                if constexpr(kIsGroupMode)
                {
                    // scale origin = packed block index * scales_per_block (contiguous per-(b,h)
                    // run, indexed by k_abs_pos / tps in the attention kernel).
                    pp.scale_out = kargs.scale_out +
                                   scale_packed_block *
                                       static_cast<long_index_t>(scales_per_block);
                }
                else
                {
                    // scale origin = [batch, head, block_id * scales_per_block].
                    const long_index_t bh =
                        static_cast<long_index_t>(batch_id) * kargs.nhead + head_id;
                    pp.scale_out = kargs.scale_out +
                                   bh * static_cast<long_index_t>(kargs.num_block_scale) +
                                   static_cast<long_index_t>(block_id) * scales_per_block;
                }
            }
        }

        Pipeline{}(slice, mean_out, sim_out, pp, smem_raw);
    }
};

// Distinct K-side / Q-side types so the launch chain issues two separate kernel calls, each
// with its own grid, while sharing the implementation.
template <typename InputType_,
          bool kIsGroupMode_                       = false,
          index_t kBlockSize_                      = 256,
          BlockSageAttentionQuantScaleEnum QScale_ = BlockSageAttentionQuantScaleEnum::NO_SCALE,
          typename QuantType_                      = int8_t>
struct FmhaFwdSpargePreprocessKKernel
    : FmhaFwdSpargePreprocessOneSideKernel<InputType_, kIsGroupMode_, kBlockSize_, QScale_,
                                           QuantType_>
{
};

template <typename InputType_,
          bool kIsGroupMode_                       = false,
          index_t kBlockSize_                      = 256,
          BlockSageAttentionQuantScaleEnum QScale_ = BlockSageAttentionQuantScaleEnum::NO_SCALE,
          typename QuantType_                      = int8_t>
struct FmhaFwdSpargePreprocessQKernel
    : FmhaFwdSpargePreprocessOneSideKernel<InputType_, kIsGroupMode_, kBlockSize_, QScale_,
                                           QuantType_>
{
};

// Fused K+Q preprocess: one launch dispatches both data-independent sides via run_side, saving a
// kernel launch (dominant cost at short seqlen). Outputs land at identical offsets.
template <typename InputType_,
          bool kIsGroupMode_                       = false,
          index_t kBlockSize_                      = 256,
          BlockSageAttentionQuantScaleEnum QScale_ = BlockSageAttentionQuantScaleEnum::NO_SCALE,
          typename QuantType_                      = int8_t>
struct FmhaFwdSpargePreprocessFusedKernel
{
    using Side =
        FmhaFwdSpargePreprocessOneSideKernel<InputType_, kIsGroupMode_, kBlockSize_, QScale_,
                                             QuantType_>;

    static constexpr bool    kIsGroupMode = kIsGroupMode_;
    static constexpr index_t kBlockSize   = Side::kBlockSize;
    static constexpr index_t kBlockPerCu  = Side::kBlockPerCu;

    CK_TILE_HOST_DEVICE static index_t k_xblocks(const SpargePreprocessFusedKargs& kargs)
    {
        return kIsGroupMode_ ? kargs.k.total_blocks : kargs.k.num_blocks;
    }
    CK_TILE_HOST_DEVICE static index_t q_xblocks(const SpargePreprocessFusedKargs& kargs)
    {
        return kIsGroupMode_ ? kargs.q.total_blocks : kargs.q.num_blocks;
    }

    CK_TILE_HOST static dim3 GridSize(const SpargePreprocessFusedKargs& kargs)
    {
        const uint32_t x  = static_cast<uint32_t>(k_xblocks(kargs) + q_xblocks(kargs));
        const uint32_t ny = static_cast<uint32_t>(
            kargs.k.nhead > kargs.q.nhead ? kargs.k.nhead : kargs.q.nhead);
        if constexpr(kIsGroupMode_)
            return dim3(x, ny, 1);
        else
            return dim3(x, ny, static_cast<uint32_t>(kargs.k.batch));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static std::size_t GetSmemSize(index_t hdim)
    {
        return Side::GetSmemSize(hdim);
    }

    CK_TILE_DEVICE void operator()(SpargePreprocessFusedKargs kargs) const
    {
        extern __shared__ char smem_raw[];
        const index_t gx  = static_cast<index_t>(blockIdx.x);
        const index_t gy  = static_cast<index_t>(blockIdx.y);
        const index_t gz  = static_cast<index_t>(blockIdx.z);
        const index_t k_x = k_xblocks(kargs);
        if(gx < k_x)
        {
            if(gy >= kargs.k.nhead)
                return;  // GQA: K side has fewer heads than grid y.
            Side::run_side(kargs.k, gx, gy, gz, smem_raw);
        }
        else
        {
            if(gy >= kargs.q.nhead)
                return;
            Side::run_side(kargs.q, gx - k_x, gy, gz, smem_raw);
        }
    }
};

// PERTENSOR Q/K INT8 quant: one work-group per (batch, head) computes the global absmax over the
// full X slice [seqlen, hdim], emits one scale = absmax/127 and INT8 X. Batch: 2D grid (head,
// batch). Group: same grid, per-batch token window from seqstart. Scale lands at
// scale_out[batch_id * nhead + head_id] for both modes.
struct SpargeQKQuantKargs
{
    const void* x_ptr;     // bf16 X (Q or K)
    void*       quant_ptr; // quant X output
    float*      scale_ptr; // [batch, nhead] one scale per (batch, head)

    index_t batch;
    index_t nhead;
    index_t seqlen;        // batch mode: per-sequence length
    index_t hdim;

    index_t batch_stride_x; // bf16 X strides
    index_t nhead_stride_x;
    index_t stride_x;       // token stride

    index_t batch_stride_quant; // quant X strides
    index_t nhead_stride_quant;
    index_t stride_quant;

    // smooth_k (K side only): per-channel K-mean km[B, nhead, hdim]; nullptr disables (Q side).
    const float* km_ptr = nullptr;

    // Group / varlen mode (batch leaves all nullptr).
    const int32_t* seqstart_ptr = nullptr; // [batch+1] token cumsum
    const int32_t* seqlen_ptr   = nullptr; // [batch] (nullptr -> diff of seqstart)
};

template <typename InputType_, bool kIsGroupMode_ = false, index_t kBlockSize_ = 256,
          typename QuantType_ = int8_t>
struct FmhaFwdSpargeQKQuantKernel
{
    using InputType = remove_cvref_t<InputType_>;
    using QuantType = remove_cvref_t<QuantType_>;
    using Pipeline  = BlockSpargeQKQuantPipeline<InputType, kBlockSize_, QuantType>;

    static constexpr bool    kIsGroupMode = kIsGroupMode_;
    static constexpr index_t kBlockSize   = Pipeline::kBlockSize;
    static constexpr index_t kBlockPerCu  = 4;

    CK_TILE_HOST static dim3 GridSize(const SpargeQKQuantKargs& kargs)
    {
        return dim3(static_cast<uint32_t>(kargs.nhead),
                    static_cast<uint32_t>(kargs.batch),
                    1);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static std::size_t GetSmemSize(index_t hdim)
    {
        return static_cast<std::size_t>(Pipeline::GetSmemSize(hdim));
    }

    CK_TILE_DEVICE void operator()(SpargeQKQuantKargs kargs) const
    {
        extern __shared__ char smem_raw[];

        const index_t head_id  = static_cast<index_t>(blockIdx.x);
        const index_t batch_id = static_cast<index_t>(blockIdx.y);

        long_index_t x_batch_off, quant_batch_off;
        index_t      seqlen_actual;
        if constexpr(kIsGroupMode)
        {
            const long_index_t start =
                static_cast<long_index_t>(kargs.seqstart_ptr[batch_id]);
            x_batch_off     = start * kargs.stride_x;
            quant_batch_off = start * kargs.stride_quant;
            seqlen_actual  = __builtin_amdgcn_readfirstlane(
                kargs.seqlen_ptr != nullptr
                    ? kargs.seqlen_ptr[batch_id]
                    : (kargs.seqstart_ptr[batch_id + 1] - kargs.seqstart_ptr[batch_id]));
        }
        else
        {
            x_batch_off     = static_cast<long_index_t>(batch_id) * kargs.batch_stride_x;
            quant_batch_off = static_cast<long_index_t>(batch_id) * kargs.batch_stride_quant;
            seqlen_actual   = kargs.seqlen;
        }

        const auto* slice =
            reinterpret_cast<const InputType*>(kargs.x_ptr) + x_batch_off +
            static_cast<long_index_t>(head_id) * kargs.nhead_stride_x;
        auto* quant_out =
            reinterpret_cast<QuantType*>(kargs.quant_ptr) + quant_batch_off +
            static_cast<long_index_t>(head_id) * kargs.nhead_stride_quant;

        // One global scale per (batch, head); packed [batch, nhead].
        float* scale_out =
            kargs.scale_ptr +
            static_cast<long_index_t>(batch_id) * kargs.nhead + head_id;

        typename Pipeline::Params pp;
        pp.seqlen          = seqlen_actual;
        pp.hdim            = kargs.hdim;
        pp.stride_seq       = kargs.stride_x;
        pp.quant_stride_seq = kargs.stride_quant;
        // smooth_k: per-channel km[batch, nhead, hdim] for this (batch, head).
        pp.km_ptr = kargs.km_ptr
            ? kargs.km_ptr +
                  (static_cast<long_index_t>(batch_id) * kargs.nhead + head_id) *
                      static_cast<long_index_t>(kargs.hdim)
            : nullptr;

        Pipeline{}(slice, quant_out, scale_out, pp, smem_raw);
    }
};

struct SpargeMaskPredictionKargs
{
    const float* k_means;
    const float* q_means;
    const float* k_sim;            // nullable when simthreshold <= 0
    const float* q_sim;
    int32_t* lut_out;              // batch: [B,Hq,nq,nk]; group: batch-outer/head-mid packed
    int32_t* valid_block_num_out;  // batch: [B,Hq,nq]; group: batch-outer/head-mid packed (q)

    index_t batch;
    index_t nhead_q;
    index_t nhead_k;
    index_t nhead_ratio_qk;
    index_t num_q_blocks;
    index_t num_k_blocks;
    index_t hdim;
    float scale;            // softmax scale for q/k mean scores (caller-provided)
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
    const int32_t* mask_batch_offset_ptr  = nullptr;
    index_t        total_q_blocks        = 0;
    index_t        total_k_blocks        = 0;
};

template <bool kIsGroupMode_ = false, index_t kMaxKBlocksPow2_ = 256, index_t kBlockSize_ = 256>
struct FmhaFwdSpargeMaskPredictionKernel
{
    using Pipeline = BlockSpargeMaskPredictionPipeline<kMaxKBlocksPow2_, kBlockSize_>;

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

            // Batch-outer/head-mid packed: per-batch [H, X_b], new_index = Xstart_b*H + head*X_b +
            // local. q_means/q_sim/vbn: H=nhead_q, X_b=num_q_blocks_b; k_means/k_sim: H=nhead_k,
            // X_b=num_k_blocks_b; lut: H=nhead_q, X_b=q_b*k_b. Pre-slice to (batch,head); pipeline
            // then runs with b=0, nhead=1.
            const long_index_t q_means_view_off =
                (static_cast<long_index_t>(qstart_b) * kargs.nhead_q +
                 static_cast<long_index_t>(head) * num_q_blocks_b) *
                kargs.hdim;
            const float* q_means_view = kargs.q_means + q_means_view_off;
            const float* q_sim_view = (kargs.q_sim != nullptr)
                ? kargs.q_sim +
                    static_cast<long_index_t>(qstart_b) * kargs.nhead_q +
                    static_cast<long_index_t>(head) * num_q_blocks_b
                : nullptr;

            const long_index_t k_means_view_off =
                (static_cast<long_index_t>(kstart_b) * kargs.nhead_k +
                 static_cast<long_index_t>(kv_head) * num_k_blocks_b) *
                kargs.hdim;
            const float* k_means_view = kargs.k_means + k_means_view_off;
            const float* k_sim_view = (kargs.k_sim != nullptr)
                ? kargs.k_sim +
                    static_cast<long_index_t>(kstart_b) * kargs.nhead_k +
                    static_cast<long_index_t>(kv_head) * num_k_blocks_b
                : nullptr;

            const long_index_t lut_xstart_b =
                static_cast<long_index_t>(kargs.mask_batch_offset_ptr[b]);
            const long_index_t lut_x_b =
                static_cast<long_index_t>(kargs.mask_batch_offset_ptr[b + 1]) - lut_xstart_b;
            int32_t* lut_row =
                kargs.lut_out +
                lut_xstart_b * kargs.nhead_q +
                static_cast<long_index_t>(head) * lut_x_b +
                static_cast<long_index_t>(q_block_in_b) * num_k_blocks_b;
            int32_t* vbn_ptr =
                kargs.valid_block_num_out +
                static_cast<long_index_t>(qstart_b) * kargs.nhead_q +
                static_cast<long_index_t>(head) * num_q_blocks_b +
                q_block_in_b;

            const float head_cdfthreshd   = kargs.cdfthreshd_per_head
                ? kargs.cdfthreshd_per_head[head] : kargs.cdfthreshd;
            const float head_topk          = kargs.topk_per_head
                ? kargs.topk_per_head[head] : kargs.topk;
            const float head_simthreshold = kargs.simthreshold_per_head
                ? kargs.simthreshold_per_head[head] : kargs.simthreshold;

            typename Pipeline::MaskRunArgs args;
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
            args.scale          = kargs.scale;
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
                       kargs.scale,
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

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using RandValOutputDataType =
        ck_tile::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    using PreprocessKKernel       = FmhaFwdSpargePreprocessKKernel<QDataType, FmhaPipeline::kIsGroupMode, FmhaPipeline::kBlockSize>;
    using PreprocessQKernel       = FmhaFwdSpargePreprocessQKernel<QDataType, FmhaPipeline::kIsGroupMode, FmhaPipeline::kBlockSize>;
    using PreprocessFusedKernel   = FmhaFwdSpargePreprocessFusedKernel<QDataType, FmhaPipeline::kIsGroupMode, FmhaPipeline::kBlockSize>;
    using MaskPredictionKernel    = FmhaFwdSpargeMaskPredictionKernel<FmhaPipeline::kIsGroupMode, 256, FmhaPipeline::kBlockSize>;

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
        const int32_t* mask_batch_offset_ptr;
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
              const int32_t* mask_batch_offset_ptr,
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
                    mask_batch_offset_ptr,
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
        // Masked M-tile reversal below assumes a single N1 tile spans hdim_v (num_tile_n1 == 1),
        // i.e. hdim_v <= kN1.
        static_assert(FmhaPipeline::kN1 >= FmhaPipeline::kQKHeaddim,
                      "sparge masked M-tile reversal assumes a single N1 tile "
                      "(hdim_v <= kN1)");
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
            // Reverse M tile so masked (top) rows run last (assumes num_tile_n1 == 1).
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
        __shared__ char smem_ptr[GetSmemSize()];

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

        // LUT / VBN. Batch: rectangular. Group: batch-outer/head-mid packed, per-batch [H, X_b]
        // (LUT X_b = q_b*k_b via mask_batch_offset_ptr, vbn X_b = q_b via seqstart_q_block_ptr);
        // new_index = Xstart_b*H + head*X_b + local.
        const long_index_t lut_xstart_b = [&]() -> long_index_t {
            if constexpr(kIsGroupMode)
                return __builtin_amdgcn_readfirstlane(
                    kargs.mask_batch_offset_ptr[i_batch]);
            return 0;
        }();
        const long_index_t lut_x_b = [&]() -> long_index_t {
            if constexpr(kIsGroupMode)
                return __builtin_amdgcn_readfirstlane(
                           kargs.mask_batch_offset_ptr[i_batch + 1]) -
                       lut_xstart_b;
            return 0;
        }();
        const long_index_t vbn_xstart_b = [&]() -> long_index_t {
            if constexpr(kIsGroupMode)
                return __builtin_amdgcn_readfirstlane(
                    kargs.seqstart_q_block_ptr[i_batch]);
            return 0;
        }();
        const long_index_t vbn_x_b = [&]() -> long_index_t {
            if constexpr(kIsGroupMode)
                return __builtin_amdgcn_readfirstlane(
                           kargs.seqstart_q_block_ptr[i_batch + 1]) -
                       vbn_xstart_b;
            return 0;
        }();
        const int* lut_row = [&]() -> const int* {
            const auto* base = reinterpret_cast<const int*>(kargs.lut_ptr);
            if constexpr(kIsGroupMode)
            {
                const index_t k_blocks_b =
                    ck_tile::integer_divide_ceil(seqlen_k_actual, FmhaPipeline::kN0);
                return base +
                       lut_xstart_b * kargs.num_head_q +
                       static_cast<long_index_t>(i_nhead) * lut_x_b +
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
                    vbn_xstart_b * kargs.num_head_q +
                    static_cast<long_index_t>(i_nhead) * vbn_x_b +
                    static_cast<long_index_t>(i_tile_m)];
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

        // ALIBI slope: stride_bias=0 -> slope[i_nhead] (shared); else slope[b*stride_bias + h].
        auto position_encoding = [&]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto* slope_arr = reinterpret_cast<const SaccDataType*>(kargs.bias_ptr);
                const long_index_t off =
                    static_cast<long_index_t>(i_batch) * kargs.stride_bias + i_nhead;
                SaccDataType slope = slope_arr ? slope_arr[off] : SaccDataType{0};
                // FAST_EXP2: pre-scale slope by log2e to match the already-scaled QK term.
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

        auto o_acc_tile = FmhaPipeline{}(q_dram_window,
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
