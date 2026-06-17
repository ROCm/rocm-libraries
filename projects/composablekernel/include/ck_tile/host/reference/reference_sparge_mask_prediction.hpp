// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

// CPU reference for sparge's mask-prediction stage.
namespace ck_tile {

template <typename T>
HostTensor<float> compute_block_means(
    const HostTensor<T>& data_bhsd,
    index_t batch,
    index_t nhead,
    index_t num_blocks,
    index_t block_size,
    index_t seqlen,
    index_t hdim)
{
    HostTensor<float> means({batch, nhead, num_blocks, hdim});
    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
            for(index_t blk = 0; blk < num_blocks; ++blk)
            {
                index_t s_start = blk * block_size;
                index_t s_end   = std::min(s_start + block_size, seqlen);
                index_t count   = s_end - s_start;
                for(index_t d = 0; d < hdim; ++d)
                {
                    float sum = 0.0f;
                    for(index_t s = s_start; s < s_end; ++s)
                        sum += type_convert<float>(data_bhsd(b, h, s, d));
                    means(b, h, blk, d) = sum / static_cast<float>(count);
                }
            }
    return means;
}

// Block self-similarity = mean of the pairwise cosine Gram matrix, matching upstream SpargeAttn
// (L2-normalize tokens, sim = sum(x@x^T)/BS^2). Via the identity
// sum_{i,j} <u_i,u_j> = ||sum_i u_i||^2 we accumulate the unit-vector sum u and report
// ||u||^2 / count^2.
// `km` (nullable): per-channel K-mean [B,H,D]. When provided (smooth_k, K side), each token
// element is centered by km[d] BEFORE both the per-token L2-norm and the unit-vector
// accumulation -- mirroring the device (block_sparge_mask_pipelines.hpp ~568 and ~598). Q side
// passes nullptr (device never centers Q); smooth_k off also passes nullptr (raw, as before).
template <typename T>
HostTensor<float> compute_block_similarity(
    const HostTensor<T>& data_bhsd,
    const HostTensor<float>* km, // [B,H,D] or nullptr (no centering)
    index_t batch,
    index_t nhead,
    index_t num_blocks,
    index_t block_size,
    index_t seqlen,
    index_t hdim)
{
    auto kmv = [&](index_t b, index_t h, index_t d) {
        return km ? (*km)(b, h, d) : 0.0f;
    };
    HostTensor<float> sim({batch, nhead, num_blocks});
    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
            for(index_t blk = 0; blk < num_blocks; ++blk)
            {
                index_t s_start = blk * block_size;
                index_t s_end   = std::min(s_start + block_size, seqlen);
                index_t count   = s_end - s_start;
                std::vector<float> u(static_cast<size_t>(hdim), 0.0f);
                for(index_t s = s_start; s < s_end; ++s)
                {
                    float tok_norm_sq = 0.0f;
                    for(index_t d = 0; d < hdim; ++d)
                    {
                        float v = type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d);
                        tok_norm_sq += v * v;
                    }
                    float inv_norm = 1.0f / std::sqrt(tok_norm_sq + 1e-8f);
                    for(index_t d = 0; d < hdim; ++d)
                    {
                        float v = type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d);
                        u[static_cast<size_t>(d)] += v * inv_norm;
                    }
                }
                float u_norm_sq = 0.0f;
                for(index_t d = 0; d < hdim; ++d)
                    u_norm_sq += u[static_cast<size_t>(d)] * u[static_cast<size_t>(d)];
                sim(b, h, blk) = u_norm_sq / (static_cast<float>(count) * static_cast<float>(count));
            }
    return sim;
}

template <typename T>
HostTensor<float>
compute_global_k_mean(const HostTensor<T>& k_bhsd,
                      index_t batch, index_t nhead_k,
                      index_t seqlen_k, index_t hdim)
{
    HostTensor<float> km({batch, nhead_k, hdim});
    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead_k; ++h)
            for(index_t d = 0; d < hdim; ++d)
            {
                float sum = 0.0f;
                for(index_t s = 0; s < seqlen_k; ++s)
                    sum += type_convert<float>(k_bhsd(b, h, s, d));
                km(b, h, d) = sum / static_cast<float>(seqlen_k);
            }
    return km;
}

// Reference for the per-warp row-wise INT8 quant fused into preprocess. Mirrors
// BlockSpargePreprocessPipeline::quantize_block bit-for-bit: (smooth_k, K side) center val by
// per-channel km[c] first; group absmax over tokens_per_scale tokens x full hidden;
// scale = absmax/127; int8 = saturate(val/scale).
// km (nullable): per-channel K-mean [B,H,D]; nullptr for Q (no centering). Outputs int8 [B,H,S,D]
// and scale [B,H,num_block_scale] with num_block_scale = num_blocks*(block_size/tokens_per_scale),
// block-packed to match the device scale_out layout.
template <typename T>
void reference_sparge_rowwise_quant(const HostTensor<T>& data_bhsd,
                                    index_t              batch,
                                    index_t              nhead,
                                    index_t              seqlen,
                                    index_t              hdim,
                                    index_t              block_size,
                                    index_t              tokens_per_scale,
                                    HostTensor<int8_t>&  int8_out,   // [B,H,S,D]
                                    HostTensor<float>&   scale_out,  // [B,H,num_block_scale]
                                    const HostTensor<float>* km = nullptr) // [B,H,D] or nullptr
{
    const index_t num_blocks      = (seqlen + block_size - 1) / block_size;
    const index_t scales_per_blk  = block_size / tokens_per_scale;
    const index_t num_block_scale = num_blocks * scales_per_blk;
    auto kmv = [&](index_t b, index_t h, index_t d) {
        return km ? (*km)(b, h, d) : 0.0f;
    };

    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
            for(index_t blk = 0; blk < num_blocks; ++blk)
            {
                const index_t s_start = blk * block_size;
                const index_t s_end   = std::min(s_start + block_size, seqlen);

                for(index_t g = 0; g < scales_per_blk; ++g)
                {
                    float amax = 0.0f;
                    for(index_t tt = 0; tt < tokens_per_scale; ++tt)
                    {
                        const index_t s = s_start + g * tokens_per_scale + tt;
                        if(s >= s_end)
                            continue;
                        for(index_t d = 0; d < hdim; ++d)
                        {
                            float v = type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d);
                            amax    = std::max(amax, std::abs(v));
                        }
                    }
                    const float scale = amax / 127.0f;
                    const index_t scale_idx =
                        (b * nhead + h) * num_block_scale + blk * scales_per_blk + g;
                    scale_out.mData[static_cast<size_t>(scale_idx)] = scale;

                    for(index_t tt = 0; tt < tokens_per_scale; ++tt)
                    {
                        const index_t s = s_start + g * tokens_per_scale + tt;
                        if(s >= s_end)
                            continue;
                        for(index_t d = 0; d < hdim; ++d)
                        {
                            float v = type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d);
                            float r = (scale > 0.0f) ? (v / scale) : 0.0f;
                            int8_out(b, h, s, d) =
                                type_convert<int8_t>(saturates<int8_t>{}(r));
                        }
                    }
                }
            }
}

// FP8 variant of reference_sparge_rowwise_quant (fp8bf16 Q/K path): scale = absmax/fp8_max,
// quant = fp8_t (the cast clamps). Outputs the fp8-roundtripped value as float so the caller
// can dequantize bit-for-bit (val_deq = float(fp8) * scale).
template <typename T>
void reference_sparge_rowwise_quant_fp8(const HostTensor<T>& data_bhsd,
                                        index_t              batch,
                                        index_t              nhead,
                                        index_t              seqlen,
                                        index_t              hdim,
                                        index_t              block_size,
                                        index_t              tokens_per_scale,
                                        HostTensor<float>&   fp8_as_float, // [B,H,S,D] float(fp8)
                                        HostTensor<float>&   scale_out,    // [B,H,num_block_scale]
                                        const HostTensor<float>* km = nullptr) // [B,H,D] or nullptr
{
    const index_t num_blocks      = (seqlen + block_size - 1) / block_size;
    const index_t scales_per_blk  = block_size / tokens_per_scale;
    const index_t num_block_scale = num_blocks * scales_per_blk;
    const float   fp8_max         = type_convert<float>(numeric<fp8_t>::max());
    auto kmv = [&](index_t b, index_t h, index_t d) {
        return km ? (*km)(b, h, d) : 0.0f;
    };

    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
            for(index_t blk = 0; blk < num_blocks; ++blk)
            {
                const index_t s_start = blk * block_size;
                const index_t s_end   = std::min(s_start + block_size, seqlen);

                for(index_t g = 0; g < scales_per_blk; ++g)
                {
                    float amax = 0.0f;
                    for(index_t tt = 0; tt < tokens_per_scale; ++tt)
                    {
                        const index_t s = s_start + g * tokens_per_scale + tt;
                        if(s >= s_end)
                            continue;
                        for(index_t d = 0; d < hdim; ++d)
                        {
                            float v = type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d);
                            amax    = std::max(amax, std::abs(v));
                        }
                    }
                    const float scale = amax / fp8_max;
                    const index_t scale_idx =
                        (b * nhead + h) * num_block_scale + blk * scales_per_blk + g;
                    scale_out.mData[static_cast<size_t>(scale_idx)] = scale;

                    for(index_t tt = 0; tt < tokens_per_scale; ++tt)
                    {
                        const index_t s = s_start + g * tokens_per_scale + tt;
                        if(s >= s_end)
                            continue;
                        for(index_t d = 0; d < hdim; ++d)
                        {
                            float v = type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d);
                            float r = (scale > 0.0f) ? (v / scale) : 0.0f;
                            fp8_t q = type_convert<fp8_t>(r);
                            fp8_as_float(b, h, s, d) = type_convert<float>(q);
                        }
                    }
                }
            }
}

// PERTENSOR reference quant (global per-(b,h) scale). Mirrors BlockSpargeQKQuantPipeline:
// scale = absmax_over_all_tokens_and_hidden / 127; one scale per (b,h). INT8 path.
template <typename T>
void reference_sparge_global_quant(const HostTensor<T>& data_bhsd,
                                   index_t              batch,
                                   index_t              nhead,
                                   index_t              seqlen,
                                   index_t              hdim,
                                   HostTensor<int8_t>&  int8_out,  // [B,H,S,D]
                                   HostTensor<float>&   scale_out, // [B,H] (one scale per b,h)
                                   const HostTensor<float>* km = nullptr) // [B,H,D] or nullptr
{
    auto kmv = [&](index_t b, index_t h, index_t d) {
        return km ? (*km)(b, h, d) : 0.0f;
    };
    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
        {
            float amax = 0.0f;
            for(index_t s = 0; s < seqlen; ++s)
                for(index_t d = 0; d < hdim; ++d)
                    amax = std::max(amax,
                        std::abs(type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d)));
            const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            scale_out.mData[static_cast<size_t>(b * nhead + h)] = scale;
            for(index_t s = 0; s < seqlen; ++s)
                for(index_t d = 0; d < hdim; ++d)
                {
                    float v = type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d);
                    int8_out(b, h, s, d) =
                        type_convert<int8_t>(saturates<int8_t>{}(v / scale));
                }
        }
}

// FP8 variant of reference_sparge_global_quant: scale = absmax / fp8_max, fp8 round-trip.
template <typename T>
void reference_sparge_global_quant_fp8(const HostTensor<T>& data_bhsd,
                                       index_t              batch,
                                       index_t              nhead,
                                       index_t              seqlen,
                                       index_t              hdim,
                                       HostTensor<float>&   fp8_as_float, // [B,H,S,D] float(fp8)
                                       HostTensor<float>&   scale_out,    // [B,H]
                                       const HostTensor<float>* km = nullptr) // [B,H,D] or nullptr
{
    const float fp8_max = type_convert<float>(numeric<fp8_t>::max());
    auto kmv = [&](index_t b, index_t h, index_t d) {
        return km ? (*km)(b, h, d) : 0.0f;
    };
    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
        {
            float amax = 0.0f;
            for(index_t s = 0; s < seqlen; ++s)
                for(index_t d = 0; d < hdim; ++d)
                    amax = std::max(amax,
                        std::abs(type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d)));
            const float scale = (amax > 0.0f) ? (amax / fp8_max) : 1.0f;
            scale_out.mData[static_cast<size_t>(b * nhead + h)] = scale;
            for(index_t s = 0; s < seqlen; ++s)
                for(index_t d = 0; d < hdim; ++d)
                {
                    float v = type_convert<float>(data_bhsd(b, h, s, d)) - kmv(b, h, d);
                    fp8_t q = type_convert<fp8_t>(v / scale);
                    fp8_as_float(b, h, s, d) = type_convert<float>(q);
                }
        }
}

// Generic inter-token causal / sliding-window masking parameters.
struct sparge_causal_params
{
    int causal_type  = 0;
    int window_left  = -1;
    int window_right = -1;
};

// Q/K block mask-prediction parameters (which K-blocks to select per Q-block).
struct sparge_block_predict_params
{
    float cdfthreshd   = 0.0f;
    float topk         = 0.0f;
    float simthreshold = 0.0f;

    std::vector<float> cdfthreshd_per_head;
    std::vector<float> topk_per_head;
    std::vector<float> simthreshold_per_head;

    bool  attention_sink = false;
    bool  smooth_k       = true;
    float scale          = 0.0f;  // 0 => default 1/sqrt(hdim); else caller-provided
};

namespace detail {
inline float lookup_per_head_host(const std::vector<float>& v, index_t h, float fb)
{
    return v.empty() ? fb : v[static_cast<size_t>(h)];
}
} // namespace detail

template <typename T>
HostTensor<uint8_t>
reference_sparge_mask_prediction(const HostTensor<T>& q_bhsd,
                               const HostTensor<T>& k_bhsd,
                               index_t batch,
                               index_t nhead,
                               index_t nhead_k,
                               index_t seqlen_q,
                               index_t seqlen_k,
                               index_t hdim,
                               index_t block_size_q,
                               index_t block_size_k,
                               const sparge_block_predict_params& params,
                               const sparge_causal_params& causal)
{
    const float topk_scalar       = params.topk;
    const float cdf_scalar        = params.cdfthreshd;
    const float sim_scalar        = params.simthreshold;
    const int causal_type         = causal.causal_type;
    const bool attention_sink     = params.attention_sink;
    const int window_left         = causal.window_left;
    const int window_right        = causal.window_right;

    index_t num_q_blocks = (seqlen_q + block_size_q - 1) / block_size_q;
    index_t num_k_blocks = (seqlen_k + block_size_k - 1) / block_size_k;
    float scale          = (params.scale != 0.0f)
                               ? params.scale
                               : 1.0f / std::sqrt(static_cast<float>(hdim));

    // Q side is never centered by the device (km_ptr == nullptr): pass null.
    auto q_means = compute_block_means(q_bhsd, batch, nhead, num_q_blocks,
                                       block_size_q, seqlen_q, hdim);
    auto q_sim = compute_block_similarity(q_bhsd, nullptr, batch, nhead, num_q_blocks,
                                          block_size_q, seqlen_q, hdim);

    // K side centers by global per-channel km iff smooth_k -- this must happen BEFORE k_sim so the
    // similarity is computed on km-centered tokens (matching the device), not just k_means.
    const bool have_km = params.smooth_k;
    std::vector<HostTensor<float>> km_storage; // 0 or 1 element (HostTensor has no default ctor)
    if(have_km)
        km_storage.push_back(
            compute_global_k_mean(k_bhsd, batch, nhead_k, seqlen_k, hdim));
    const HostTensor<float>* km = have_km ? &km_storage.front() : nullptr;

    auto k_means = compute_block_means(k_bhsd, batch, nhead_k, num_k_blocks,
                                       block_size_k, seqlen_k, hdim);
    auto k_sim   = compute_block_similarity(k_bhsd, km,
                                            batch, nhead_k, num_k_blocks,
                                            block_size_k, seqlen_k, hdim);
    if(have_km)
    {
        for(index_t b = 0; b < batch; ++b)
            for(index_t h = 0; h < nhead_k; ++h)
                for(index_t blk = 0; blk < num_k_blocks; ++blk)
                    for(index_t d = 0; d < hdim; ++d)
                        k_means(b, h, blk, d) -= (*km)(b, h, d);
    }

    HostTensor<float> k_means_expand({batch, nhead, num_k_blocks, hdim});
    HostTensor<float> k_sim_expand({batch, nhead, num_k_blocks});
    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
        {
            index_t h_k = h / (nhead / nhead_k);
            for(index_t blk = 0; blk < num_k_blocks; ++blk)
            {
                k_sim_expand(b, h, blk) = k_sim(b, h_k, blk);
                for(index_t d = 0; d < hdim; ++d)
                    k_means_expand(b, h, blk, d) = k_means(b, h_k, blk, d);
            }
        }

    HostTensor<uint8_t> mask({batch, nhead, num_q_blocks, num_k_blocks});
    const index_t causal_delta =
        (causal_type == 2) ? (seqlen_k - seqlen_q) : index_t{0};
    const index_t right_ext =
        (causal_type && window_right >= 0) ? window_right : index_t{0};

    for(index_t b = 0; b < batch; ++b)
    {
        for(index_t h = 0; h < nhead; ++h)
        {
            const float head_cdfthreshd   = detail::lookup_per_head_host(params.cdfthreshd_per_head,   h, cdf_scalar);
            const float head_topk         = detail::lookup_per_head_host(params.topk_per_head,         h, topk_scalar);
            const float head_simthreshold = detail::lookup_per_head_host(params.simthreshold_per_head, h, sim_scalar);
            const bool topk_mode = (head_topk > 0.0f);

            for(index_t qi = 0; qi < num_q_blocks; ++qi)
            {
                index_t last_q  = qi * block_size_q + block_size_q - 1;
                index_t first_q = qi * block_size_q;
                index_t causal_max_kj = causal_type
                    ? std::min(num_k_blocks - 1,
                               (last_q + right_ext + causal_delta) / block_size_k)
                    : (num_k_blocks - 1);
                index_t causal_min_kj = (causal_type && window_left >= 0)
                    ? std::max(index_t{0},
                               (first_q - window_left + causal_delta) / block_size_k)
                    : index_t{0};

                // Low-sim qi block (official ~sim_qblocks, sim <= thr): attend all causal-valid K.
                if(head_simthreshold > 0.0f && q_sim(b, h, qi) <= head_simthreshold)
                {
                    for(index_t kj = 0; kj < num_k_blocks; ++kj)
                        mask(b, h, qi, kj) =
                            (kj >= causal_min_kj && kj <= causal_max_kj) ? 1 : 0;
                    continue;
                }

                std::vector<float> row(static_cast<size_t>(num_k_blocks));
                for(index_t kj = 0; kj < num_k_blocks; ++kj)
                {
                    if(causal_type && (kj > causal_max_kj || kj < causal_min_kj))
                    {
                        row[static_cast<size_t>(kj)] = -1e30f;
                    }
                    // Low-sim kj block (official pooled_score[~sim_kblocks]=-inf): exclude from
                    // the softmax competition; force-selected after the loop.
                    else if(head_simthreshold > 0.0f && k_sim_expand(b, h, kj) <= head_simthreshold)
                    {
                        row[static_cast<size_t>(kj)] = -1e30f;
                    }
                    else
                    {
                        float dot = 0.0f;
                        for(index_t d = 0; d < hdim; ++d)
                            dot += q_means(b, h, qi, d) * k_means_expand(b, h, kj, d);
                        row[static_cast<size_t>(kj)] = dot * scale;
                    }
                }

                float mx = *std::max_element(row.begin(), row.end());
                std::vector<float> probs(static_cast<size_t>(num_k_blocks));
                float se = 0.0f;
                for(index_t kj = 0; kj < num_k_blocks; ++kj)
                {
                    probs[static_cast<size_t>(kj)] = std::exp(row[static_cast<size_t>(kj)] - mx);
                    se += probs[static_cast<size_t>(kj)];
                }
                for(auto& p : probs)
                    p /= se;

                std::vector<index_t> indices(static_cast<size_t>(num_k_blocks));
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&probs](auto lhs, auto rhs) {
                    return probs[static_cast<size_t>(lhs)] > probs[static_cast<size_t>(rhs)];
                });

                for(index_t kj = 0; kj < num_k_blocks; ++kj)
                    mask(b, h, qi, kj) = 0;

                const index_t n_target = topk_mode
                    ? std::max(index_t{1},
                               static_cast<index_t>(head_topk * static_cast<float>(num_k_blocks)))
                    : num_k_blocks;
                float cum = 0.0f;
                index_t n_selected = 0;
                for(index_t idx = 0; idx < num_k_blocks; ++idx)
                {
                    auto kj = indices[static_cast<size_t>(idx)];
                    if(causal_type && (kj > causal_max_kj || kj < causal_min_kj))
                        continue;
                    if(topk_mode)
                    {
                        mask(b, h, qi, kj) = 1;
                        if(++n_selected >= n_target)
                            break;
                    }
                    else
                    {
                        // CDF (official searchsorted right=True): stop before the block that pushes
                        // the cumulative past cdfthreshd; keep at least one.
                        const float next = cum + probs[static_cast<size_t>(kj)];
                        if(n_selected != 0 && next > head_cdfthreshd)
                            break;
                        mask(b, h, qi, kj) = 1;
                        cum = next;
                        ++n_selected;
                    }
                }

                // Force-include low-sim K blocks (official final_map[~sim_kblocks]=1; excluded
                // from the competition above).
                if(head_simthreshold > 0.0f)
                {
                    for(index_t kj = 0; kj < num_k_blocks; ++kj)
                        if(k_sim_expand(b, h, kj) <= head_simthreshold)
                            mask(b, h, qi, kj) = 1;
                }

                if(attention_sink && num_k_blocks > 0)
                    mask(b, h, qi, 0) = 1;

                if(causal_type)
                {
                    for(index_t kj = 0; kj < causal_min_kj; ++kj)
                        mask(b, h, qi, kj) = 0;
                    for(index_t kj = causal_max_kj + 1; kj < num_k_blocks; ++kj)
                        mask(b, h, qi, kj) = 0;
                }
            }
        }
    }
    return mask;
}

} // namespace ck_tile
