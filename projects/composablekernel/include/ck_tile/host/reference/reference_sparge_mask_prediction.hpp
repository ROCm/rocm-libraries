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
                        sum += static_cast<float>(data_bhsd(b, h, s, d));
                    means(b, h, blk, d) = sum / static_cast<float>(count);
                }
            }
    return means;
}

template <typename T>
HostTensor<float> compute_block_similarity(
    const HostTensor<T>& data_bhsd,
    const HostTensor<float>& means,
    index_t batch,
    index_t nhead,
    index_t num_blocks,
    index_t block_size,
    index_t seqlen,
    index_t hdim)
{
    HostTensor<float> sim({batch, nhead, num_blocks});
    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
            for(index_t blk = 0; blk < num_blocks; ++blk)
            {
                index_t s_start = blk * block_size;
                index_t s_end   = std::min(s_start + block_size, seqlen);
                index_t count   = s_end - s_start;
                float mean_norm_sq       = 0.0f;
                for(index_t d = 0; d < hdim; ++d)
                {
                    float m = means(b, h, blk, d);
                    mean_norm_sq += m * m;
                }
                float mean_norm = std::sqrt(mean_norm_sq + 1e-8f);
                float total_sim = 0.0f;
                for(index_t s = s_start; s < s_end; ++s)
                {
                    float dot      = 0.0f;
                    float tok_norm = 0.0f;
                    for(index_t d = 0; d < hdim; ++d)
                    {
                        float v = static_cast<float>(data_bhsd(b, h, s, d));
                        dot += v * means(b, h, blk, d);
                        tok_norm += v * v;
                    }
                    tok_norm = std::sqrt(tok_norm + 1e-8f);
                    total_sim += dot / (tok_norm * mean_norm);
                }
                sim(b, h, blk) = total_sim / static_cast<float>(count);
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
                    sum += static_cast<float>(k_bhsd(b, h, s, d));
                km(b, h, d) = sum / static_cast<float>(seqlen_k);
            }
    return km;
}

struct sparge_mask_prediction_params
{
    float cdfthreshd   = 0.0f;
    float topk         = 0.0f;
    float simthreshold = 0.0f;

    std::vector<float> cdfthreshd_per_head;
    std::vector<float> topk_per_head;
    std::vector<float> simthreshold_per_head;

    int  causal_type    = 0;
    bool attention_sink = false;
    int  window_left    = -1;
    int  window_right   = -1;
    bool smooth_k       = true;
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
                               const sparge_mask_prediction_params& params)
{
    const float topk_scalar       = params.topk;
    const float cdf_scalar        = params.cdfthreshd;
    const float sim_scalar        = params.simthreshold;
    const int causal_type         = params.causal_type;
    const bool attention_sink     = params.attention_sink;
    const int window_left         = params.window_left;
    const int window_right        = params.window_right;

    index_t num_q_blocks = (seqlen_q + block_size_q - 1) / block_size_q;
    index_t num_k_blocks = (seqlen_k + block_size_k - 1) / block_size_k;
    float scale          = 1.0f / std::sqrt(static_cast<float>(hdim));

    auto q_means = compute_block_means(q_bhsd, batch, nhead, num_q_blocks,
                                       block_size_q, seqlen_q, hdim);
    auto q_sim = compute_block_similarity(q_bhsd, q_means, batch, nhead, num_q_blocks,
                                          block_size_q, seqlen_q, hdim);

    HostTensor<float> k_means({batch, nhead_k, num_k_blocks, hdim});
    HostTensor<float> k_sim({batch, nhead_k, num_k_blocks});
    k_means = compute_block_means(k_bhsd, batch, nhead_k, num_k_blocks,
                                  block_size_k, seqlen_k, hdim);
    k_sim   = compute_block_similarity(k_bhsd, k_means, batch, nhead_k, num_k_blocks,
                                       block_size_k, seqlen_k, hdim);
    if(params.smooth_k)
    {
        const auto km = compute_global_k_mean(k_bhsd, batch, nhead_k, seqlen_k, hdim);
        for(index_t b = 0; b < batch; ++b)
            for(index_t h = 0; h < nhead_k; ++h)
                for(index_t blk = 0; blk < num_k_blocks; ++blk)
                    for(index_t d = 0; d < hdim; ++d)
                        k_means(b, h, blk, d) -= km(b, h, d);
    }

    HostTensor<float> k_means_exp({batch, nhead, num_k_blocks, hdim});
    HostTensor<float> k_sim_exp({batch, nhead, num_k_blocks});
    for(index_t b = 0; b < batch; ++b)
        for(index_t h = 0; h < nhead; ++h)
        {
            index_t h_k = h / (nhead / nhead_k);
            for(index_t blk = 0; blk < num_k_blocks; ++blk)
            {
                k_sim_exp(b, h, blk) = k_sim(b, h_k, blk);
                for(index_t d = 0; d < hdim; ++d)
                    k_means_exp(b, h, blk, d) = k_means(b, h_k, blk, d);
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

                if(head_simthreshold > 0.0f && q_sim(b, h, qi) < head_simthreshold)
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
                    else if(head_simthreshold > 0.0f && k_sim_exp(b, h, kj) < head_simthreshold)
                    {
                        row[static_cast<size_t>(kj)] = 1e6f;
                    }
                    else
                    {
                        float dot = 0.0f;
                        for(index_t d = 0; d < hdim; ++d)
                            dot += q_means(b, h, qi, d) * k_means_exp(b, h, kj, d);
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
                    mask(b, h, qi, kj) = 1;
                    cum += probs[static_cast<size_t>(kj)];
                    n_selected++;
                    if(topk_mode ? (n_selected >= n_target) : (cum >= head_cdfthreshd))
                        break;
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
