// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// Reference implementation: blocked attention (for sparse attention tests).
template <typename T, typename MaskT, typename BiasT = T, typename AccT = float>
void reference_blocked_attention(
    const HostTensor<T>& q,
    const HostTensor<T>& k,
    const HostTensor<T>& v,
    const HostTensor<MaskT>& block_relation,
    HostTensor<T>& output,
    index_t BLKQ,
    index_t BLKK,
    AccT scale,
    int causal_type           = 0,
    int window_left           = -1,
    int window_right          = -1,
    AccT logits_soft_cap      = AccT{0},
    const HostTensor<BiasT>* bias = nullptr,
    int bias_rank             = 0)
{
    auto q_lengths   = q.get_lengths();
    index_t batch    = q_lengths[0];
    index_t nhead_q  = q_lengths[1];
    index_t seqlen_q = q_lengths[2];
    index_t hdim     = q_lengths[3];

    auto k_lengths  = k.get_lengths();
    index_t nhead_k = k_lengths[1];

    auto v_lengths   = v.get_lengths();
    index_t seqlen_k = v_lengths[2];
    index_t hdim_v   = v_lengths[3];

    index_t num_q_blocks = (seqlen_q + BLKQ - 1) / BLKQ;
    index_t num_k_blocks = (seqlen_k + BLKK - 1) / BLKK;

    const index_t qk_head_ratio = (nhead_k > 0) ? (nhead_q / nhead_k) : index_t{1};

    const index_t causal_delta = (causal_type == 2) ? (seqlen_k - seqlen_q) : index_t{0};

    const bool has_left_bound  = (window_left >= 0);
    const bool has_right_bound = (window_right >= 0);

    const bool has_soft_cap = (logits_soft_cap > AccT{0});
    const bool has_bias     = (bias != nullptr);
    const AccT inv_cap      = has_soft_cap ? (AccT{1} / logits_soft_cap) : AccT{0};

    for(index_t b = 0; b < batch; ++b)
    {
        for(index_t h = 0; h < nhead_q; ++h)
        {
            const index_t hk     = h / qk_head_ratio;
            const index_t bias_b = (bias_rank == 2) ? b : index_t{0};
            const index_t bias_h = (bias_rank == 0) ? index_t{0} : h;
            for(index_t qb = 0; qb < num_q_blocks; ++qb)
            {
                index_t q_start = qb * BLKQ;
                if(q_start >= seqlen_q)
                {
                    continue;
                }
                index_t q_end = std::min<index_t>(q_start + BLKQ, seqlen_q);

                std::vector<index_t> relevant_k_indices;
                for(index_t kb = 0; kb < num_k_blocks; ++kb)
                {
                    if(static_cast<float>(block_relation(b, h, qb, kb)) > 0.5f)
                    {
                        relevant_k_indices.push_back(kb);
                    }
                }

                if(relevant_k_indices.empty())
                {
                    continue;
                }

                for(index_t sq = q_start; sq < q_end; ++sq)
                {
                    std::vector<AccT> scores;
                    AccT max_score = -std::numeric_limits<AccT>::infinity();

                    for(auto kb : relevant_k_indices)
                    {
                        index_t k_start = kb * BLKK;
                        if(k_start >= seqlen_k)
                        {
                            continue;
                        }
                        index_t k_end = std::min<index_t>(k_start + BLKK, seqlen_k);

                        for(index_t sk = k_start; sk < k_end; ++sk)
                        {
                            bool masked = false;
                            if(has_right_bound &&
                               sk > sq + causal_delta + static_cast<index_t>(window_right))
                            {
                                masked = true;
                            }
                            if(has_left_bound &&
                               sk < sq + causal_delta - static_cast<index_t>(window_left))
                            {
                                masked = true;
                            }
                            if(masked)
                            {
                                scores.push_back(-std::numeric_limits<AccT>::infinity());
                                continue;
                            }
                            AccT score = AccT{0};
                            for(index_t d = 0; d < hdim; ++d)
                            {
                                score +=
                                    type_convert<AccT>(q(b, h, sq, d)) * type_convert<AccT>(k(b, hk, sk, d));
                            }
                            if(has_soft_cap)
                            {
                                score = logits_soft_cap * std::tanh(score * scale * inv_cap);
                            }
                            else
                            {
                                score = score * scale;
                                if(has_bias)
                                {
                                    score += type_convert<AccT>((*bias)(bias_b, bias_h, sq, sk));
                                }
                            }
                            scores.push_back(score);
                            max_score = std::max(max_score, score);
                        }
                    }

                    const bool all_masked =
                        (max_score == -std::numeric_limits<AccT>::infinity());

                    AccT sum_exp = 0.0f;
                    if(!all_masked)
                    {
                        for(auto& s : scores)
                        {
                            s = std::exp(s - max_score);
                            sum_exp += s;
                        }
                        for(auto& s : scores)
                        {
                            s /= sum_exp;
                        }
                    }

                    for(index_t dv = 0; dv < hdim_v; ++dv)
                    {
                        AccT out_val = 0.0f;
                        if(!all_masked)
                        {
                            size_t score_idx = 0;
                            for(auto kb : relevant_k_indices)
                            {
                                index_t k_start = kb * BLKK;
                                if(k_start >= seqlen_k)
                                {
                                    continue;
                                }
                                index_t k_end = std::min<index_t>(k_start + BLKK, seqlen_k);

                                for(index_t sk = k_start; sk < k_end; ++sk)
                                {
                                    out_val += scores[score_idx] *
                                               type_convert<AccT>(v(b, hk, sk, dv));
                                    score_idx++;
                                }
                            }
                        }
                        output(b, h, sq, dv) = type_convert<T>(out_val);
                    }
                }
            }
        }
    }
}

} // namespace ck_tile
