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
#include "ck_tile/ops/fmha/block/block_masking.hpp"

namespace ck_tile {

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
    int causal_type               = 0,
    int window_left               = -1,
    int window_right              = -1,
    AccT logits_soft_cap          = AccT{0},
    const HostTensor<BiasT>* bias = nullptr,
    int bias_rank                 = 0,
    // Stage-2 pv-skip: skip a selected K-block when every Q-row's block-peak (raw QK + bias/scale)
    // is > pvthreshd below the running max. <=0 disables; pvthreshd_per_head overrides per head.
    // Works with bias too; disabled on the soft-cap path.
    AccT pvthreshd                               = AccT{0},
    const std::vector<float>* pvthreshd_per_head = nullptr,
    // sage path: round-trip each softmax prob through fp8_t before PV, mirroring the device P->fp8.
    bool quant_p_fp8 = false)
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

    // Masking mirrors the device FMHA exactly (same make_generic_attention_mask_from_lr_window +
    // IsOutOfBound predicate). causal_type: 0 = no mask, 1 = top-left, 2 = bottom-right.
    const bool has_mask    = (causal_type != 0) || (window_left >= 0) || (window_right >= 0);
    const bool is_top_left = (causal_type != 2);
    const auto mask =
        make_generic_attention_mask_from_lr_window<SimplifiedGenericAttentionMask<true>>(
            window_left, window_right, seqlen_q, seqlen_k, is_top_left);

    const bool has_soft_cap = (logits_soft_cap > AccT{0});
    const bool has_bias     = (bias != nullptr);
    const AccT inv_cap      = has_soft_cap ? (AccT{1} / logits_soft_cap) : AccT{0};

    // pv-skip predicate is on raw-QK (+bias/scale) units, matching the device block-peak (taken
    // after adding bias). Disabled on the nonlinear soft-cap path (which is NO_BIAS-only).
    const bool pv_skip_path = !has_soft_cap;

    for(index_t b = 0; b < batch; ++b)
    {
        for(index_t h = 0; h < nhead_q; ++h)
        {
            const index_t hk     = h / qk_head_ratio;
            const index_t bias_b = (bias_rank == 2) ? b : index_t{0};
            const index_t bias_h = (bias_rank == 0) ? index_t{0} : h;
            const AccT pvthreshd_eff =
                (pvthreshd_per_head && !pvthreshd_per_head->empty())
                    ? static_cast<AccT>((*pvthreshd_per_head)[static_cast<size_t>(h)])
                    : pvthreshd;
            const bool pv_skip_enabled = pv_skip_path && (pvthreshd_eff > AccT{0});
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

                // Replay the device running-max skip over the selected blocks (ascending). Per
                // Q-row keep run_max of the raw-QK block-peak; skip a block when every row's peak
                // is > pvthreshd below run_max. Skipped blocks leave run_max unchanged.
                if(pv_skip_enabled)
                {
                    const index_t n_rows = q_end - q_start;
                    std::vector<AccT> run_max(static_cast<size_t>(n_rows),
                                              -std::numeric_limits<AccT>::infinity());
                    std::vector<index_t> active_k_indices;
                    active_k_indices.reserve(relevant_k_indices.size());
                    for(auto kb : relevant_k_indices)
                    {
                        index_t k_start = kb * BLKK;
                        if(k_start >= seqlen_k)
                            continue;
                        index_t k_end = std::min<index_t>(k_start + BLKK, seqlen_k);

                        std::vector<AccT> m_local(static_cast<size_t>(n_rows),
                                                  -std::numeric_limits<AccT>::infinity());
                        for(index_t sq = q_start; sq < q_end; ++sq)
                        {
                            AccT row_peak = -std::numeric_limits<AccT>::infinity();
                            for(index_t sk = k_start; sk < k_end; ++sk)
                            {
                                if(has_mask && mask.IsOutOfBound(sq, sk))
                                    continue;
                                AccT raw = AccT{0};
                                for(index_t d = 0; d < hdim; ++d)
                                    raw += type_convert<AccT>(q(b, h, sq, d)) *
                                           type_convert<AccT>(k(b, hk, sk, d));
                                // device adds bias (in raw-QK units == bias/scale) before the peak.
                                if(has_bias)
                                    raw +=
                                        type_convert<AccT>((*bias)(bias_b, bias_h, sq, sk)) / scale;
                                row_peak = std::max(row_peak, raw);
                            }
                            m_local[static_cast<size_t>(sq - q_start)] = row_peak;
                        }

                        AccT block_max_diff = -std::numeric_limits<AccT>::infinity();
                        for(index_t r = 0; r < n_rows; ++r)
                        {
                            const AccT ml  = m_local[static_cast<size_t>(r)];
                            const AccT mn  = std::max(run_max[static_cast<size_t>(r)], ml);
                            block_max_diff = std::max(block_max_diff, ml - mn);
                        }
                        const bool skip = (block_max_diff < -pvthreshd_eff);
                        if(skip)
                            continue; // run_max unchanged
                        for(index_t r = 0; r < n_rows; ++r)
                            run_max[static_cast<size_t>(r)] = std::max(
                                run_max[static_cast<size_t>(r)], m_local[static_cast<size_t>(r)]);
                        active_k_indices.push_back(kb);
                    }
                    relevant_k_indices.swap(active_k_indices);
                    if(relevant_k_indices.empty())
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
                            if(has_mask && mask.IsOutOfBound(sq, sk))
                            {
                                scores.push_back(-std::numeric_limits<AccT>::infinity());
                                continue;
                            }
                            AccT score = AccT{0};
                            for(index_t d = 0; d < hdim; ++d)
                            {
                                score += type_convert<AccT>(q(b, h, sq, d)) *
                                         type_convert<AccT>(k(b, hk, sk, d));
                            }
                            // fmha order: scale -> soft-cap -> +bias (cap and bias both optional).
                            score = has_soft_cap
                                        ? logits_soft_cap * std::tanh(score * scale * inv_cap)
                                        : score * scale;
                            if(has_bias)
                            {
                                score += type_convert<AccT>((*bias)(bias_b, bias_h, sq, sk));
                            }
                            scores.push_back(score);
                            max_score = std::max(max_score, score);
                        }
                    }

                    const bool all_masked = (max_score == -std::numeric_limits<AccT>::infinity());

                    // Keep P unnormalized; divide PV by p_denom at the end (device flow never
                    // normalizes P before PV, so fp8-P slots in cleanly).
                    AccT sum_exp = 0.0f;
                    if(!all_masked)
                    {
                        for(auto& s : scores)
                        {
                            s = std::exp(s - max_score);
                            sum_exp += s;
                        }
                    }
                    AccT p_denom = sum_exp;
                    if(quant_p_fp8 && !all_masked)
                    {
                        // p_shift lifts P into fp8 range before the cast; a power of two, it
                        // cancels in the final /p_denom and only sets the underflow floor.
#if CK_TILE_USE_OCP_FP8
                        constexpr AccT p_shift = AccT{256}; // 2^8
#else
                        constexpr AccT p_shift = AccT{128}; // 2^7
#endif
                        for(auto& s : scores)
                        {
                            s = type_convert<AccT>(type_convert<fp8_t>(s * p_shift));
                        }
                        p_denom = p_shift * sum_exp;
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
                                    out_val +=
                                        scores[score_idx] * type_convert<AccT>(v(b, hk, sk, dv));
                                    score_idx++;
                                }
                            }
                            out_val /= p_denom;
                        }
                        output(b, h, sq, dv) = type_convert<T>(out_val);
                    }
                }
            }
        }
    }
}

} // namespace ck_tile
