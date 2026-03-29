// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceSdpa
{
public:
    /// SDPA forward: O = softmax(Q @ K^T * scale) @ V
    ///
    /// Supports GQA/MQA: numHeads must be divisible by both numHeadsK and numHeadsV.
    /// Optionally adds an additive attention mask before softmax.
    ///
    /// @param q              Query tensor [B, H, Sq, D]
    /// @param k              Key tensor   [B, Hkv, Skv, D]
    /// @param v              Value tensor [B, Hkv, Skv, Dv]
    /// @param o              Output tensor [B, H, Sq, Dv]
    /// @param attnScaleValue Optional scale factor; defaults to 1/sqrt(D)
    /// @param attnMask       Optional additive attention mask with dims right-aligned
    ///                       to [B, H, Sq, Skv] (rank 1–4), with broadcasting on size-1 dims
    /// @param causalMask     When true, applies a lower-triangular causal mask so each
    ///                       query position sq can only attend to kv positions skv <= sq
    template <class QDataType,
              class KDataType,
              class VDataType,
              class ODataType,
              class ComputeDataType = float>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                        const hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                        const hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                        hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                        std::optional<float> attnScaleValue = std::nullopt,
                        const hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* attnMask
                        = nullptr,
                        bool causalMask = false)
    {
        if(q.dims().size() != 4)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: q must be rank-4 [B, H, Sq, D]");
        }
        if(k.dims().size() != 4)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: k must be rank-4 [B, Hkv, Skv, D]");
        }
        if(v.dims().size() != 4)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: v must be rank-4 [B, Hkv, Skv, Dv]");
        }
        if(o.dims().size() != 4)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: o must be rank-4 [B, H, Sq, Dv]");
        }
        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[1];
        const auto seqQ = q.dims()[2];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto seqKv = k.dims()[2];
        const auto headDimV = v.dims()[3];
        if(batch <= 0 || numHeads <= 0 || seqQ <= 0 || headDim <= 0 || numHeadsK <= 0
           || numHeadsV <= 0 || seqKv <= 0 || headDimV <= 0)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: all dimensions must be positive");
        }

        // ===== CROSS-TENSOR CHECKS (insert HERE) =====
        if(k.dims()[0] != batch || v.dims()[0] != batch || o.dims()[0] != batch)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: batch dimension mismatch");
        }
        if(k.dims()[3] != headDim)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: Q head_dim (" + std::to_string(headDim)
                                        + ") != K head_dim (" + std::to_string(k.dims()[3]) + ")");
        }
        if(numHeads % numHeadsV != 0)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa: numHeads must be divisible by numHeadsV");
        }
        if(v.dims()[2] != seqKv)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: K and V sequence lengths must match");
        }
        if(o.dims()[1] != numHeads || o.dims()[2] != seqQ || o.dims()[3] != headDimV)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: output shape must be [B, H, Sq, Dv]");
        }
        if(numHeads % numHeadsK != 0)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa: numHeads must be divisible by numHeadsK");
        }

        const auto headsPerHeadK = numHeads / numHeadsK;
        const auto headsPerHeadV = numHeads / numHeadsV;

        const float scale = attnScaleValue.has_value()
                                ? attnScaleValue.value()
                                : (1.0f / std::sqrt(static_cast<float>(headDim)));

        const std::vector<int64_t> parallelDims = {batch, numHeads, seqQ};

        auto sdpaFwdFunc = [&](const std::vector<int64_t>& indices) {
            const auto b = indices[0];
            const auto h = indices[1];
            const auto sq = indices[2];
            const auto kvHeadK = h / headsPerHeadK;
            const auto kvHeadV = h / headsPerHeadV;

            // Step 1: Compute scaled dot-product scores S[skv]
            std::vector<float> scores(static_cast<size_t>(seqKv));
            for(int64_t skv = 0; skv < seqKv; ++skv)
            {
                float dot = 0.0f;
                for(int64_t d = 0; d < headDim; ++d)
                {
                    dot += static_cast<float>(q.getHostValue(std::vector<int64_t>{b, h, sq, d}))
                           * static_cast<float>(
                               k.getHostValue(std::vector<int64_t>{b, kvHeadK, skv, d}));
                }
                scores[static_cast<size_t>(skv)] = dot * scale;
            }

            // Step 2: Add additive attention mask (if provided)
            if(attnMask != nullptr)
            {
                const auto& maskDims = attnMask->dims();
                const auto maskRank = static_cast<int64_t>(maskDims.size());

                // Mask dims are right-aligned to the output context [b, h, sq, skv]
                constexpr int64_t K_OUTPUT_CONTEXT_RANK = 4;

                for(int64_t skv = 0; skv < seqKv; ++skv)
                {
                    const std::array<int64_t, K_OUTPUT_CONTEXT_RANK> ctxIdxs = {b, h, sq, skv};
                    std::vector<int64_t> maskIndices(static_cast<size_t>(maskRank));
                    for(int64_t i = 0; i < maskRank; ++i)
                    {
                        const auto outputDimIdx
                            = static_cast<size_t>(K_OUTPUT_CONTEXT_RANK - maskRank + i);
                        const auto outputIdx = ctxIdxs[outputDimIdx];
                        maskIndices[static_cast<size_t>(i)]
                            = (maskDims[static_cast<size_t>(i)] == 1) ? 0 : outputIdx;
                    }
                    scores[static_cast<size_t>(skv)]
                        += static_cast<float>(attnMask->getHostValue(maskIndices));
                }
            }

            // Step 3: Apply causal mask (skv > sq → -inf)
            if(causalMask)
            {
                for(int64_t skv = sq + 1; skv < seqKv; ++skv)
                {
                    scores[static_cast<size_t>(skv)] = -std::numeric_limits<float>::infinity();
                }
            }

            // Step 4: Numerically stable softmax over skv
            const float maxVal = *std::max_element(scores.begin(), scores.end());
            std::vector<float> probs(static_cast<size_t>(seqKv));
            float sumExp = 0.0f;
            for(int64_t skv = 0; skv < seqKv; ++skv)
            {
                probs[static_cast<size_t>(skv)]
                    = std::exp(scores[static_cast<size_t>(skv)] - maxVal);
                sumExp += probs[static_cast<size_t>(skv)];
            }
            for(int64_t skv = 0; skv < seqKv; ++skv)
            {
                probs[static_cast<size_t>(skv)] /= sumExp;
            }

            // Step 5: Weighted sum over V to produce O
            for(int64_t dv = 0; dv < headDimV; ++dv)
            {
                float acc = 0.0f;
                for(int64_t skv = 0; skv < seqKv; ++skv)
                {
                    acc += probs[static_cast<size_t>(skv)]
                           * static_cast<float>(
                               v.getHostValue(std::vector<int64_t>{b, kvHeadV, skv, dv}));
                }
                o.setHostValue(hipdnn_test_sdk::detail::safeConvert<ODataType>(acc),
                               std::vector<int64_t>{b, h, sq, dv});
            }
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(sdpaFwdFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        o.memory().markHostModified();
    }
};

} // namespace hipdnn_test_sdk::utilities
