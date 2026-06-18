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
    /// Optionally outputs LSE (log-sum-exp) for backward pass recomputation.
    ///
    /// @param q                Query tensor [B, H, Sq, D]
    /// @param k                Key tensor   [B, Hkv, Skv, D]
    /// @param v                Value tensor [B, Hkv, Skv, Dv]
    /// @param o                Output tensor [B, H, Sq, Dv]
    /// @param attnScaleValue   Optional scale factor; defaults to 1/sqrt(D)
    /// @param attnMask         Optional additive attention mask with dims right-aligned
    ///                         to [B, H, Sq, Skv] (rank 1–4), with broadcasting on size-1 dims
    /// @param leftBound        Number of sequence elements to the left that are unmasked
    ///                         -1 is no mask to the left
    /// @param rightBound        Number of sequence elements to the right that are unmasked
    ///                         -1 is no mask to the right
    /// @param topLeftAlignment If true, diagonal is measured from top left
    ///                         If false, diagonal is measured from bottom right
    /// @param lse              Optional log-sum-exp output [B, H, Sq] (always float type).
    ///                         Stores maxVal + log(sumExp) for each query position.
    ///                         Used for memory-efficient backward pass recomputation.
    ///                         Pass nullptr (default) to disable LSE output.
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
                        int64_t leftBound = -1,
                        int64_t rightBound = -1,
                        bool topLeftAlignment = true,
                        hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr)
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

        // Validate LSE tensor if provided
        if(lse != nullptr)
        {
            if(lse->dims().size() != 3)
            {
                throw std::invalid_argument("CpuFpReferenceSdpa: lse must be rank-3 [B, H, Sq]");
            }
            if(lse->dims()[0] != batch || lse->dims()[1] != numHeads || lse->dims()[2] != seqQ)
            {
                throw std::invalid_argument(
                    "CpuFpReferenceSdpa: lse shape must be [" + std::to_string(batch) + ", "
                    + std::to_string(numHeads) + ", " + std::to_string(seqQ) + "] but got ["
                    + std::to_string(lse->dims()[0]) + ", " + std::to_string(lse->dims()[1]) + ", "
                    + std::to_string(lse->dims()[2]) + "]");
            }
        }

        const auto headsPerHeadK = numHeads / numHeadsK;
        const auto headsPerHeadV = numHeads / numHeadsV;

        const auto scale = attnScaleValue.has_value()
                               ? static_cast<ComputeDataType>(attnScaleValue.value())
                               : (static_cast<ComputeDataType>(1.0)
                                  / std::sqrt(static_cast<ComputeDataType>(headDim)));

        const std::vector<int64_t> parallelDims = {batch, numHeads, seqQ};

        auto sdpaFwdFunc = [&](const std::vector<int64_t>& indices) {
            const auto b = indices[0];
            const auto h = indices[1];
            const auto sq = indices[2];
            const auto kvHeadK = h / headsPerHeadK;
            const auto kvHeadV = h / headsPerHeadV;

            // Step 1: Compute scaled dot-product scores S[skv]
            std::vector<ComputeDataType> scores(static_cast<size_t>(seqKv));
            for(int64_t skv = 0; skv < seqKv; ++skv)
            {
                auto dot = static_cast<ComputeDataType>(0.0);
                for(int64_t d = 0; d < headDim; ++d)
                {
                    dot += static_cast<ComputeDataType>(
                               q.getHostValue(std::vector<int64_t>{b, h, sq, d}))
                           * static_cast<ComputeDataType>(
                               k.getHostValue(std::vector<int64_t>{b, kvHeadK, skv, d}));
                }
                scores[static_cast<size_t>(skv)] = dot * scale;
            }

            // Step 2: Add additive attention mask (if provided)
            if(attnMask != nullptr)
            {
                for(int64_t skv = 0; skv < seqKv; ++skv)
                {
                    const auto maskIndices = computeMaskIndex(attnMask->dims(), b, h, sq, skv);
                    scores[static_cast<size_t>(skv)]
                        += static_cast<ComputeDataType>(attnMask->getHostValue(maskIndices));
                }
            }

            // Step 3: Apply sliding-window mask.
            // For topLeftAlignment, the diagonal is at skv == sq.
            // For bottomRightAlignment, the diagonal is at skv == sq + (seqKv - seqQ),
            // which is negative (i.e. nothing visible) for early query positions when seqKv < seqQ.
            if(rightBound >= 0)
            {
                // Offset to account for bottomRightAlignment
                const int64_t offset = (topLeftAlignment) ? 0 : seqKv - seqQ;
                const int64_t startKv = std::max<int64_t>(sq + 1 + offset + rightBound, 0);
                for(int64_t skv = startKv; skv < seqKv; ++skv)
                {
                    scores[static_cast<size_t>(skv)]
                        = -std::numeric_limits<ComputeDataType>::infinity();
                }
            }
            if(leftBound >= 0)
            {
                // Offset to account for bottomRightAlignment
                const int64_t offset = (topLeftAlignment) ? 0 : seqKv - seqQ;
                for(int64_t skv = 0; skv < sq + offset - leftBound; ++skv)
                {
                    scores[static_cast<size_t>(skv)]
                        = -std::numeric_limits<ComputeDataType>::infinity();
                }
            }

            // Step 4: Numerically stable softmax over skv
            const auto maxVal = *std::max_element(scores.begin(), scores.end());
            std::vector<ComputeDataType> probs(static_cast<size_t>(seqKv));
            auto sumExp = static_cast<ComputeDataType>(0.0);

            // If row is not all -inf, probs should be calculated normally
            if(maxVal != -std::numeric_limits<ComputeDataType>::infinity())
            {
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
            }
            // Else, probabilities should remain all zero

            // Step 4b: Write LSE (log-sum-exp) if requested
            // LSE = maxVal + log(sumExp) enables backward pass to recompute softmax
            // probabilities without storing the full [B, H, Sq, Skv] attention matrix.
            // For fully masked rows (sumExp = 0), log(0) = -inf which is correct.
            if(lse != nullptr)
            {
                const auto lseVal = static_cast<float>(maxVal + std::log(sumExp));
                lse->setHostValue(lseVal, std::vector<int64_t>{b, h, sq});
            }

            // Step 5: Weighted sum over V to produce O
            for(int64_t dv = 0; dv < headDimV; ++dv)
            {
                auto acc = static_cast<ComputeDataType>(0.0);
                for(int64_t skv = 0; skv < seqKv; ++skv)
                {
                    acc += probs[static_cast<size_t>(skv)]
                           * static_cast<ComputeDataType>(
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
        if(lse != nullptr)
        {
            lse->memory().markHostModified();
        }
    }

    /// SDPA forward: O = softmax(Q @ K^T * scale) @ V
    ///
    /// Supports GQA/MQA: numHeads must be divisible by both numHeadsK and numHeadsV.
    /// Optionally adds an additive attention mask before softmax.
    /// Optionally outputs LSE (log-sum-exp) for backward pass recomputation.
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
    /// @param lse            Optional log-sum-exp output [B, H, Sq] (always float type).
    ///                       Stores maxVal + log(sumExp) for each query position.
    ///                       Used for memory-efficient backward pass recomputation.
    ///                       Pass nullptr (default) to disable LSE output.
    template <class QDataType,
              class KDataType,
              class VDataType,
              class ODataType,
              class ComputeDataType = float>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                        const hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                        const hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                        hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                        std::optional<float> attnScaleValue,
                        const hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* attnMask,
                        bool causalMask,
                        hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr)
    {
        const int64_t leftBound = -1;
        const int64_t rightBound = (causalMask) ? 0 : -1;
        const bool topLeftAlignment = true;

        forward(q, k, v, o, attnScaleValue, attnMask, leftBound, rightBound, topLeftAlignment, lse);
    }

    /// SDPA backward: computes dQ, dK, dV from upstream gradient dO
    ///
    /// Implements Flash Attention backward pass algorithm. Recomputes attention
    /// probabilities from Q, K to avoid storing the full attention matrix.
    /// Optionally uses LSE (log-sum-exp) from forward pass for efficient softmax
    /// recomputation.
    ///
    /// @param q              Query tensor [B, H_q, Sq, D]
    /// @param k              Key tensor   [B, H_k, Skv, D]
    /// @param v              Value tensor [B, H_v, Skv, Dv]
    /// @param o              Output from forward pass [B, H_q, Sq, Dv]
    /// @param dO             Upstream gradient [B, H_q, Sq, Dv]
    /// @param dQ             Output: gradient w.r.t. Q [B, H_q, Sq, D]
    /// @param dK             Output: gradient w.r.t. K [B, H_k, Skv, D]
    /// @param dV             Output: gradient w.r.t. V [B, H_v, Skv, Dv]
    /// @param attnScaleValue Optional scale factor; defaults to 1/sqrt(D)
    /// @param lse            Optional log-sum-exp from forward [B, H_q, Sq] (FP32).
    ///                       When provided, enables efficient softmax recomputation.
    ///                       When nullptr, recomputes softmax from scratch.
    /// @param attnMask       Optional additive attention mask (same as forward)
    /// @param causalMask     When true, applies causal masking (same as forward)
    ///
    /// Note: For GQA (H_q > H_kv), multiple query heads accumulate gradients to
    /// the same KV heads. This implementation is sequential to ensure correctness.
    template <class QDataType,
              class KDataType,
              class VDataType,
              class ODataType,
              class DODataType,
              class DQDataType,
              class DKDataType,
              class DVDataType,
              class ComputeDataType = float>
    static void backward(const hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                         const hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                         const hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                         const hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                         const hipdnn_data_sdk::utilities::TensorBase<DODataType>& dO,
                         hipdnn_data_sdk::utilities::TensorBase<DQDataType>& dQ,
                         hipdnn_data_sdk::utilities::TensorBase<DKDataType>& dK,
                         hipdnn_data_sdk::utilities::TensorBase<DVDataType>& dV,
                         std::optional<float> attnScaleValue = std::nullopt,
                         const hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr,
                         const hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* attnMask
                         = nullptr,
                         bool causalMask = false)
    {
        // Validate input tensor ranks
        if(q.dims().size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: q must be rank-4 [B, H_q, Sq, D]");
        }
        if(k.dims().size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: k must be rank-4 [B, H_kv, Skv, D]");
        }
        if(v.dims().size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: v must be rank-4 [B, H_kv, Skv, Dv]");
        }
        if(o.dims().size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: o must be rank-4 [B, H_q, Sq, Dv]");
        }
        if(dO.dims().size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dO must be rank-4 [B, H_q, Sq, Dv]");
        }
        if(dQ.dims().size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dQ must be rank-4 [B, H_q, Sq, D]");
        }
        if(dK.dims().size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dK must be rank-4 [B, H_kv, Skv, D]");
        }
        if(dV.dims().size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dV must be rank-4 [B, H_kv, Skv, Dv]");
        }

        // Extract dimensions
        const auto batch = q.dims()[0];
        const auto numHeadsQ = q.dims()[1];
        const auto seqQ = q.dims()[2];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto seqKv = k.dims()[2];
        const auto headDimV = v.dims()[3];

        if(batch <= 0 || numHeadsQ <= 0 || seqQ <= 0 || headDim <= 0 || numHeadsK <= 0
           || numHeadsV <= 0 || seqKv <= 0 || headDimV <= 0)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: all dimensions must be positive");
        }

        // Cross-tensor dimension validation
        if(k.dims()[0] != batch || v.dims()[0] != batch || o.dims()[0] != batch
           || dO.dims()[0] != batch || dQ.dims()[0] != batch || dK.dims()[0] != batch
           || dV.dims()[0] != batch)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa::backward: batch dimension mismatch");
        }
        if(k.dims()[3] != headDim)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa::backward: Q head_dim != K head_dim");
        }
        if(v.dims()[2] != seqKv)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: K and V sequence lengths must match");
        }
        if(o.dims()[1] != numHeadsQ || o.dims()[2] != seqQ || o.dims()[3] != headDimV)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: o shape must be [B, H_q, Sq, Dv]");
        }
        if(dO.dims()[1] != numHeadsQ || dO.dims()[2] != seqQ || dO.dims()[3] != headDimV)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dO shape must be [B, H_q, Sq, Dv]");
        }
        if(dQ.dims()[1] != numHeadsQ || dQ.dims()[2] != seqQ || dQ.dims()[3] != headDim)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dQ shape must be [B, H_q, Sq, D]");
        }
        if(dK.dims()[1] != numHeadsK || dK.dims()[2] != seqKv || dK.dims()[3] != headDim)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dK shape must be [B, H_k, Skv, D]");
        }
        if(dV.dims()[1] != numHeadsV || dV.dims()[2] != seqKv || dV.dims()[3] != headDimV)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dV shape must be [B, H_v, Skv, Dv]");
        }
        if(numHeadsQ % numHeadsK != 0)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: numHeadsQ must be divisible by numHeadsK (GQA)");
        }
        if(numHeadsQ % numHeadsV != 0)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: numHeadsQ must be divisible by numHeadsV (GQA)");
        }

        // Validate LSE tensor if provided
        if(lse != nullptr)
        {
            if(lse->dims().size() != 3)
            {
                throw std::invalid_argument(
                    "CpuFpReferenceSdpa::backward: lse must be rank-3 [B, H_q, Sq]");
            }
            if(lse->dims()[0] != batch || lse->dims()[1] != numHeadsQ || lse->dims()[2] != seqQ)
            {
                throw std::invalid_argument(
                    "CpuFpReferenceSdpa::backward: lse shape must be [B, H_q, Sq]");
            }
        }

        const auto headsPerHeadK = numHeadsQ / numHeadsK;
        const auto headsPerHeadV = numHeadsQ / numHeadsV;
        const auto scale = attnScaleValue.has_value()
                               ? static_cast<ComputeDataType>(attnScaleValue.value())
                               : (static_cast<ComputeDataType>(1.0)
                                  / std::sqrt(static_cast<ComputeDataType>(headDim)));

        // Accumulate gradients in FP32 to match GPU kernel behavior.
        // The GPU ASM kernel uses FP32 accumulators (A32 path) and only converts
        // to the output type at the very end. Accumulating directly in the output
        // type (e.g. BF16) truncates the running sum at every iteration, producing
        // significantly worse precision than the GPU — especially for BF16 where
        // the 7-bit mantissa causes large cumulative rounding error over S iterations.
        const auto totalDqElements = static_cast<size_t>(batch * numHeadsQ * seqQ * headDim);
        const auto totalDkElements = static_cast<size_t>(batch * numHeadsK * seqKv * headDim);
        const auto totalDvElements = static_cast<size_t>(batch * numHeadsV * seqKv * headDimV);

        std::vector<ComputeDataType> dqAccum(totalDqElements, ComputeDataType(0));
        std::vector<ComputeDataType> dkAccum(totalDkElements, ComputeDataType(0));
        std::vector<ComputeDataType> dvAccum(totalDvElements, ComputeDataType(0));

        // Helper lambdas for contiguous indexing into accumulation buffers.
        // Layout: [B, H, S, D] with contiguous strides.
        auto dqIdx = [&](int64_t bIdx, int64_t hIdx, int64_t sIdx, int64_t dIdx) -> size_t {
            return static_cast<size_t>(((bIdx * numHeadsQ + hIdx) * seqQ + sIdx) * headDim + dIdx);
        };
        auto dkIdx = [&](int64_t bIdx, int64_t hIdx, int64_t sIdx, int64_t dIdx) -> size_t {
            return static_cast<size_t>(((bIdx * numHeadsK + hIdx) * seqKv + sIdx) * headDim + dIdx);
        };
        auto dvIdx = [&](int64_t bIdx, int64_t hIdx, int64_t sIdx, int64_t dIdx) -> size_t {
            return static_cast<size_t>(((bIdx * numHeadsV + hIdx) * seqKv + sIdx) * headDimV
                                       + dIdx);
        };

        // Sequential loop over [B, H_q, Sq]: multiple sq positions accumulate
        // into the same dK/dV entries, and in GQA multiple Q heads also share
        // K/V heads, so this cannot be parallelized without atomics.
        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t hQ = 0; hQ < numHeadsQ; ++hQ)
            {
                for(int64_t sq = 0; sq < seqQ; ++sq)
                {
                    const auto kvHeadK = hQ / headsPerHeadK;
                    const auto kvHeadV = hQ / headsPerHeadV;

                    // Allocate temporary buffers for this query position
                    std::vector<ComputeDataType> scores(static_cast<size_t>(seqKv));
                    std::vector<ComputeDataType> probs(static_cast<size_t>(seqKv));
                    std::vector<ComputeDataType> dpBuffer(static_cast<size_t>(seqKv));

                    // Step 1: Compute D[b,hQ,sq] = sum(dO[b,hQ,sq,:] * O[b,hQ,sq,:])
                    auto dVal = static_cast<ComputeDataType>(0.0);
                    for(int64_t dv = 0; dv < headDimV; ++dv)
                    {
                        const auto doVal = static_cast<ComputeDataType>(
                            dO.getHostValue(std::vector<int64_t>{b, hQ, sq, dv}));
                        const auto oVal = static_cast<ComputeDataType>(
                            o.getHostValue(std::vector<int64_t>{b, hQ, sq, dv}));
                        dVal += doVal * oVal;
                    }

                    // Step 2: Recompute attention scores and probabilities
                    // Option A: Use LSE if available (efficient)
                    // Option B: Recompute softmax from scratch
                    if(lse != nullptr)
                    {
                        // Efficient recomputation using LSE from forward pass
                        const float lseVal = lse->getHostValue(std::vector<int64_t>{b, hQ, sq});

                        for(int64_t skv = 0; skv < seqKv; ++skv)
                        {
                            // Apply causal mask
                            if(causalMask && skv > sq)
                            {
                                scores[static_cast<size_t>(skv)]
                                    = -std::numeric_limits<ComputeDataType>::infinity();
                                probs[static_cast<size_t>(skv)] = static_cast<ComputeDataType>(0.0);
                                continue;
                            }

                            // Compute QK dot product
                            auto qkDot = static_cast<ComputeDataType>(0.0);
                            for(int64_t d = 0; d < headDim; ++d)
                            {
                                qkDot += static_cast<ComputeDataType>(
                                             q.getHostValue(std::vector<int64_t>{b, hQ, sq, d}))
                                         * static_cast<ComputeDataType>(k.getHostValue(
                                             std::vector<int64_t>{b, kvHeadK, skv, d}));
                            }
                            scores[static_cast<size_t>(skv)]
                                = qkDot * static_cast<ComputeDataType>(scale);

                            // Apply additive attention mask if provided
                            if(attnMask != nullptr)
                            {
                                const auto maskIndices
                                    = computeMaskIndex(attnMask->dims(), b, hQ, sq, skv);
                                scores[static_cast<size_t>(skv)] += static_cast<ComputeDataType>(
                                    attnMask->getHostValue(maskIndices));
                            }

                            // Recompute probability using LSE:
                            // P[skv] = exp(score[skv] - lse)
                            // where lse = maxVal + log(sumExp)
                            probs[static_cast<size_t>(skv)]
                                = std::exp(scores[static_cast<size_t>(skv)]
                                           - static_cast<ComputeDataType>(lseVal));
                        }
                    }
                    else
                    {
                        // Recompute softmax from scratch (no LSE available)
                        auto maxVal = -std::numeric_limits<ComputeDataType>::infinity();

                        for(int64_t skv = 0; skv < seqKv; ++skv)
                        {
                            // Apply causal mask
                            if(causalMask && skv > sq)
                            {
                                scores[static_cast<size_t>(skv)]
                                    = -std::numeric_limits<ComputeDataType>::infinity();
                                continue;
                            }

                            // Compute QK dot product
                            auto qkDot = static_cast<ComputeDataType>(0.0);
                            for(int64_t d = 0; d < headDim; ++d)
                            {
                                qkDot += static_cast<ComputeDataType>(
                                             q.getHostValue(std::vector<int64_t>{b, hQ, sq, d}))
                                         * static_cast<ComputeDataType>(k.getHostValue(
                                             std::vector<int64_t>{b, kvHeadK, skv, d}));
                            }
                            scores[static_cast<size_t>(skv)]
                                = qkDot * static_cast<ComputeDataType>(scale);

                            // Apply additive attention mask if provided
                            if(attnMask != nullptr)
                            {
                                const auto maskIndices
                                    = computeMaskIndex(attnMask->dims(), b, hQ, sq, skv);
                                scores[static_cast<size_t>(skv)] += static_cast<ComputeDataType>(
                                    attnMask->getHostValue(maskIndices));
                            }

                            maxVal = std::max(maxVal, scores[static_cast<size_t>(skv)]);
                        }

                        // Compute softmax probabilities
                        auto sumExp = static_cast<ComputeDataType>(0.0);
                        for(int64_t skv = 0; skv < seqKv; ++skv)
                        {
                            if(scores[static_cast<size_t>(skv)]
                               == -std::numeric_limits<ComputeDataType>::infinity())
                            {
                                probs[static_cast<size_t>(skv)] = static_cast<ComputeDataType>(0.0);
                            }
                            else
                            {
                                probs[static_cast<size_t>(skv)]
                                    = std::exp(scores[static_cast<size_t>(skv)] - maxVal);
                                sumExp += probs[static_cast<size_t>(skv)];
                            }
                        }
                        for(int64_t skv = 0; skv < seqKv; ++skv)
                        {
                            probs[static_cast<size_t>(skv)] /= sumExp;
                        }
                    }

                    // Step 3: Compute dP = dO @ V^T
                    for(int64_t skv = 0; skv < seqKv; ++skv)
                    {
                        auto dpVal = static_cast<ComputeDataType>(0.0);
                        for(int64_t dv = 0; dv < headDimV; ++dv)
                        {
                            const auto doVal = static_cast<ComputeDataType>(
                                dO.getHostValue(std::vector<int64_t>{b, hQ, sq, dv}));
                            const auto vVal = static_cast<ComputeDataType>(
                                v.getHostValue(std::vector<int64_t>{b, kvHeadV, skv, dv}));
                            dpVal += doVal * vVal;
                        }
                        dpBuffer[static_cast<size_t>(skv)] = dpVal;
                    }

                    // Step 4: Compute dS = P * (dP - D)  [softmax backward]
                    // and accumulate gradients
                    for(int64_t skv = 0; skv < seqKv; ++skv)
                    {
                        if(causalMask && skv > sq)
                        {
                            continue;
                        }

                        const ComputeDataType dsVal = probs[static_cast<size_t>(skv)]
                                                      * (dpBuffer[static_cast<size_t>(skv)] - dVal);
                        const ComputeDataType dsScaled
                            = dsVal * static_cast<ComputeDataType>(scale);

                        // dQ[b, hQ, sq, d] += dS[skv] * K[b, kvHeadK, skv, d] * scale
                        for(int64_t d = 0; d < headDim; ++d)
                        {
                            const auto kVal = static_cast<ComputeDataType>(
                                k.getHostValue(std::vector<int64_t>{b, kvHeadK, skv, d}));
                            const auto dqContrib = dsScaled * kVal;

                            dqAccum[dqIdx(b, hQ, sq, d)] += dqContrib;
                        }

                        // dK[b, kvHeadK, skv, d] += dS[skv] * Q[b, hQ, sq, d] * scale
                        // Note: For GQA, multiple Q heads accumulate to same K head
                        for(int64_t d = 0; d < headDim; ++d)
                        {
                            const auto qVal = static_cast<ComputeDataType>(
                                q.getHostValue(std::vector<int64_t>{b, hQ, sq, d}));
                            const auto dkContrib = dsScaled * qVal;

                            dkAccum[dkIdx(b, kvHeadK, skv, d)] += dkContrib;
                        }

                        // dV[b, kvHeadV, skv, dv] += P[skv] * dO[b, hQ, sq, dv]
                        for(int64_t dv = 0; dv < headDimV; ++dv)
                        {
                            const auto doVal = static_cast<ComputeDataType>(
                                dO.getHostValue(std::vector<int64_t>{b, hQ, sq, dv}));
                            const auto dvContrib = probs[static_cast<size_t>(skv)] * doVal;

                            dvAccum[dvIdx(b, kvHeadV, skv, dv)] += dvContrib;
                        }
                    }
                }
            }
        }

        // Convert FP32 accumulators to output type — single conversion at the end,
        // matching the GPU kernel's dq_convert behavior.
        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t h = 0; h < numHeadsQ; ++h)
            {
                for(int64_t s = 0; s < seqQ; ++s)
                {
                    for(int64_t d = 0; d < headDim; ++d)
                    {
                        dQ.setHostValue(hipdnn_test_sdk::detail::safeConvert<DQDataType>(
                                            dqAccum[dqIdx(b, h, s, d)]),
                                        std::vector<int64_t>{b, h, s, d});
                    }
                }
            }
        }
        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t h = 0; h < numHeadsK; ++h)
            {
                for(int64_t s = 0; s < seqKv; ++s)
                {
                    for(int64_t d = 0; d < headDim; ++d)
                    {
                        dK.setHostValue(hipdnn_test_sdk::detail::safeConvert<DKDataType>(
                                            dkAccum[dkIdx(b, h, s, d)]),
                                        std::vector<int64_t>{b, h, s, d});
                    }
                }
            }
        }
        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t h = 0; h < numHeadsV; ++h)
            {
                for(int64_t s = 0; s < seqKv; ++s)
                {
                    for(int64_t d = 0; d < headDimV; ++d)
                    {
                        dV.setHostValue(hipdnn_test_sdk::detail::safeConvert<DVDataType>(
                                            dvAccum[dvIdx(b, h, s, d)]),
                                        std::vector<int64_t>{b, h, s, d});
                    }
                }
            }
        }

        dQ.memory().markHostModified();
        dK.memory().markHostModified();
        dV.memory().markHostModified();
    }

private:
    /// Compute broadcastable mask indices by right-aligning mask dims to [b, h, sq, skv].
    /// Dimensions of size 1 are broadcast (index clamped to 0).
    static std::vector<int64_t> computeMaskIndex(
        const std::vector<int64_t>& maskDims, int64_t b, int64_t h, int64_t sq, int64_t skv)
    {
        constexpr int64_t K_OUTPUT_CONTEXT_RANK = 4;
        const auto maskRank = static_cast<int64_t>(maskDims.size());
        const std::array<int64_t, K_OUTPUT_CONTEXT_RANK> ctxIdxs = {b, h, sq, skv};

        std::vector<int64_t> maskIndices(static_cast<size_t>(maskRank));
        for(int64_t i = 0; i < maskRank; ++i)
        {
            const auto outputDimIdx = static_cast<size_t>(K_OUTPUT_CONTEXT_RANK - maskRank + i);
            const auto outputIdx = ctxIdxs[outputDimIdx];
            maskIndices[static_cast<size_t>(i)]
                = (maskDims[static_cast<size_t>(i)] == 1) ? 0 : outputIdx;
        }
        return maskIndices;
    }
};

} // namespace hipdnn_test_sdk::utilities
