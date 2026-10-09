// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

namespace hipdnn_test_sdk::utilities
{

// Ragged forward SDPA CPU reference (RFC-0014: packed [B, S, H, D] + ragged_offset), the host
// mirror of GpuFpReferenceSdpaRagged. It computes in plain fp32 with no provider P-storage
// rounding, so GPU-vs-CPU checks use gpuRefFwdTolerance.
//
// q/k/v/o are ragged tensors (ShallowRaggedTensor / RaggedTensor) with dims [B, S, H, D] and the
// sequence at BSHD_SEQ_AXIS. The SDK does the packed addressing: getHostValue({b, s, h, d})
// starts at ragged_offset[b], and per-batch lengths come from raggedIterationInfo().
//
// Supports GQA/MQA, causal and sliding-window masks, fp8 descales, and an optional LSE that is
// ragged or dense [B, Sq_max, H, 1]. No bias, alibi or dropout, as on the ASM v3 path.
// Descales are scalar [1] or per KV head [B, H_kv, 1, 1]. Q and K descales are both indexed by
// the KV head of the query head, as in CpuFpReferenceSdpa and AITER.
//
// Q/O (and a ragged LSE) must have the same per-batch lengths, and so must K/V. Their element
// offsets can differ because their token widths differ.
class CpuFpReferenceSdpaRagged
{
public:
    // q/k/v/o must be ragged. lse may be ragged or dense. Descales are plain tensors.
    template <class QDataType,
              class KDataType = QDataType,
              class VDataType = QDataType,
              class ODataType = QDataType,
              class ComputeDataType = float>
    static void forward(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                        hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                        hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                        hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                        std::optional<float> attnScaleValue = std::nullopt,
                        int64_t leftBound = -1,
                        int64_t rightBound = -1,
                        bool topLeftAlignment = true,
                        hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr,
                        hipdnn_data_sdk::utilities::TensorBase<float>* descaleQ = nullptr,
                        hipdnn_data_sdk::utilities::TensorBase<float>* descaleK = nullptr,
                        hipdnn_data_sdk::utilities::TensorBase<float>* descaleV = nullptr)
    {
        const auto qInfo = q.raggedIterationInfo();
        const auto kInfo = k.raggedIterationInfo();
        const auto vInfo = v.raggedIterationInfo();
        const auto oInfo = o.raggedIterationInfo();
        if(!qInfo.has_value() || !kInfo.has_value() || !vInfo.has_value() || !oInfo.has_value())
        {
            throw std::invalid_argument("CpuFpReferenceSdpaRagged: q/k/v/o must be ragged tensors "
                                        "(ShallowRaggedTensor / RaggedTensor)");
        }

        validateInput(q.dims(), k.dims(), v.dims(), o.dims());

        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[2];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[2];
        const auto numHeadsV = v.dims()[2];
        const auto headDimV = v.dims()[3];
        const auto headsPerHeadK = numHeads / numHeadsK;
        const auto headsPerHeadV = numHeads / numHeadsV;

        // An absent scale is 1.0 (no scaling), as in cuDNN and the dense reference.
        const auto scale = static_cast<ComputeDataType>(attnScaleValue.value_or(1.0F));

        // Every ragged tensor must use the RFC-0014 layout, and tensors that share a packing must
        // agree on every batch's length.
        const std::string who = "CpuFpReferenceSdpaRagged";
        const auto tokens = [&](const hipdnn_data_sdk::utilities::RaggedIterationInfo& info,
                                const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& strides,
                                const char* name) {
            if(info.seqAxis != hipdnn_data_sdk::utilities::BSHD_SEQ_AXIS)
            {
                throw std::invalid_argument(who + ": " + name
                                            + " must be ragged along BSHD_SEQ_AXIS (1), got "
                                            + std::to_string(info.seqAxis));
            }
            detail::requireTokenMajorRaggedLayout(dims, strides, who, name);
            return detail::raggedTokenBoundaries(
                info.rowOffsets, info.seqStride, dims[1], who, name);
        };
        const auto qTokens = tokens(*qInfo, q.dims(), q.strides(), "Q");
        const auto kTokens = tokens(*kInfo, k.dims(), k.strides(), "K");
        detail::requireMatchingTokenBoundaries(
            qTokens, "Q", tokens(*oInfo, o.dims(), o.strides(), "O"), "O", who);
        detail::requireMatchingTokenBoundaries(
            kTokens, "K", tokens(*vInfo, v.dims(), v.strides(), "V"), "V", who);

        if(lse != nullptr)
        {
            validateLse(lse->dims(), q.dims());
            if(const auto lseInfo = lse->raggedIterationInfo())
            {
                detail::requireMatchingTokenBoundaries(
                    qTokens, "Q", tokens(*lseInfo, lse->dims(), lse->strides(), "LSE"), "LSE", who);
            }
        }

        // Q descale is per KV head, like K (AITER's [B, H_kv] contract).
        const DescaleBinding dq = bindDescale(descaleQ, batch, numHeadsK, "Q");
        const DescaleBinding dk = bindDescale(descaleK, batch, numHeadsK, "K");
        const DescaleBinding dv = bindDescale(descaleV, batch, numHeadsV, "V");

        const auto negInf = -std::numeric_limits<ComputeDataType>::infinity();

        for(int64_t b = 0; b < batch; ++b)
        {
            const auto bIdx = static_cast<size_t>(b);
            const int64_t seqQ = qTokens[bIdx + 1] - qTokens[bIdx];
            const int64_t seqKv = kTokens[bIdx + 1] - kTokens[bIdx];
            const int64_t windowOffset = topLeftAlignment ? 0 : (seqKv - seqQ);

            for(int64_t h = 0; h < numHeads; ++h)
            {
                const int64_t kvHeadK = h / headsPerHeadK;
                const int64_t kvHeadV = h / headsPerHeadV;

                const auto descaleQK
                    = static_cast<ComputeDataType>(dq.value(b, kvHeadK) * dk.value(b, kvHeadK));
                const auto descaleVVal = dv.value(b, kvHeadV);

                for(int64_t sq = 0; sq < seqQ; ++sq)
                {
                    std::vector<ComputeDataType> scores(static_cast<size_t>(seqKv));
                    for(int64_t skv = 0; skv < seqKv; ++skv)
                    {
                        if(isMasked(sq, skv, leftBound, rightBound, windowOffset))
                        {
                            scores[static_cast<size_t>(skv)] = negInf;
                            continue;
                        }
                        auto dot = static_cast<ComputeDataType>(0);
                        for(int64_t d = 0; d < headDim; ++d)
                        {
                            const auto qv = static_cast<ComputeDataType>(
                                q.getHostValue(std::vector<int64_t>{b, sq, h, d}));
                            const auto kv = static_cast<ComputeDataType>(
                                k.getHostValue(std::vector<int64_t>{b, skv, kvHeadK, d}));
                            dot += qv * kv;
                        }
                        scores[static_cast<size_t>(skv)] = dot * descaleQK * scale;
                    }

                    auto maxVal = negInf;
                    for(const auto s : scores)
                    {
                        maxVal = std::max(maxVal, s);
                    }

                    if(maxVal == negInf)
                    {
                        // Fully-masked row: output zero, LSE = -inf.
                        for(int64_t dvIdx = 0; dvIdx < headDimV; ++dvIdx)
                        {
                            o.setHostValue(hipdnn_test_sdk::detail::safeConvert<ODataType>(
                                               static_cast<ComputeDataType>(0)),
                                           std::vector<int64_t>{b, sq, h, dvIdx});
                        }
                        if(lse != nullptr)
                        {
                            lse->setHostValue(static_cast<float>(negInf),
                                              std::vector<int64_t>{b, sq, h, 0});
                        }
                        continue;
                    }

                    auto sumExp = static_cast<ComputeDataType>(0);
                    std::vector<ComputeDataType> probs(static_cast<size_t>(seqKv));
                    for(int64_t skv = 0; skv < seqKv; ++skv)
                    {
                        const auto s = scores[static_cast<size_t>(skv)];
                        const auto e = (s == negInf) ? static_cast<ComputeDataType>(0)
                                                     : std::exp(s - maxVal);
                        probs[static_cast<size_t>(skv)] = e;
                        sumExp += e;
                    }
                    for(auto& p : probs)
                    {
                        p /= sumExp;
                    }

                    // V descale is applied once, after accumulation.
                    for(int64_t dvIdx = 0; dvIdx < headDimV; ++dvIdx)
                    {
                        auto acc = static_cast<ComputeDataType>(0);
                        for(int64_t skv = 0; skv < seqKv; ++skv)
                        {
                            const auto vv = static_cast<ComputeDataType>(
                                v.getHostValue(std::vector<int64_t>{b, skv, kvHeadV, dvIdx}));
                            acc += probs[static_cast<size_t>(skv)] * vv;
                        }
                        acc *= static_cast<ComputeDataType>(descaleVVal);
                        o.setHostValue(hipdnn_test_sdk::detail::safeConvert<ODataType>(acc),
                                       std::vector<int64_t>{b, sq, h, dvIdx});
                    }

                    if(lse != nullptr)
                    {
                        lse->setHostValue(static_cast<float>(maxVal + std::log(sumExp)),
                                          std::vector<int64_t>{b, sq, h, 0});
                    }
                }
            }
        }

        o.memory().markHostModified();
        if(lse != nullptr)
        {
            lse->memory().markHostModified();
        }
    }

private:
    // fp8 descale lookup by (batch, head). A missing descale reads as 1, a scalar has zero
    // strides. Descale tensors are not ragged.
    struct DescaleBinding
    {
        const float* ptr = nullptr;
        int64_t batchStride = 0;
        int64_t headStride = 0;

        float value(int64_t b, int64_t head) const
        {
            return ptr != nullptr ? ptr[b * batchStride + head * headStride] : 1.0F;
        }
    };

    static DescaleBinding bindDescale(hipdnn_data_sdk::utilities::TensorBase<float>* descale,
                                      int64_t batch,
                                      int64_t heads,
                                      const char* name)
    {
        DescaleBinding binding;
        if(descale == nullptr)
        {
            return binding;
        }
        if(descale->elementCount() == 1)
        {
            binding.ptr = descale->memory().hostData(); // strides stay 0 for a scalar
            return binding;
        }
        const auto& dims = descale->dims();
        if(dims.size() == 4 && dims[0] == batch && dims[1] == heads && dims[2] == 1 && dims[3] == 1)
        {
            binding.ptr = descale->memory().hostData();
            binding.batchStride = descale->strides()[0];
            binding.headStride = descale->strides()[1];
            return binding;
        }
        throw std::invalid_argument(std::string("CpuFpReferenceSdpaRagged: ") + name
                                    + " descale must be scalar [1] or per-KV-head [B, H_kv, 1, 1]");
    }

    // Same per-batch window mask as the kernel. Keeps skv in
    // [sq + windowOffset - leftBound, sq + windowOffset + rightBound]. A negative bound is open.
    static bool isMasked(
        int64_t sq, int64_t skv, int64_t leftBound, int64_t rightBound, int64_t windowOffset)
    {
        if(rightBound >= 0)
        {
            const int64_t startKv = std::max<int64_t>(sq + 1 + windowOffset + rightBound, 0);
            if(skv >= startKv)
            {
                return true;
            }
        }
        if(leftBound >= 0 && skv < sq + windowOffset - leftBound)
        {
            return true;
        }
        return false;
    }

    // LSE must be [B, Sq, H, 1] with Q's Sq. A shorter Sq would spill rows into the next batch.
    static void validateLse(const std::vector<int64_t>& lseDims, const std::vector<int64_t>& qDims)
    {
        if(lseDims.size() != 4 || lseDims[0] != qDims[0] || lseDims[1] != qDims[1]
           || lseDims[2] != qDims[2] || lseDims[3] != 1)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: lse must be rank-4 [B, Sq, H, 1] with Q's B, Sq, H");
        }
    }

    static void validateInput(const std::vector<int64_t>& qDims,
                              const std::vector<int64_t>& kDims,
                              const std::vector<int64_t>& vDims,
                              const std::vector<int64_t>& oDims)
    {
        if(qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: q/k/v/o must all be rank-4 [B, S, H, D]");
        }
        const auto batch = qDims[0];
        // S_max may be 0 (every batch empty); the other dims may not.
        if(batch <= 0 || qDims[2] <= 0 || qDims[3] <= 0 || kDims[2] <= 0 || vDims[2] <= 0
           || vDims[3] <= 0)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: all dimensions must be positive");
        }
        if(kDims[0] != batch || vDims[0] != batch || oDims[0] != batch)
        {
            throw std::invalid_argument("CpuFpReferenceSdpaRagged: batch dimension mismatch");
        }
        if(kDims[3] != qDims[3])
        {
            throw std::invalid_argument("CpuFpReferenceSdpaRagged: Q head_dim != K head_dim");
        }
        if(vDims[1] != kDims[1])
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: K and V sequence extents (S_max) must match");
        }
        const auto numHeads = qDims[2];
        if(numHeads % kDims[2] != 0 || numHeads % vDims[2] != 0)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: numHeads must be divisible by numHeadsK and numHeadsV");
        }
        if(oDims[1] != qDims[1] || oDims[2] != numHeads || oDims[3] != vDims[3])
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: output shape must be [B, Sq, H, Dv]");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
