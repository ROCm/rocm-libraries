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

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

// Ragged (RFC-0014: packed [B,H,S,D] + ragged_offset) forward SDPA CPU reference — the host mirror
// of GpuFpReferenceSdpaRagged / GpuRefSdpaRaggedFwd.cpp. It is the pure fp32 oracle (no provider
// P-storage rounding), so GPU-vs-CPU comparisons run under gpuRefFwdTolerance.
//
// q/k/v/o are ragged-aware tensors (RFC-0014: `ShallowRaggedTensor<T>` / `RaggedTensor<T>`) with
// logical dims `[B, H, S, D]`, seqAxis=2, and BSHD-layout strides. Each carries its own
// `ragged_offset` aux, so packed addressing is delegated to the SDK: `getHostValue({b,h,s,d})` bases
// at `ragged_offset[b]` and adds the (batch-relative) strided offset, and per-batch sequence lengths
// come from `raggedIterationInfo()` (`rowOffsets`, `seqStride`). No manual global-token arithmetic.
// Supports GQA/MQA, per-batch causal/sliding-window, fp8 Q/K/V descale, and optional LSE (also a
// ragged tensor). No additive bias/alibi/dropout (gated off on the ASM v3 path).
class CpuFpReferenceSdpaRagged
{
public:
    // q/k/v/o (and lse) must be ragged tensors; descale tensors are ordinary (scalar/per-head).
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
        // Per-batch offsets/strides come from the ragged tensors themselves (RFC-0014).
        const auto qInfo = q.raggedIterationInfo();
        const auto kInfo = k.raggedIterationInfo();
        const auto vInfo = v.raggedIterationInfo();
        if(!qInfo.has_value() || !kInfo.has_value() || !vInfo.has_value())
        {
            throw std::invalid_argument("CpuFpReferenceSdpaRagged: q/k/v must be ragged tensors "
                                        "(ShallowRaggedTensor / RaggedTensor)");
        }

        validateInput(q.dims(), k.dims(), v.dims(), o.dims());

        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[1];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto headDimV = v.dims()[3];
        const auto headsPerHeadK = numHeads / numHeadsK;
        const auto headsPerHeadV = numHeads / numHeadsV;

        const auto scale = attnScaleValue.has_value()
                               ? static_cast<ComputeDataType>(attnScaleValue.value())
                               : (static_cast<ComputeDataType>(1.0)
                                  / std::sqrt(static_cast<ComputeDataType>(headDim)));

        // Element-offset tables (rowOffsets) + sequence strides for the Q and K/V axes.
        const auto& qRows = qInfo->rowOffsets;
        const auto& kRows = kInfo->rowOffsets;
        const auto qSeqStride = qInfo->seqStride;
        const auto kSeqStride = kInfo->seqStride;

        if(lse != nullptr)
        {
            validateLse(lse->dims(), batch, numHeads);
        }

        const DescaleBinding dq = bindDescale(descaleQ, batch, numHeads, "Q");
        const DescaleBinding dk = bindDescale(descaleK, batch, numHeadsK, "K");
        const DescaleBinding dv = bindDescale(descaleV, batch, numHeadsV, "V");

        const auto negInf = -std::numeric_limits<ComputeDataType>::infinity();

        for(int64_t b = 0; b < batch; ++b)
        {
            const int64_t seqQ
                = (qRows[static_cast<size_t>(b) + 1] - qRows[static_cast<size_t>(b)]) / qSeqStride;
            const int64_t seqKv
                = (kRows[static_cast<size_t>(b) + 1] - kRows[static_cast<size_t>(b)]) / kSeqStride;
            const int64_t windowOffset = topLeftAlignment ? 0 : (seqKv - seqQ);

            for(int64_t h = 0; h < numHeads; ++h)
            {
                const int64_t kvHeadK = h / headsPerHeadK;
                const int64_t kvHeadV = h / headsPerHeadV;

                const auto descaleQK
                    = static_cast<ComputeDataType>(dq.value(b, h) * dk.value(b, kvHeadK));
                const auto descaleVVal = dv.value(b, kvHeadV);

                for(int64_t sq = 0; sq < seqQ; ++sq)
                {
                    // Scaled, masked scores over this batch's key range.
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
                                q.getHostValue(std::vector<int64_t>{b, h, sq, d}));
                            const auto kv = static_cast<ComputeDataType>(
                                k.getHostValue(std::vector<int64_t>{b, kvHeadK, skv, d}));
                            dot += qv * kv;
                        }
                        scores[static_cast<size_t>(skv)] = dot * descaleQK * scale;
                    }

                    // Numerically stable softmax over skv.
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
                                           std::vector<int64_t>{b, h, sq, dvIdx});
                        }
                        if(lse != nullptr)
                        {
                            lse->setHostValue(static_cast<float>(negInf),
                                              std::vector<int64_t>{b, h, sq, 0});
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

                    // Weighted sum over V (fp32 accumulate), then fold in the V descale.
                    for(int64_t dvIdx = 0; dvIdx < headDimV; ++dvIdx)
                    {
                        auto acc = static_cast<ComputeDataType>(0);
                        for(int64_t skv = 0; skv < seqKv; ++skv)
                        {
                            const auto vv = static_cast<ComputeDataType>(
                                v.getHostValue(std::vector<int64_t>{b, kvHeadV, skv, dvIdx}));
                            acc += probs[static_cast<size_t>(skv)] * vv;
                        }
                        acc *= static_cast<ComputeDataType>(descaleVVal);
                        o.setHostValue(hipdnn_test_sdk::detail::safeConvert<ODataType>(acc),
                                       std::vector<int64_t>{b, h, sq, dvIdx});
                    }

                    if(lse != nullptr)
                    {
                        lse->setHostValue(static_cast<float>(maxVal + std::log(sumExp)),
                                          std::vector<int64_t>{b, h, sq, 0});
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
    // Resolved fp8 descale binding: host pointer + (batch, head) index strides. Absent or scalar
    // descale reports value 1 with zero strides. Descale tensors are ordinary (not ragged).
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
            binding.ptr = descale->memory().hostData(); // scalar; zero strides
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
                                    + " descale must be scalar [1] or per-head [B, heads, 1, 1]");
    }

    // Mirrors the kernel's per-batch window mask: asymmetric +1 on the right bound.
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

    static void validateLse(const std::vector<int64_t>& lseDims, int64_t batch, int64_t numHeads)
    {
        if(lseDims.size() != 4 || lseDims[0] != batch || lseDims[1] != numHeads || lseDims[3] != 1)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: lse must be rank-4 [B, H, Sq, 1]");
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
                "CpuFpReferenceSdpaRagged: q/k/v/o must all be rank-4 [B, H, S, D]");
        }
        const auto batch = qDims[0];
        if(kDims[0] != batch || vDims[0] != batch || oDims[0] != batch)
        {
            throw std::invalid_argument("CpuFpReferenceSdpaRagged: batch dimension mismatch");
        }
        if(kDims[3] != qDims[3])
        {
            throw std::invalid_argument("CpuFpReferenceSdpaRagged: Q head_dim != K head_dim");
        }
        if(vDims[2] != kDims[2])
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: K and V sequence extents (S_max) must match");
        }
        const auto numHeads = qDims[1];
        if(numHeads % kDims[1] != 0 || numHeads % vDims[1] != 0)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: numHeads must be divisible by numHeadsK and numHeadsV");
        }
        if(oDims[1] != numHeads || oDims[2] != qDims[2] || oDims[3] != vDims[3])
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: output shape must be [B, H, Sq, Dv]");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
