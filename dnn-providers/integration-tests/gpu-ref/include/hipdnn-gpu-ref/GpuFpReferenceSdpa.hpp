// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

enum class SdpaSoftmaxProbabilityMode : int
{
    FLOAT = 0,
    BFLOAT16_RTNE = 1,
    BFLOAT16_RTZ = 2,
};

namespace detail
{

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType>
inline std::vector<std::string> buildSdpaDefines(SdpaSoftmaxProbabilityMode probabilityMode)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DQ_TYPE=") + HipRtcTypeName<QDataType>::VALUE);
    defines.emplace_back(std::string("-DK_TYPE=") + HipRtcTypeName<KDataType>::VALUE);
    defines.emplace_back(std::string("-DV_TYPE=") + HipRtcTypeName<VDataType>::VALUE);
    defines.emplace_back(std::string("-DO_TYPE=") + HipRtcTypeName<ODataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DSDPA_SOFTMAX_PROBABILITY_MODE=")
                         + std::to_string(static_cast<int>(probabilityMode)));
    return defines;
}

} // namespace detail

class GpuFpReferenceSdpa
{
public:
    // --- Forward SDPA (fprop): O = softmax(Q @ K^T * scale + mask) @ V ---
    //
    // By default this mirrors the fp32-softmax numerical contract of
    // CpuFpReferenceSdpa::forward. Callers that compare against provider kernels
    // can request a provider-attuned softmax probability storage mode, e.g. bf16
    // RTNE for AITER BF16 forward where P is rounded before P@V.
    // Supports GQA/MQA: numHeads must be divisible by both numHeadsK and numHeadsV.
    //
    // Takes non-const references because deviceData() may trigger host→device sync.
    // attnMask is non-const for the same reason (uploading it needs deviceData()).
    template <class QDataType,
              class KDataType = QDataType,
              class VDataType = QDataType,
              class ODataType = QDataType,
              class ComputeDataType = float>
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                      hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                      hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                      hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                      std::optional<float> attnScaleValue = std::nullopt,
                      hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* attnMask = nullptr,
                      int64_t leftBound = -1,
                      int64_t rightBound = -1,
                      bool topLeftAlignment = true,
                      hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr,
                      SdpaSoftmaxProbabilityMode probabilityMode
                      = SdpaSoftmaxProbabilityMode::FLOAT)
    {
        validateInput(q.dims(), k.dims(), v.dims(), o.dims());

        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[1];
        const auto seqQ = q.dims()[2];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto seqKv = k.dims()[2];
        const auto headDimV = v.dims()[3];

        const float scale = attnScaleValue.has_value()
                                ? attnScaleValue.value()
                                : (1.0F / std::sqrt(static_cast<float>(headDim)));

        auto defines
            = detail::buildSdpaDefines<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                probabilityMode);

        const void* maskPtr = nullptr;
        std::vector<int64_t> maskDims;
        std::vector<int64_t> maskStrides;
        if(attnMask != nullptr)
        {
            // The kernel right-aligns the mask to [B, H, Sq, Skv] and indexes
            // args.maskDims/maskStr (size 4); a rank>4 mask would read out of bounds.
            if(attnMask->dims().empty() || attnMask->dims().size() > 4)
            {
                throw std::invalid_argument("GpuFpReferenceSdpa: attention mask must be rank 1..4");
            }
            maskPtr = attnMask->memory().deviceData();
            maskDims = attnMask->dims();
            maskStrides = attnMask->strides();
        }

        void* lsePtr = nullptr;
        std::vector<int64_t> lseStrides;
        if(lse != nullptr)
        {
            // LSE is one value per query position; mirror the CPU oracle's shape check.
            if(lse->dims().size() != 3 || lse->dims()[0] != batch || lse->dims()[1] != numHeads
               || lse->dims()[2] != seqQ)
            {
                throw std::invalid_argument("GpuFpReferenceSdpa: lse must be rank-3 [B, H, Sq]");
            }
            lsePtr = lse->memory().deviceData();
            lseStrides = lse->strides();
        }

        launchSdpaFwd(q.memory().deviceData(),
                      k.memory().deviceData(),
                      v.memory().deviceData(),
                      maskPtr,
                      o.memory().deviceData(),
                      lsePtr,
                      q.strides(),
                      k.strides(),
                      v.strides(),
                      o.strides(),
                      maskStrides,
                      maskDims,
                      lseStrides,
                      batch,
                      numHeads,
                      numHeadsK,
                      numHeadsV,
                      seqQ,
                      seqKv,
                      headDim,
                      headDimV,
                      scale,
                      leftBound,
                      rightBound,
                      topLeftAlignment,
                      defines);

        o.memory().markDeviceModified();
        if(lse != nullptr)
        {
            lse->memory().markDeviceModified();
        }
    }

private:
    // --- Validation ---
    // No SDPA param validator exists; perform minimal shape checks only.

    static void validateInput(const std::vector<int64_t>& qDims,
                              const std::vector<int64_t>& kDims,
                              const std::vector<int64_t>& vDims,
                              const std::vector<int64_t>& oDims)
    {
        if(qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: q/k/v/o must all be rank-4 tensors");
        }

        const auto batch = qDims[0];
        const auto numHeads = qDims[1];
        const auto seqQ = qDims[2];
        const auto headDim = qDims[3];
        const auto numHeadsK = kDims[1];
        const auto numHeadsV = vDims[1];
        const auto seqKv = kDims[2];
        const auto headDimV = vDims[3];

        if(batch <= 0 || numHeads <= 0 || seqQ <= 0 || headDim <= 0 || numHeadsK <= 0
           || numHeadsV <= 0 || seqKv <= 0 || headDimV <= 0)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: all dimensions must be positive");
        }

        if(kDims[0] != batch || vDims[0] != batch || oDims[0] != batch)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: batch dimension mismatch");
        }
        if(kDims[3] != headDim)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: Q head_dim != K head_dim");
        }
        if(vDims[2] != seqKv)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: K and V sequence lengths must match");
        }
        if(numHeadsK == 0 || numHeadsV == 0 || numHeads % numHeadsK != 0
           || numHeads % numHeadsV != 0)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpa: numHeads must be divisible by numHeadsK and numHeadsV");
        }
        if(oDims[1] != numHeads || oDims[2] != seqQ || oDims[3] != headDimV)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: output shape must be [B, H, Sq, Dv]");
        }
    }

    // --- Kernel launcher (defined in GpuFpReferenceSdpa.cpp) ---

    static void launchSdpaFwd(const void* qPtr,
                              const void* kPtr,
                              const void* vPtr,
                              const void* maskPtr,
                              void* oPtr,
                              void* lsePtr,
                              const std::vector<int64_t>& qTensorStrides,
                              const std::vector<int64_t>& kTensorStrides,
                              const std::vector<int64_t>& vTensorStrides,
                              const std::vector<int64_t>& oTensorStrides,
                              const std::vector<int64_t>& maskTensorStrides,
                              const std::vector<int64_t>& maskDims,
                              const std::vector<int64_t>& lseTensorStrides,
                              int64_t batch,
                              int64_t numHeads,
                              int64_t numHeadsK,
                              int64_t numHeadsV,
                              int64_t seqQ,
                              int64_t seqKv,
                              int64_t headDim,
                              int64_t headDimV,
                              float scale,
                              int64_t leftBound,
                              int64_t rightBound,
                              bool topLeftAlignment,
                              const std::vector<std::string>& defines);
};

} // namespace hipdnn_gpu_ref
