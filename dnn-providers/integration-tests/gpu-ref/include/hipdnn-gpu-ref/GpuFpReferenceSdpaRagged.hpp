// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_gpu_ref
{

// RFC-0014 ragged_offset_multiplier of each ragged tensor: element offset = stored offset *
// multiplier. 1 means the table holds element offsets. AITER's cu_seqlens are token offsets, so
// they bind with multiplier = H*D (the tensor's seq stride), and one table can then serve Q/O
// and another K/V.
struct RaggedOffsetMultipliers
{
    int64_t q = 1;
    int64_t k = 1;
    int64_t v = 1;
    int64_t o = 1;
    int64_t lse = 1;
};

// GPU reference for ragged forward SDPA (RFC-0014), the ragged twin of GpuFpReferenceSdpa.
// Tensors are rank-4 [B, S, H, D] with the sequence at axis 1, packed token by token with no
// per-batch padding:
//   q = [B, Sq,  H,  D ]   k = [B, Skv, Hk, D ]
//   v = [B, Skv, Hv, Dv]   o = [B, Sq,  H,  Dv]
// raggedOffsetQ/K/V/O are INT32 [B+1, 1, 1, 1] cumulative offsets, scaled to elements by
// `offsetMultipliers`. Batch b starts at element offset[b] and has
// (offset[b+1] - offset[b]) / strides[1] tokens. O must match Q's token boundaries and V must
// match K's.
// The optional LSE is [B, Sq, H, 1], ragged if raggedOffsetLse is given, else dense.
// Softmax numerics match the dense reference. Supports GQA/MQA and per-batch causal and
// sliding window. No bias, alibi or dropout, since the ASM v3 path gates them off.
class GpuFpReferenceSdpaRagged
{
public:
    // Takes non-const references because deviceData() may trigger host→device sync.
    template <class QDataType,
              class KDataType = QDataType,
              class VDataType = QDataType,
              class ODataType = QDataType,
              class ComputeDataType = float>
    static void
        fpropRagged(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                    hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                    hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                    hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>& raggedOffsetQ,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>& raggedOffsetK,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>& raggedOffsetV,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>& raggedOffsetO,
                    std::optional<float> attnScaleValue = std::nullopt,
                    int64_t leftBound = -1,
                    int64_t rightBound = -1,
                    bool topLeftAlignment = true,
                    hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>* raggedOffsetLse = nullptr,
                    SdpaSoftmaxProbabilityMode probabilityMode = SdpaSoftmaxProbabilityMode::FLOAT,
                    hipdnn_data_sdk::utilities::TensorBase<float>* descaleQ = nullptr,
                    hipdnn_data_sdk::utilities::TensorBase<float>* descaleK = nullptr,
                    hipdnn_data_sdk::utilities::TensorBase<float>* descaleV = nullptr,
                    const RaggedOffsetMultipliers& offsetMultipliers = {})
    {
        // fp8 is input-only: the kernel can decode fp8 but not encode it (AITER writes bf16).
        static_assert(!std::is_same_v<ODataType, hipdnn_data_sdk::types::fp8_e4m3>,
                      "fpropRagged: fp8 output is not supported; pass a bf16 or float ODataType");
        validateInput(q.dims(),
                      k.dims(),
                      v.dims(),
                      o.dims(),
                      {raggedOffsetQ.dims(),
                       raggedOffsetK.dims(),
                       raggedOffsetV.dims(),
                       raggedOffsetO.dims()});

        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[2];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[2];
        const auto numHeadsV = v.dims()[2];
        const auto headDimV = v.dims()[3];

        // Elements per token: H*D for Q, Hk*D for K.
        const auto seqStrideQ = q.strides()[1];
        const auto seqStrideKv = k.strides()[1];

        // Offsets may live only on the device (plan path), so read them back and check them here.
        // Every ragged tensor must use the RFC-0014 layout, and tensors that share a packing must
        // agree on every batch length.
        const std::string who = "GpuFpReferenceSdpaRagged";
        const auto tokenBoundaries = [&](hipdnn_data_sdk::utilities::TensorBase<int32_t>& offsets,
                                         int64_t multiplier,
                                         const std::vector<int64_t>& dims,
                                         const std::vector<int64_t>& strides,
                                         const char* name) {
            hipdnn_test_sdk::detail::requireTokenMajorRaggedLayout(dims, strides, who, name);
            if(offsets.strides()[0] != 1)
            {
                throw std::invalid_argument(who + ": " + name
                                            + " ragged_offset must be contiguous");
            }
            if(multiplier < 1)
            {
                throw std::invalid_argument(who + ": " + name
                                            + " ragged_offset_multiplier must be >= 1 (got "
                                            + std::to_string(multiplier) + ")");
            }
            auto elementOffsets = readRaggedOffsets(offsets.memory().deviceData(), batch + 1);
            for(auto& offset : elementOffsets)
            {
                offset *= multiplier;
            }
            return hipdnn_test_sdk::detail::raggedTokenBoundaries(
                elementOffsets, strides[1], dims[1], who, name);
        };
        const auto& mult = offsetMultipliers;
        const auto qTokens = tokenBoundaries(raggedOffsetQ, mult.q, q.dims(), q.strides(), "Q");
        const auto kTokens = tokenBoundaries(raggedOffsetK, mult.k, k.dims(), k.strides(), "K");
        hipdnn_test_sdk::detail::requireMatchingTokenBoundaries(
            qTokens,
            "Q",
            tokenBoundaries(raggedOffsetO, mult.o, o.dims(), o.strides(), "O"),
            "O",
            who);
        hipdnn_test_sdk::detail::requireMatchingTokenBoundaries(
            kTokens,
            "K",
            tokenBoundaries(raggedOffsetV, mult.v, v.dims(), v.strides(), "V"),
            "V",
            who);
        const int64_t totalQ = qTokens.back();

        // An absent scale is 1.0 (no scaling), as in cuDNN and the dense reference.
        const float scale = attnScaleValue.value_or(1.0F);

        auto defines
            = detail::buildSdpaDefines<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                probabilityMode);

        void* lsePtr = nullptr;
        void* raggedOffsetLsePtr = nullptr;
        std::vector<int64_t> lseStrides;
        if(lse != nullptr)
        {
            // One value per query token. Sq must match Q's, or a dense LSE's rows would spill
            // into the next batch.
            const auto& lseDims = lse->dims();
            if(lseDims.size() != 4 || lseDims[0] != batch || lseDims[1] != q.dims()[1]
               || lseDims[2] != numHeads || lseDims[3] != 1)
            {
                throw std::invalid_argument("GpuFpReferenceSdpaRagged: lse must be rank-4 [B, Sq, "
                                            "H, 1] with Q's B, Sq, H");
            }
            lsePtr = lse->memory().deviceData();
            lseStrides = lse->strides();
            if(raggedOffsetLse != nullptr)
            {
                if(raggedOffsetLse->dims() != raggedOffsetQ.dims())
                {
                    throw std::invalid_argument(
                        "GpuFpReferenceSdpaRagged: raggedOffsetLse must be rank-4 [B+1, 1, 1, 1]");
                }
                raggedOffsetLsePtr = raggedOffsetLse->memory().deviceData();
                hipdnn_test_sdk::detail::requireMatchingTokenBoundaries(
                    qTokens,
                    "Q",
                    tokenBoundaries(*raggedOffsetLse, mult.lse, lseDims, lseStrides, "LSE"),
                    "LSE",
                    who);
            }
        }
        else if(raggedOffsetLse != nullptr)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: raggedOffsetLse given without an lse tensor");
        }

        // Optional fp8 descale: scalar or per KV head [B, H_kv, 1, 1]. As in AITER, the Q
        // descale is indexed by K head.
        const DescaleBinding dq = bindDescale(descaleQ, batch, numHeadsK, "Q");
        const DescaleBinding dk = bindDescale(descaleK, batch, numHeadsK, "K");
        const DescaleBinding dv = bindDescale(descaleV, batch, numHeadsV, "V");

        // Empty Q: skip the launch, since a zero-size grid is invalid.
        if(totalQ == 0)
        {
            return;
        }

        launchSdpaRaggedFwd(q.memory().deviceData(),
                            k.memory().deviceData(),
                            v.memory().deviceData(),
                            o.memory().deviceData(),
                            lsePtr,
                            raggedOffsetLsePtr,
                            raggedOffsetQ.memory().deviceData(),
                            raggedOffsetK.memory().deviceData(),
                            mult.q,
                            mult.k,
                            mult.lse,
                            seqStrideQ,
                            seqStrideKv,
                            dq.ptr,
                            dq.batchStride,
                            dq.headStride,
                            dk.ptr,
                            dk.batchStride,
                            dk.headStride,
                            dv.ptr,
                            dv.batchStride,
                            dv.headStride,
                            q.strides(),
                            k.strides(),
                            v.strides(),
                            o.strides(),
                            lseStrides,
                            batch,
                            totalQ,
                            numHeads,
                            numHeadsK,
                            numHeadsV,
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
    // fp8 descale device pointer and its (batch, head) strides. Strides stay zero for a scalar
    // or absent descale.
    struct DescaleBinding
    {
        const void* ptr = nullptr;
        long long batchStride = 0;
        long long headStride = 0;
    };

    // Accepts a scalar (one element) or [B, heads, 1, 1]. `heads` is H_k for Q and K, H_v for V.
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
            binding.ptr = descale->memory().deviceData(); // scalar, zero strides
            return binding;
        }
        const auto& dims = descale->dims();
        if(dims.size() == 4 && dims[0] == batch && dims[1] == heads && dims[2] == 1 && dims[3] == 1)
        {
            binding.ptr = descale->memory().deviceData();
            binding.batchStride = static_cast<long long>(descale->strides()[0]);
            binding.headStride = static_cast<long long>(descale->strides()[1]);
            return binding;
        }
        throw std::invalid_argument(std::string("GpuFpReferenceSdpaRagged: ") + name
                                    + " descale must be scalar [1] or per-KV-head [B, H_kv, 1, 1]");
    }

    static void validateInput(const std::vector<int64_t>& qDims,
                              const std::vector<int64_t>& kDims,
                              const std::vector<int64_t>& vDims,
                              const std::vector<int64_t>& oDims,
                              const std::vector<std::vector<int64_t>>& raggedOffsetDims)
    {
        if(qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: q/k/v/o must all be rank-4 [B, S, H, D] tensors");
        }
        // RFC-0014: ragged_offset is INT32 [B+1, 1, 1, 1].
        for(const auto& d : raggedOffsetDims)
        {
            if(d.size() != 4 || d[0] != qDims[0] + 1 || d[1] != 1 || d[2] != 1 || d[3] != 1)
            {
                throw std::invalid_argument(
                    "GpuFpReferenceSdpaRagged: raggedOffsetQ/K/V/O must be rank-4 [B+1, 1, 1, 1]");
            }
        }

        const auto batch = qDims[0];
        const auto numHeads = qDims[2];
        const auto headDim = qDims[3];
        const auto numHeadsK = kDims[2];
        const auto numHeadsV = vDims[2];
        const auto headDimV = vDims[3];

        if(batch <= 0 || numHeads <= 0 || headDim <= 0 || numHeadsK <= 0 || numHeadsV <= 0
           || headDimV <= 0)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: all dimensions must be positive");
        }
        if(kDims[0] != batch || vDims[0] != batch || oDims[0] != batch)
        {
            throw std::invalid_argument("GpuFpReferenceSdpaRagged: batch dimension mismatch");
        }
        if(vDims[1] != kDims[1])
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: K and V sequence extents (S_max) must match");
        }
        if(kDims[3] != headDim)
        {
            throw std::invalid_argument("GpuFpReferenceSdpaRagged: Q head_dim != K head_dim");
        }
        if(numHeads % numHeadsK != 0 || numHeads % numHeadsV != 0)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: numHeads must be divisible by numHeadsK and numHeadsV");
        }
        if(oDims[1] != qDims[1] || oDims[2] != numHeads || oDims[3] != headDimV)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: output shape must be [B, Sq, H, Dv]");
        }
    }

    // Copies `count` INT32 offsets from a contiguous ragged_offset table on the device.
    static std::vector<int64_t> readRaggedOffsets(const void* raggedOffsetPtr, int64_t count);

    // --- Kernel launcher, defined in GpuFpReferenceSdpaRagged.cpp ---

    static void launchSdpaRaggedFwd(const void* qPtr,
                                    const void* kPtr,
                                    const void* vPtr,
                                    void* oPtr,
                                    void* lsePtr,
                                    const void* raggedOffsetLsePtr,
                                    const void* raggedOffsetQPtr,
                                    const void* raggedOffsetKvPtr,
                                    int64_t offsetMultiplierQ,
                                    int64_t offsetMultiplierKv,
                                    int64_t offsetMultiplierLse,
                                    int64_t seqStrideQ,
                                    int64_t seqStrideKv,
                                    const void* descaleQPtr,
                                    int64_t descaleQBatchStride,
                                    int64_t descaleQHeadStride,
                                    const void* descaleKPtr,
                                    int64_t descaleKBatchStride,
                                    int64_t descaleKHeadStride,
                                    const void* descaleVPtr,
                                    int64_t descaleVBatchStride,
                                    int64_t descaleVHeadStride,
                                    const std::vector<int64_t>& qTensorStrides,
                                    const std::vector<int64_t>& kTensorStrides,
                                    const std::vector<int64_t>& vTensorStrides,
                                    const std::vector<int64_t>& oTensorStrides,
                                    const std::vector<int64_t>& lseTensorStrides,
                                    int64_t batch,
                                    int64_t totalQ,
                                    int64_t numHeads,
                                    int64_t numHeadsK,
                                    int64_t numHeadsV,
                                    int64_t headDim,
                                    int64_t headDimV,
                                    float scale,
                                    int64_t leftBound,
                                    int64_t rightBound,
                                    bool topLeftAlignment,
                                    const std::vector<std::string>& defines);
};

} // namespace hipdnn_gpu_ref
