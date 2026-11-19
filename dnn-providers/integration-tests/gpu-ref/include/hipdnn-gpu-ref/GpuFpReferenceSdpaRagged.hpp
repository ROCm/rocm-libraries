// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
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
// Tensors are logical rank-4 [B, H, S, D] with the sequence at SDPA_SEQ_AXIS (2) and any
// non-aliasing strides (token-major, head-major, padded token stride, shifted or non-monotonic
// origins):
//   q = [B, H,  Sq,  D ]   k = [B, Hk, Skv, D ]
//   v = [B, Hv, Skv, Dv]   o = [B, H,  Sq,  Dv]
// raggedOffsetQ/K/V/O are INT32 [B+1, 1, 1, 1] offset tables (entry b at b * strides()[0]), scaled to elements by
// `offsetMultipliers`: batch b starts at element offset[b]. Its length is seqLenQ[b] (Q, O, LSE)
// or seqLenKv[b] (K, V) when those INT32 [B, 1, 1, 1] tensors are given (element b at
// b * strides()[0]), else
// min((offset[b+1] - offset[b]) / strides[2], S_max). Each layout must satisfy the Data SDK
// ragged container rules (raggedFootprint). O and the LSE must have Q's per-batch lengths and V
// K's.
// The optional LSE is [B, H, Sq, 1], ragged if raggedOffsetLse is given, else dense.
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
                    const RaggedOffsetMultipliers& offsetMultipliers = {},
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenQ = nullptr,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenKv = nullptr)
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
        const auto numHeads = q.dims()[1];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto headDimV = v.dims()[3];

        // Offsets and lengths may live only on the device (plan path), so read them back and
        // check them here. Every ragged tensor must be a valid ragged container layout, and
        // tensors that share a length must agree on every batch length.
        constexpr auto SEQ_AXIS = hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS;
        const std::string who = "GpuFpReferenceSdpaRagged";
        const auto readSeqLens = [&](hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLen,
                                     const char* name) -> std::optional<std::vector<int64_t>> {
            if(seqLen == nullptr)
            {
                return std::nullopt;
            }
            if(!hipdnn_data_sdk::utilities::detail::isBatchAxisVector(seqLen->dims(), batch))
            {
                throw std::invalid_argument(who + ": " + name + " must have shape [B, 1, 1, 1]");
            }
            return readStridedValues(
                seqLen->memory().deviceData(), batch, seqLen->strides()[0], who + ": " + name);
        };
        const auto seqLensQ = readSeqLens(seqLenQ, "seq_len_q");
        const auto seqLensKv = readSeqLens(seqLenKv, "seq_len_kv");
        const auto batchesOf = [&](hipdnn_data_sdk::utilities::TensorBase<int32_t>& offsets,
                                   int64_t multiplier,
                                   const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides,
                                   const std::optional<std::vector<int64_t>>& seqLens,
                                   const char* name) {
            if(multiplier < 1)
            {
                throw std::invalid_argument(who + ": " + name
                                            + " ragged_offset_multiplier must be >= 1 (got "
                                            + std::to_string(multiplier) + ")");
            }
            if(strides[SEQ_AXIS] <= 0)
            {
                throw std::invalid_argument(who + ": " + name
                                            + " sequence stride must be positive (got "
                                            + std::to_string(strides[SEQ_AXIS]) + ")");
            }
            auto elementOffsets = readStridedValues(offsets.memory().deviceData(),
                                                    batch + 1,
                                                    offsets.strides()[0],
                                                    who + ": " + name + " ragged_offset");
            for(auto& offset : elementOffsets)
            {
                offset *= multiplier;
            }
            try
            {
                auto batches = hipdnn_data_sdk::utilities::detail::raggedBatches(
                    elementOffsets,
                    seqLens.has_value() ? &*seqLens : nullptr,
                    strides[SEQ_AXIS],
                    dims[SEQ_AXIS]);
                hipdnn_data_sdk::utilities::detail::raggedFootprint(
                    dims, strides, SEQ_AXIS, batches);
                return batches;
            }
            catch(const std::invalid_argument& e)
            {
                throw std::invalid_argument(who + ": " + name + ": " + e.what());
            }
        };
        const auto& mult = offsetMultipliers;
        const auto qBatches
            = batchesOf(raggedOffsetQ, mult.q, q.dims(), q.strides(), seqLensQ, "Q");
        const auto kBatches
            = batchesOf(raggedOffsetK, mult.k, k.dims(), k.strides(), seqLensKv, "K");
        hipdnn_test_sdk::detail::requireMatchingRaggedLengths(
            qBatches.lengths,
            "Q",
            batchesOf(raggedOffsetO, mult.o, o.dims(), o.strides(), seqLensQ, "O").lengths,
            "O",
            who);
        hipdnn_test_sdk::detail::requireMatchingRaggedLengths(
            kBatches.lengths,
            "K",
            batchesOf(raggedOffsetV, mult.v, v.dims(), v.strides(), seqLensKv, "V").lengths,
            "V",
            who);
        int64_t totalQ = 0;
        for(const auto length : qBatches.lengths)
        {
            totalQ += length;
        }

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
            if(lseDims.size() != 4 || lseDims[0] != batch || lseDims[1] != numHeads
               || lseDims[2] != q.dims()[SEQ_AXIS] || lseDims[3] != 1)
            {
                throw std::invalid_argument("GpuFpReferenceSdpaRagged: lse must be rank-4 [B, H, "
                                            "Sq, 1] with Q's B, H, Sq");
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
                hipdnn_test_sdk::detail::requireMatchingRaggedLengths(
                    qBatches.lengths,
                    "Q",
                    batchesOf(*raggedOffsetLse, mult.lse, lseDims, lseStrides, seqLensQ, "LSE")
                        .lengths,
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

        RaggedLaunchTensors tensors;
        tensors.q = q.memory().deviceData();
        tensors.k = k.memory().deviceData();
        tensors.v = v.memory().deviceData();
        tensors.o = o.memory().deviceData();
        tensors.lse = lsePtr;
        tensors.offsetQ = raggedOffsetQ.memory().deviceData();
        tensors.offsetK = raggedOffsetK.memory().deviceData();
        tensors.offsetV = raggedOffsetV.memory().deviceData();
        tensors.offsetO = raggedOffsetO.memory().deviceData();
        tensors.offsetLse = raggedOffsetLsePtr;
        tensors.multipliers = mult;
        tensors.offsetStrideQ = raggedOffsetQ.strides()[0];
        tensors.offsetStrideK = raggedOffsetK.strides()[0];
        tensors.offsetStrideV = raggedOffsetV.strides()[0];
        tensors.offsetStrideO = raggedOffsetO.strides()[0];
        tensors.offsetStrideLse = raggedOffsetLse != nullptr ? raggedOffsetLse->strides()[0] : 0;
        tensors.seqLenQ = seqLenQ != nullptr ? seqLenQ->memory().deviceData() : nullptr;
        tensors.seqLenKv = seqLenKv != nullptr ? seqLenKv->memory().deviceData() : nullptr;
        tensors.seqLenQStride = seqLenQ != nullptr ? seqLenQ->strides()[0] : 0;
        tensors.seqLenKvStride = seqLenKv != nullptr ? seqLenKv->strides()[0] : 0;
        launchSdpaRaggedFwd(tensors,
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
                            q.dims()[SEQ_AXIS],
                            k.dims()[SEQ_AXIS],
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
                "GpuFpReferenceSdpaRagged: q/k/v/o must all be rank-4 [B, H, S, D] tensors");
        }
        // RFC-0014: ragged_offset is INT32 [B+1, 1, 1, 1].
        for(const auto& d : raggedOffsetDims)
        {
            if(!hipdnn_data_sdk::utilities::detail::isBatchAxisVector(d, qDims[0] + 1))
            {
                throw std::invalid_argument("GpuFpReferenceSdpaRagged: raggedOffsetQ/K/V/O must "
                                            "have shape [B + 1, 1, 1, 1]");
            }
        }

        const auto batch = qDims[0];
        const auto numHeads = qDims[1];
        const auto headDim = qDims[3];
        const auto numHeadsK = kDims[1];
        const auto numHeadsV = vDims[1];
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
        if(vDims[2] != kDims[2])
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
        if(oDims[1] != numHeads || oDims[2] != qDims[2] || oDims[3] != headDimV)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: output shape must be [B, H, Sq, Dv]");
        }
    }

    // Copies the `count` INT32 values of a ragged_offset table or seq_len tensor, value b at
    // b * stride on the device. A stride is only meaningful, and so only validated, with more than one value.
    static std::vector<int64_t>
        readStridedValues(const void* ptr, int64_t count, int64_t stride, const std::string& name);

    // --- Kernel launcher, defined in GpuFpReferenceSdpaRagged.cpp ---

    // Device pointers of one launch. Optional ones are nullptr.
    struct RaggedLaunchTensors
    {
        const void* q = nullptr;
        const void* k = nullptr;
        const void* v = nullptr;
        void* o = nullptr;
        void* lse = nullptr;
        const void* offsetQ = nullptr;
        const void* offsetK = nullptr;
        const void* offsetV = nullptr;
        const void* offsetO = nullptr;
        const void* offsetLse = nullptr;
        RaggedOffsetMultipliers multipliers;
        // Batch-axis strides (elements) of the offset tables; 0 when not bound.
        int64_t offsetStrideQ = 0;
        int64_t offsetStrideK = 0;
        int64_t offsetStrideV = 0;
        int64_t offsetStrideO = 0;
        int64_t offsetStrideLse = 0;
        const void* seqLenQ = nullptr;
        const void* seqLenKv = nullptr;
        // Batch-axis strides (elements) of the seq_len tensors; 0 when not bound.
        int64_t seqLenQStride = 0;
        int64_t seqLenKvStride = 0;
    };

    static void launchSdpaRaggedFwd(const RaggedLaunchTensors& tensors,
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
                                    int64_t seqMaxQ,
                                    int64_t seqMaxKv,
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
