// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <array>
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

// Puts a type in a non-deduced context so a template parameter is never deduced from
// the argument (a literal nullptr mask still compiles, a wrong-dtype mask does not).
template <typename T>
struct NonDeduced
{
    using type = T;
};

} // namespace detail

// One paged K or V side for the SDPA forward kernel: page-table pointer, its batch and
// table-index strides, and the container's block size. table == nullptr means the side is dense.
struct SdpaPagedSide
{
    const int32_t* table = nullptr;
    int64_t strideBatch = 0;
    int64_t strideTable = 0;
    int64_t blockSize = 0;
};

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
    //
    // The mask dtype is MaskDataType, independent of ComputeDataType (float, half or
    // bfloat16; the kernel converts to float on read). It is passed to the kernel as
    // -DMASK_TYPE after the buildSdpaDefines set (Q/K/V/O/COMPUTE_TYPE/probability mode).
    // The mask parameter is non-deduced: name MaskDataType when passing a non-float mask.
    template <class QDataType,
              class KDataType = QDataType,
              class VDataType = QDataType,
              class ODataType = QDataType,
              class ComputeDataType = float,
              class MaskDataType = ComputeDataType>
    static void fprop(
        hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
        hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
        hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
        hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
        std::optional<float> attnScaleValue = std::nullopt,
        hipdnn_data_sdk::utilities::TensorBase<typename detail::NonDeduced<MaskDataType>::type>*
            attnMask
        = nullptr,
        int64_t leftBound = -1,
        int64_t rightBound = -1,
        bool topLeftAlignment = true,
        hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr,
        SdpaSoftmaxProbabilityMode probabilityMode = SdpaSoftmaxProbabilityMode::FLOAT,
        hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenQ = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenKv = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<int32_t>* pageTableK = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<int32_t>* pageTableV = nullptr,
        int64_t pagedMaxSeqKv = 0,
        hipdnn_data_sdk::utilities::TensorBase<float>* sinkToken = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<float>* dropoutMask = nullptr,
        std::optional<float> dropoutScale = std::nullopt,
        std::optional<float> softcap = std::nullopt,
        hipdnn_data_sdk::utilities::TensorBase<float>* rowMax = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<float>* rowSumExp = nullptr)
    {
        validateSoftcap(softcap);
        // Paged K / V (cuDNN): k / v are containers [blocks, H, blockSize, D] addressed through an
        // INT32 [B, 1, T, 1] page table; pagedMaxSeqKv is the logical key extent. Paging needs
        // padding. Table entries must name existing blocks; the kernel does not check them.
        const bool pagedK = pageTableK != nullptr;
        const bool pagedV = pageTableV != nullptr;
        if((pagedK || pagedV) && (seqLenKv == nullptr || pagedMaxSeqKv <= 0))
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: paged K/V needs seq_len tensors and a "
                                        "positive max key length");
        }
        validateInput(q.dims(), k.dims(), v.dims(), o.dims(), pagedK, pagedV, pagedMaxSeqKv);

        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[1];
        const auto seqQ = q.dims()[2];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto seqKv = (pagedK || pagedV) ? pagedMaxSeqKv : k.dims()[2];

        SdpaPagedSide pagedKSide;
        SdpaPagedSide pagedVSide;
        if(pagedK)
        {
            pagedKSide = pagedSide(*pageTableK, k.dims(), batch, seqKv);
        }
        if(pagedV)
        {
            pagedVSide = pagedSide(*pageTableV, v.dims(), batch, seqKv);
        }
        const auto headDimV = v.dims()[3];

        // An absent scale is 1.0 (no scaling), as in cuDNN.
        const float scale = attnScaleValue.value_or(1.0F);

        auto defines
            = detail::buildSdpaDefines<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                probabilityMode);
        defines.emplace_back(std::string("-DMASK_TYPE=")
                             + detail::HipRtcTypeName<MaskDataType>::VALUE);

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
        // cuDNN Max / Sum_exp row statistics: rank-3 [B, H, Sq], as the LSE.
        for(auto* stats : {rowMax, rowSumExp})
        {
            if(stats != nullptr
               && (stats->dims().size() != 3 || stats->dims()[0] != batch
                   || stats->dims()[1] != numHeads || stats->dims()[2] != seqQ))
            {
                throw std::invalid_argument(
                    "GpuFpReferenceSdpa: rowMax / rowSumExp must be rank-3 [B, H, Sq]");
            }
        }

        // Padding (cuDNN seq_len_q / seq_len_kv): one INT32 per batch, both or neither.
        if((seqLenQ == nullptr) != (seqLenKv == nullptr))
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpa: seqLenQ and seqLenKv must be given together");
        }
        const int32_t* seqLenQPtr = nullptr;
        const int32_t* seqLenKvPtr = nullptr;
        int64_t seqLenQStride = 0;
        int64_t seqLenKvStride = 0;
        if(seqLenQ != nullptr)
        {
            for(auto* len : {seqLenQ, seqLenKv})
            {
                if(static_cast<int64_t>(len->elementCount()) != batch || len->dims().empty()
                   || len->dims()[0] != batch)
                {
                    throw std::invalid_argument(
                        "GpuFpReferenceSdpa: seq_len tensors must hold one value per batch");
                }
            }
            seqLenQPtr = static_cast<const int32_t*>(seqLenQ->memory().deviceData());
            seqLenKvPtr = static_cast<const int32_t*>(seqLenKv->memory().deviceData());
            seqLenQStride = seqLenQ->strides()[0];
            seqLenKvStride = seqLenKv->strides()[0];
        }

        // Attention sink (cuDNN): FLOAT [1, H, 1, 1], one logit per query head.
        const float* sinkPtr = nullptr;
        int64_t sinkStride = 0;
        if(sinkToken != nullptr)
        {
            if(sinkToken->dims().size() != 4 || sinkToken->dims()[1] != numHeads
               || static_cast<int64_t>(sinkToken->elementCount()) != numHeads)
            {
                throw std::invalid_argument("GpuFpReferenceSdpa: sinkToken must be [1, H, 1, 1]");
            }
            sinkPtr = static_cast<const float*>(sinkToken->memory().deviceData());
            sinkStride = sinkToken->strides()[1];
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
                      defines,
                      seqLenQPtr,
                      seqLenKvPtr,
                      seqLenQStride,
                      seqLenKvStride,
                      pagedKSide,
                      pagedVSide,
                      sinkPtr,
                      sinkStride,
                      dropoutMaskPointer(dropoutMask, batch, numHeads, seqQ, seqKv),
                      dropoutMaskStrides(dropoutMask),
                      dropoutScale.value_or(1.0F),
                      softcap.value_or(0.0F),
                      rowMax != nullptr ? rowMax->memory().deviceData() : nullptr,
                      rowMax != nullptr ? rowMax->strides() : std::vector<int64_t>{},
                      rowSumExp != nullptr ? rowSumExp->memory().deviceData() : nullptr,
                      rowSumExp != nullptr ? rowSumExp->strides() : std::vector<int64_t>{});

        o.memory().markDeviceModified();
        for(auto* stats : {lse, rowMax, rowSumExp})
        {
            if(stats != nullptr)
            {
                stats->memory().markDeviceModified();
            }
        }
    }

    // --- Backward SDPA (bprop): dQ, dK, dV and optional dBias ---
    //
    // Same parameters, order and semantics as CpuFpReferenceSdpa::backward: fp32 accumulation,
    // D = sum(dO * O) from the given O, LSE (rank 3 [B, H, Sq] or rank 4 [B, H, Sq, 1]) or
    // softmax recompute when lse is null, additive bias with broadcast and dBias (unscaled dS
    // summed into the bias shape), diagonal band, GQA/MQA with independent K and V heads,
    // Dqk != Dv, and seq_len padding (padded rows and keys get exactly 0 and are never read).
    template <class QDataType,
              class KDataType = QDataType,
              class VDataType = QDataType,
              class ODataType = QDataType,
              class DODataType = ODataType,
              class DQDataType = QDataType,
              class DKDataType = KDataType,
              class DVDataType = VDataType,
              class ComputeDataType = float,
              class MaskDataType = ComputeDataType>
    static void bprop(
        hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
        hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
        hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
        hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
        hipdnn_data_sdk::utilities::TensorBase<DODataType>& dO,
        hipdnn_data_sdk::utilities::TensorBase<DQDataType>& dQ,
        hipdnn_data_sdk::utilities::TensorBase<DKDataType>& dK,
        hipdnn_data_sdk::utilities::TensorBase<DVDataType>& dV,
        std::optional<float> attnScaleValue = std::nullopt,
        hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<typename detail::NonDeduced<MaskDataType>::type>*
            attnMask
        = nullptr,
        int64_t leftBound = -1,
        int64_t rightBound = -1,
        bool topLeftAlignment = true,
        hipdnn_data_sdk::utilities::TensorBase<typename detail::NonDeduced<MaskDataType>::type>*
            dBias
        = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenQ = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenKv = nullptr,
        hipdnn_data_sdk::utilities::TensorBase<float>* dropoutMask = nullptr,
        std::optional<float> dropoutScale = std::nullopt,
        std::optional<float> dropoutScaleInv = std::nullopt,
        std::optional<float> softcap = std::nullopt)
    {
        validateSoftcap(softcap);
        validateInput(q.dims(), k.dims(), v.dims(), o.dims());
        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[1];
        const auto seqQ = q.dims()[2];
        const auto seqKv = k.dims()[2];
        const auto headDim = q.dims()[3];
        const auto headDimV = v.dims()[3];
        const auto expect = [](bool ok, const char* what) {
            if(!ok)
            {
                throw std::invalid_argument(std::string("GpuFpReferenceSdpa::bprop: ") + what);
            }
        };
        expect(dO.dims() == o.dims(), "dO must have O's shape");
        expect(dQ.dims() == q.dims(), "dQ must have Q's shape");
        expect(dK.dims() == k.dims(), "dK must have K's shape");
        expect(dV.dims() == v.dims(), "dV must have V's shape");
        if(lse != nullptr)
        {
            const auto& ld = lse->dims();
            expect((ld.size() == 3 || (ld.size() == 4 && ld[3] == 1)) && ld[0] == batch
                       && ld[1] == numHeads && ld[2] == seqQ,
                   "lse must be [B, H, Sq] or [B, H, Sq, 1]");
        }
        std::vector<int64_t> maskDims;
        if(attnMask != nullptr)
        {
            maskDims = attnMask->dims();
            expect(!maskDims.empty() && maskDims.size() <= 4, "attention mask must be rank 1..4");
            const std::array<int64_t, 4> full = {batch, numHeads, seqQ, seqKv};
            for(size_t i = 0; i < maskDims.size(); ++i)
            {
                const auto ctx = full[4 - maskDims.size() + i];
                expect(maskDims[i] == 1 || maskDims[i] == ctx,
                       "each attention mask dim must be 1 or the full size");
            }
        }
        expect(dBias == nullptr || (attnMask != nullptr && dBias->dims() == maskDims),
               "dBias requires an attention mask of the same shape");
        expect((seqLenQ == nullptr) == (seqLenKv == nullptr),
               "seqLenQ and seqLenKv must be given together");
        for(auto* len : {seqLenQ, seqLenKv})
        {
            expect(len == nullptr
                       || (static_cast<int64_t>(len->elementCount()) == batch
                           && !len->dims().empty() && len->dims()[0] == batch),
                   "seq_len tensors must hold one value per batch");
        }

        std::vector<std::string> defines
            = {std::string("-DQ_TYPE=") + detail::HipRtcTypeName<QDataType>::VALUE,
               std::string("-DK_TYPE=") + detail::HipRtcTypeName<KDataType>::VALUE,
               std::string("-DV_TYPE=") + detail::HipRtcTypeName<VDataType>::VALUE,
               std::string("-DO_TYPE=") + detail::HipRtcTypeName<ODataType>::VALUE,
               std::string("-DDO_TYPE=") + detail::HipRtcTypeName<DODataType>::VALUE,
               std::string("-DDQ_TYPE=") + detail::HipRtcTypeName<DQDataType>::VALUE,
               std::string("-DDK_TYPE=") + detail::HipRtcTypeName<DKDataType>::VALUE,
               std::string("-DDV_TYPE=") + detail::HipRtcTypeName<DVDataType>::VALUE,
               std::string("-DMASK_TYPE=") + detail::HipRtcTypeName<MaskDataType>::VALUE,
               std::string("-DCOMPUTE_TYPE=") + detail::HipRtcTypeName<ComputeDataType>::VALUE};

        SdpaBwdLaunch launch;
        launch.q = q.memory().deviceData();
        launch.k = k.memory().deviceData();
        launch.v = v.memory().deviceData();
        launch.o = o.memory().deviceData();
        launch.dO = dO.memory().deviceData();
        launch.dQ = dQ.memory().deviceData();
        launch.dK = dK.memory().deviceData();
        launch.dV = dV.memory().deviceData();
        launch.mask = attnMask != nullptr ? attnMask->memory().deviceData() : nullptr;
        launch.dBias = dBias != nullptr ? dBias->memory().deviceData() : nullptr;
        launch.lse
            = lse != nullptr ? static_cast<const float*>(lse->memory().deviceData()) : nullptr;
        launch.seqLenQ = seqLenQ != nullptr
                             ? static_cast<const int32_t*>(seqLenQ->memory().deviceData())
                             : nullptr;
        launch.seqLenKv = seqLenKv != nullptr
                              ? static_cast<const int32_t*>(seqLenKv->memory().deviceData())
                              : nullptr;
        launch.strides = {q.strides(),
                          k.strides(),
                          v.strides(),
                          o.strides(),
                          dO.strides(),
                          dQ.strides(),
                          dK.strides(),
                          dV.strides(),
                          attnMask != nullptr ? attnMask->strides() : std::vector<int64_t>{},
                          dBias != nullptr ? dBias->strides() : std::vector<int64_t>{},
                          lse != nullptr ? lse->strides() : std::vector<int64_t>{}};
        launch.seqLenQStride = seqLenQ != nullptr ? seqLenQ->strides()[0] : 0;
        launch.seqLenKvStride = seqLenKv != nullptr ? seqLenKv->strides()[0] : 0;
        launch.maskDims = maskDims;
        launch.batch = batch;
        launch.numHeads = numHeads;
        launch.numHeadsK = k.dims()[1];
        launch.numHeadsV = v.dims()[1];
        launch.seqQ = seqQ;
        launch.seqKv = seqKv;
        launch.headDim = headDim;
        launch.headDimV = headDimV;
        launch.scale = attnScaleValue.value_or(1.0F);
        launch.leftBound = leftBound;
        launch.rightBound = rightBound;
        launch.topLeftAlignment = topLeftAlignment;
        launch.dropoutMask = dropoutMaskPointer(dropoutMask, batch, numHeads, seqQ, seqKv);
        launch.dropoutMaskStrides = dropoutMaskStrides(dropoutMask);
        launch.dropoutScale = dropoutScale.value_or(1.0F);
        launch.dropoutScaleInv = dropoutScaleInv.value_or(1.0F);
        launch.softcap = softcap.value_or(0.0F);
        launchSdpaBwd(launch, defines);

        dQ.memory().markDeviceModified();
        dK.memory().markDeviceModified();
        dV.memory().markDeviceModified();
        if(dBias != nullptr)
        {
            dBias->memory().markDeviceModified();
        }
    }

private:
    // Host-side description of one backward launch (device pointers, strides and sizes).
    struct SdpaBwdLaunch
    {
        const void* q = nullptr;
        const void* k = nullptr;
        const void* v = nullptr;
        const void* o = nullptr;
        const void* dO = nullptr;
        void* dQ = nullptr;
        void* dK = nullptr;
        void* dV = nullptr;
        const void* mask = nullptr;
        void* dBias = nullptr;
        const float* lse = nullptr;
        const int32_t* seqLenQ = nullptr;
        const int32_t* seqLenKv = nullptr;
        // q, k, v, o, dO, dQ, dK, dV, mask, dBias, lse
        std::array<std::vector<int64_t>, 11> strides;
        int64_t seqLenQStride = 0;
        int64_t seqLenKvStride = 0;
        std::vector<int64_t> maskDims;
        int64_t batch = 0;
        int64_t numHeads = 0;
        int64_t numHeadsK = 0;
        int64_t numHeadsV = 0;
        int64_t seqQ = 0;
        int64_t seqKv = 0;
        int64_t headDim = 0;
        int64_t headDimV = 0;
        float scale = 1.0F;
        // Explicit-mask dropout; scales stay 1 without dropout (cuDNN's defaults).
        const float* dropoutMask = nullptr;
        std::vector<int64_t> dropoutMaskStrides;
        float dropoutScale = 1.0F;
        float dropoutScaleInv = 1.0F;
        float softcap = 0.0F; // 0 = no softcap
        int64_t leftBound = -1;
        int64_t rightBound = -1;
        bool topLeftAlignment = true;
    };

    static void launchSdpaBwd(const SdpaBwdLaunch& launch, const std::vector<std::string>& defines);

    // --- Validation ---
    // No SDPA param validator exists; perform minimal shape checks only.

    static SdpaPagedSide pagedSide(hipdnn_data_sdk::utilities::TensorBase<int32_t>& table,
                                   const std::vector<int64_t>& containerDims,
                                   int64_t batch,
                                   int64_t maxSeqKv)
    {
        const auto& tableDims = table.dims();
        if(tableDims.size() != 4 || tableDims[0] != batch || tableDims[1] != 1 || tableDims[3] != 1)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: page table must be [B, 1, T, 1]");
        }
        if(tableDims[2] * containerDims[2] < maxSeqKv)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpa: page table capacity is below the max key length");
        }
        return {static_cast<const int32_t*>(table.memory().deviceData()),
                table.strides()[0],
                table.strides()[2],
                containerDims[2]};
    }

    static void validateInput(const std::vector<int64_t>& qDims,
                              const std::vector<int64_t>& kDims,
                              const std::vector<int64_t>& vDims,
                              const std::vector<int64_t>& oDims,
                              bool pagedK = false,
                              bool pagedV = false,
                              int64_t pagedMaxSeqKv = 0)
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
        const auto seqKv = (pagedK || pagedV) ? pagedMaxSeqKv : kDims[2];
        const auto headDimV = vDims[3];

        if(batch <= 0 || numHeads <= 0 || seqQ <= 0 || headDim <= 0 || numHeadsK <= 0
           || numHeadsV <= 0 || seqKv <= 0 || headDimV <= 0)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: all dimensions must be positive");
        }

        // A paged side is a block container, so only dense sides carry the batch and key length.
        if((!pagedK && (kDims[0] != batch || kDims[2] != seqKv))
           || (!pagedV && (vDims[0] != batch || vDims[2] != seqKv)) || oDims[0] != batch)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: batch or key length mismatch");
        }
        if(kDims[3] != headDim)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: Q head_dim != K head_dim");
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
                              const std::vector<std::string>& defines,
                              const int32_t* seqLenQPtr = nullptr,
                              const int32_t* seqLenKvPtr = nullptr,
                              int64_t seqLenQStride = 0,
                              int64_t seqLenKvStride = 0,
                              const SdpaPagedSide& pagedK = {},
                              const SdpaPagedSide& pagedV = {},
                              const float* sinkPtr = nullptr,
                              int64_t sinkStride = 0,
                              const float* dropoutMaskPtr = nullptr,
                              const std::vector<int64_t>& dropoutMaskStrides = {},
                              float dropoutScale = 1.0F,
                              float softcap = 0.0F,
                              void* rowMaxPtr = nullptr,
                              const std::vector<int64_t>& rowMaxStrides = {},
                              void* rowSumExpPtr = nullptr,
                              const std::vector<int64_t>& rowSumExpStrides = {});

    // Logit softcap (reference-only extension, SEM-020): finite and > 0 when given.
    static void validateSoftcap(std::optional<float> softcap)
    {
        if(softcap.has_value() && !(softcap.value() > 0.0F && std::isfinite(softcap.value())))
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: softcap must be finite and > 0");
        }
    }

    // Explicit dropout mask (cuDNN custom dropout): FLOAT [B|1, H|1, Sq, Skv] on the device.
    static const float* dropoutMaskPointer(hipdnn_data_sdk::utilities::TensorBase<float>* mask,
                                           int64_t batch,
                                           int64_t heads,
                                           int64_t seqQ,
                                           int64_t seqKv)
    {
        if(mask == nullptr)
        {
            return nullptr;
        }
        const auto& d = mask->dims();
        if(d.size() != 4 || (d[0] != batch && d[0] != 1) || (d[1] != heads && d[1] != 1)
           || d[2] != seqQ || d[3] != seqKv)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpa: dropoutMask must be [B or 1, H or 1, Sq, Skv]");
        }
        return static_cast<const float*>(mask->memory().deviceData());
    }

    // Mask strides with broadcast (size-1) dims set to 0, so the kernels index it directly.
    static std::vector<int64_t>
        dropoutMaskStrides(const hipdnn_data_sdk::utilities::TensorBase<float>* mask)
    {
        if(mask == nullptr)
        {
            return {};
        }
        auto strides = mask->strides();
        for(size_t i = 0; i < 2; ++i)
        {
            if(mask->dims()[i] == 1)
            {
                strides[i] = 0;
            }
        }
        return strides;
    }
};

} // namespace hipdnn_gpu_ref
