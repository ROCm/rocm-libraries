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
#include <type_traits>
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

/// The fp8 format softmax probabilities are stored in before P @ V (cuDNN sdpa_fp8 quantizes P
/// to the input fp8 type). The values are the kernel's SdpaFwdArgs::pFp8Format.
enum class SdpaFp8Format : int
{
    NONE = 0,
    E4M3 = 1,
    E4M3_FNUZ = 2,
    E5M2 = 3,
    E5M2_FNUZ = 4,
};

// cuDNN sdpa_fp8 softmax scaling for one forward launch (SdpaFwdArgs::pFp8Format and friends).
struct SdpaFp8Softmax
{
    SdpaFp8Format format = SdpaFp8Format::NONE;
    float scaleS = 1.0F;
    float descaleS = 1.0F;
    float* amaxS = nullptr;
};

// One paged K or V side for the SDPA forward kernel: page-table pointer, its batch and
// table-index strides, and the container's block size. table == nullptr means the side is dense.
struct SdpaPagedSide
{
    const int32_t* table = nullptr;
    int64_t strideBatch = 0;
    int64_t strideTable = 0;
    int64_t blockSize = 0;
};

/// Named options for GpuFpReferenceSdpa::fprop, mirroring the CPU reference's SdpaFwdOptions (same
/// fields and setters) plus the GPU-only probability mode and paged K/V. Every field defaults to
/// "feature off". Pointers are non-const because uploading a tensor needs deviceData().
template <class MaskDataType = float>
struct GpuSdpaFwdOptions
{
    std::optional<float> attnScale;
    hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* attnMask = nullptr;
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeftAlignment = true;
    hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenQ = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenKv = nullptr;
    /// Paged K / V (cuDNN): k / v are containers [blocks, H, blockSize, D] addressed through an
    /// INT32 [B, 1, T, 1] page table; pagedMaxSeqKv is the logical key extent. Needs padding.
    hipdnn_data_sdk::utilities::TensorBase<int32_t>* pageTableK = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<int32_t>* pageTableV = nullptr;
    int64_t pagedMaxSeqKv = 0;
    hipdnn_data_sdk::utilities::TensorBase<float>* sinkToken = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* dropoutMask = nullptr;
    std::optional<float> dropoutScale;
    std::optional<float> softcap;
    bool alibi = false;
    /// cuDNN block-sparse mask, as CpuFpReferenceSdpa's SdpaFwdOptions::blockMask (SEM-025).
    hipdnn_data_sdk::utilities::TensorBase<uint8_t>* blockMask = nullptr;
    /// Outputs, FP32 rank 3 [B, H, Sq].
    hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* rowMax = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* rowSumExp = nullptr;
    /// Provider-attuned emulation of P storage (opt-in; FLOAT is the oracle, SEM-012).
    SdpaSoftmaxProbabilityMode probabilityMode = SdpaSoftmaxProbabilityMode::FLOAT;
    /// fp8 (AITER contract): FLOAT scalar or [B, H_kv, 1, 1] Q/K/V descales. Q/K/V are
    /// dequantized first, then the float pipeline runs.
    hipdnn_data_sdk::utilities::TensorBase<float>* descaleQ = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* descaleK = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* descaleV = nullptr;
    /// fp8 softmax and output scaling and amax outputs, as CpuFpReferenceSdpa's SdpaFwdOptions
    /// (SEM-024). fprop() sets pFormat from Q's fp8 dtype.
    std::optional<float> scaleS;
    std::optional<float> descaleS;
    SdpaFp8Format pFormat = SdpaFp8Format::NONE;
    std::optional<float> scaleO;
    hipdnn_data_sdk::utilities::TensorBase<float>* amaxS = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* amaxO = nullptr;

    GpuSdpaFwdOptions& setScale(std::optional<float> value)
    {
        attnScale = value;
        return *this;
    }
    GpuSdpaFwdOptions& setMask(hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* value)
    {
        attnMask = value;
        return *this;
    }
    GpuSdpaFwdOptions& setBand(int64_t left, int64_t right, bool topLeft)
    {
        leftBound = left;
        rightBound = right;
        topLeftAlignment = topLeft;
        return *this;
    }
    GpuSdpaFwdOptions& setSeqLens(hipdnn_data_sdk::utilities::TensorBase<int32_t>* q,
                                  hipdnn_data_sdk::utilities::TensorBase<int32_t>* kv)
    {
        seqLenQ = q;
        seqLenKv = kv;
        return *this;
    }
    GpuSdpaFwdOptions& setPaged(hipdnn_data_sdk::utilities::TensorBase<int32_t>* tableK,
                                hipdnn_data_sdk::utilities::TensorBase<int32_t>* tableV,
                                int64_t maxSeqKv)
    {
        pageTableK = tableK;
        pageTableV = tableV;
        pagedMaxSeqKv = maxSeqKv;
        return *this;
    }
    GpuSdpaFwdOptions& setSink(hipdnn_data_sdk::utilities::TensorBase<float>* value)
    {
        sinkToken = value;
        return *this;
    }
    GpuSdpaFwdOptions& setDropout(hipdnn_data_sdk::utilities::TensorBase<float>* mask,
                                  std::optional<float> scale)
    {
        dropoutMask = mask;
        dropoutScale = scale;
        return *this;
    }
    GpuSdpaFwdOptions& setSoftcap(std::optional<float> value)
    {
        softcap = value;
        return *this;
    }
    GpuSdpaFwdOptions& setAlibi(bool value)
    {
        alibi = value;
        return *this;
    }
    GpuSdpaFwdOptions& setBlockMask(hipdnn_data_sdk::utilities::TensorBase<uint8_t>* value)
    {
        blockMask = value;
        return *this;
    }
    GpuSdpaFwdOptions& setLse(hipdnn_data_sdk::utilities::TensorBase<float>* value)
    {
        lse = value;
        return *this;
    }
    GpuSdpaFwdOptions& setRowStats(hipdnn_data_sdk::utilities::TensorBase<float>* max,
                                   hipdnn_data_sdk::utilities::TensorBase<float>* sumExp)
    {
        rowMax = max;
        rowSumExp = sumExp;
        return *this;
    }
    GpuSdpaFwdOptions& setProbabilityMode(SdpaSoftmaxProbabilityMode value)
    {
        probabilityMode = value;
        return *this;
    }
    GpuSdpaFwdOptions& setDescales(hipdnn_data_sdk::utilities::TensorBase<float>* q,
                                   hipdnn_data_sdk::utilities::TensorBase<float>* k,
                                   hipdnn_data_sdk::utilities::TensorBase<float>* v)
    {
        descaleQ = q;
        descaleK = k;
        descaleV = v;
        return *this;
    }
    GpuSdpaFwdOptions& setSoftmaxScales(std::optional<float> scale, std::optional<float> descale)
    {
        scaleS = scale;
        descaleS = descale;
        return *this;
    }
    GpuSdpaFwdOptions& setOutputScale(std::optional<float> value)
    {
        scaleO = value;
        return *this;
    }
    GpuSdpaFwdOptions& setAmax(hipdnn_data_sdk::utilities::TensorBase<float>* s,
                               hipdnn_data_sdk::utilities::TensorBase<float>* out)
    {
        amaxS = s;
        amaxO = out;
        return *this;
    }
};

/// Named options for GpuFpReferenceSdpa::bprop, mirroring the CPU reference's SdpaBwdOptions.
template <class MaskDataType = float>
struct GpuSdpaBwdOptions
{
    std::optional<float> attnScale;
    /// Forward LSE, rank 3 [B, H, Sq] or rank 4 [B, H, Sq, 1]; null recomputes the softmax.
    hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* attnMask = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* dBias = nullptr;
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeftAlignment = true;
    hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenQ = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenKv = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* dropoutMask = nullptr;
    std::optional<float> dropoutScale;
    std::optional<float> dropoutScaleInv;
    std::optional<float> softcap;
    bool alibi = false;

    GpuSdpaBwdOptions& setScale(std::optional<float> value)
    {
        attnScale = value;
        return *this;
    }
    GpuSdpaBwdOptions& setLse(hipdnn_data_sdk::utilities::TensorBase<float>* value)
    {
        lse = value;
        return *this;
    }
    GpuSdpaBwdOptions& setMask(hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* value)
    {
        attnMask = value;
        return *this;
    }
    GpuSdpaBwdOptions& setDBias(hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* value)
    {
        dBias = value;
        return *this;
    }
    GpuSdpaBwdOptions& setBand(int64_t left, int64_t right, bool topLeft)
    {
        leftBound = left;
        rightBound = right;
        topLeftAlignment = topLeft;
        return *this;
    }
    GpuSdpaBwdOptions& setSeqLens(hipdnn_data_sdk::utilities::TensorBase<int32_t>* q,
                                  hipdnn_data_sdk::utilities::TensorBase<int32_t>* kv)
    {
        seqLenQ = q;
        seqLenKv = kv;
        return *this;
    }
    GpuSdpaBwdOptions& setDropout(hipdnn_data_sdk::utilities::TensorBase<float>* mask,
                                  std::optional<float> scale,
                                  std::optional<float> scaleInv)
    {
        dropoutMask = mask;
        dropoutScale = scale;
        dropoutScaleInv = scaleInv;
        return *this;
    }
    GpuSdpaBwdOptions& setSoftcap(std::optional<float> value)
    {
        softcap = value;
        return *this;
    }
    GpuSdpaBwdOptions& setAlibi(bool value)
    {
        alibi = value;
        return *this;
    }
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
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                      hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                      hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                      hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                      const GpuSdpaFwdOptions<MaskDataType>& options = {})
    {
        validateSoftcap(options.softcap);
        // Paged K / V (cuDNN): k / v are containers [blocks, H, blockSize, D] addressed through an
        // INT32 [B, 1, T, 1] page table; options.pagedMaxSeqKv is the logical key extent. Paging needs
        // padding. Table entries must name existing blocks; the kernel does not check them.
        const bool pagedK = options.pageTableK != nullptr;
        const bool pagedV = options.pageTableV != nullptr;
        if((pagedK || pagedV) && (options.seqLenKv == nullptr || options.pagedMaxSeqKv <= 0))
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: paged K/V needs seq_len tensors and a "
                                        "positive max key length");
        }
        validateInput(
            q.dims(), k.dims(), v.dims(), o.dims(), pagedK, pagedV, options.pagedMaxSeqKv);

        // fp8 output scaling (cuDNN sdpa_fp8): the pipeline writes a float O_f, then
        // amaxO = max |O_f| and O = O_f * scaleO in O's dtype.
        if(options.scaleO.has_value() || options.amaxO != nullptr)
        {
            auto* amaxO = singleElement(options.amaxO, "amaxO");
            hipdnn_data_sdk::utilities::Tensor<float> oFloat(o.dims());
            auto inner = options;
            inner.scaleO.reset();
            inner.amaxO = nullptr;
            fprop<QDataType, KDataType, VDataType, float, ComputeDataType, MaskDataType>(
                q, k, v, oFloat, inner);
            launchSdpaScaleOut(static_cast<const float*>(oFloat.memory().deviceData()),
                               o.dims(),
                               o.strides(),
                               o.memory().deviceData(),
                               options.scaleO.value_or(1.0F),
                               amaxO,
                               detail::HipRtcTypeName<ODataType>::VALUE);
            o.memory().markDeviceModified();
            if(options.amaxO != nullptr)
            {
                options.amaxO->memory().markDeviceModified();
            }
            return;
        }
        // fp8 softmax scaling stores P in Q's fp8 format.
        if((options.scaleS.has_value() || options.descaleS.has_value())
           && options.pFormat == SdpaFp8Format::NONE)
        {
            constexpr auto FORMAT = fp8FormatOf<QDataType>();
            if constexpr(FORMAT == SdpaFp8Format::NONE)
            {
                throw std::invalid_argument(
                    "GpuFpReferenceSdpa: scaleS / descaleS need fp8 Q (P is stored in Q's format)");
            }
            else
            {
                auto inner = options;
                inner.pFormat = FORMAT;
                fprop<QDataType, KDataType, VDataType, ODataType, ComputeDataType, MaskDataType>(
                    q, k, v, o, inner);
                return;
            }
        }
        if(options.pFormat != SdpaFp8Format::NONE
           && options.probabilityMode != SdpaSoftmaxProbabilityMode::FLOAT)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpa: fp8 softmax scaling replaces the probability mode");
        }

        // fp8 (AITER contract): dequantize Q/K/V on the device, then run the float pipeline
        // (CpuFpReferenceSdpa::forward does the same on the host).
        if(options.descaleQ != nullptr || options.descaleK != nullptr
           || options.descaleV != nullptr)
        {
            const auto batch = q.dims()[0];
            const auto headsPerK = q.dims()[1] / k.dims()[1];
            auto qd = dequantize(q, options.descaleQ, batch, k.dims()[1], headsPerK, false, "Q");
            auto kd = dequantize(k, options.descaleK, batch, k.dims()[1], 1, pagedK, "K");
            auto vd = dequantize(v, options.descaleV, batch, v.dims()[1], 1, pagedV, "V");
            auto plain = options;
            plain.descaleQ = nullptr;
            plain.descaleK = nullptr;
            plain.descaleV = nullptr;
            fprop<float, float, float, ODataType, ComputeDataType, MaskDataType>(
                qd, kd, vd, o, plain);
            return;
        }

        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[1];
        const auto seqQ = q.dims()[2];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto seqKv = (pagedK || pagedV) ? options.pagedMaxSeqKv : k.dims()[2];

        SdpaPagedSide pagedKSide;
        SdpaPagedSide pagedVSide;
        if(pagedK)
        {
            pagedKSide = pagedSide(*options.pageTableK, k.dims(), batch, seqKv);
        }
        if(pagedV)
        {
            pagedVSide = pagedSide(*options.pageTableV, v.dims(), batch, seqKv);
        }
        const auto headDimV = v.dims()[3];

        // An absent scale is 1.0 (no scaling), as in cuDNN.
        const float scale = options.attnScale.value_or(1.0F);

        auto defines
            = detail::buildSdpaDefines<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                options.probabilityMode);
        defines.emplace_back(std::string("-DMASK_TYPE=")
                             + detail::HipRtcTypeName<MaskDataType>::VALUE);

        const void* maskPtr = nullptr;
        std::vector<int64_t> maskDims;
        std::vector<int64_t> maskStrides;
        if(options.attnMask != nullptr)
        {
            // The kernel right-aligns the mask to [B, H, Sq, Skv] and indexes
            // args.maskDims/maskStr (size 4); a rank>4 mask would read out of bounds.
            if(options.attnMask->dims().empty() || options.attnMask->dims().size() > 4)
            {
                throw std::invalid_argument("GpuFpReferenceSdpa: attention mask must be rank 1..4");
            }
            maskPtr = options.attnMask->memory().deviceData();
            maskDims = options.attnMask->dims();
            maskStrides = options.attnMask->strides();
        }

        void* lsePtr = nullptr;
        std::vector<int64_t> lseStrides;
        if(options.lse != nullptr)
        {
            // LSE is one value per query position; mirror the CPU oracle's shape check.
            if(options.lse->dims().size() != 3 || options.lse->dims()[0] != batch
               || options.lse->dims()[1] != numHeads || options.lse->dims()[2] != seqQ)
            {
                throw std::invalid_argument("GpuFpReferenceSdpa: lse must be rank-3 [B, H, Sq]");
            }
            lsePtr = options.lse->memory().deviceData();
            lseStrides = options.lse->strides();
        }
        // cuDNN Max / Sum_exp row statistics: rank-3 [B, H, Sq], as the LSE.
        for(auto* stats : {options.rowMax, options.rowSumExp})
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
        if((options.seqLenQ == nullptr) != (options.seqLenKv == nullptr))
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpa: seqLenQ and seqLenKv must be given together");
        }
        const int32_t* seqLenQPtr = nullptr;
        const int32_t* seqLenKvPtr = nullptr;
        int64_t seqLenQStride = 0;
        int64_t seqLenKvStride = 0;
        if(options.seqLenQ != nullptr)
        {
            for(auto* len : {options.seqLenQ, options.seqLenKv})
            {
                if(static_cast<int64_t>(len->elementCount()) != batch || len->dims().empty()
                   || len->dims()[0] != batch)
                {
                    throw std::invalid_argument(
                        "GpuFpReferenceSdpa: seq_len tensors must hold one value per batch");
                }
            }
            seqLenQPtr = static_cast<const int32_t*>(options.seqLenQ->memory().deviceData());
            seqLenKvPtr = static_cast<const int32_t*>(options.seqLenKv->memory().deviceData());
            seqLenQStride = options.seqLenQ->strides()[0];
            seqLenKvStride = options.seqLenKv->strides()[0];
        }

        // Attention sink (cuDNN): FLOAT [1, H, 1, 1], one logit per query head.
        const float* sinkPtr = nullptr;
        int64_t sinkStride = 0;
        if(options.sinkToken != nullptr)
        {
            if(options.sinkToken->dims().size() != 4 || options.sinkToken->dims()[1] != numHeads
               || static_cast<int64_t>(options.sinkToken->elementCount()) != numHeads)
            {
                throw std::invalid_argument("GpuFpReferenceSdpa: sinkToken must be [1, H, 1, 1]");
            }
            sinkPtr = static_cast<const float*>(options.sinkToken->memory().deviceData());
            sinkStride = options.sinkToken->strides()[1];
        }

        launchSdpaFwd(
            q.memory().deviceData(),
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
            options.leftBound,
            options.rightBound,
            options.topLeftAlignment,
            defines,
            seqLenQPtr,
            seqLenKvPtr,
            seqLenQStride,
            seqLenKvStride,
            pagedKSide,
            pagedVSide,
            sinkPtr,
            sinkStride,
            dropoutMaskPointer(options.dropoutMask, batch, numHeads, seqQ, seqKv),
            dropoutMaskStrides(options.dropoutMask),
            options.dropoutScale.value_or(1.0F),
            options.softcap.value_or(0.0F),
            options.rowMax != nullptr ? options.rowMax->memory().deviceData() : nullptr,
            options.rowMax != nullptr ? options.rowMax->strides() : std::vector<int64_t>{},
            options.rowSumExp != nullptr ? options.rowSumExp->memory().deviceData() : nullptr,
            options.rowSumExp != nullptr ? options.rowSumExp->strides() : std::vector<int64_t>{},
            options.alibi,
            SdpaFp8Softmax{options.pFormat,
                           options.scaleS.value_or(1.0F),
                           options.descaleS.value_or(1.0F),
                           singleElement(options.amaxS, "amaxS")},
            blockMaskPointer(options.blockMask, batch, numHeads, seqQ, seqKv),
            options.blockMask != nullptr ? options.blockMask->strides() : std::vector<int64_t>{});

        o.memory().markDeviceModified();
        for(auto* stats : {options.lse, options.rowMax, options.rowSumExp, options.amaxS})
        {
            if(stats != nullptr)
            {
                stats->memory().markDeviceModified();
            }
        }
    }

    // --- Backward SDPA (bprop): dQ, dK, dV and optional dBias ---
    //
    // Same options and semantics as CpuFpReferenceSdpa::backward: fp32 accumulation,
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
    static void bprop(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                      hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                      hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                      hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                      hipdnn_data_sdk::utilities::TensorBase<DODataType>& dO,
                      hipdnn_data_sdk::utilities::TensorBase<DQDataType>& dQ,
                      hipdnn_data_sdk::utilities::TensorBase<DKDataType>& dK,
                      hipdnn_data_sdk::utilities::TensorBase<DVDataType>& dV,
                      const GpuSdpaBwdOptions<MaskDataType>& options = {})
    {
        validateSoftcap(options.softcap);
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
        if(options.lse != nullptr)
        {
            const auto& ld = options.lse->dims();
            expect((ld.size() == 3 || (ld.size() == 4 && ld[3] == 1)) && ld[0] == batch
                       && ld[1] == numHeads && ld[2] == seqQ,
                   "lse must be [B, H, Sq] or [B, H, Sq, 1]");
        }
        std::vector<int64_t> maskDims;
        if(options.attnMask != nullptr)
        {
            maskDims = options.attnMask->dims();
            expect(!maskDims.empty() && maskDims.size() <= 4, "attention mask must be rank 1..4");
            const std::array<int64_t, 4> full = {batch, numHeads, seqQ, seqKv};
            for(size_t i = 0; i < maskDims.size(); ++i)
            {
                const auto ctx = full[4 - maskDims.size() + i];
                expect(maskDims[i] == 1 || maskDims[i] == ctx,
                       "each attention mask dim must be 1 or the full size");
            }
        }
        expect(options.dBias == nullptr
                   || (options.attnMask != nullptr && options.dBias->dims() == maskDims),
               "dBias requires an attention mask of the same shape");
        expect((options.seqLenQ == nullptr) == (options.seqLenKv == nullptr),
               "seqLenQ and seqLenKv must be given together");
        for(auto* len : {options.seqLenQ, options.seqLenKv})
        {
            expect(len == nullptr
                       || (static_cast<int64_t>(len->elementCount()) == batch
                           && !len->dims().empty() && len->dims()[0] == batch),
                   "seq_len tensors must hold one value per batch");
        }

        const std::vector<std::string> defines
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
        launch.mask
            = options.attnMask != nullptr ? options.attnMask->memory().deviceData() : nullptr;
        launch.dBias = options.dBias != nullptr ? options.dBias->memory().deviceData() : nullptr;
        launch.lse = options.lse != nullptr
                         ? static_cast<const float*>(options.lse->memory().deviceData())
                         : nullptr;
        launch.seqLenQ = options.seqLenQ != nullptr
                             ? static_cast<const int32_t*>(options.seqLenQ->memory().deviceData())
                             : nullptr;
        launch.seqLenKv = options.seqLenKv != nullptr
                              ? static_cast<const int32_t*>(options.seqLenKv->memory().deviceData())
                              : nullptr;
        launch.strides
            = {q.strides(),
               k.strides(),
               v.strides(),
               o.strides(),
               dO.strides(),
               dQ.strides(),
               dK.strides(),
               dV.strides(),
               options.attnMask != nullptr ? options.attnMask->strides() : std::vector<int64_t>{},
               options.dBias != nullptr ? options.dBias->strides() : std::vector<int64_t>{},
               options.lse != nullptr ? options.lse->strides() : std::vector<int64_t>{}};
        launch.seqLenQStride = options.seqLenQ != nullptr ? options.seqLenQ->strides()[0] : 0;
        launch.seqLenKvStride = options.seqLenKv != nullptr ? options.seqLenKv->strides()[0] : 0;
        launch.maskDims = maskDims;
        launch.batch = batch;
        launch.numHeads = numHeads;
        launch.numHeadsK = k.dims()[1];
        launch.numHeadsV = v.dims()[1];
        launch.seqQ = seqQ;
        launch.seqKv = seqKv;
        launch.headDim = headDim;
        launch.headDimV = headDimV;
        launch.scale = options.attnScale.value_or(1.0F);
        launch.leftBound = options.leftBound;
        launch.rightBound = options.rightBound;
        launch.topLeftAlignment = options.topLeftAlignment;
        launch.dropoutMask = dropoutMaskPointer(options.dropoutMask, batch, numHeads, seqQ, seqKv);
        launch.dropoutMaskStrides = dropoutMaskStrides(options.dropoutMask);
        launch.dropoutScale = options.dropoutScale.value_or(1.0F);
        launch.dropoutScaleInv = options.dropoutScaleInv.value_or(1.0F);
        launch.softcap = options.softcap.value_or(0.0F);
        launch.alibi = options.alibi;
        launchSdpaBwd(launch, defines);

        dQ.memory().markDeviceModified();
        dK.memory().markDeviceModified();
        dV.memory().markDeviceModified();
        if(options.dBias != nullptr)
        {
            options.dBias->memory().markDeviceModified();
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
        bool alibi = false; // cuDNN ALiBi (SEM-021)
        int64_t leftBound = -1;
        int64_t rightBound = -1;
        bool topLeftAlignment = true;
    };

    static void launchSdpaBwd(const SdpaBwdLaunch& launch, const std::vector<std::string>& defines);

    static void launchSdpaDequant(const void* in,
                                  const std::vector<int64_t>& dims,
                                  const std::vector<int64_t>& strides,
                                  float* out,
                                  const float* descale,
                                  int64_t descaleBatchStride,
                                  int64_t descaleHeadStride,
                                  int64_t headsPerKv,
                                  const char* inType);

    // fp8 dequantization on the device: a packed float copy of x * descale, with a FLOAT scalar
    // or [B, H_kv, 1, 1] descale; head h of x uses KV head h / headsPerKv. A per-head descale is
    // undefined on a paged container (its blocks are not batches), so paging needs a scalar.
    template <class T>
    static hipdnn_data_sdk::utilities::Tensor<float>
        dequantize(hipdnn_data_sdk::utilities::TensorBase<T>& x,
                   hipdnn_data_sdk::utilities::TensorBase<float>* descale,
                   int64_t batch,
                   int64_t numHeadsKv,
                   int64_t headsPerKv,
                   bool paged,
                   const char* name)
    {
        const float* descalePtr = nullptr;
        int64_t batchStride = 0;
        int64_t headStride = 0;
        if(descale != nullptr)
        {
            descalePtr = static_cast<const float*>(descale->memory().deviceData());
            if(descale->elementCount() != 1)
            {
                const auto& d = descale->dims();
                if(paged || d.size() != 4 || d[0] != batch || d[1] != numHeadsKv || d[2] != 1
                   || d[3] != 1)
                {
                    throw std::invalid_argument(
                        std::string("GpuFpReferenceSdpa: descale") + name
                        + " must be a scalar or [B, H_kv, 1, 1] (a scalar when paged)");
                }
                batchStride = descale->strides()[0];
                headStride = descale->strides()[1];
            }
        }
        hipdnn_data_sdk::utilities::Tensor<float> out(x.dims());
        launchSdpaDequant(x.memory().deviceData(),
                          x.dims(),
                          x.strides(),
                          static_cast<float*>(out.memory().deviceData()),
                          descalePtr,
                          batchStride,
                          headStride,
                          headsPerKv,
                          detail::HipRtcTypeName<T>::VALUE);
        out.memory().markDeviceModified();
        return out;
    }

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
                              const std::vector<int64_t>& rowSumExpStrides = {},
                              bool alibi = false,
                              const SdpaFp8Softmax& fp8 = {},
                              const uint8_t* blockMask = nullptr,
                              const std::vector<int64_t>& blockMaskStrides = {});

    // cuDNN block-sparse mask on the device, after checking its shape (CpuFpReferenceSdpa's
    // validateBlockMask): UINT8 [B, H, ceil(Sq / 128), ceil(ceil(Skv / 128) / 8)].
    static const uint8_t* blockMaskPointer(hipdnn_data_sdk::utilities::TensorBase<uint8_t>* mask,
                                           int64_t batch,
                                           int64_t heads,
                                           int64_t seqQ,
                                           int64_t seqKv)
    {
        if(mask == nullptr)
        {
            return nullptr;
        }
        const auto tilesKv = (seqKv + 127) / 128;
        const std::vector<int64_t> want = {batch, heads, (seqQ + 127) / 128, (tilesKv + 7) / 8};
        if(mask->dims() != want)
        {
            throw std::invalid_argument("GpuFpReferenceSdpa: blockMask must be UINT8 [B, H, "
                                        "ceil(Sq / 128), ceil(ceil(Skv / 128) / 8)]");
        }
        return static_cast<const uint8_t*>(mask->memory().deviceData());
    }

    static void launchSdpaScaleOut(const float* in,
                                   const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& outStrides,
                                   void* out,
                                   float scale,
                                   float* amax,
                                   const char* outType);

    /// The fp8 format of an fp8 element type, NONE for any other type.
    template <class T>
    static constexpr SdpaFp8Format fp8FormatOf()
    {
        if constexpr(std::is_same_v<T, hipdnn_data_sdk::types::fp8_e4m3>)
        {
            return SdpaFp8Format::E4M3;
        }
        else if constexpr(std::is_same_v<T, hipdnn_data_sdk::types::fp8_e4m3_fnuz>)
        {
            return SdpaFp8Format::E4M3_FNUZ;
        }
        else if constexpr(std::is_same_v<T, hipdnn_data_sdk::types::fp8_e5m2>)
        {
            return SdpaFp8Format::E5M2;
        }
        else if constexpr(std::is_same_v<T, hipdnn_data_sdk::types::fp8_e5m2_fnuz>)
        {
            return SdpaFp8Format::E5M2_FNUZ;
        }
        else
        {
            return SdpaFp8Format::NONE;
        }
    }

    // The device pointer of a one-element FLOAT output (amax), or nullptr when absent.
    static float* singleElement(hipdnn_data_sdk::utilities::TensorBase<float>* t, const char* name)
    {
        if(t == nullptr)
        {
            return nullptr;
        }
        if(t->elementCount() != 1)
        {
            throw std::invalid_argument(std::string("GpuFpReferenceSdpa: ") + name
                                        + " must have one element");
        }
        return static_cast<float*>(t->memory().deviceData());
    }

    // Logit softcap (reference-only extension, SEM-020): finite and > 0 when given.
    static void validateSoftcap(std::optional<float> softcap)
    {
        if(softcap.has_value() && (softcap.value() <= 0.0F || !std::isfinite(softcap.value())))
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
