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
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

// Non-deduced context: the mask element type comes from explicit template arguments only,
// so a literal nullptr mask compiles and a wrong-dtype mask pointer is a compile error.
template <class T>
struct NonDeduced
{
    using type = T;
};

/// The fp8 format the softmax probabilities are stored in before P @ V (cuDNN sdpa_fp8 quantizes
/// P to the input fp8 type).
enum class SdpaFp8Format
{
    NONE,
    E4M3,
    E4M3_FNUZ,
};

/// Named options for CpuFpReferenceSdpa::forward. Every field defaults to "feature off".
template <class MaskDataType = float>
struct SdpaFwdOptions
{
    /// Attention scale; 1.0 (no scaling) when unset, as in cuDNN.
    std::optional<float> attnScale;
    /// Additive bias, dims right-aligned to [B, H, Sq, Skv] (rank 1-4), broadcast on size-1 dims.
    const hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* attnMask = nullptr;
    /// Diagonal band (cuDNN, SEM-001): leftBound = L keeps L keys up to and including the
    /// diagonal, rightBound = R keeps R keys past it; -1 is unbounded. Bottom-right alignment
    /// measures the diagonal from each batch's live lengths.
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeftAlignment = true;
    /// Padding (cuDNN seq_len_q / seq_len_kv): INT32 [B, 1, 1, 1], both or neither. Rows past
    /// seqLenQ output O = 0 and LSE = -inf; keys past seqLenKv are masked and never read.
    const hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenQ = nullptr;
    const hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenKv = nullptr;
    /// Attention sink, FP32 [1, H, 1, 1]: one unscaled logit per query head in the softmax max and
    /// denominator, with no value row. The LSE includes it (SEM-022).
    const hipdnn_data_sdk::utilities::TensorBase<float>* sinkToken = nullptr;
    /// Explicit dropout (cuDNN custom dropout): FP32 mask [B or 1, H or 1, Sq, Skv];
    /// O uses P * dropoutScale * mask (dropoutScale is 1 when unset).
    const hipdnn_data_sdk::utilities::TensorBase<float>* dropoutMask = nullptr;
    std::optional<float> dropoutScale;
    /// Logit softcap c > 0: s becomes c * tanh(s / c) after the scale, before bias and masks.
    /// Not expressible in the graph API yet (SEM-020).
    std::optional<float> softcap;
    /// cuDNN ALiBi: (skv - sq) * alibiSlope(h) after the bias (SEM-021).
    bool alibi = false;
    /// cuDNN block-sparse mask (SEM-025): UINT8 [B, H, ceil(Sq / 128), ceil(ceil(Skv / 128) / 8)],
    /// one bit per 128 x 128 score tile, least significant bit first; a clear bit masks the tile.
    const hipdnn_data_sdk::utilities::TensorBase<uint8_t>* blockMask = nullptr;
    /// Outputs, FP32 [B, H, Sq, 1]: LSE = Max + log(Sum_exp) (SEM-023).
    hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* rowMax = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* rowSumExp = nullptr;
    /// fp8 dequantization (AITER's contract): scalar or per-(batch, KV-head) [B, H_kv, 1, 1]
    /// descales. Q/K/V are dequantized first, then the float pipeline runs.
    const hipdnn_data_sdk::utilities::TensorBase<float>* descaleQ = nullptr;
    const hipdnn_data_sdk::utilities::TensorBase<float>* descaleK = nullptr;
    const hipdnn_data_sdk::utilities::TensorBase<float>* descaleV = nullptr;
    /// fp8 softmax scaling (cuDNN sdpa_fp8, SEM-024): P @ V uses
    /// fp8(exp(s - Max) * scaleS) * descaleS / Sum_exp in place of P, in pFormat. forward() sets
    /// pFormat from Q's fp8 dtype; both scales are needed and default to 1 when the other is set.
    std::optional<float> scaleS;
    std::optional<float> descaleS;
    SdpaFp8Format pFormat = SdpaFp8Format::NONE;
    /// fp8 output scaling: O = O_f * scaleO, stored in O's dtype.
    std::optional<float> scaleO;
    /// Outputs, FP32 single element: amaxS = max exp(s - Max) over live keys (before scaleS),
    /// amaxO = max |O_f| (before scaleO).
    hipdnn_data_sdk::utilities::TensorBase<float>* amaxS = nullptr;
    hipdnn_data_sdk::utilities::TensorBase<float>* amaxO = nullptr;

    // Chainable setters, so a call can name its options inline:
    //   forward(q, k, v, o, SdpaFwdOptions<float>{}.setScale(0.5F).setLse(&lse))
    SdpaFwdOptions& setScale(std::optional<float> value)
    {
        attnScale = value;
        return *this;
    }
    SdpaFwdOptions& setMask(const hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* value)
    {
        attnMask = value;
        return *this;
    }
    SdpaFwdOptions& setBand(int64_t left, int64_t right, bool topLeft)
    {
        leftBound = left;
        rightBound = right;
        topLeftAlignment = topLeft;
        return *this;
    }
    SdpaFwdOptions& setSeqLens(const hipdnn_data_sdk::utilities::TensorBase<int32_t>* q,
                               const hipdnn_data_sdk::utilities::TensorBase<int32_t>* kv)
    {
        seqLenQ = q;
        seqLenKv = kv;
        return *this;
    }
    SdpaFwdOptions& setSink(const hipdnn_data_sdk::utilities::TensorBase<float>* value)
    {
        sinkToken = value;
        return *this;
    }
    SdpaFwdOptions& setDropout(const hipdnn_data_sdk::utilities::TensorBase<float>* mask,
                               std::optional<float> scale)
    {
        dropoutMask = mask;
        dropoutScale = scale;
        return *this;
    }
    SdpaFwdOptions& setSoftcap(std::optional<float> value)
    {
        softcap = value;
        return *this;
    }
    SdpaFwdOptions& setAlibi(bool value)
    {
        alibi = value;
        return *this;
    }
    SdpaFwdOptions& setBlockMask(const hipdnn_data_sdk::utilities::TensorBase<uint8_t>* value)
    {
        blockMask = value;
        return *this;
    }
    SdpaFwdOptions& setLse(hipdnn_data_sdk::utilities::TensorBase<float>* value)
    {
        lse = value;
        return *this;
    }
    SdpaFwdOptions& setRowStats(hipdnn_data_sdk::utilities::TensorBase<float>* max,
                                hipdnn_data_sdk::utilities::TensorBase<float>* sumExp)
    {
        rowMax = max;
        rowSumExp = sumExp;
        return *this;
    }
    SdpaFwdOptions& setDescales(const hipdnn_data_sdk::utilities::TensorBase<float>* q,
                                const hipdnn_data_sdk::utilities::TensorBase<float>* k,
                                const hipdnn_data_sdk::utilities::TensorBase<float>* v)
    {
        descaleQ = q;
        descaleK = k;
        descaleV = v;
        return *this;
    }
    SdpaFwdOptions& setSoftmaxScales(std::optional<float> scale, std::optional<float> descale)
    {
        scaleS = scale;
        descaleS = descale;
        return *this;
    }
    SdpaFwdOptions& setOutputScale(std::optional<float> value)
    {
        scaleO = value;
        return *this;
    }
    SdpaFwdOptions& setAmax(hipdnn_data_sdk::utilities::TensorBase<float>* s,
                            hipdnn_data_sdk::utilities::TensorBase<float>* out)
    {
        amaxS = s;
        amaxO = out;
        return *this;
    }
};

/// Named options for CpuFpReferenceSdpa::backward. Fields shared with SdpaFwdOptions mean the same.
template <class MaskDataType = float>
struct SdpaBwdOptions
{
    std::optional<float> attnScale;
    /// Forward LSE [B, H, Sq, 1]: P = exp(score - LSE). When null, the softmax is recomputed.
    const hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr;
    const hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* attnMask = nullptr;
    /// Gradient of attnMask, same shape and dtype: the unscaled dS summed over every dim the mask
    /// broadcasts (cuDNN). Masked positions and dead rows contribute 0. Requires attnMask.
    hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* dBias = nullptr;
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeftAlignment = true;
    const hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenQ = nullptr;
    const hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenKv = nullptr;
    /// Explicit dropout as cuDNN's backward composes it: dP is masked, D is scaled by
    /// dropoutScaleInv (1 when unset), dS by dropoutScale, and dV uses P * dropoutScale * mask.
    const hipdnn_data_sdk::utilities::TensorBase<float>* dropoutMask = nullptr;
    std::optional<float> dropoutScale;
    std::optional<float> dropoutScaleInv;
    std::optional<float> softcap;
    bool alibi = false;

    SdpaBwdOptions& setScale(std::optional<float> value)
    {
        attnScale = value;
        return *this;
    }
    SdpaBwdOptions& setLse(const hipdnn_data_sdk::utilities::TensorBase<float>* value)
    {
        lse = value;
        return *this;
    }
    SdpaBwdOptions& setMask(const hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* value)
    {
        attnMask = value;
        return *this;
    }
    SdpaBwdOptions& setDBias(hipdnn_data_sdk::utilities::TensorBase<MaskDataType>* value)
    {
        dBias = value;
        return *this;
    }
    SdpaBwdOptions& setBand(int64_t left, int64_t right, bool topLeft)
    {
        leftBound = left;
        rightBound = right;
        topLeftAlignment = topLeft;
        return *this;
    }
    SdpaBwdOptions& setSeqLens(const hipdnn_data_sdk::utilities::TensorBase<int32_t>* q,
                               const hipdnn_data_sdk::utilities::TensorBase<int32_t>* kv)
    {
        seqLenQ = q;
        seqLenKv = kv;
        return *this;
    }
    SdpaBwdOptions& setDropout(const hipdnn_data_sdk::utilities::TensorBase<float>* mask,
                               std::optional<float> scale,
                               std::optional<float> scaleInv)
    {
        dropoutMask = mask;
        dropoutScale = scale;
        dropoutScaleInv = scaleInv;
        return *this;
    }
    SdpaBwdOptions& setSoftcap(std::optional<float> value)
    {
        softcap = value;
        return *this;
    }
    SdpaBwdOptions& setAlibi(bool value)
    {
        alibi = value;
        return *this;
    }
};

class CpuFpReferenceSdpa
{
public:
    /// Paged KV cache (cuDNN): gathers a paged K or V container into a dense
    /// [B, H, maxSeqKv, D] tensor that forward() consumes with seq_len padding.
    ///
    /// The container is [numBlocks, H, blockSize, D] with arbitrary strides (so vLLM's NHD and
    /// HND cache layouts are both just strides). The page table is INT32 [B, 1, tableSize, 1]:
    /// logical key j of batch b lives in block pageTable[b, 0, j / blockSize, 0] at slot
    /// j % blockSize. Only keys below seqLenKv[b] are read, so unused table entries and unused
    /// slots (which may hold any value, including NaN) are never touched. Positions past
    /// seqLenKv[b] in the result are left at zero; forward() never reads them under padding.
    template <class T>
    static hipdnn_data_sdk::utilities::Tensor<T>
        gatherPagedKv(const hipdnn_data_sdk::utilities::TensorBase<T>& container,
                      const hipdnn_data_sdk::utilities::TensorBase<int32_t>& pageTable,
                      const hipdnn_data_sdk::utilities::TensorBase<int32_t>& seqLenKv,
                      int64_t maxSeqKv)
    {
        const auto& cDims = container.dims();
        const auto& tDims = pageTable.dims();
        if(cDims.size() != 4)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::gatherPagedKv: container must be rank-4 [blocks, H, bs, D]");
        }
        if(tDims.size() != 4 || tDims[1] != 1 || tDims[3] != 1)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::gatherPagedKv: page table must be [B, 1, tableSize, 1]");
        }
        const auto numBlocks = cDims[0];
        const auto heads = cDims[1];
        const auto blockSize = cDims[2];
        const auto headDim = cDims[3];
        const auto batch = tDims[0];
        const auto tableSize = tDims[2];
        if(maxSeqKv <= 0 || maxSeqKv > tableSize * blockSize)
        {
            throw std::invalid_argument("CpuFpReferenceSdpa::gatherPagedKv: maxSeqKv must be in "
                                        "[1, tableSize * blockSize]");
        }
        const auto lens = resolveSeqLens(&seqLenKv, &seqLenKv, batch, maxSeqKv, "seqLenKv");

        hipdnn_data_sdk::utilities::Tensor<T> dense({batch, heads, maxSeqKv, headDim});
        dense.fillWithValue(hipdnn_test_sdk::detail::safeConvert<T>(0.0F));
        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t j = 0; j < lens[static_cast<size_t>(b)]; ++j)
            {
                const auto block = static_cast<int64_t>(
                    pageTable.getHostValue(std::vector<int64_t>{b, 0, j / blockSize, 0}));
                if(block < 0 || block >= numBlocks)
                {
                    throw std::invalid_argument(
                        "CpuFpReferenceSdpa::gatherPagedKv: page table entry "
                        + std::to_string(block) + " for batch " + std::to_string(b)
                        + " is outside [0, " + std::to_string(numBlocks) + ")");
                }
                const auto slot = j % blockSize;
                for(int64_t h = 0; h < heads; ++h)
                {
                    for(int64_t d = 0; d < headDim; ++d)
                    {
                        dense.setHostValue(
                            container.getHostValue(std::vector<int64_t>{block, h, slot, d}),
                            std::vector<int64_t>{b, h, j, d});
                    }
                }
            }
        }
        return dense;
    }

    /// SDPA forward: O = softmax(score(Q, K)) @ V, one pipeline for every feature.
    ///
    /// The score of key skv for query sq is, in order (SdpaSemanticsRegister): (q . k) * scale,
    /// softcap, + bias, + ALiBi, then the diagonal band and padding mask it to -inf. The softmax
    /// takes an attention sink in its max and denominator, dropout scales P, and P @ V gives O.
    /// Each feature is a field of SdpaFwdOptions that is off by default.
    ///
    /// Supports GQA/MQA: H must be divisible by both H_k and H_v.
    ///
    /// @param q        Query tensor [B, H, Sq, D]
    /// @param k        Key tensor   [B, H_k, Skv, D]
    /// @param v        Value tensor [B, H_v, Skv, Dv]
    /// @param o        Output tensor [B, H, Sq, Dv]
    /// @param options  Everything else (SdpaFwdOptions).
    template <class QDataType,
              class KDataType,
              class VDataType,
              class ODataType,
              class ComputeDataType = float,
              class MaskDataType = ComputeDataType>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                        const hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                        const hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                        hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                        const SdpaFwdOptions<MaskDataType>& options = {})
    {
        requireRank4(q, "q");
        requireRank4(k, "k");
        requireRank4(v, "v");
        requireRank4(o, "o");
        // fp8 output scaling (cuDNN sdpa_fp8): the pipeline writes a float O_f, then
        // amaxO = max |O_f| and O = O_f * scaleO in O's dtype.
        if(options.scaleO.has_value() || options.amaxO != nullptr)
        {
            requireSingleElement(options.amaxO, "amaxO");
            hipdnn_data_sdk::utilities::Tensor<float> oFloat(o.dims());
            auto inner = options;
            inner.scaleO.reset();
            inner.amaxO = nullptr;
            forward<QDataType, KDataType, VDataType, float, ComputeDataType, MaskDataType>(
                q, k, v, oFloat, inner);
            const auto scaleO = options.scaleO.value_or(1.0F);
            float amax = 0.0F;
            const auto& d = o.dims();
            for(int64_t b = 0; b < d[0]; ++b)
            {
                for(int64_t h = 0; h < d[1]; ++h)
                {
                    for(int64_t sq = 0; sq < d[2]; ++sq)
                    {
                        for(int64_t dv = 0; dv < d[3]; ++dv)
                        {
                            const std::vector<int64_t> idx = {b, h, sq, dv};
                            const float value = oFloat.getHostValue(idx);
                            amax = std::max(amax, std::fabs(value));
                            o.setHostValue(
                                hipdnn_test_sdk::detail::safeConvert<ODataType>(value * scaleO),
                                idx);
                        }
                    }
                }
            }
            o.memory().markHostModified();
            writeAmax(options.amaxO, amax);
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
                    "CpuFpReferenceSdpa: scaleS / descaleS need fp8 Q (P is stored in Q's format)");
            }
            else
            {
                auto inner = options;
                inner.pFormat = FORMAT;
                forward<QDataType, KDataType, VDataType, ODataType, ComputeDataType, MaskDataType>(
                    q, k, v, o, inner);
                return;
            }
        }
        // fp8 (AITER contract): Q/K/V descales dequantize the inputs, then the float pipeline runs.
        if(options.descaleQ != nullptr || options.descaleK != nullptr
           || options.descaleV != nullptr)
        {
            const auto numHeads = q.dims()[1];
            const auto numHeadsK = k.dims()[1];
            const auto numHeadsV = v.dims()[1];
            if(numHeadsK <= 0 || numHeadsV <= 0 || numHeads % numHeadsK != 0
               || numHeads % numHeadsV != 0)
            {
                throw std::invalid_argument(
                    "CpuFpReferenceSdpa: numHeads must be divisible by numHeadsK and numHeadsV");
            }
            validateDescaleShape(options.descaleQ, "descaleQ", q.dims()[0], numHeadsK);
            validateDescaleShape(options.descaleK, "descaleK", q.dims()[0], numHeadsK);
            validateDescaleShape(options.descaleV, "descaleV", q.dims()[0], numHeadsV);
            const auto qd = dequantize<ComputeDataType>(q, options.descaleQ, numHeads / numHeadsK);
            const auto kd = dequantize<ComputeDataType>(k, options.descaleK, 1);
            const auto vd = dequantize<ComputeDataType>(v, options.descaleV, 1);
            auto plain = options;
            plain.descaleQ = nullptr;
            plain.descaleK = nullptr;
            plain.descaleV = nullptr;
            forward<ComputeDataType,
                    ComputeDataType,
                    ComputeDataType,
                    ODataType,
                    ComputeDataType,
                    MaskDataType>(qd, kd, vd, o, plain);
            return;
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
        validateRowStats(options.lse, "lse", batch, numHeads, seqQ);
        validateRowStats(options.rowMax, "rowMax", batch, numHeads, seqQ);
        validateRowStats(options.rowSumExp, "rowSumExp", batch, numHeads, seqQ);
        requireSingleElement(options.amaxS, "amaxS");

        // Validated on the calling thread so a bad input is a catchable exception rather than a
        // failure inside the parallel region.
        const auto validQ
            = resolveSeqLens(options.seqLenQ, options.seqLenKv, batch, seqQ, "seqLenQ");
        const auto validKv
            = resolveSeqLens(options.seqLenKv, options.seqLenQ, batch, seqKv, "seqLenKv");
        validateDropoutMask(options.dropoutMask, batch, numHeads, seqQ, seqKv);
        validateSoftcap(options.softcap);
        // Attention sink (cuDNN): one unscaled logit per query head, [1, H, 1, 1].
        if(options.sinkToken != nullptr
           && (static_cast<int64_t>(options.sinkToken->elementCount()) != numHeads
               || options.sinkToken->dims().size() != 4
               || options.sinkToken->dims()[1] != numHeads))
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: sinkToken must be [1, H, 1, 1]");
        }
        validateBlockMask(options.blockMask, batch, numHeads, seqQ, seqKv);

        const auto headsPerHeadK = numHeads / numHeadsK;
        const auto headsPerHeadV = numHeads / numHeadsV;
        const auto scale = static_cast<ComputeDataType>(options.attnScale.value_or(1.0F));
        // Dropout with an explicit mask (cuDNN): O uses P * dropoutScale * mask; LSE does not.
        const auto dropoutFactor
            = static_cast<ComputeDataType>(options.dropoutScale.value_or(1.0F));
        const auto negInf = -std::numeric_limits<ComputeDataType>::infinity();
        const bool quantizeP = options.pFormat != SdpaFp8Format::NONE;
        const auto scaleS = static_cast<ComputeDataType>(options.scaleS.value_or(1.0F));
        const auto descaleS = static_cast<ComputeDataType>(options.descaleS.value_or(1.0F));
        // max exp(s - Max) per row, reduced after the parallel loop (amaxS).
        std::vector<ComputeDataType> rowAmaxS(static_cast<size_t>(batch * numHeads * seqQ),
                                              static_cast<ComputeDataType>(0.0));

        auto rowFunc = [&](const std::vector<int64_t>& indices) {
            const auto b = indices[0];
            const auto h = indices[1];
            const auto sq = indices[2];
            const auto kvHeadK = h / headsPerHeadK;
            const auto kvHeadV = h / headsPerHeadV;
            const std::vector<int64_t> statIndex = {b, h, sq, 0};

            // Padding (cuDNN): a query row past seq_len_q outputs O = 0, LSE = Max = -inf and
            // Sum_exp = 0. Keys past seq_len_kv are masked and never read.
            const auto liveQ = validQ[static_cast<size_t>(b)];
            const auto liveKv = validKv[static_cast<size_t>(b)];
            if(sq >= liveQ)
            {
                for(int64_t dv = 0; dv < headDimV; ++dv)
                {
                    o.setHostValue(hipdnn_test_sdk::detail::safeConvert<ODataType>(0.0F),
                                   std::vector<int64_t>{b, h, sq, dv});
                }
                writeRowStats(options, statIndex, negInf, static_cast<ComputeDataType>(0.0));
                return;
            }

            std::vector<ComputeDataType> scores(static_cast<size_t>(seqKv));
            scoreRow(q, k, options, b, h, kvHeadK, sq, liveQ, liveKv, scale, scores, nullptr);

            // An attention sink (cuDNN) is one more logit in the max and the denominator with no
            // value row, so it only rescales P; the LSE includes it. A row with no live key but a
            // finite sink gets O = 0, LSE = sink.
            const auto sink = options.sinkToken != nullptr
                                  ? static_cast<ComputeDataType>(options.sinkToken->getHostValue(
                                        std::vector<int64_t>{0, h, 0, 0}))
                                  : negInf;
            std::vector<ComputeDataType> probs(static_cast<size_t>(seqKv));
            const auto [maxVal, sumExp] = softmaxRow(scores, sink, probs, /*normalize=*/false);
            writeRowStats(options, statIndex, maxVal, sumExp);
            // probs holds exp(s - Max). cuDNN sdpa_fp8 stores it as fp8(e * scaleS) * descaleS;
            // either way P is then divided by Sum_exp. A row with no live key keeps P = 0.
            auto& rowAmax = rowAmaxS[static_cast<size_t>((b * numHeads + h) * seqQ + sq)];
            if(maxVal != negInf)
            {
                for(auto& p : probs)
                {
                    rowAmax = std::max(rowAmax, p);
                    if(quantizeP)
                    {
                        p = roundToFp8(p * scaleS, options.pFormat) * descaleS;
                    }
                    p /= sumExp;
                }
            }

            for(int64_t dv = 0; dv < headDimV; ++dv)
            {
                auto acc = static_cast<ComputeDataType>(0.0);
                for(int64_t skv = 0; skv < liveKv; ++skv)
                {
                    acc += probs[static_cast<size_t>(skv)]
                           * dropoutKeep<ComputeDataType>(
                               options.dropoutMask, dropoutFactor, b, h, sq, skv)
                           * static_cast<ComputeDataType>(
                               v.getHostValue(std::vector<int64_t>{b, kvHeadV, skv, dv}));
                }
                o.setHostValue(hipdnn_test_sdk::detail::safeConvert<ODataType>(acc),
                               std::vector<int64_t>{b, h, sq, dv});
            }
        };

        auto parallelFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            rowFunc, std::vector<int64_t>{batch, numHeads, seqQ});
        parallelFunc(std::thread::hardware_concurrency());

        o.memory().markHostModified();
        if(options.amaxS != nullptr)
        {
            writeAmax(options.amaxS,
                      static_cast<float>(*std::max_element(rowAmaxS.begin(), rowAmaxS.end())));
        }
        for(auto* stats : {options.lse, options.rowMax, options.rowSumExp})
        {
            if(stats != nullptr)
            {
                stats->memory().markHostModified();
            }
        }
    }

    /// SDPA backward: dQ, dK, dV (and dBias) from the upstream gradient dO, through the same score
    /// pipeline as forward. P comes from the forward LSE when given, else from a recomputed
    /// softmax. Each feature is a field of SdpaBwdOptions that is off by default.
    ///
    /// GQA: several query heads accumulate into one K/V head, so the loop is sequential.
    /// Gradients accumulate in ComputeDataType and convert to the output types once at the end.
    ///
    /// @param q, k, v, o, dO   Forward inputs and output, and the upstream gradient.
    /// @param dQ, dK, dV       Outputs, shaped like q, k, v.
    /// @param options          Everything else (SdpaBwdOptions).
    template <class QDataType,
              class KDataType,
              class VDataType,
              class ODataType,
              class DODataType,
              class DQDataType,
              class DKDataType,
              class DVDataType,
              class ComputeDataType = float,
              class MaskDataType = ComputeDataType>
    static void backward(const hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                         const hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                         const hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                         const hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                         const hipdnn_data_sdk::utilities::TensorBase<DODataType>& dO,
                         hipdnn_data_sdk::utilities::TensorBase<DQDataType>& dQ,
                         hipdnn_data_sdk::utilities::TensorBase<DKDataType>& dK,
                         hipdnn_data_sdk::utilities::TensorBase<DVDataType>& dV,
                         const SdpaBwdOptions<MaskDataType>& options = {})
    {
        for(const auto& [dims, name] : {std::pair{&q.dims(), "q"},
                                        std::pair{&k.dims(), "k"},
                                        std::pair{&v.dims(), "v"},
                                        std::pair{&o.dims(), "o"},
                                        std::pair{&dO.dims(), "dO"},
                                        std::pair{&dQ.dims(), "dQ"},
                                        std::pair{&dK.dims(), "dK"},
                                        std::pair{&dV.dims(), "dV"}})
        {
            if(dims->size() != 4)
            {
                throw std::invalid_argument(std::string("CpuFpReferenceSdpa::backward: ") + name
                                            + " must be rank-4");
            }
        }
        const auto* attnMask = options.attnMask;
        auto* dBias = options.dBias;
        if(dBias != nullptr && (attnMask == nullptr || dBias->dims() != attnMask->dims()))
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa::backward: dBias requires attnMask and must match its shape");
        }

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
        validateRowStats(options.lse, "lse", batch, numHeadsQ, seqQ);

        const auto headsPerHeadK = numHeadsQ / numHeadsK;
        const auto headsPerHeadV = numHeadsQ / numHeadsV;
        const auto scale = static_cast<ComputeDataType>(options.attnScale.value_or(1.0F));
        const auto validQ
            = resolveSeqLens(options.seqLenQ, options.seqLenKv, batch, seqQ, "seqLenQ");
        const auto validKv
            = resolveSeqLens(options.seqLenKv, options.seqLenQ, batch, seqKv, "seqLenKv");
        validateDropoutMask(options.dropoutMask, batch, numHeadsQ, seqQ, seqKv);
        validateSoftcap(options.softcap);
        // Dropout with an explicit mask, composed as cuDNN's backward does: D is scaled by
        // dropoutScaleInv (1 when unset), dP is masked, dS is scaled by dropoutScale, and dV uses
        // P * dropoutScale * mask. dBias reduces the dropout-scaled dS.
        const auto dropoutFactor
            = static_cast<ComputeDataType>(options.dropoutScale.value_or(1.0F));
        const auto dropoutInv
            = static_cast<ComputeDataType>(options.dropoutScaleInv.value_or(1.0F));

        std::vector<ComputeDataType> dqAccum(
            static_cast<size_t>(batch * numHeadsQ * seqQ * headDim), ComputeDataType(0));
        std::vector<ComputeDataType> dkAccum(
            static_cast<size_t>(batch * numHeadsK * seqKv * headDim), ComputeDataType(0));
        std::vector<ComputeDataType> dvAccum(
            static_cast<size_t>(batch * numHeadsV * seqKv * headDimV), ComputeDataType(0));
        const auto dqIdx = [&](int64_t b, int64_t h, int64_t s, int64_t d) {
            return static_cast<size_t>(((b * numHeadsQ + h) * seqQ + s) * headDim + d);
        };
        const auto dkIdx = [&](int64_t b, int64_t h, int64_t s, int64_t d) {
            return static_cast<size_t>(((b * numHeadsK + h) * seqKv + s) * headDim + d);
        };
        const auto dvIdx = [&](int64_t b, int64_t h, int64_t s, int64_t d) {
            return static_cast<size_t>(((b * numHeadsV + h) * seqKv + s) * headDimV + d);
        };

        // dBias accumulator over the mask's own (possibly broadcast) shape, row-major.
        std::vector<ComputeDataType> dBiasAccum;
        const auto dBiasIdx = [&](const std::vector<int64_t>& maskIndices) {
            int64_t linear = 0;
            for(size_t i = 0; i < maskIndices.size(); ++i)
            {
                linear = linear * attnMask->dims()[i] + maskIndices[i];
            }
            return static_cast<size_t>(linear);
        };
        if(dBias != nullptr)
        {
            int64_t count = 1;
            for(const auto dim : attnMask->dims())
            {
                count *= dim;
            }
            dBiasAccum.assign(static_cast<size_t>(count), ComputeDataType(0));
        }

        std::vector<ComputeDataType> scores(static_cast<size_t>(seqKv));
        std::vector<ComputeDataType> probs(static_cast<size_t>(seqKv));
        std::vector<ComputeDataType> dP(static_cast<size_t>(seqKv));
        // d(softcap)/d(score) per key: 1 - tanh^2(score / c), or 1 without softcap.
        std::vector<ComputeDataType> softcapGrad(static_cast<size_t>(seqKv));
        for(int64_t b = 0; b < batch; ++b)
        {
            const auto liveQ = validQ[static_cast<size_t>(b)];
            const auto liveKv = validKv[static_cast<size_t>(b)];
            for(int64_t hQ = 0; hQ < numHeadsQ; ++hQ)
            {
                const auto kvHeadK = hQ / headsPerHeadK;
                const auto kvHeadV = hQ / headsPerHeadV;
                // Padding: a query row past seq_len_q has no output, so it contributes nothing
                // (its dQ stays 0); keys past seq_len_kv are never read.
                for(int64_t sq = 0; sq < liveQ; ++sq)
                {
                    // D = rowsum(dO * O), scaled by dropoutScaleInv.
                    auto rowD = static_cast<ComputeDataType>(0.0);
                    for(int64_t dv = 0; dv < headDimV; ++dv)
                    {
                        rowD += static_cast<ComputeDataType>(
                                    dO.getHostValue(std::vector<int64_t>{b, hQ, sq, dv}))
                                * static_cast<ComputeDataType>(
                                    o.getHostValue(std::vector<int64_t>{b, hQ, sq, dv}));
                    }
                    rowD *= dropoutInv;

                    std::fill(softcapGrad.begin(), softcapGrad.end(), ComputeDataType(1.0));
                    scoreRow(q,
                             k,
                             options,
                             b,
                             hQ,
                             kvHeadK,
                             sq,
                             liveQ,
                             liveKv,
                             scale,
                             scores,
                             &softcapGrad);
                    if(options.lse != nullptr)
                    {
                        // P = exp(score - LSE). A row with no live key has LSE = -inf (+inf from
                        // kernels that use that convention); its P is all zero.
                        const float lseVal
                            = options.lse->getHostValue(std::vector<int64_t>{b, hQ, sq, 0});
                        const bool deadRow = std::isinf(lseVal);
                        for(int64_t skv = 0; skv < liveKv; ++skv)
                        {
                            probs[static_cast<size_t>(skv)]
                                = deadRow ? ComputeDataType(0.0)
                                          : std::exp(scores[static_cast<size_t>(skv)]
                                                     - static_cast<ComputeDataType>(lseVal));
                        }
                    }
                    else
                    {
                        std::fill(probs.begin(), probs.end(), ComputeDataType(0.0));
                        softmaxRow(
                            scores, -std::numeric_limits<ComputeDataType>::infinity(), probs);
                    }

                    // dP = (dO @ V^T) * mask (dropout).
                    for(int64_t skv = 0; skv < liveKv; ++skv)
                    {
                        auto dpVal = static_cast<ComputeDataType>(0.0);
                        for(int64_t dv = 0; dv < headDimV; ++dv)
                        {
                            dpVal += static_cast<ComputeDataType>(
                                         dO.getHostValue(std::vector<int64_t>{b, hQ, sq, dv}))
                                     * static_cast<ComputeDataType>(
                                         v.getHostValue(std::vector<int64_t>{b, kvHeadV, skv, dv}));
                        }
                        dP[static_cast<size_t>(skv)]
                            = dpVal
                              * dropoutKeep(options.dropoutMask,
                                            static_cast<ComputeDataType>(1.0),
                                            b,
                                            hQ,
                                            sq,
                                            skv);
                    }

                    // dS = dropoutScale * P * (dP - D); accumulate dQ, dK, dV and dBias.
                    for(int64_t skv = 0; skv < liveKv; ++skv)
                    {
                        if(isMasked(sq,
                                    skv,
                                    liveQ,
                                    liveKv,
                                    options.leftBound,
                                    options.rightBound,
                                    options.topLeftAlignment))
                        {
                            continue;
                        }
                        const auto s = static_cast<size_t>(skv);
                        const ComputeDataType dsVal = dropoutFactor * probs[s] * (dP[s] - rowD);
                        // The bias is added after scale and softcap, so dBias is the unscaled dS;
                        // Q and K also go through the softcap derivative and the scale.
                        const ComputeDataType dsScaled = dsVal * softcapGrad[s] * scale;
                        if(dBias != nullptr)
                        {
                            dBiasAccum[dBiasIdx(computeMaskIndex(attnMask->dims(), b, hQ, sq, skv))]
                                += dsVal;
                        }
                        for(int64_t d = 0; d < headDim; ++d)
                        {
                            dqAccum[dqIdx(b, hQ, sq, d)]
                                += dsScaled
                                   * static_cast<ComputeDataType>(
                                       k.getHostValue(std::vector<int64_t>{b, kvHeadK, skv, d}));
                            dkAccum[dkIdx(b, kvHeadK, skv, d)]
                                += dsScaled
                                   * static_cast<ComputeDataType>(
                                       q.getHostValue(std::vector<int64_t>{b, hQ, sq, d}));
                        }
                        const auto pKeep
                            = probs[s]
                              * dropoutKeep(options.dropoutMask, dropoutFactor, b, hQ, sq, skv);
                        for(int64_t dv = 0; dv < headDimV; ++dv)
                        {
                            dvAccum[dvIdx(b, kvHeadV, skv, dv)]
                                += pKeep
                                   * static_cast<ComputeDataType>(
                                       dO.getHostValue(std::vector<int64_t>{b, hQ, sq, dv}));
                        }
                    }
                }
            }
        }

        storeAccumulated(dqAccum, dQ);
        storeAccumulated(dkAccum, dK);
        storeAccumulated(dvAccum, dV);
        if(dBias != nullptr)
        {
            storeAccumulated(dBiasAccum, *dBias);
        }
    }

    /// cuDNN's ALiBi slope for query head h of numHeads (cudnn-frontend get_alibi_slope and
    /// fp16_ref.py): 2^(-8 (h + 1) / n) for the closest power of two n <= numHeads, then the
    /// interleaved 2^(-4 (2 j + 1) / n) slopes for the remaining heads.
    static double alibiSlope(int64_t h, int64_t numHeads)
    {
        int64_t n = 1;
        while(n * 2 <= numHeads)
        {
            n *= 2;
        }
        if(h < n)
        {
            return std::pow(2.0, -8.0 * static_cast<double>(h + 1) / static_cast<double>(n));
        }
        const auto j = h - n;
        return std::pow(2.0, -4.0 * static_cast<double>(2 * j + 1) / static_cast<double>(n));
    }

private:
    template <class T>
    static void requireRank4(const hipdnn_data_sdk::utilities::TensorBase<T>& t, const char* name)
    {
        if(t.dims().size() != 4)
        {
            throw std::invalid_argument(std::string("CpuFpReferenceSdpa: ") + name
                                        + " must be rank-4");
        }
    }

    /// LSE, Max and Sum_exp are [B, H, Sq, 1].
    static void validateRowStats(const hipdnn_data_sdk::utilities::TensorBase<float>* stats,
                                 const char* name,
                                 int64_t batch,
                                 int64_t numHeads,
                                 int64_t seqQ)
    {
        if(stats == nullptr)
        {
            return;
        }
        const auto& d = stats->dims();
        if(d.size() != 4 || d[0] != batch || d[1] != numHeads || d[2] != seqQ || d[3] != 1)
        {
            throw std::invalid_argument(std::string("CpuFpReferenceSdpa: ") + name
                                        + " shape must be [" + std::to_string(batch) + ", "
                                        + std::to_string(numHeads) + ", " + std::to_string(seqQ)
                                        + ", 1]");
        }
    }

    /// The score pipeline forward and backward share. For each live key skv of row (b, h, sq):
    /// s = (q . k) * scale, then softcap, + bias, + ALiBi; keys the diagonal band or the forward's
    /// block mask masks, and keys at or past liveKv, are -inf. softcapGrad, when given, receives d(softcap)/d(s) at each key
    /// the band keeps (the caller initialises it to 1).
    template <class QDataType, class KDataType, class ComputeDataType, class OptionsT>
    static void scoreRow(const hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                         const hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                         const OptionsT& options,
                         int64_t b,
                         int64_t h,
                         int64_t kvHeadK,
                         int64_t sq,
                         int64_t liveQ,
                         int64_t liveKv,
                         ComputeDataType scale,
                         std::vector<ComputeDataType>& scores,
                         std::vector<typename NonDeduced<ComputeDataType>::type>* softcapGrad)
    {
        std::fill(scores.begin(), scores.end(), -std::numeric_limits<ComputeDataType>::infinity());
        const auto headDim = q.dims()[3];
        const auto slope = options.alibi ? static_cast<ComputeDataType>(alibiSlope(h, q.dims()[1]))
                                         : static_cast<ComputeDataType>(0.0);
        const auto* blockMask = blockMaskOf(options);
        for(int64_t skv = 0; skv < liveKv; ++skv)
        {
            if(isMasked(sq,
                        skv,
                        liveQ,
                        liveKv,
                        options.leftBound,
                        options.rightBound,
                        options.topLeftAlignment)
               || isBlockMasked(blockMask, b, h, sq, skv))
            {
                continue;
            }
            auto dot = static_cast<ComputeDataType>(0.0);
            for(int64_t d = 0; d < headDim; ++d)
            {
                dot += static_cast<ComputeDataType>(
                           q.getHostValue(std::vector<int64_t>{b, h, sq, d}))
                       * static_cast<ComputeDataType>(
                           k.getHostValue(std::vector<int64_t>{b, kvHeadK, skv, d}));
            }
            auto s = applySoftcap(dot * scale, options.softcap);
            if(softcapGrad != nullptr)
            {
                (*softcapGrad)[static_cast<size_t>(skv)] = softcapDerivative(s, options.softcap);
            }
            if(options.attnMask != nullptr)
            {
                s += static_cast<ComputeDataType>(options.attnMask->getHostValue(
                    computeMaskIndex(options.attnMask->dims(), b, h, sq, skv)));
            }
            if(options.alibi)
            {
                s += static_cast<ComputeDataType>(skv - sq) * slope;
            }
            scores[static_cast<size_t>(skv)] = s;
        }
    }

    /// Numerically stable softmax of one score row with an optional sink logit (-inf for none):
    /// the sink joins the max and the denominator but has no value row. Writes P (or exp(s - max)
    /// when !normalize) into probs (left untouched for a row with no live key and no sink) and
    /// returns {max, sum of exp}.
    template <class ComputeDataType>
    static std::pair<ComputeDataType, ComputeDataType>
        softmaxRow(const std::vector<ComputeDataType>& scores,
                   ComputeDataType sink,
                   std::vector<ComputeDataType>& probs,
                   bool normalize = true)
    {
        const auto negInf = -std::numeric_limits<ComputeDataType>::infinity();
        const auto maxVal = std::max(*std::max_element(scores.begin(), scores.end()), sink);
        auto sumExp = sink != negInf ? std::exp(sink - maxVal) : static_cast<ComputeDataType>(0.0);
        if(maxVal != negInf)
        {
            for(size_t i = 0; i < scores.size(); ++i)
            {
                probs[i] = std::exp(scores[i] - maxVal);
                sumExp += probs[i];
            }
            for(auto& p : probs)
            {
                p = normalize ? p / sumExp : p;
            }
        }
        return {maxVal, sumExp};
    }

    /// cuDNN's block-sparse tile size: each block-mask bit covers BLOCK_MASK_TILE x BLOCK_MASK_TILE.
    static constexpr int64_t BLOCK_MASK_TILE = 128;

    template <class MaskDataType>
    static const hipdnn_data_sdk::utilities::TensorBase<uint8_t>*
        blockMaskOf(const SdpaFwdOptions<MaskDataType>& options)
    {
        return options.blockMask;
    }
    // The backward has no block mask (cuDNN's backward node takes none).
    template <class MaskDataType>
    static const hipdnn_data_sdk::utilities::TensorBase<uint8_t>*
        blockMaskOf(const SdpaBwdOptions<MaskDataType>& /*options*/)
    {
        return nullptr;
    }

    static void validateBlockMask(const hipdnn_data_sdk::utilities::TensorBase<uint8_t>* mask,
                                  int64_t batch,
                                  int64_t numHeads,
                                  int64_t seqQ,
                                  int64_t seqKv)
    {
        if(mask == nullptr)
        {
            return;
        }
        const auto tilesKv = (seqKv + BLOCK_MASK_TILE - 1) / BLOCK_MASK_TILE;
        const std::vector<int64_t> want
            = {batch, numHeads, (seqQ + BLOCK_MASK_TILE - 1) / BLOCK_MASK_TILE, (tilesKv + 7) / 8};
        if(mask->dims() != want)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa: blockMask must be UINT8 [B, H, ceil(Sq / 128), "
                "ceil(ceil(Skv / 128) / 8)]");
        }
    }

    /// Whether the block mask clears the tile holding score (sq, skv).
    static bool isBlockMasked(const hipdnn_data_sdk::utilities::TensorBase<uint8_t>* mask,
                              int64_t b,
                              int64_t h,
                              int64_t sq,
                              int64_t skv)
    {
        if(mask == nullptr)
        {
            return false;
        }
        const auto tileKv = skv / BLOCK_MASK_TILE;
        const auto byte
            = mask->getHostValue(std::vector<int64_t>{b, h, sq / BLOCK_MASK_TILE, tileKv / 8});
        return ((static_cast<unsigned>(byte) >> static_cast<unsigned>(tileKv % 8)) & 1U) == 0;
    }

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
        else
        {
            return SdpaFp8Format::NONE;
        }
    }

    /// x rounded to the nearest value of an fp8 format (RNE, saturating) and back to float.
    template <class ComputeDataType>
    static ComputeDataType roundToFp8(ComputeDataType x, SdpaFp8Format format)
    {
        const auto f = static_cast<float>(x);
        return static_cast<ComputeDataType>(
            format == SdpaFp8Format::E4M3_FNUZ
                ? static_cast<float>(hipdnn_data_sdk::types::fp8_e4m3_fnuz(f))
                : static_cast<float>(hipdnn_data_sdk::types::fp8_e4m3(f)));
    }

    static void requireSingleElement(const hipdnn_data_sdk::utilities::TensorBase<float>* t,
                                     const char* name)
    {
        if(t != nullptr && t->elementCount() != 1)
        {
            throw std::invalid_argument(std::string("CpuFpReferenceSdpa: ") + name
                                        + " must have one element");
        }
    }

    static void writeAmax(hipdnn_data_sdk::utilities::TensorBase<float>* amax, float value)
    {
        if(amax != nullptr)
        {
            amax->setHostValue(value, std::vector<int64_t>(amax->dims().size(), 0));
            amax->memory().markHostModified();
        }
    }

    /// LSE = max + log(sum of exp) (-inf for a row with no live key), Max and Sum_exp (cuDNN).
    template <class MaskDataType, class ComputeDataType>
    static void writeRowStats(const SdpaFwdOptions<MaskDataType>& options,
                              const std::vector<int64_t>& index,
                              ComputeDataType maxVal,
                              ComputeDataType sumExp)
    {
        if(options.lse != nullptr)
        {
            options.lse->setHostValue(static_cast<float>(maxVal + std::log(sumExp)), index);
        }
        if(options.rowMax != nullptr)
        {
            options.rowMax->setHostValue(static_cast<float>(maxVal), index);
        }
        if(options.rowSumExp != nullptr)
        {
            options.rowSumExp->setHostValue(static_cast<float>(sumExp), index);
        }
    }

    /// Writes a row-major accumulator into a tensor of the same dims, converting once.
    template <class ComputeDataType, class T>
    static void storeAccumulated(const std::vector<ComputeDataType>& accum,
                                 hipdnn_data_sdk::utilities::TensorBase<T>& out)
    {
        const auto& dims = out.dims();
        std::vector<int64_t> idx(dims.size(), 0);
        for(const auto value : accum)
        {
            out.setHostValue(hipdnn_test_sdk::detail::safeConvert<T>(value), idx);
            for(auto d = static_cast<int64_t>(dims.size()) - 1; d >= 0; --d)
            {
                const auto du = static_cast<size_t>(d);
                if(++idx[du] < dims[du])
                {
                    break;
                }
                idx[du] = 0;
            }
        }
        out.memory().markHostModified();
    }

    /// fp8 dequantization: x * descale, with a scalar or per-(batch, KV-head) [B, H_kv, 1, 1]
    /// descale; head h of x uses KV head h / headsPerKvHead. A null descale leaves x unchanged.
    template <class ComputeDataType, class T>
    static hipdnn_data_sdk::utilities::Tensor<ComputeDataType>
        dequantize(const hipdnn_data_sdk::utilities::TensorBase<T>& x,
                   const hipdnn_data_sdk::utilities::TensorBase<float>* descale,
                   int64_t headsPerKvHead)
    {
        const auto& d = x.dims();
        hipdnn_data_sdk::utilities::Tensor<ComputeDataType> out(d);
        for(int64_t b = 0; b < d[0]; ++b)
        {
            for(int64_t h = 0; h < d[1]; ++h)
            {
                const auto factor = static_cast<ComputeDataType>(
                    getDescaleFactor(descale, b, h / headsPerKvHead));
                for(int64_t s = 0; s < d[2]; ++s)
                {
                    for(int64_t e = 0; e < d[3]; ++e)
                    {
                        out.setHostValue(static_cast<ComputeDataType>(
                                             x.getHostValue(std::vector<int64_t>{b, h, s, e}))
                                             * factor,
                                         std::vector<int64_t>{b, h, s, e});
                    }
                }
            }
        }
        return out;
    }

    static void validateSoftcap(std::optional<float> softcap)
    {
        if(softcap.has_value() && !(softcap.value() > 0.0F && std::isfinite(softcap.value())))
        {
            throw std::invalid_argument("CpuFpReferenceSdpa: softcap must be finite and > 0");
        }
    }

    /// Logit softcap: c * tanh(s / c), applied after the attention scale and before bias and masks
    /// (vLLM / FlexAttention order). Not expressible in the hipDNN graph yet (register SEM-020).
    template <class ComputeDataType>
    static ComputeDataType applySoftcap(ComputeDataType score, std::optional<float> softcap)
    {
        if(!softcap.has_value())
        {
            return score;
        }
        const auto c = static_cast<ComputeDataType>(softcap.value());
        return c * std::tanh(score / c);
    }

    /// Derivative of the softcap at a capped score s' = c * tanh(s / c): 1 - (s' / c)^2.
    template <class ComputeDataType>
    static ComputeDataType softcapDerivative(ComputeDataType capped, std::optional<float> softcap)
    {
        if(!softcap.has_value())
        {
            return static_cast<ComputeDataType>(1.0);
        }
        const auto t = capped / static_cast<ComputeDataType>(softcap.value());
        return static_cast<ComputeDataType>(1.0) - t * t;
    }

    /// Explicit dropout mask (cuDNN custom dropout): rank-4 [B or 1, H or 1, Sq, Skv].
    static void validateDropoutMask(const hipdnn_data_sdk::utilities::TensorBase<float>* mask,
                                    int64_t batch,
                                    int64_t heads,
                                    int64_t seqQ,
                                    int64_t seqKv)
    {
        if(mask == nullptr)
        {
            return;
        }
        const auto& d = mask->dims();
        if(d.size() != 4 || (d[0] != batch && d[0] != 1) || (d[1] != heads && d[1] != 1)
           || d[2] != seqQ || d[3] != seqKv)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa: dropoutMask must be [B or 1, H or 1, Sq, Skv]");
        }
    }

    /// Dropout factor for one (b, h, sq, skv): scale * mask, or 1 without dropout.
    template <class ComputeDataType>
    static ComputeDataType dropoutKeep(const hipdnn_data_sdk::utilities::TensorBase<float>* mask,
                                       ComputeDataType scale,
                                       int64_t b,
                                       int64_t h,
                                       int64_t sq,
                                       int64_t skv)
    {
        if(mask == nullptr)
        {
            return static_cast<ComputeDataType>(1.0);
        }
        const auto& d = mask->dims();
        return scale
               * static_cast<ComputeDataType>(mask->getHostValue(
                   std::vector<int64_t>{d[0] == 1 ? 0 : b, d[1] == 1 ? 0 : h, sq, skv}));
    }

    /// Per-batch valid sequence lengths. With no seq_len tensor every batch uses the full
    /// length. The two seq_len tensors must be given together (cuDNN: padding needs both); each
    /// holds one INT32 value per batch, and every value must lie in [0, maxLen].
    static std::vector<int64_t>
        resolveSeqLens(const hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLen,
                       const hipdnn_data_sdk::utilities::TensorBase<int32_t>* otherSeqLen,
                       int64_t batch,
                       int64_t maxLen,
                       const char* name)
    {
        if((seqLen == nullptr) != (otherSeqLen == nullptr))
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpa: seqLenQ and seqLenKv must be given together");
        }
        std::vector<int64_t> lens(static_cast<size_t>(batch), maxLen);
        if(seqLen == nullptr)
        {
            return lens;
        }
        if(static_cast<int64_t>(seqLen->elementCount()) != batch || seqLen->dims().empty()
           || seqLen->dims()[0] != batch)
        {
            throw std::invalid_argument(std::string("CpuFpReferenceSdpa: ") + name
                                        + " must hold one value per batch ([B, 1, 1, 1])");
        }
        for(int64_t b = 0; b < batch; ++b)
        {
            std::vector<int64_t> idx(seqLen->dims().size(), 0);
            idx[0] = b;
            const auto len = static_cast<int64_t>(seqLen->getHostValue(idx));
            if(len < 0 || len > maxLen)
            {
                throw std::invalid_argument(std::string("CpuFpReferenceSdpa: ") + name + "["
                                            + std::to_string(b) + "] = " + std::to_string(len)
                                            + " is outside [0, " + std::to_string(maxLen) + "]");
            }
            lens[static_cast<size_t>(b)] = len;
        }
        return lens;
    }

    /// Returns true if position (sq, skv) should be masked given the window bounds.
    /// leftBound = -1 means no left mask; rightBound = -1 means no right mask.
    /// cuDNN's band: leftBound = L keeps L keys up to and including the diagonal
    /// (skv > sq + offset - L); rightBound = R keeps R keys past it (SdpaSemanticsRegister SEM-001).
    /// topLeftAlignment: if true, diagonal at skv==sq; if false, diagonal at
    /// skv==sq+(seqKv-seqQ).
    static bool isMasked(int64_t sq,
                         int64_t skv,
                         int64_t seqQ,
                         int64_t seqKv,
                         int64_t leftBound,
                         int64_t rightBound,
                         bool topLeftAlignment)
    {
        const int64_t offset = topLeftAlignment ? 0 : seqKv - seqQ;
        if(rightBound >= 0 && skv >= sq + 1 + offset + rightBound)
        {
            return true;
        }
        if(leftBound >= 0 && skv <= sq + offset - leftBound)
        {
            return true;
        }
        return false;
    }

    /// Returns true if a descale tensor is a per-tensor scalar (its shape collapses
    /// to a single element, e.g. [1] or [1, 1]).
    static bool isScalarDescale(const hipdnn_data_sdk::utilities::TensorBase<float>& descale)
    {
        int64_t numElements = 1;
        for(const auto dim : descale.dims())
        {
            numElements *= dim;
        }
        return numElements == 1;
    }

    /// Validate an FP8 descale tensor's shape on the calling thread. A descale tensor is
    /// either a per-tensor scalar ([1] or [1, 1, 1, 1]) or a per-(batch, KV-head) tensor.
    /// AITER's fp8 forward contract uses a 2-D [B, H_kv] descale, but hipDNN follows an
    /// equal-rank convention for per-channel/broadcast scale operands (e.g. batchnorm's
    /// [1, C, 1, 1]), so the per-head descale is shaped [B, H_kv, 1, 1]. A null pointer is
    /// a no-op. Throwing here (rather than inside the parallel region) keeps the error
    /// catchable by the caller, and rejecting a mismatched shape prevents getDescaleFactor()
    /// from silently reading the wrong (or out-of-bounds) element.
    static void validateDescaleShape(const hipdnn_data_sdk::utilities::TensorBase<float>* descale,
                                     const char* name,
                                     int64_t batch,
                                     int64_t numHeadsKv)
    {
        if(descale == nullptr)
        {
            return;
        }
        if(isScalarDescale(*descale))
        {
            return;
        }
        const auto& dims = descale->dims();
        if(dims.size() != 4 || dims[0] != batch || dims[1] != numHeadsKv || dims[2] != 1
           || dims[3] != 1)
        {
            throw std::invalid_argument(
                std::string("CpuFpReferenceSdpa: ") + name
                + " descale tensor must be a per-tensor scalar ([1]) or rank-4 [B, H_kv, 1, 1] = ["
                + std::to_string(batch) + ", " + std::to_string(numHeadsKv) + ", 1, 1]");
        }
    }

    /// Look up an FP8 descale factor for a given batch and KV head. Assumes the
    /// tensor shape was already validated by validateDescaleShape(). A null pointer
    /// yields a neutral factor of 1 (no dequantization), so the BF16/FP16 paths are
    /// unaffected.
    static float getDescaleFactor(const hipdnn_data_sdk::utilities::TensorBase<float>* descale,
                                  int64_t batch,
                                  int64_t kvHead)
    {
        if(descale == nullptr)
        {
            return 1.0f;
        }
        // Per-tensor scalar: shape collapses to a single element (e.g. [1] or [1,1]).
        if(isScalarDescale(*descale))
        {
            return descale->getHostValue(std::vector<int64_t>(descale->dims().size(), 0));
        }
        // Per-(batch, KV-head): [B, H_kv, 1, 1].
        return descale->getHostValue(std::vector<int64_t>{batch, kvHead, 0, 0});
    }

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
