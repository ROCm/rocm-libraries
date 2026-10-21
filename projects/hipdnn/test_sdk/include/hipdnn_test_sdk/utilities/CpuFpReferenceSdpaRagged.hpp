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

// Ragged forward SDPA CPU reference (RFC-0014: packed BSHD + ragged_offset), the host
// mirror of GpuFpReferenceSdpaRagged. It computes in plain fp32 with no provider P-storage
// rounding, so GPU-vs-CPU checks use gpuRefFwdTolerance.
//
// q/k/v/o are ragged tensors (ShallowRaggedTensor / RaggedTensor) with logical dims [B, H, S, D],
// BSHD strides and the sequence at SDPA_SEQ_AXIS. The SDK does the packed addressing:
// getHostValue({b, h, s, d}) starts at ragged_offset[b], and per-batch lengths come from
// raggedIterationInfo().
//
// Supports GQA/MQA, causal and sliding-window masks, fp8 descales, and an optional LSE that is
// ragged or dense [B, H, Sq_max, 1]. No bias, alibi or dropout, as on the ASM v3 path.
// Descales are scalar [1] or per KV head [B, H_kv, 1, 1]. Q and K descales are both indexed by
// the KV head of the query head, as in CpuFpReferenceSdpa and AITER.
//
// Q/O (and a ragged LSE) must have the same per-batch lengths, and so must K/V. Their element
// offsets can differ because their token widths differ.
//
// forwardPaged() is the paged K/V variant (cuDNN paged attention, vLLM block tables): K/V are
// block pools addressed through per-batch page tables, Q/O are dense or ragged, and the per-batch
// lengths come from seq_len tensors. It shares the softmax with forward().
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
        const auto numHeads = q.dims()[1];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto headsPerHeadK = numHeads / numHeadsK;
        const auto headsPerHeadV = numHeads / numHeadsV;

        // Every ragged tensor must use the RFC-0014 layout, and tensors that share a packing must
        // agree on every batch's length.
        const auto qTokens = raggedTokens(*qInfo, q.dims(), q.strides(), "Q");
        const auto kTokens = raggedTokens(*kInfo, k.dims(), k.strides(), "K");
        detail::requireMatchingTokenBoundaries(
            qTokens, "Q", raggedTokens(*oInfo, o.dims(), o.strides(), "O"), "O", WHO);
        detail::requireMatchingTokenBoundaries(
            kTokens, "K", raggedTokens(*vInfo, v.dims(), v.strides(), "V"), "V", WHO);

        if(lse != nullptr)
        {
            validateLse(lse->dims(), q.dims());
            if(const auto lseInfo = lse->raggedIterationInfo())
            {
                detail::requireMatchingTokenBoundaries(
                    qTokens,
                    "Q",
                    raggedTokens(*lseInfo, lse->dims(), lse->strides(), "LSE"),
                    "LSE",
                    WHO);
            }
        }

        // Q descale is per KV head, like K (AITER's [B, H_kv] contract).
        const DescaleBinding dq = bindDescale(descaleQ, batch, numHeadsK, "Q");
        const DescaleBinding dk = bindDescale(descaleK, batch, numHeadsK, "K");
        const DescaleBinding dv = bindDescale(descaleV, batch, numHeadsV, "V");

        // An absent scale is 1.0 (no scaling), as in cuDNN and the dense reference.
        HeadParams params;
        params.scale = attnScaleValue.value_or(1.0F);
        params.leftBound = leftBound;
        params.rightBound = rightBound;

        for(int64_t b = 0; b < batch; ++b)
        {
            const auto bIdx = static_cast<size_t>(b);
            params.batch = b;
            params.seqQ = qTokens[bIdx + 1] - qTokens[bIdx];
            params.seqKv = kTokens[bIdx + 1] - kTokens[bIdx];
            params.windowOffset = topLeftAlignment ? 0 : (params.seqKv - params.seqQ);

            for(int64_t h = 0; h < numHeads; ++h)
            {
                const int64_t kvHeadK = h / headsPerHeadK;
                const int64_t kvHeadV = h / headsPerHeadV;
                params.head = h;
                params.descaleQK = dq.value(b, kvHeadK) * dk.value(b, kvHeadK);
                params.descaleV = dv.value(b, kvHeadV);

                const auto kAt = [&](int64_t skv, int64_t d) {
                    return static_cast<ComputeDataType>(
                        k.getHostValue(std::vector<int64_t>{b, kvHeadK, skv, d}));
                };
                const auto vAt = [&](int64_t skv, int64_t dvIdx) {
                    return static_cast<ComputeDataType>(
                        v.getHostValue(std::vector<int64_t>{b, kvHeadV, skv, dvIdx}));
                };
                attendHead<ComputeDataType>(q, o, lse, params, kAt, vAt);
            }
        }

        o.memory().markHostModified();
        if(lse != nullptr)
        {
            lse->memory().markHostModified();
        }
    }

    // Paged K/V forward. kContainer / vContainer are block pools [num_blocks, H_kv, block_size, D]
    // (num_blocks may differ between K and V) with any strides. Logical key row skv of batch b is
    // row skv % block_size of block pageTable[b, 0, skv / block_size, 0]. The page tables are INT32
    // [B, 1, M, 1] with any strides and may be the same tensor. Only the first
    // ceil(seqLenKv[b] / block_size) entries of a row are read, so unused entries may hold anything
    // (e.g. -1); each used entry must be in [0, num_blocks).
    //
    // q is dense [B, H, Sq, D] or ragged (RFC-0014, detected by raggedIterationInfo()); o has q's
    // layout kind and token boundaries, with V's head dim. seqLenKv / seqLenQ are INT32
    // [B, 1, 1, 1]. Batch b attends seqQ[b] = seqLenQ[b] query rows (Sq when seqLenQ is null, which
    // only a dense q allows) against seqKv[b] = seqLenKv[b] <= maxSeqLenKv keys. Rows past seqQ[b]
    // (dense padding rows or ragged gap rows) are not written. A live row that is fully masked,
    // including seqKv[b] == 0, gets zero output and LSE = -inf.
    //
    // Mask alignment uses the per-batch lengths. GQA, descales and LSE ([B, H, Sq, 1], dense or
    // ragged like q) are as in forward(). Every input is validated before any output is written.
    template <class QDataType,
              class KDataType = QDataType,
              class VDataType = QDataType,
              class ODataType = QDataType,
              class ComputeDataType = float>
    static void forwardPaged(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                             hipdnn_data_sdk::utilities::TensorBase<KDataType>& kContainer,
                             hipdnn_data_sdk::utilities::TensorBase<VDataType>& vContainer,
                             hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                             hipdnn_data_sdk::utilities::TensorBase<int32_t>& pageTableK,
                             hipdnn_data_sdk::utilities::TensorBase<int32_t>& pageTableV,
                             hipdnn_data_sdk::utilities::TensorBase<int32_t>& seqLenKv,
                             hipdnn_data_sdk::utilities::TensorBase<int32_t>* seqLenQ,
                             int64_t maxSeqLenKv,
                             std::optional<float> attnScaleValue = std::nullopt,
                             int64_t leftBound = -1,
                             int64_t rightBound = -1,
                             bool topLeftAlignment = true,
                             hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr,
                             hipdnn_data_sdk::utilities::TensorBase<float>* descaleQ = nullptr,
                             hipdnn_data_sdk::utilities::TensorBase<float>* descaleK = nullptr,
                             hipdnn_data_sdk::utilities::TensorBase<float>* descaleV = nullptr)
    {
        validatePagedInput(q.dims(), kContainer.dims(), vContainer.dims(), o.dims());

        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[1];
        const auto numHeadsK = kContainer.dims()[1];
        const auto numHeadsV = vContainer.dims()[1];
        const auto blockSize = kContainer.dims()[2];
        const auto headsPerHeadK = numHeads / numHeadsK;
        const auto headsPerHeadV = numHeads / numHeadsV;

        // Q and O are both dense, or both ragged with the same token boundaries.
        const auto qInfo = q.raggedIterationInfo();
        const auto oInfo = o.raggedIterationInfo();
        if(qInfo.has_value() != oInfo.has_value())
        {
            throw std::invalid_argument(std::string(WHO)
                                        + ": o must be ragged exactly when q is ragged");
        }
        std::vector<int64_t> qTokens;
        if(qInfo.has_value())
        {
            qTokens = raggedTokens(*qInfo, q.dims(), q.strides(), "Q");
            detail::requireMatchingTokenBoundaries(
                qTokens, "Q", raggedTokens(*oInfo, o.dims(), o.strides(), "O"), "O", WHO);
            if(seqLenQ == nullptr)
            {
                throw std::invalid_argument(std::string(WHO)
                                            + ": seq_len_q is required for a ragged q");
            }
        }

        if(lse != nullptr)
        {
            validateLse(lse->dims(), q.dims());
            if(const auto lseInfo = lse->raggedIterationInfo())
            {
                if(!qInfo.has_value())
                {
                    throw std::invalid_argument(std::string(WHO)
                                                + ": a ragged lse needs a ragged q");
                }
                detail::requireMatchingTokenBoundaries(
                    qTokens,
                    "Q",
                    raggedTokens(*lseInfo, lse->dims(), lse->strides(), "LSE"),
                    "LSE",
                    WHO);
            }
        }

        if(maxSeqLenKv < 0)
        {
            throw std::invalid_argument(std::string(WHO) + ": max_seq_len_kv must be >= 0");
        }
        const int64_t maxBlocks = pageTableWidth(pageTableK.dims(), batch, "page_table_k");
        if(pageTableWidth(pageTableV.dims(), batch, "page_table_v") != maxBlocks)
        {
            throw std::invalid_argument(std::string(WHO)
                                        + ": page_table_k and page_table_v widths must match");
        }
        validateSeqLen(seqLenKv.dims(), batch, "seq_len_kv");
        if(seqLenQ != nullptr)
        {
            validateSeqLen(seqLenQ->dims(), batch, "seq_len_q");
        }

        const DescaleBinding dq = bindDescale(descaleQ, batch, numHeadsK, "Q");
        const DescaleBinding dk = bindDescale(descaleK, batch, numHeadsK, "K");
        const DescaleBinding dv = bindDescale(descaleV, batch, numHeadsV, "V");

        // Resolve and check every batch's lengths and used blocks before writing anything.
        const auto batchCount = static_cast<size_t>(batch);
        std::vector<int64_t> seqQ(batchCount);
        std::vector<int64_t> seqKv(batchCount);
        std::vector<std::vector<int64_t>> blocksK(batchCount);
        std::vector<std::vector<int64_t>> blocksV(batchCount);
        for(int64_t b = 0; b < batch; ++b)
        {
            const auto bIdx = static_cast<size_t>(b);
            const std::vector<int64_t> lengthIndex = {b, 0, 0, 0};

            const int64_t extentQ
                = qInfo.has_value() ? qTokens[bIdx + 1] - qTokens[bIdx] : q.dims()[2];
            seqQ[bIdx] = seqLenQ != nullptr ? seqLenQ->getHostValue(lengthIndex) : extentQ;
            if(seqQ[bIdx] < 0 || seqQ[bIdx] > extentQ)
            {
                throw std::invalid_argument(std::string(WHO) + ": seq_len_q[" + std::to_string(b)
                                            + "] = " + std::to_string(seqQ[bIdx])
                                            + " is outside [0, " + std::to_string(extentQ)
                                            + "], the rows q holds for that batch");
            }

            seqKv[bIdx] = seqLenKv.getHostValue(lengthIndex);
            if(seqKv[bIdx] < 0 || seqKv[bIdx] > maxSeqLenKv)
            {
                throw std::invalid_argument(std::string(WHO) + ": seq_len_kv[" + std::to_string(b)
                                            + "] = " + std::to_string(seqKv[bIdx])
                                            + " is outside [0, max_seq_len_kv = "
                                            + std::to_string(maxSeqLenKv) + "]");
            }

            const int64_t usedBlocks = (seqKv[bIdx] + blockSize - 1) / blockSize;
            if(usedBlocks > maxBlocks)
            {
                throw std::invalid_argument(std::string(WHO) + ": seq_len_kv[" + std::to_string(b)
                                            + "] needs " + std::to_string(usedBlocks)
                                            + " blocks but the page tables hold "
                                            + std::to_string(maxBlocks));
            }
            blocksK[bIdx]
                = usedBlockIds(pageTableK, b, usedBlocks, kContainer.dims()[0], "page_table_k");
            blocksV[bIdx]
                = usedBlockIds(pageTableV, b, usedBlocks, vContainer.dims()[0], "page_table_v");
        }

        HeadParams params;
        params.scale = attnScaleValue.value_or(1.0F);
        params.leftBound = leftBound;
        params.rightBound = rightBound;

        for(int64_t b = 0; b < batch; ++b)
        {
            const auto bIdx = static_cast<size_t>(b);
            const auto& rowBlocksK = blocksK[bIdx];
            const auto& rowBlocksV = blocksV[bIdx];
            params.batch = b;
            params.seqQ = seqQ[bIdx];
            params.seqKv = seqKv[bIdx];
            params.windowOffset = topLeftAlignment ? 0 : (params.seqKv - params.seqQ);

            for(int64_t h = 0; h < numHeads; ++h)
            {
                const int64_t kvHeadK = h / headsPerHeadK;
                const int64_t kvHeadV = h / headsPerHeadV;
                params.head = h;
                params.descaleQK = dq.value(b, kvHeadK) * dk.value(b, kvHeadK);
                params.descaleV = dv.value(b, kvHeadV);

                const auto kAt = [&](int64_t skv, int64_t d) {
                    const int64_t block = rowBlocksK[static_cast<size_t>(skv / blockSize)];
                    return static_cast<ComputeDataType>(kContainer.getHostValue(
                        std::vector<int64_t>{block, kvHeadK, skv % blockSize, d}));
                };
                const auto vAt = [&](int64_t skv, int64_t dvIdx) {
                    const int64_t block = rowBlocksV[static_cast<size_t>(skv / blockSize)];
                    return static_cast<ComputeDataType>(vContainer.getHostValue(
                        std::vector<int64_t>{block, kvHeadV, skv % blockSize, dvIdx}));
                };
                attendHead<ComputeDataType>(q, o, lse, params, kAt, vAt);
            }
        }

        o.memory().markHostModified();
        if(lse != nullptr)
        {
            lse->memory().markHostModified();
        }
    }

private:
    static constexpr auto WHO = "CpuFpReferenceSdpaRagged";

    // Inputs of attendHead for one (batch, query head).
    struct HeadParams
    {
        int64_t batch = 0;
        int64_t head = 0;
        int64_t seqQ = 0;
        int64_t seqKv = 0;
        int64_t leftBound = -1;
        int64_t rightBound = -1;
        int64_t windowOffset = 0;
        float scale = 1.0F;
        float descaleQK = 1.0F; // Q descale * K descale
        float descaleV = 1.0F;
    };

    // Attention for query rows [0, seqQ) of (batch, head) over seqKv keys. kAt(skv, d) and
    // vAt(skv, dv) return K and V as ComputeDataType, so ragged and paged K/V share this softmax.
    // A fully masked row, including seqKv == 0, writes zero output and LSE = -inf. Rows at or past
    // seqQ are not written.
    template <class ComputeDataType, class QDataType, class ODataType, class KAt, class VAt>
    static void attendHead(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                           hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                           hipdnn_data_sdk::utilities::TensorBase<float>* lse,
                           const HeadParams& p,
                           const KAt& kAt,
                           const VAt& vAt)
    {
        const auto headDim = q.dims()[3];
        const auto headDimV = o.dims()[3];
        const auto b = p.batch;
        const auto h = p.head;
        const auto scale = static_cast<ComputeDataType>(p.scale);
        const auto descaleQK = static_cast<ComputeDataType>(p.descaleQK);
        const auto negInf = -std::numeric_limits<ComputeDataType>::infinity();

        for(int64_t sq = 0; sq < p.seqQ; ++sq)
        {
            std::vector<ComputeDataType> scores(static_cast<size_t>(p.seqKv));
            for(int64_t skv = 0; skv < p.seqKv; ++skv)
            {
                if(isMasked(sq, skv, p.leftBound, p.rightBound, p.windowOffset))
                {
                    scores[static_cast<size_t>(skv)] = negInf;
                    continue;
                }
                auto dot = static_cast<ComputeDataType>(0);
                for(int64_t d = 0; d < headDim; ++d)
                {
                    const auto qv = static_cast<ComputeDataType>(
                        q.getHostValue(std::vector<int64_t>{b, h, sq, d}));
                    dot += qv * kAt(skv, d);
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
            std::vector<ComputeDataType> probs(static_cast<size_t>(p.seqKv));
            for(int64_t skv = 0; skv < p.seqKv; ++skv)
            {
                const auto s = scores[static_cast<size_t>(skv)];
                const auto e
                    = (s == negInf) ? static_cast<ComputeDataType>(0) : std::exp(s - maxVal);
                probs[static_cast<size_t>(skv)] = e;
                sumExp += e;
            }
            for(auto& prob : probs)
            {
                prob /= sumExp;
            }

            // V descale is applied once, after accumulation.
            for(int64_t dvIdx = 0; dvIdx < headDimV; ++dvIdx)
            {
                auto acc = static_cast<ComputeDataType>(0);
                for(int64_t skv = 0; skv < p.seqKv; ++skv)
                {
                    acc += probs[static_cast<size_t>(skv)] * vAt(skv, dvIdx);
                }
                acc *= static_cast<ComputeDataType>(p.descaleV);
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

    // Token boundaries of a ragged SDPA tensor, which must be ragged along SDPA_SEQ_AXIS and use
    // the RFC-0014 token-major layout.
    static std::vector<int64_t>
        raggedTokens(const hipdnn_data_sdk::utilities::RaggedIterationInfo& info,
                     const std::vector<int64_t>& dims,
                     const std::vector<int64_t>& strides,
                     const char* name)
    {
        if(info.seqAxis != hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS)
        {
            throw std::invalid_argument(std::string(WHO) + ": " + name
                                        + " must be ragged along SDPA_SEQ_AXIS (2), got "
                                        + std::to_string(info.seqAxis));
        }
        detail::requireTokenMajorRaggedLayout(dims, strides, WHO, name);
        return detail::raggedTokenBoundaries(info.rowOffsets,
                                             info.seqStride,
                                             dims[hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS],
                                             WHO,
                                             name);
    }

    // Shapes for forwardPaged: q [B, H, Sq, D], K/V containers [num_blocks, H_kv, block_size, D]
    // with one block size, o [B, H, Sq, Dv].
    static void validatePagedInput(const std::vector<int64_t>& qDims,
                                   const std::vector<int64_t>& kDims,
                                   const std::vector<int64_t>& vDims,
                                   const std::vector<int64_t>& oDims)
    {
        if(qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4)
        {
            throw std::invalid_argument(std::string(WHO)
                                        + ": q, o and the K/V containers must all be rank-4");
        }
        // Sq (S_max for a ragged q) may be 0; the other dims may not.
        if(qDims[0] <= 0 || qDims[1] <= 0 || qDims[2] < 0 || qDims[3] <= 0 || kDims[0] <= 0
           || kDims[1] <= 0 || kDims[2] <= 0 || vDims[0] <= 0 || vDims[1] <= 0 || vDims[3] <= 0)
        {
            throw std::invalid_argument(std::string(WHO) + ": all dimensions must be positive");
        }
        if(kDims[3] != qDims[3])
        {
            throw std::invalid_argument(std::string(WHO) + ": Q head_dim != K head_dim");
        }
        if(vDims[2] != kDims[2])
        {
            throw std::invalid_argument(std::string(WHO)
                                        + ": K and V containers must share one block size");
        }
        const auto numHeads = qDims[1];
        if(numHeads % kDims[1] != 0 || numHeads % vDims[1] != 0)
        {
            throw std::invalid_argument(
                std::string(WHO) + ": numHeads must be divisible by numHeadsK and numHeadsV");
        }
        if(oDims[0] != qDims[0] || oDims[1] != numHeads || oDims[2] != qDims[2]
           || oDims[3] != vDims[3])
        {
            throw std::invalid_argument(std::string(WHO) + ": output shape must be [B, H, Sq, Dv]");
        }
    }

    // A page table is [B, 1, M, 1]. Returns M, the blocks per sequence.
    static int64_t pageTableWidth(const std::vector<int64_t>& dims, int64_t batch, const char* name)
    {
        if(dims.size() != 4 || dims[0] != batch || dims[1] != 1 || dims[3] != 1)
        {
            throw std::invalid_argument(std::string(WHO) + ": " + name
                                        + " must be [B, 1, max_blocks_per_seq, 1]");
        }
        return dims[2];
    }

    // seq_len_q / seq_len_kv are [B, 1, 1, 1].
    static void validateSeqLen(const std::vector<int64_t>& dims, int64_t batch, const char* name)
    {
        if(dims.size() != 4 || dims[0] != batch || dims[1] != 1 || dims[2] != 1 || dims[3] != 1)
        {
            throw std::invalid_argument(std::string(WHO) + ": " + name + " must be [B, 1, 1, 1]");
        }
    }

    // Block ids of entries [0, usedBlocks) of page-table row b. Each must be in [0, numBlocks).
    static std::vector<int64_t>
        usedBlockIds(const hipdnn_data_sdk::utilities::TensorBase<int32_t>& pageTable,
                     int64_t b,
                     int64_t usedBlocks,
                     int64_t numBlocks,
                     const char* name)
    {
        std::vector<int64_t> blocks;
        blocks.reserve(static_cast<size_t>(usedBlocks));
        for(int64_t j = 0; j < usedBlocks; ++j)
        {
            const int64_t block = pageTable.getHostValue(std::vector<int64_t>{b, 0, j, 0});
            if(block < 0 || block >= numBlocks)
            {
                throw std::invalid_argument(std::string(WHO) + ": " + name + "[" + std::to_string(b)
                                            + ", " + std::to_string(j)
                                            + "] = " + std::to_string(block) + " is outside [0, "
                                            + std::to_string(numBlocks) + ")");
            }
            blocks.push_back(block);
        }
        return blocks;
    }

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

    // LSE must be [B, H, Sq, 1] with Q's Sq. A shorter Sq would spill rows into the next batch.
    static void validateLse(const std::vector<int64_t>& lseDims, const std::vector<int64_t>& qDims)
    {
        if(lseDims.size() != 4 || lseDims[0] != qDims[0] || lseDims[1] != qDims[1]
           || lseDims[2] != qDims[2] || lseDims[3] != 1)
        {
            throw std::invalid_argument(
                "CpuFpReferenceSdpaRagged: lse must be rank-4 [B, H, Sq, 1] with Q's B, H, Sq");
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
        // S_max may be 0 (every batch empty); the other dims may not.
        if(batch <= 0 || qDims[1] <= 0 || qDims[3] <= 0 || kDims[1] <= 0 || vDims[1] <= 0
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
