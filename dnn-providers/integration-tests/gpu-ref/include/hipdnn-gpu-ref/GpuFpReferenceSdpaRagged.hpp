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
// Tensors are logical rank-4 [B, H, S, D] with the sequence at SDPA_SEQ_AXIS (2) and BSHD
// strides, packed token by token with no per-batch padding:
//   q = [B, H,  Sq,  D ]   k = [B, Hk, Skv, D ]
//   v = [B, Hv, Skv, Dv]   o = [B, H,  Sq,  Dv]
// raggedOffsetQ/K/V/O are INT32 [B+1, 1, 1, 1] cumulative offsets, scaled to elements by
// `offsetMultipliers`. Batch b starts at element offset[b] and has
// (offset[b+1] - offset[b]) / strides[2] tokens. O must match Q's token boundaries and V must
// match K's.
// The optional LSE is [B, H, Sq, 1], ragged if raggedOffsetLse is given, else dense.
// Softmax numerics match the dense reference. Supports GQA/MQA and per-batch causal and
// sliding window. No bias, alibi or dropout, since the ASM v3 path gates them off.
// fpropPaged runs the same kernel with paged K/V (vLLM block tables) and dense or ragged Q.
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
        const auto numHeads = q.dims()[1];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto headDimV = v.dims()[3];

        // Elements per token: H*D for Q, Hk*D for K.
        constexpr auto SEQ_AXIS = hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS;
        const auto seqStrideQ = q.strides()[SEQ_AXIS];
        const auto seqStrideKv = k.strides()[SEQ_AXIS];

        // Offsets may live only on the device (plan path), so read them back and check them here.
        // Every ragged tensor must use the RFC-0014 layout, and tensors that share a packing must
        // agree on every batch length.
        const std::string who = "GpuFpReferenceSdpaRagged";
        const auto tokenBoundaries = [&](hipdnn_data_sdk::utilities::TensorBase<int32_t>& offsets,
                                         int64_t multiplier,
                                         const std::vector<int64_t>& dims,
                                         const std::vector<int64_t>& strides,
                                         const char* name) {
            return readTokenBoundaries(offsets, multiplier, dims, strides, batch, who, name);
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
                            PagedKvLaunch{},
                            defines);

        o.memory().markDeviceModified();
        if(lse != nullptr)
        {
            lse->memory().markDeviceModified();
        }
    }

    // Paged-KV forward (vLLM block tables), with dense or ragged Q.
    //   q = [B, H, Sq, D] dense, or ragged (RFC-0014) when raggedOffsetQ is given
    //   kContainer = [num_blocks_k, Hk, block_size, D], vContainer = [num_blocks_v, Hv, block_size,
    //   Dv], any strides
    //   o = [B, H, Sq, Dv], same layout kind as q (raggedOffsetO non-null iff raggedOffsetQ)
    //   pageTableK/V = INT32 [B, 1, M, 1], any strides, M == ceil(maxSeqLenKv / block_size)
    //   seqLenKv = INT32 [B, 1, 1, 1]; seqLenQ = INT32 [B, 1, 1, 1], nullptr (Sq rows) for dense Q
    // Key row skv of batch b is container row skv % block_size of block pageTable[b, 0, skv /
    // block_size, 0]. Only the first ceil(seqLenKv[b] / block_size) entries are read; each must be
    // in [0, num_blocks). Rows sq >= seqQ[b] are left untouched; a live row with no unmasked key
    // (including seqLenKv[b] == 0) writes zeros and LSE -inf. The optional LSE is [B, H, Sq, 1],
    // ragged (with raggedOffsetLse, ragged Q only) or dense.
    template <class QDataType,
              class KDataType = QDataType,
              class VDataType = QDataType,
              class ODataType = QDataType,
              class ComputeDataType = float>
    static void
        fpropPaged(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                   hipdnn_data_sdk::utilities::TensorBase<KDataType>& kContainer,
                   hipdnn_data_sdk::utilities::TensorBase<VDataType>& vContainer,
                   hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                   hipdnn_data_sdk::utilities::TensorBase<int32_t>* raggedOffsetQ,
                   hipdnn_data_sdk::utilities::TensorBase<int32_t>* raggedOffsetO,
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
                   hipdnn_data_sdk::utilities::TensorBase<int32_t>* raggedOffsetLse = nullptr,
                   SdpaSoftmaxProbabilityMode probabilityMode = SdpaSoftmaxProbabilityMode::FLOAT,
                   hipdnn_data_sdk::utilities::TensorBase<float>* descaleQ = nullptr,
                   hipdnn_data_sdk::utilities::TensorBase<float>* descaleK = nullptr,
                   hipdnn_data_sdk::utilities::TensorBase<float>* descaleV = nullptr,
                   const RaggedOffsetMultipliers& offsetMultipliers = {})
    {
        static_assert(!std::is_same_v<ODataType, hipdnn_data_sdk::types::fp8_e4m3>,
                      "fpropPaged: fp8 output is not supported; pass a bf16 or float ODataType");
        const std::string who = "GpuFpReferenceSdpaRagged::fpropPaged";
        const auto fail
            = [&who](const std::string& msg) { throw std::invalid_argument(who + ": " + msg); };

        const auto& qDims = q.dims();
        const auto& kDims = kContainer.dims();
        const auto& vDims = vContainer.dims();
        const auto& oDims = o.dims();
        if(qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4)
        {
            fail("q/o must be rank-4 [B, H, Sq, D] and K/V containers rank-4 [num_blocks, H, "
                 "block_size, D]");
        }

        constexpr auto SEQ_AXIS = hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS;
        const auto batch = qDims[0];
        const auto numHeads = qDims[1];
        const auto seqQMax = qDims[SEQ_AXIS];
        const auto headDim = qDims[3];
        const auto numBlocksK = kDims[0];
        const auto numHeadsK = kDims[1];
        const auto blockSize = kDims[SEQ_AXIS];
        const auto numBlocksV = vDims[0];
        const auto numHeadsV = vDims[1];
        const auto headDimV = vDims[3];

        if(batch <= 0 || numHeads <= 0 || seqQMax <= 0 || headDim <= 0 || numBlocksK <= 0
           || numHeadsK <= 0 || blockSize <= 0 || numBlocksV <= 0 || numHeadsV <= 0
           || headDimV <= 0)
        {
            fail("all dimensions must be positive");
        }
        if(kDims[3] != headDim)
        {
            fail("Q head_dim != K head_dim");
        }
        if(vDims[SEQ_AXIS] != blockSize)
        {
            fail("K and V block sizes must match");
        }
        if(numHeads % numHeadsK != 0 || numHeads % numHeadsV != 0)
        {
            fail("numHeads must be divisible by numHeadsK and numHeadsV");
        }
        if(oDims[0] != batch || oDims[1] != numHeads || oDims[SEQ_AXIS] != seqQMax
           || oDims[3] != headDimV)
        {
            fail("output shape must be [B, H, Sq, Dv]");
        }
        if((raggedOffsetQ == nullptr) != (raggedOffsetO == nullptr))
        {
            fail("Q and O must both be ragged or both dense");
        }
        if(raggedOffsetQ != nullptr && seqLenQ == nullptr)
        {
            fail("ragged Q needs seq_len_q");
        }
        if(maxSeqLenKv < 1)
        {
            fail("max_seq_len_kv must be >= 1 (got " + std::to_string(maxSeqLenKv) + ")");
        }

        const int64_t maxBlocks = (maxSeqLenKv + blockSize - 1) / blockSize;
        const std::vector<int64_t> pageTableDims{batch, 1, maxBlocks, 1};
        if(pageTableK.dims() != pageTableDims || pageTableV.dims() != pageTableDims)
        {
            fail("page tables must be [B, 1, ceil(max_seq_len_kv / block_size), 1] = [B, 1, "
                 + std::to_string(maxBlocks) + ", 1]");
        }
        const std::vector<int64_t> seqLenDims{batch, 1, 1, 1};
        if(seqLenKv.dims() != seqLenDims || (seqLenQ != nullptr && seqLenQ->dims() != seqLenDims))
        {
            fail("seq_len_kv and seq_len_q must be [B, 1, 1, 1]");
        }

        // Q token boundaries: packed for ragged Q, Sq per batch for dense Q.
        const auto& mult = offsetMultipliers;
        const std::vector<int64_t> offsetDims{batch + 1, 1, 1, 1};
        std::vector<int64_t> qTokens;
        if(raggedOffsetQ != nullptr)
        {
            if(raggedOffsetQ->dims() != offsetDims || raggedOffsetO->dims() != offsetDims)
            {
                fail("raggedOffsetQ/O must be rank-4 [B+1, 1, 1, 1]");
            }
            qTokens
                = readTokenBoundaries(*raggedOffsetQ, mult.q, qDims, q.strides(), batch, who, "Q");
            hipdnn_test_sdk::detail::requireMatchingTokenBoundaries(
                qTokens,
                "Q",
                readTokenBoundaries(*raggedOffsetO, mult.o, oDims, o.strides(), batch, who, "O"),
                "O",
                who);
        }
        else
        {
            qTokens.resize(static_cast<size_t>(batch + 1));
            for(int64_t b = 0; b <= batch; ++b)
            {
                qTokens[static_cast<size_t>(b)] = b * seqQMax;
            }
        }
        const int64_t totalQ = qTokens.back();

        // Lengths and page tables may live only on the device (plan path): read them back and
        // check every entry the kernel will read.
        const auto seqLenKvHost = readInt32Span(seqLenKv, who, "seq_len_kv");
        const auto seqLenQHost = seqLenQ != nullptr ? readInt32Span(*seqLenQ, who, "seq_len_q")
                                                    : std::vector<int64_t>{};
        const auto pageTableKHost = readInt32Span(pageTableK, who, "page_table_k");
        const auto pageTableVHost = readInt32Span(pageTableV, who, "page_table_v");
        const auto& ptKStr = pageTableK.strides();
        const auto& ptVStr = pageTableV.strides();
        for(int64_t b = 0; b < batch; ++b)
        {
            const auto bIdx = static_cast<size_t>(b);
            const int64_t seqKv = seqLenKvHost[static_cast<size_t>(b * seqLenKv.strides()[0])];
            if(seqKv < 0 || seqKv > maxSeqLenKv)
            {
                fail("seq_len_kv[" + std::to_string(b) + "] = " + std::to_string(seqKv)
                     + " is outside [0, max_seq_len_kv = " + std::to_string(maxSeqLenKv) + "]");
            }
            const int64_t qExtent = qTokens[bIdx + 1] - qTokens[bIdx];
            if(seqLenQ != nullptr)
            {
                const int64_t seqQ = seqLenQHost[static_cast<size_t>(b * seqLenQ->strides()[0])];
                if(seqQ < 0 || seqQ > qExtent)
                {
                    fail("seq_len_q[" + std::to_string(b) + "] = " + std::to_string(seqQ)
                         + " is outside [0, " + std::to_string(qExtent) + "]");
                }
            }
            const int64_t usedBlocks = (seqKv + blockSize - 1) / blockSize;
            for(int64_t j = 0; j < usedBlocks; ++j)
            {
                const int64_t blockK
                    = pageTableKHost[static_cast<size_t>(b * ptKStr[0] + j * ptKStr[2])];
                const int64_t blockV
                    = pageTableVHost[static_cast<size_t>(b * ptVStr[0] + j * ptVStr[2])];
                if(blockK < 0 || blockK >= numBlocksK || blockV < 0 || blockV >= numBlocksV)
                {
                    fail("page table entry [" + std::to_string(b) + ", " + std::to_string(j)
                         + "] (k=" + std::to_string(blockK) + ", v=" + std::to_string(blockV)
                         + ") is outside [0, num_blocks)");
                }
            }
        }

        const float scale = attnScaleValue.value_or(1.0F);
        auto defines
            = detail::buildSdpaDefines<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                probabilityMode);

        void* lsePtr = nullptr;
        void* raggedOffsetLsePtr = nullptr;
        std::vector<int64_t> lseStrides;
        if(lse != nullptr)
        {
            const auto& lseDims = lse->dims();
            if(lseDims.size() != 4 || lseDims[0] != batch || lseDims[1] != numHeads
               || lseDims[2] != seqQMax || lseDims[3] != 1)
            {
                fail("lse must be rank-4 [B, H, Sq, 1] with Q's B, H, Sq");
            }
            lsePtr = lse->memory().deviceData();
            lseStrides = lse->strides();
            if(raggedOffsetLse != nullptr)
            {
                if(raggedOffsetQ == nullptr)
                {
                    fail("a ragged LSE needs a ragged Q");
                }
                if(raggedOffsetLse->dims() != offsetDims)
                {
                    fail("raggedOffsetLse must be rank-4 [B+1, 1, 1, 1]");
                }
                raggedOffsetLsePtr = raggedOffsetLse->memory().deviceData();
                hipdnn_test_sdk::detail::requireMatchingTokenBoundaries(
                    qTokens,
                    "Q",
                    readTokenBoundaries(
                        *raggedOffsetLse, mult.lse, lseDims, lseStrides, batch, who, "LSE"),
                    "LSE",
                    who);
            }
        }
        else if(raggedOffsetLse != nullptr)
        {
            fail("raggedOffsetLse given without an lse tensor");
        }

        const DescaleBinding dq = bindDescale(descaleQ, batch, numHeadsK, "Q");
        const DescaleBinding dk = bindDescale(descaleK, batch, numHeadsK, "K");
        const DescaleBinding dv = bindDescale(descaleV, batch, numHeadsV, "V");

        if(totalQ == 0)
        {
            return;
        }

        PagedKvLaunch paged;
        paged.seqLenQ = seqLenQ != nullptr ? seqLenQ->memory().deviceData() : nullptr;
        paged.seqLenQStride = seqLenQ != nullptr ? seqLenQ->strides()[0] : 0;
        paged.seqLenKv = seqLenKv.memory().deviceData();
        paged.seqLenKvStride = seqLenKv.strides()[0];
        paged.pageTableK = pageTableK.memory().deviceData();
        paged.pageTableKStr0 = ptKStr[0];
        paged.pageTableKStr2 = ptKStr[2];
        paged.pageTableV = pageTableV.memory().deviceData();
        paged.pageTableVStr0 = ptVStr[0];
        paged.pageTableVStr2 = ptVStr[2];
        paged.blockSize = blockSize;
        paged.seqQMax = seqQMax;

        launchSdpaRaggedFwd(q.memory().deviceData(),
                            kContainer.memory().deviceData(),
                            vContainer.memory().deviceData(),
                            o.memory().deviceData(),
                            lsePtr,
                            raggedOffsetLsePtr,
                            raggedOffsetQ != nullptr ? raggedOffsetQ->memory().deviceData()
                                                     : nullptr,
                            nullptr,
                            mult.q,
                            mult.k,
                            mult.lse,
                            q.strides()[SEQ_AXIS],
                            kContainer.strides()[SEQ_AXIS],
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
                            kContainer.strides(),
                            vContainer.strides(),
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
                            paged,
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

    // Paged-KV / dense-Q kernel inputs. Default-constructed for the plain ragged path.
    struct PagedKvLaunch
    {
        const void* seqLenQ = nullptr;
        int64_t seqLenQStride = 0;
        const void* seqLenKv = nullptr;
        int64_t seqLenKvStride = 0;
        const void* pageTableK = nullptr;
        int64_t pageTableKStr0 = 0;
        int64_t pageTableKStr2 = 0;
        const void* pageTableV = nullptr;
        int64_t pageTableVStr0 = 0;
        int64_t pageTableVStr2 = 0;
        int64_t blockSize = 0;
        int64_t seqQMax = 0;
    };

    // Reads an RFC-0014 ragged_offset table back from the device and turns it into per-batch
    // token boundaries, checking the layout, multiplier and monotonicity.
    static std::vector<int64_t>
        readTokenBoundaries(hipdnn_data_sdk::utilities::TensorBase<int32_t>& offsets,
                            int64_t multiplier,
                            const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            int64_t batch,
                            const std::string& who,
                            const char* name)
    {
        hipdnn_test_sdk::detail::requireTokenMajorRaggedLayout(dims, strides, who, name);
        if(offsets.strides()[0] != 1)
        {
            throw std::invalid_argument(who + ": " + name + " ragged_offset must be contiguous");
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
        constexpr auto SEQ_AXIS = hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS;
        return hipdnn_test_sdk::detail::raggedTokenBoundaries(
            elementOffsets, strides[SEQ_AXIS], dims[SEQ_AXIS], who, name);
    }

    // Reads the whole element span of a small INT32 device tensor (non-negative strides). Index
    // the result by the tensor's strides.
    static std::vector<int64_t> readInt32Span(hipdnn_data_sdk::utilities::TensorBase<int32_t>& t,
                                              const std::string& who,
                                              const char* name)
    {
        const auto& dims = t.dims();
        const auto& strides = t.strides();
        int64_t span = 1;
        for(size_t i = 0; i < dims.size(); ++i)
        {
            if(strides[i] < 0)
            {
                throw std::invalid_argument(who + ": " + name + " must have non-negative strides");
            }
            span += (dims[i] - 1) * strides[i];
        }
        return readRaggedOffsets(t.memory().deviceData(), span);
    }

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
            if(d.size() != 4 || d[0] != qDims[0] + 1 || d[1] != 1 || d[2] != 1 || d[3] != 1)
            {
                throw std::invalid_argument(
                    "GpuFpReferenceSdpaRagged: raggedOffsetQ/K/V/O must be rank-4 [B+1, 1, 1, 1]");
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
                                    const PagedKvLaunch& paged,
                                    const std::vector<std::string>& defines);
};

} // namespace hipdnn_gpu_ref
