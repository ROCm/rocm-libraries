// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/MoeGroupedMatmulValidation.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

/// CPU reference for the forward MoE grouped matmul.
///
/// A group of routed rows is defined per `FirstTokenOffset` entry; group `g` uses
/// expert `g % E` (the offset table is laid out `[batch][expert]`, expert-minor).
/// Per mode, the source token row and destination output row for routed row `r`
/// are:
///   NONE:    src = r,                 dst = r
///   GATHER:  src = TokenIndex[r],     dst = r
///   SCATTER: src = r,                 dst = TokenIndex[r] * top_k + TokenKs[r]
/// `Output[0, dst, n] = sum_k Token[0, src, k] * Weight[expert, k, n]`. Output rows
/// no group ever writes are zero-filled.
class CpuFpReferenceMoeGroupedMatmul
{
public:
    template <class TokenDataType,
              class WeightDataType,
              class OutputDataType,
              class ComputeDataType = float>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<TokenDataType>& token,
                        const hipdnn_data_sdk::utilities::TensorBase<WeightDataType>& weight,
                        const hipdnn_data_sdk::utilities::TensorBase<int32_t>& firstTokenOffset,
                        hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output,
                        hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode mode,
                        int32_t topK,
                        const hipdnn_data_sdk::utilities::TensorBase<int32_t>* tokenIndex = nullptr,
                        const hipdnn_data_sdk::utilities::TensorBase<int32_t>* tokenKs = nullptr)
    {
        using Mode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;

        validateInput(token, weight, firstTokenOffset, output, mode, topK, tokenIndex, tokenKs);

        const int64_t expertCount = weight.dims()[0];
        const int64_t tokenRows = token.dims()[1];
        const int64_t outputRows = output.dims()[1];
        const int64_t hiddenK = weight.dims()[1];
        const int64_t outputN = weight.dims()[2];
        const int64_t groupCount = firstTokenOffset.dims()[0];
        const int64_t routedRows = (mode == Mode::NONE) ? 0 : tokenIndex->dims()[1];
        const int64_t rowsTotal = (mode == Mode::GATHER) ? routedRows : tokenRows;

        // --- Routing prepass: reduce all three modes to a single "one output row =
        // one independent work item" mapping, so the compute loop below is
        // race-free even for SCATTER. O(rowsTotal), negligible next to the compute
        // loop's O(rowsTotal * K * N).
        std::vector<int64_t> sourceRow(static_cast<size_t>(outputRows), -1);
        std::vector<int64_t> expertOfRow(static_cast<size_t>(outputRows), -1);

        // Validate the whole offset table before consuming any of it: the mapping
        // loop below reads routing rows across [start, end), but `end` is otherwise
        // only range-checked on the next iteration, as that iteration's `start` -
        // after the inner loop has already read past it.
        int64_t previousOffset = 0;
        for(int64_t group = 0; group < groupCount; ++group)
        {
            const int64_t offset = firstTokenOffset.getHostValue({group, 0, 0});
            if(offset < 0 || offset > rowsTotal)
            {
                throw std::runtime_error("CpuFpReferenceMoeGroupedMatmul: FirstTokenOffset["
                                         + std::to_string(group) + "] = " + std::to_string(offset)
                                         + " is out of range [0, " + std::to_string(rowsTotal)
                                         + "]");
            }
            if(group > 0 && offset < previousOffset)
            {
                throw std::runtime_error(
                    "CpuFpReferenceMoeGroupedMatmul: FirstTokenOffset must be non-decreasing "
                    "(violated at group "
                    + std::to_string(group) + ")");
            }
            previousOffset = offset;
        }

        for(int64_t group = 0; group < groupCount; ++group)
        {
            const int64_t start = firstTokenOffset.getHostValue({group, 0, 0});
            const int64_t end = (group + 1 < groupCount)
                                    ? firstTokenOffset.getHostValue({group + 1, 0, 0})
                                    : rowsTotal;

            const int64_t expert = group % expertCount;

            for(int64_t r = start; r < end; ++r)
            {
                int64_t src = 0;
                int64_t dst = 0;
                switch(mode)
                {
                case Mode::NONE:
                    src = r;
                    dst = r;
                    break;
                case Mode::GATHER:
                    src = tokenIndex->getHostValue({0, r, 0});
                    dst = r;
                    break;
                case Mode::SCATTER:
                {
                    const int64_t tokenKsValue = tokenKs->getHostValue({0, r, 0});
                    if(tokenKsValue < 0 || tokenKsValue >= topK)
                    {
                        throw std::runtime_error(
                            "CpuFpReferenceMoeGroupedMatmul: TokenKs[" + std::to_string(r)
                            + "] = " + std::to_string(tokenKsValue) + " is out of range [0, "
                            + std::to_string(topK) + ")");
                    }
                    const int64_t tokenIndexValue = tokenIndex->getHostValue({0, r, 0});
                    src = r;
                    dst = tokenIndexValue * topK + tokenKsValue;
                    break;
                }
                default:
                    throw std::runtime_error(
                        "CpuFpReferenceMoeGroupedMatmul: unknown routing mode");
                }

                if(src < 0 || src >= tokenRows)
                {
                    throw std::runtime_error("CpuFpReferenceMoeGroupedMatmul: routed source row "
                                             + std::to_string(src) + " is out of range [0, "
                                             + std::to_string(tokenRows) + ")");
                }
                if(dst < 0 || dst >= outputRows)
                {
                    throw std::runtime_error(
                        "CpuFpReferenceMoeGroupedMatmul: routed destination row "
                        + std::to_string(dst) + " is out of range [0, " + std::to_string(outputRows)
                        + ")");
                }
                if(expertOfRow[static_cast<size_t>(dst)] >= 0)
                {
                    throw std::runtime_error(
                        "CpuFpReferenceMoeGroupedMatmul: duplicate SCATTER destination row "
                        + std::to_string(dst));
                }

                sourceRow[static_cast<size_t>(dst)] = src;
                expertOfRow[static_cast<size_t>(dst)] = expert;
            }
        }

        // --- Compute, parallel over output rows only. Raw pointers/strides are
        // hoisted once; the inner loops never call getHostValue/setHostValue,
        // which each cost a virtual memory() call plus a stride reduction.
        const TokenDataType* tokenBase = token.memory().hostData();
        const WeightDataType* weightBase = weight.memory().hostData();
        OutputDataType* outputBase = output.memory().hostData();

        const auto& tokenStrides = token.strides();
        const auto& weightStrides = weight.strides();
        const auto& outputStrides = output.strides();

        const int64_t tokenStride1 = tokenStrides[1];
        const int64_t tokenStride2 = tokenStrides[2];
        const int64_t weightStride0 = weightStrides[0];
        const int64_t weightStride1 = weightStrides[1];
        const int64_t weightStride2 = weightStrides[2];
        const int64_t outputStride1 = outputStrides[1];
        const int64_t outputStride2 = outputStrides[2];

        auto moeRowFunc = [&](const std::vector<int64_t>& indices) {
            const int64_t o = indices[0];
            OutputDataType* outRow = outputBase + o * outputStride1;
            const int64_t expert = expertOfRow[static_cast<size_t>(o)];

            if(expert < 0)
            {
                for(int64_t nIdx = 0; nIdx < outputN; ++nIdx)
                {
                    outRow[nIdx * outputStride2]
                        = hipdnn_test_sdk::detail::safeConvert<OutputDataType>(0.0F);
                }
                return;
            }

            const TokenDataType* tokRow
                = tokenBase + sourceRow[static_cast<size_t>(o)] * tokenStride1;
            const WeightDataType* wExpert = weightBase + expert * weightStride0;

            if(weightStride2 == 1)
            {
                // Row-major weight layout {K*N, N, 1}: unit stride along N, so the K
                // loop accumulates across a whole output row at once and needs a
                // scratch row. thread_local keeps that off the per-row allocation
                // path - this functor body runs once per output row, on every worker
                // thread.
                thread_local std::vector<ComputeDataType> s_acc;
                s_acc.assign(static_cast<size_t>(outputN), ComputeDataType{0});

                for(int64_t kIdx = 0; kIdx < hiddenK; ++kIdx)
                {
                    const auto a = static_cast<ComputeDataType>(tokRow[kIdx * tokenStride2]);
                    const WeightDataType* wRow = wExpert + kIdx * weightStride1;
                    for(int64_t nIdx = 0; nIdx < outputN; ++nIdx)
                    {
                        s_acc[static_cast<size_t>(nIdx)]
                            += a * static_cast<ComputeDataType>(wRow[nIdx]);
                    }
                }

                for(int64_t nIdx = 0; nIdx < outputN; ++nIdx)
                {
                    outRow[nIdx * outputStride2]
                        = hipdnn_test_sdk::detail::safeConvert<OutputDataType>(
                            s_acc[static_cast<size_t>(nIdx)]);
                }
            }
            else
            {
                // Column-major weight layout {K*N, 1, K} (cuDNN's documented MoE
                // layout): unit stride along K, so each output element is a
                // self-contained dot product and no scratch row is needed.
                for(int64_t nIdx = 0; nIdx < outputN; ++nIdx)
                {
                    ComputeDataType s{0};
                    const WeightDataType* wCol = wExpert + nIdx * weightStride2;
                    for(int64_t kIdx = 0; kIdx < hiddenK; ++kIdx)
                    {
                        s += static_cast<ComputeDataType>(tokRow[kIdx * tokenStride2])
                             * static_cast<ComputeDataType>(wCol[kIdx * weightStride1]);
                    }
                    outRow[nIdx * outputStride2]
                        = hipdnn_test_sdk::detail::safeConvert<OutputDataType>(s);
                }
            }
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(moeRowFunc, {outputRows});
        parallelFunc(std::thread::hardware_concurrency());

        output.memory().markHostModified();
    }

private:
    template <class TokenDataType, class WeightDataType, class OutputDataType>
    static void
        validateInput(const hipdnn_data_sdk::utilities::TensorBase<TokenDataType>& token,
                      const hipdnn_data_sdk::utilities::TensorBase<WeightDataType>& weight,
                      const hipdnn_data_sdk::utilities::TensorBase<int32_t>& firstTokenOffset,
                      const hipdnn_data_sdk::utilities::TensorBase<OutputDataType>& output,
                      hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode mode,
                      int32_t topK,
                      const hipdnn_data_sdk::utilities::TensorBase<int32_t>* tokenIndex,
                      const hipdnn_data_sdk::utilities::TensorBase<int32_t>* tokenKs)
    {
        using Mode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;
        const std::string prefix = "CpuFpReferenceMoeGroupedMatmul: ";

        const auto validateNoRaggedTensor
            = [&prefix](const hipdnn_data_sdk::utilities::ITensor& tensor, const char* name) {
                  if(tensor.raggedIterationInfo().has_value())
                  {
                      throw std::runtime_error(prefix + "ragged " + name
                                               + " tensor is not supported");
                  }
              };
        validateNoRaggedTensor(token, "token");
        validateNoRaggedTensor(weight, "weight");
        validateNoRaggedTensor(firstTokenOffset, "first_token_offset");
        validateNoRaggedTensor(output, "output");
        if(tokenIndex != nullptr)
        {
            validateNoRaggedTensor(*tokenIndex, "token-index");
        }
        if(tokenKs != nullptr)
        {
            validateNoRaggedTensor(*tokenKs, "token-ks");
        }

        if(token.dims().size() != 3 || weight.dims().size() != 3
           || firstTokenOffset.dims().size() != 3 || output.dims().size() != 3)
        {
            throw std::runtime_error(
                prefix + "token, weight, first_token_offset and output must all be rank 3");
        }
        if(token.dims()[0] != 1)
        {
            throw std::runtime_error(prefix + "token must have a singleton leading dimension");
        }
        if(output.dims()[0] != 1)
        {
            throw std::runtime_error(prefix + "output must have a singleton leading dimension");
        }
        if(firstTokenOffset.dims()[1] != 1 || firstTokenOffset.dims()[2] != 1)
        {
            throw std::runtime_error(prefix + "first_token_offset must have shape [G, 1, 1]");
        }
        if(token.dims()[2] != weight.dims()[1])
        {
            throw std::runtime_error(prefix
                                     + "token hidden size (dims[2]) must equal weight.dims[1]");
        }
        if(output.dims()[2] != weight.dims()[2])
        {
            throw std::runtime_error(prefix + "output width (dims[2]) must equal weight.dims[2]");
        }

        const int64_t expertCount = weight.dims()[0];
        if(expertCount <= 0)
        {
            throw std::runtime_error(prefix + "expert count (weight.dims[0]) must be positive");
        }
        if(firstTokenOffset.dims()[0] % expertCount != 0)
        {
            throw std::runtime_error(
                prefix + "first_token_offset row count must be a multiple of the expert count");
        }

        // Presence/top_k rules are exactly the shared FlatBuffers routing contract
        // (backend descriptor and CPU plan builder evaluate the same function).
        // Routing tensors are statically int32_t here, so their dtype fields hold.
        namespace fb_utilities = hipdnn_flatbuffers_sdk::utilities;
        using FbDataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
        const fb_utilities::MoeGroupedMatmulRouting routing{mode,
                                                            tokenIndex != nullptr,
                                                            tokenKs != nullptr,
                                                            FbDataType::INT32,
                                                            FbDataType::INT32,
                                                            FbDataType::INT32,
                                                            topK,
                                                            expertCount};
        if(const char* reason = fb_utilities::checkMoeGroupedMatmulRouting(routing))
        {
            throw std::runtime_error(prefix + reason);
        }

        const auto validateRoutingTensorShape
            = [&prefix](const hipdnn_data_sdk::utilities::TensorBase<int32_t>* tensor,
                        const char* name) {
                  if(tensor == nullptr)
                  {
                      return;
                  }
                  if(tensor->dims().size() != 3 || tensor->dims()[0] != 1 || tensor->dims()[2] != 1)
                  {
                      throw std::runtime_error(prefix + name + " must have shape [1, rows, 1]");
                  }
              };
        validateRoutingTensorShape(tokenIndex, "token-index tensor");
        validateRoutingTensorShape(tokenKs, "token-ks tensor");

        const int64_t tokenRows = token.dims()[1];

        if(mode == Mode::SCATTER)
        {
            if(tokenIndex->dims()[1] != tokenKs->dims()[1])
            {
                throw std::runtime_error(
                    prefix + "token-index and token-ks tensors must have equal length");
            }
            // SCATTER walks r over [0, tokenRows) and indexes both routing tensors
            // at r, so their length is a memory bound, not a formality.
            if(tokenIndex->dims()[1] != tokenRows)
            {
                throw std::runtime_error(
                    prefix
                    + "SCATTER mode requires the routed row count to equal the token row count");
            }
        }

        if(mode == Mode::GATHER)
        {
            if(output.dims()[1] != tokenIndex->dims()[1])
            {
                throw std::runtime_error(
                    prefix + "GATHER mode requires output row count to equal the routed row count");
            }
        }
        else if(output.dims()[1] != tokenRows)
        {
            throw std::runtime_error(prefix + "output row count must equal the token row count");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
