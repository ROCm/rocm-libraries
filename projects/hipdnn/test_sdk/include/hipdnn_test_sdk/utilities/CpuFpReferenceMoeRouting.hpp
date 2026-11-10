// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_test_sdk::utilities
{

/// CPU reference for MoE routing (moe_routing_attributes.fbs).
///
/// `TopkIds` is `[S, top_k]`. Every pair `(s, k)` whose expert `e = TopkIds[s, k]` lies in
/// `[0, numExperts)` becomes one routed row; routed rows are ordered by expert, then by `s`,
/// then by `k` (a stable counting sort over the row-major `(s, k)` order). For routed row
/// `r`: `TokenIndex[0, r, 0] = s` and `TokenKs[0, r, 0] = k`, and
/// `FirstTokenOffset[e, 0, 0]` is the number of routed rows with expert `< e`, for `e` in
/// `[0, numExperts]`. Ids outside `[0, numExperts)` are not routed (expert parallelism):
/// `FirstTokenOffset[numExperts]` is then below `S * top_k` and the trailing `TokenIndex` /
/// `TokenKs` rows are left untouched.
class CpuFpReferenceMoeRouting
{
public:
    template <class OffsetDataType>
    static void route(const hipdnn_data_sdk::utilities::TensorBase<int32_t>& topkIds,
                      int64_t numExperts,
                      hipdnn_data_sdk::utilities::TensorBase<OffsetDataType>& firstTokenOffset,
                      hipdnn_data_sdk::utilities::TensorBase<int32_t>& tokenIndex,
                      hipdnn_data_sdk::utilities::TensorBase<int32_t>& tokenKs)
    {
        validateInput(topkIds, numExperts, firstTokenOffset, tokenIndex, tokenKs);

        const int64_t tokens = topkIds.dims()[0];
        const int64_t topK = topkIds.dims()[1];

        std::vector<int64_t> offsets(static_cast<size_t>(numExperts) + 1, 0);
        for(int64_t s = 0; s < tokens; ++s)
        {
            for(int64_t k = 0; k < topK; ++k)
            {
                const int64_t expert = topkIds.getHostValue({s, k});
                if(expert >= 0 && expert < numExperts)
                {
                    ++offsets[static_cast<size_t>(expert) + 1];
                }
            }
        }
        for(int64_t e = 0; e < numExperts; ++e)
        {
            offsets[static_cast<size_t>(e) + 1] += offsets[static_cast<size_t>(e)];
        }
        for(int64_t e = 0; e <= numExperts; ++e)
        {
            firstTokenOffset.setHostValue(
                static_cast<OffsetDataType>(offsets[static_cast<size_t>(e)]), {e, 0, 0});
        }

        std::vector<int64_t> cursor(offsets.begin(), offsets.end() - 1);
        for(int64_t s = 0; s < tokens; ++s)
        {
            for(int64_t k = 0; k < topK; ++k)
            {
                const int64_t expert = topkIds.getHostValue({s, k});
                if(expert < 0 || expert >= numExperts)
                {
                    continue;
                }
                const int64_t row = cursor[static_cast<size_t>(expert)]++;
                tokenIndex.setHostValue(static_cast<int32_t>(s), {0, row, 0});
                tokenKs.setHostValue(static_cast<int32_t>(k), {0, row, 0});
            }
        }

        firstTokenOffset.memory().markHostModified();
        tokenIndex.memory().markHostModified();
        tokenKs.memory().markHostModified();
    }

private:
    template <class OffsetDataType>
    static void validateInput(
        const hipdnn_data_sdk::utilities::TensorBase<int32_t>& topkIds,
        int64_t numExperts,
        const hipdnn_data_sdk::utilities::TensorBase<OffsetDataType>& firstTokenOffset,
        const hipdnn_data_sdk::utilities::TensorBase<int32_t>& tokenIndex,
        const hipdnn_data_sdk::utilities::TensorBase<int32_t>& tokenKs)
    {
        const std::string prefix = "CpuFpReferenceMoeRouting: ";

        const auto validateNoRaggedTensor
            = [&prefix](const hipdnn_data_sdk::utilities::ITensor& tensor, const char* name) {
                  if(tensor.raggedIterationInfo().has_value())
                  {
                      throw std::runtime_error(prefix + "ragged " + name
                                               + " tensor is not supported");
                  }
              };
        validateNoRaggedTensor(topkIds, "topk_ids");
        validateNoRaggedTensor(firstTokenOffset, "first_token_offset");
        validateNoRaggedTensor(tokenIndex, "token_index");
        validateNoRaggedTensor(tokenKs, "token_ks");

        if(numExperts < 1)
        {
            throw std::runtime_error(prefix + "num_experts must be at least 1");
        }
        if(topkIds.dims().size() != 2)
        {
            throw std::runtime_error(prefix + "topk_ids must have shape [S, top_k]");
        }

        const int64_t routedCapacity = topkIds.dims()[0] * topkIds.dims()[1];
        if(routedCapacity > static_cast<int64_t>(std::numeric_limits<int32_t>::max())
           || numExperts > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        {
            throw std::runtime_error(prefix + "S * top_k and num_experts must fit in INT32");
        }

        if(firstTokenOffset.dims() != std::vector<int64_t>{numExperts + 1, 1, 1})
        {
            throw std::runtime_error(
                prefix + "first_token_offset must have shape [num_experts + 1, 1, 1]");
        }
        const std::vector<int64_t> rowDims = {1, routedCapacity, 1};
        if(tokenIndex.dims() != rowDims)
        {
            throw std::runtime_error(prefix + "token_index must have shape [1, S * top_k, 1]");
        }
        if(tokenKs.dims() != rowDims)
        {
            throw std::runtime_error(prefix + "token_ks must have shape [1, S * top_k, 1]");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
