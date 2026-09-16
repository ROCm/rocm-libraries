// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_test_sdk::utilities
{

// Shared geometry helpers for ragged (RFC-0014: packed + ragged_offset) SDPA reference tests, used by
// both the CPU-only test_sdk suite and the integration gpu-ref suite.

// Sequence axis for ragged SDPA tensors. SDPA logical dims are [B, H, S, D] ("BHSD ordering", see
// hipdnn_frontend/attributes/SdpaAttributes.hpp) with the sequence axis at index 2; the packed/ragged
// case expresses the BSHD (sequence-major) memory layout purely via strides. We therefore pass
// seqAxis=2 rather than the SDK's generic BSHD_SEQ_AXIS=1 constant (which labels dims literally as
// [B, S, H, D]) — both are valid for ShallowRaggedTensor, and [B, H, S, D] matches the SDPA frontend.
inline constexpr int SEQ_AXIS = 2;

// BSHD-layout element strides for a packed rank-4 [B, H, S, D] tensor: seq stride = strides[2] = H*D.
inline std::vector<int64_t> bshd(const std::vector<int64_t>& dims)
{
    return {dims[1] * dims[2] * dims[3], dims[3], dims[1] * dims[3], 1};
}

// Exclusive prefix sum of per-batch token counts: cum[0]=0, cum[b+1]=cum[b]+lengths[b].
inline std::vector<int64_t> cumTokens(const std::vector<int64_t>& lengths)
{
    std::vector<int64_t> cum(lengths.size() + 1, 0);
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        cum[i + 1] = cum[i] + lengths[i];
    }
    return cum;
}

// Rank-4 [B+1,1,1,1] INT32 ragged_offset aux = cumTokens * seqStride (element units, RFC-0014),
// returned as a shared ITensor for ShallowRaggedTensor / RaggedTensor construction.
inline std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    makeRaggedOffsetAux(const std::vector<int64_t>& cum, int64_t seqStride)
{
    auto aux = std::make_shared<hipdnn_data_sdk::utilities::Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(cum.size()), 1, 1, 1});
    for(size_t i = 0; i < cum.size(); ++i)
    {
        aux->setHostValue(
            static_cast<int32_t>(cum[i] * seqStride), static_cast<int64_t>(i), 0, 0, 0);
    }
    return aux;
}

} // namespace hipdnn_test_sdk::utilities
