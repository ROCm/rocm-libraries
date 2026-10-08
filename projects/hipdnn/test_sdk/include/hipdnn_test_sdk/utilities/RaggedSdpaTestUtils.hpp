// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_test_sdk::utilities
{

// Geometry helpers for ragged SDPA tests (RFC-0014), shared by the CPU test_sdk suite and the
// gpu-ref integration suite. Ragged tensors are [B, S, H, D] with the sequence at
// BSHD_SEQ_AXIS (1), packed token by token. Ragged tests build every ragged shape, stride and
// element index through these helpers.

inline std::vector<int64_t> raggedDims(int64_t batch, int64_t seq, int64_t heads, int64_t dim)
{
    return {batch, seq, heads, dim};
}

// Logical index of element (b, s, h, d) in a tensor shaped by raggedDims.
inline std::vector<int64_t> raggedIndex(int64_t b, int64_t s, int64_t h, int64_t d)
{
    return {b, s, h, d};
}

inline int64_t raggedSeqExtent(const std::vector<int64_t>& dims)
{
    return dims[hipdnn_data_sdk::utilities::BSHD_SEQ_AXIS];
}

inline int64_t raggedHeads(const std::vector<int64_t>& dims)
{
    return dims[2];
}

// Contiguous strides of a raggedDims tensor. The seq stride is H * D.
inline std::vector<int64_t> raggedStrides(const std::vector<int64_t>& dims)
{
    return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

// Exclusive prefix sum: cum[0] = 0, cum[b + 1] = cum[b] + lengths[b].
inline std::vector<int64_t> cumTokens(const std::vector<int64_t>& lengths)
{
    std::vector<int64_t> cum(lengths.size() + 1, 0);
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        cum[i + 1] = cum[i] + lengths[i];
    }
    return cum;
}

// RFC-0014 ragged_offset aux: int32 [B + 1, 1, 1, 1] holding cum * seqStride in element units.
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
