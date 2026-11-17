// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>

namespace hipdnn_test_sdk::detail
{

// RFC-0014 packs a ragged tensor token by token: logical rank-4 [B, H, S, D] ragged along
// SDPA_SEQ_AXIS, physically BSHD so the sequence is the outermost axis after batch. The seq
// stride (strides[2]) must therefore cover one token's whole H x D block, so tokens never
// overlap. A contiguous BHSD tensor fails this, because its seq stride is D.
inline bool isTokenMajorRaggedLayout(const std::vector<int64_t>& dims,
                                     const std::vector<int64_t>& strides)
{
    constexpr auto SEQ_AXIS = static_cast<size_t>(hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS);
    if(dims.size() != 4 || strides.size() != 4)
    {
        return false;
    }
    int64_t tokenSpan = 1; // elements from a token's first to last element, inclusive
    for(size_t i = 1; i < dims.size(); ++i)
    {
        if(i != SEQ_AXIS && dims[i] > 0)
        {
            tokenSpan += (dims[i] - 1) * strides[i];
        }
    }
    return strides[SEQ_AXIS] >= tokenSpan;
}

inline void requireTokenMajorRaggedLayout(const std::vector<int64_t>& dims,
                                          const std::vector<int64_t>& strides,
                                          const std::string& who,
                                          const char* name)
{
    if(!isTokenMajorRaggedLayout(dims, strides))
    {
        throw std::invalid_argument(who + ": " + name
                                    + " must be [B, H, S, D] packed token by token (RFC-0014): "
                                      "strides[2] must cover one token's H x D block");
    }
}

// Converts an RFC-0014 ragged_offset table (B + 1 element offsets) to token boundaries,
// offset[b] / seqStride. Tensors with different token widths share a packing but not element
// offsets, so cross-tensor checks compare tokens.
// Throws std::invalid_argument unless seqStride > 0, offset[0] == 0, each offset is a whole
// number of tokens, offsets never decrease, and no batch is longer than sMax (dims()[2]).
inline std::vector<int64_t> raggedTokenBoundaries(const std::vector<int64_t>& elementOffsets,
                                                  int64_t seqStride,
                                                  int64_t sMax,
                                                  const std::string& who,
                                                  const char* name)
{
    if(seqStride <= 0)
    {
        throw std::invalid_argument(who + ": " + name + " sequence stride must be positive");
    }
    if(!elementOffsets.empty() && elementOffsets.front() != 0)
    {
        throw std::invalid_argument(who + ": " + name + " ragged_offset[0] must be 0 (got "
                                    + std::to_string(elementOffsets.front()) + ")");
    }
    std::vector<int64_t> tokens;
    tokens.reserve(elementOffsets.size());
    for(size_t b = 0; b < elementOffsets.size(); ++b)
    {
        const auto offset = elementOffsets[b];
        if(offset % seqStride != 0)
        {
            throw std::invalid_argument(who + ": " + name + " ragged_offset[" + std::to_string(b)
                                        + "] = " + std::to_string(offset)
                                        + " is not a whole number of tokens (seq stride "
                                        + std::to_string(seqStride) + ")");
        }
        tokens.push_back(offset / seqStride);
        if(b == 0)
        {
            continue;
        }
        const auto length = tokens[b] - tokens[b - 1];
        if(length < 0)
        {
            throw std::invalid_argument(who + ": " + name
                                        + " ragged_offset must be non-decreasing");
        }
        if(length > sMax)
        {
            throw std::invalid_argument(who + ": " + name + " batch " + std::to_string(b - 1)
                                        + " has " + std::to_string(length)
                                        + " tokens, more than S_max = " + std::to_string(sMax));
        }
    }
    return tokens;
}

// Throws std::invalid_argument unless two tensors that share a packing (Q/O, K/V, Q/LSE) have the
// same token boundaries.
inline void requireMatchingTokenBoundaries(const std::vector<int64_t>& a,
                                           const char* aName,
                                           const std::vector<int64_t>& b,
                                           const char* bName,
                                           const std::string& who)
{
    if(a.size() != b.size())
    {
        throw std::invalid_argument(who + ": " + aName + " and " + bName
                                    + " ragged_offset tables differ in length");
    }
    for(size_t i = 0; i < a.size(); ++i)
    {
        if(a[i] != b[i])
        {
            throw std::invalid_argument(who + ": " + aName + " and " + bName
                                        + " per-batch sequence lengths differ (token boundary "
                                        + std::to_string(i) + ": " + std::to_string(a[i]) + " vs "
                                        + std::to_string(b[i]) + ")");
        }
    }
}

} // namespace hipdnn_test_sdk::detail
