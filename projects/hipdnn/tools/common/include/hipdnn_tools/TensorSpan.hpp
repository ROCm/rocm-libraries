// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

/// @file TensorSpan.hpp
/// @brief The elements a strided tensor spans: what the benchmark allocates and what the
///        corpus generator charges against its byte ceiling.
namespace hipdnn_tools
{

/// @brief Elements spanned by @p dims with @p strides: `sum_i((dim_i - 1) * stride_i) + 1`,
///        which equals the element count only for packed tensors.
///
/// Returns 0 for a malformed tensor (rank mismatch, non-positive extent, negative stride) or
/// one whose span overflows int64_t; overflow is checked before each product is formed.
inline int64_t elementSpan(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    if(dims.empty() || dims.size() != strides.size())
    {
        return 0;
    }

    constexpr int64_t LIMIT = std::numeric_limits<int64_t>::max();
    int64_t furthest = 0;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(dims[i] <= 0 || strides[i] < 0)
        {
            return 0;
        }
        const int64_t lastIndex = dims[i] - 1;
        // lastIndex * stride + furthest <= LIMIT, tested without forming either side.
        if(lastIndex != 0 && strides[i] > (LIMIT - furthest) / lastIndex)
        {
            return 0;
        }
        furthest += lastIndex * strides[i];
    }
    return furthest == LIMIT ? 0 : furthest + 1;
}

} // namespace hipdnn_tools
