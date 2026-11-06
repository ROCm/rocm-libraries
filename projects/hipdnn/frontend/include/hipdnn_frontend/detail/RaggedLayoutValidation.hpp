// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace hipdnn_frontend::detail
{

// Ragged tensors use the logical SDPA dim order [B, H, S, D], so the sequence axis is 2.
inline constexpr size_t RAGGED_SEQUENCE_AXIS = 2;

// Rejects a ragged layout whose token rows cannot be addressed or overlap. The sequence stride
// must be positive. Over the non-batch axes with extent > 1 and a non-zero stride, sorted by
// stride, each stride must be at least the span of the axes below it. A zero stride on another
// axis is a broadcast and is legal. The declared sequence extent dims[2] is the row count.
// A tensor without a sequence axis is not checked. Expects dims and strides of equal size.
inline Error validateRaggedLayout(const std::string& tensorName,
                                  const std::vector<int64_t>& dims,
                                  const std::vector<int64_t>& strides)
{
    if(dims.size() <= RAGGED_SEQUENCE_AXIS)
    {
        return {ErrorCode::OK, ""};
    }

    const std::string prefix = "Ragged tensor " + tensorName;
    const int64_t seqStride = strides[RAGGED_SEQUENCE_AXIS];
    if(seqStride <= 0)
    {
        return {ErrorCode::INVALID_VALUE,
                prefix + " sequence stride (axis " + std::to_string(RAGGED_SEQUENCE_AXIS)
                    + ") must be positive, got " + std::to_string(seqStride)};
    }

    std::vector<std::tuple<int64_t, int64_t, size_t>> axes; // (stride, extent, axis)
    for(size_t axis = 1; axis < dims.size(); ++axis)
    {
        if(dims[axis] > 1 && strides[axis] != 0)
        {
            axes.emplace_back(strides[axis], dims[axis], axis);
        }
    }
    std::sort(axes.begin(), axes.end());

    int64_t span = 1;
    for(const auto& [stride, extent, axis] : axes)
    {
        if(stride < span)
        {
            return {ErrorCode::INVALID_VALUE,
                    prefix + " violates the sorted-stride rule: axis " + std::to_string(axis)
                        + " has stride " + std::to_string(stride) + ", below the span "
                        + std::to_string(span)
                        + " of the axes with smaller strides, so its rows overlap"};
        }
        span += (extent - 1) * stride;
    }
    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::detail
