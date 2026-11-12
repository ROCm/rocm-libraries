// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>

namespace hipdnn_test_sdk::detail
{

// The part of the ragged container's layout rule that depends on the tensor attributes alone,
// not on the per-batch origins and lengths: a rank-4 [B, H, S, D] tensor ragged along
// SDPA_SEQ_AXIS with non-negative strides, a positive sequence stride, and non-aliasing head and
// feature axes (sorted-stride rule on H and D; zero-stride axes are exempt). The sequence axis
// against the other axes depends on the dense view's row count, which only the origins and
// lengths give, so the container checks it when the tensor is built.
inline bool hasValidRaggedStrides(const std::vector<int64_t>& dims,
                                  const std::vector<int64_t>& strides)
{
    constexpr auto SEQ_AXIS = static_cast<size_t>(hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS);
    if(dims.size() != 4 || strides.size() != 4)
    {
        return false;
    }
    if(std::any_of(strides.begin() + 1, strides.end(), [](int64_t stride) { return stride < 0; })
       || strides[SEQ_AXIS] <= 0)
    {
        return false;
    }

    constexpr size_t HEAD_AXIS = 1;
    constexpr size_t FEATURE_AXIS = 3;
    const bool headCounts = dims[HEAD_AXIS] > 1 && strides[HEAD_AXIS] != 0;
    const bool featureCounts = dims[FEATURE_AXIS] > 1 && strides[FEATURE_AXIS] != 0;
    if(!headCounts || !featureCounts)
    {
        return true;
    }

    const size_t outer = strides[HEAD_AXIS] >= strides[FEATURE_AXIS] ? HEAD_AXIS : FEATURE_AXIS;
    const size_t inner = outer == HEAD_AXIS ? FEATURE_AXIS : HEAD_AXIS;
    return strides[outer] >= (dims[inner] - 1) * strides[inner] + 1;
}

// Throws std::invalid_argument unless two tensors that share a packing (Q/O, K/V, Q/LSE) have the
// same per-batch sequence lengths. Their origins are independent: token widths and layouts
// differ between tensors.
inline void requireMatchingRaggedLengths(const std::vector<int64_t>& a,
                                         const char* aName,
                                         const std::vector<int64_t>& b,
                                         const char* bName,
                                         const std::string& who)
{
    if(a.size() != b.size())
    {
        throw std::invalid_argument(who + ": " + aName + " and " + bName
                                    + " have different batch counts (" + std::to_string(a.size())
                                    + " vs " + std::to_string(b.size()) + ")");
    }
    for(size_t i = 0; i < a.size(); ++i)
    {
        if(a[i] != b[i])
        {
            throw std::invalid_argument(who + ": " + aName + " and " + bName
                                        + " per-batch sequence lengths differ (batch "
                                        + std::to_string(i) + ": " + std::to_string(a[i]) + " vs "
                                        + std::to_string(b[i]) + ")");
        }
    }
}

// Elements an INT32 aux tensor (ragged_offset or seq_len) of `count` values spans from its base
// pointer, value b at b * `stride`. A stride is only meaningful, and so only validated, with more
// than one value: std::invalid_argument naming `name` for a stride below 1.
inline size_t stridedAuxSpan(int64_t count, int64_t stride, const std::string& name)
{
    if(count <= 0)
    {
        return 0;
    }
    if(count > 1 && stride < 1)
    {
        throw std::invalid_argument(name + " batch stride must be >= 1 (got "
                                    + std::to_string(stride) + ")");
    }
    return count > 1 ? static_cast<size_t>((count - 1) * stride + 1) : 1;
}

// The `count` values of an INT32 aux tensor read from a host copy of its stridedAuxSpan elements.
inline std::vector<int64_t>
    gatherStridedAux(const int32_t* raw, int64_t count, int64_t stride, const std::string& name)
{
    (void)stridedAuxSpan(count, stride, name);
    std::vector<int64_t> values;
    values.reserve(static_cast<size_t>(std::max<int64_t>(count, 0)));
    for(int64_t b = 0; b < count; ++b)
    {
        values.push_back(raw[b * stride]);
    }
    return values;
}

} // namespace hipdnn_test_sdk::detail
