// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_ragged_test
{

using namespace hipdnn_data_sdk::utilities;

// Canonical BSHD-packed geometry shared across the ragged-tensor and iterator tests:
// dims [B, S_max, H, D], strides {S*H*D, H*D, D, 1}. B=2 (batch0 seq=2, batch1 seq=3),
// seqStride = H*D = 4, off[B] = 20.
inline const std::vector<int64_t> K_DIMS = {2, 3, 2, 2};
inline const std::vector<int64_t> K_STRIDES = {12, 4, 2, 1};
inline const std::vector<int64_t> K_OFFSETS = {0, 8, 20};

/// Build a rank-4 ragged_offset aux (dims {B+1, 1, 1, 1}) of the given index element
/// type and populate it with @p offsets (element units).
template <typename IndexT>
inline std::shared_ptr<ITensor> makeOffsetAux(const std::vector<int64_t>& offsets)
{
    const auto count = static_cast<int64_t>(offsets.size());
    auto aux = std::make_shared<Tensor<IndexT>>(std::vector<int64_t>{count, 1, 1, 1});
    for(int64_t i = 0; i < count; ++i)
    {
        aux->setHostValue(static_cast<IndexT>(offsets[static_cast<size_t>(i)]), i, 0, 0, 0);
    }
    return aux;
}

/// Per-batch sequence extent for BSHD-packed ragged geometry.
inline int64_t seqExtent(const std::vector<int64_t>& offsets, int64_t seqStride, int64_t b)
{
    return (offsets[static_cast<size_t>(b) + 1] - offsets[static_cast<size_t>(b)]) / seqStride;
}

/// elementCount/elementSpace/isPacked reporting (RFC 0014 §4.5.6/4.5.7).
template <typename T>
inline void checkReporting(const TensorBase<T>& tensor, int64_t total)
{
    EXPECT_EQ(tensor.elementCount(), static_cast<size_t>(total));
    EXPECT_EQ(tensor.elementSpace(), static_cast<size_t>(total));
    EXPECT_FALSE(tensor.isPacked());
}

/// Writes a unique value to every valid logical position and confirms it lands at the
/// expected physical slot `ragged_offset[b] + s*stride1 + h*stride2 + d*stride3`.
template <typename T>
inline void checkAddressing(TensorBase<T>& tensor,
                            const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& offsets)
{
    const int64_t batchCount = dims[0];
    const int64_t heads = dims[2];
    const int64_t headDim = dims[3];
    const int64_t seqStride = strides[1];

    T counter{1};
    for(int64_t b = 0; b < batchCount; ++b)
    {
        for(int64_t s = 0; s < seqExtent(offsets, seqStride, b); ++s)
        {
            for(int64_t h = 0; h < heads; ++h)
            {
                for(int64_t d = 0; d < headDim; ++d)
                {
                    tensor.setHostValue(counter, b, s, h, d);
                    const int64_t expectedPhysical
                        = offsets[static_cast<size_t>(b)] + s * seqStride + h * strides[2] + d;
                    const auto* base = static_cast<const T*>(tensor.memory().hostData());
                    EXPECT_EQ(base[expectedPhysical], counter)
                        << "b=" << b << " s=" << s << " h=" << h << " d=" << d;
                    ++counter;
                }
            }
        }
    }
}

/// Iterating begin()..end() visits exactly `ragged_offset[B]` physical elements, each
/// exactly once. The visited offsets are the union of all per-batch ranges
/// `[ragged_offset[b], ragged_offset[b+1])`, which partitions `[0, off[B])`. This checks
/// the visited *set*, not the order.
template <typename T>
inline void checkIteration(TensorBase<T>& tensor, const std::vector<int64_t>& offsets)
{
    const int64_t total = offsets.back();
    const auto* base = static_cast<const T*>(tensor.memory().hostData());

    std::vector<int64_t> visited;
    for(auto it = tensor.begin(); it != tensor.end(); ++it)
    {
        const auto* ptr = static_cast<const T*>(*it);
        visited.push_back(ptr - base);
    }

    ASSERT_EQ(static_cast<int64_t>(visited.size()), total);
    std::sort(visited.begin(), visited.end());
    for(int64_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(visited[static_cast<size_t>(i)], i) << "missing/duplicate physical offset";
    }
}

} // namespace hipdnn_ragged_test
