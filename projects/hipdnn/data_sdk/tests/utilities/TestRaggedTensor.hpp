// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_ragged_test
{

using namespace hipdnn_data_sdk::utilities;

// Canonical geometry shared across the ragged-tensor and iterator tests: logical SDPA dims
// [B, H, S_max, D] with physical BSHD strides {S*H*D, D, H*D, 1}. B=2 (batch0 seq=2,
// batch1 seq=3), seqStride = H*D = 4, off[B] = 20.
inline const std::vector<int64_t> K_DIMS = {2, 2, 3, 2};
inline const std::vector<int64_t> K_STRIDES = {12, 2, 4, 1};
inline constexpr int K_SEQ_AXIS = SDPA_SEQ_AXIS;
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

/// Per-batch sequence extent for sequence-outermost ragged geometry.
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

/// Writes a unique value to every in-block logical position, walking each batch in
/// (s, inner, inner) order with the two non-sequence axes ascending, and confirms each lands
/// at the next physical slot counted from `ragged_offset[b]` (@p offsets in element units).
/// Assumes rank 4 with rows packed in that order.
template <typename T>
inline void checkAddressing(TensorBase<T>& tensor,
                            const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            int seqAxis,
                            const std::vector<int64_t>& offsets)
{
    ASSERT_EQ(dims.size(), 4u);
    const auto seqIdx = static_cast<size_t>(seqAxis);
    std::vector<size_t> innerAxes;
    for(size_t axis = 1; axis < dims.size(); ++axis)
    {
        if(axis != seqIdx)
        {
            innerAxes.push_back(axis);
        }
    }
    const size_t outerInner = innerAxes[0];
    const size_t innermost = innerAxes[1];
    const int64_t seqStride = strides[seqIdx];

    T value{1};
    for(int64_t b = 0; b < dims[0]; ++b)
    {
        int64_t expectedSlot = offsets[static_cast<size_t>(b)];
        for(int64_t s = 0; s < seqExtent(offsets, seqStride, b); ++s)
        {
            for(int64_t i = 0; i < dims[outerInner]; ++i)
            {
                for(int64_t j = 0; j < dims[innermost]; ++j)
                {
                    std::vector<int64_t> indices(dims.size());
                    indices[0] = b;
                    indices[seqIdx] = s;
                    indices[outerInner] = i;
                    indices[innermost] = j;

                    tensor.setHostValue(value, indices);
                    const auto* base = static_cast<const T*>(tensor.memory().hostData());
                    EXPECT_EQ(base[expectedSlot], value)
                        << "b=" << b << " s=" << s << " inner=(" << i << ", " << j << ")";
                    ++value;
                    ++expectedSlot;
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
