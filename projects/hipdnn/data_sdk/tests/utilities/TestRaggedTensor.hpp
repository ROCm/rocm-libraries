// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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
inline const std::vector<int64_t> K_ORIGINS = {0, 8};
inline const std::vector<int64_t> K_LENGTHS = {2, 3};

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

/// elementCount/elementSpace/isPacked reporting (RFC 0014 §4.5 items 6 and 7).
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

/// One ragged layout with the values the layout must satisfy. Offsets are the stored
/// values (the multiplier is not applied); `seqLen` binds per-batch lengths when present.
/// Rank 4, dims [B, H, S, D] unless `seqAxis` says otherwise.
struct LayoutCase
{
    std::string name;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> offsets;
    int64_t multiplier{1};
    std::optional<std::vector<int64_t>> seqLen;
    int seqAxis{SDPA_SEQ_AXIS};

    // Expected: lengths/count/space for an accepted layout, or `rule` (a substring of the
    // failure message) for a rejected one.
    std::vector<int64_t> lengths;
    size_t count{0};
    size_t space{0};
    std::string rule;
};

/// The layout rows of RFC 0014 §4.5.1 (L1-L15 and their variants); the values come from the
/// RFC formulas, not from the container.
inline const std::vector<LayoutCase>& layoutTable()
{
    static const std::vector<LayoutCase> s_table = {
        {"L1",
         {2, 2, 4, 2},
         {0, 2, 4, 1},
         {0, 8, 20},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         20,
         20,
         ""},
        {"L2",
         {2, 4, 3, 8},
         {0, 8, 96, 1},
         {0, 192, 480},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         160,
         416,
         ""},
        {"L3",
         {2, 4, 3, 8},
         {0, 8, 96, 1},
         {0, 0, 0},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {0, 0},
         0,
         0,
         ""},
        {"L4",
         {2, 4, 8, 1},
         {20, 1, 4, 1},
         {0, 20, 40},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {5, 5},
         40,
         40,
         ""},
        {"L5",
         {2, 4, 5, 8},
         {0, 8, 32, 1},
         {0, 256, 512},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {5, 5},
         320,
         416,
         ""},
        {"L5s",
         {2, 4, 5, 8},
         {0, 8, 32, 1},
         {0, 256, 512},
         1,
         std::vector<int64_t>{3, 5},
         SDPA_SEQ_AXIS,
         {3, 5},
         256,
         416,
         ""},
        {"L6",
         {2, 2, 5, 2},
         {0, 10, 2, 1},
         {0, 4, 10},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         20,
         20,
         ""},
        {"L7a",
         {2, 4, 3, 1},
         {0, 5, 1, 1},
         {0, 2, 5},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         20,
         20,
         ""},
        {"L7b",
         {2, 4, 3, 1},
         {0, 1, 4, 1},
         {0, 8, 20},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         20,
         20,
         ""},
        {"L7c",
         {2, 4, 3, 1},
         {0, 10, 1, 1},
         {0, 2, 5},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         20,
         35,
         ""},
        {"L7d",
         {2, 2, 3, 1},
         {0, 7, 1, 1},
         {0, 4, 7},
         1,
         std::vector<int64_t>{2, 3},
         SDPA_SEQ_AXIS,
         {2, 3},
         10,
         14,
         ""},
        {"L8",
         {2, 1, 4, 2},
         {0, 2, 2, 1},
         {4, 8, 14},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         10,
         14,
         ""},
        {"L9",
         {2, 1, 4, 2},
         {0, 2, 2, 1},
         {10, 0, 0},
         1,
         std::vector<int64_t>{2, 3},
         SDPA_SEQ_AXIS,
         {2, 3},
         10,
         14,
         ""},
        {"L9m",
         {2, 1, 4, 2},
         {0, 2, 2, 1},
         {5, 0, 0},
         2,
         std::vector<int64_t>{2, 3},
         SDPA_SEQ_AXIS,
         {2, 3},
         10,
         14,
         ""},
        {"L7bm",
         {2, 4, 3, 1},
         {0, 1, 4, 1},
         {0, 2, 5},
         4,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         20,
         20,
         ""},
        {"L10",
         {2, 1, 4, 2},
         {0, 2, 2, 1},
         {0, 2, 0},
         1,
         std::vector<int64_t>{3, 3},
         SDPA_SEQ_AXIS,
         {},
         0,
         0,
         "overlap"},
        {"L11",
         {2, 2, 4, 2},
         {0, 8, 2, 1},
         {0, 4, 10},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {},
         0,
         0,
         "sorted-stride"},
        {"L12",
         {2, 2, 3, 2},
         {0, 0, 2, 1},
         {0, 4, 10},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         20,
         10,
         ""},
        {"L13",
         {2, 1, 4, 2},
         {0, 2, 2, 1},
         {0, 0, 6},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {0, 3},
         6,
         6,
         ""},
        {"L14",
         {2, 1, 4, 2},
         {0, 2, 2, 1},
         {0, 4, 12},
         1,
         std::vector<int64_t>{2, 3},
         SDPA_SEQ_AXIS,
         {2, 3},
         10,
         10,
         ""},
        {"L15",
         {2, 1, 4, 2},
         {0, 2, 2, 1},
         {6, 0, 0},
         1,
         std::vector<int64_t>{0, 3},
         SDPA_SEQ_AXIS,
         {0, 3},
         6,
         6,
         ""},
    };
    return s_table;
}

/// Packed layouts the container accepted before the footprint model, with count == space ==
/// the last offset; the footprint derivation must give the same values.
inline const std::vector<LayoutCase>& packedFixtures()
{
    static const std::vector<LayoutCase> s_fixtures = {
        {"PackedCanonical",
         K_DIMS,
         K_STRIDES,
         K_OFFSETS,
         1,
         std::nullopt,
         K_SEQ_AXIS,
         K_LENGTHS,
         20,
         20,
         ""},
        {"PackedTokenOffsets",
         K_DIMS,
         K_STRIDES,
         {0, 2, 5},
         4,
         std::nullopt,
         K_SEQ_AXIS,
         K_LENGTHS,
         20,
         20,
         ""},
        {"PackedEmptyBatch",
         {3, 2, 3, 2},
         {12, 2, 4, 1},
         {0, 4, 4, 8},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {1, 0, 1},
         8,
         8,
         ""},
        {"PackedLeadingTrailingEmpty",
         {4, 2, 2, 2},
         {8, 2, 4, 1},
         {0, 0, 4, 8, 8},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {0, 1, 1, 0},
         8,
         8,
         ""},
        {"PackedAllEmpty",
         {2, 2, 2, 2},
         {8, 2, 4, 1},
         {0, 0, 0},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {0, 0},
         0,
         0,
         ""},
        {"PackedSingleBatch",
         {1, 2, 3, 2},
         {12, 2, 4, 1},
         {0, 8},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2},
         8,
         8,
         ""},
        {"PackedSingleRow",
         {2, 2, 1, 2},
         {4, 2, 4, 1},
         {0, 4, 8},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {1, 1},
         8,
         8,
         ""},
        {"PackedSingletonHead",
         {2, 1, 3, 2},
         {6, 2, 2, 1},
         {0, 4, 10},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         10,
         10,
         ""},
        {"PackedTokenMajorStats",
         {2, 2, 3, 1},
         {6, 1, 2, 1},
         {0, 4, 10},
         1,
         std::nullopt,
         SDPA_SEQ_AXIS,
         {2, 3},
         10,
         10,
         ""},
        {"PackedPhysicalBshd",
         {2, 3, 2, 2},
         {12, 4, 2, 1},
         K_OFFSETS,
         1,
         std::nullopt,
         BSHD_SEQ_AXIS,
         {2, 3},
         20,
         20,
         ""},
    };
    return s_fixtures;
}

/// Brute-force oracle for a layout: every logical element `(b, s < len[b], other axes)` at
/// `offsets[b] * multiplier + sum(idx * stride)`, in no particular order.
inline std::vector<int64_t> expectedAddresses(const LayoutCase& layout)
{
    std::vector<size_t> otherAxes;
    for(size_t axis = 1; axis < layout.dims.size(); ++axis)
    {
        if(axis != static_cast<size_t>(layout.seqAxis))
        {
            otherAxes.push_back(axis);
        }
    }

    std::vector<int64_t> addresses;
    for(size_t b = 0; b < layout.lengths.size(); ++b)
    {
        const int64_t origin = layout.offsets[b] * layout.multiplier;
        for(int64_t s = 0; s < layout.lengths[b]; ++s)
        {
            std::vector<int64_t> idx(otherAxes.size(), 0);
            while(true)
            {
                int64_t address = origin + s * layout.strides[static_cast<size_t>(layout.seqAxis)];
                for(size_t i = 0; i < otherAxes.size(); ++i)
                {
                    address += idx[i] * layout.strides[otherAxes[i]];
                }
                addresses.push_back(address);

                size_t i = otherAxes.size();
                while(i > 0 && ++idx[i - 1] == layout.dims[otherAxes[i - 1]])
                {
                    idx[--i] = 0;
                }
                if(i == 0)
                {
                    break;
                }
            }
        }
    }
    return addresses;
}

/// Iterates the tensor and checks the addresses against the layout: the visit count equals
/// the element count; distinct addresses equal count / prod(zero-stride extents); the largest
/// address plus one equals the element space (so every address is inside the buffer); and the
/// visited multiset equals the brute-force enumeration.
template <typename T>
inline void checkBruteForce(TensorBase<T>& tensor, const LayoutCase& layout)
{
    tensor.fillWithValue(T{});
    const auto* base = static_cast<const T*>(tensor.memory().hostData());

    std::vector<int64_t> visited;
    for(auto it = tensor.begin(); it != tensor.end(); ++it)
    {
        visited.push_back(static_cast<const T*>(*it) - base);
    }
    ASSERT_EQ(visited.size(), layout.count);

    auto expected = expectedAddresses(layout);
    std::sort(visited.begin(), visited.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(visited, expected);

    if(layout.count == 0)
    {
        EXPECT_EQ(layout.space, 0u);
        return;
    }

    int64_t zeroStrideExtent = 1;
    for(size_t axis = 1; axis < layout.dims.size(); ++axis)
    {
        if(layout.strides[axis] == 0)
        {
            zeroStrideExtent *= layout.dims[axis];
        }
    }
    EXPECT_EQ(visited.back() + 1, static_cast<int64_t>(layout.space));
    const auto distinct
        = static_cast<int64_t>(std::unique(visited.begin(), visited.end()) - visited.begin());
    EXPECT_EQ(distinct, static_cast<int64_t>(layout.count) / zeroStrideExtent);
}

} // namespace hipdnn_ragged_test
