// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestRaggedTensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_ragged_test;

namespace
{

// BSHD-packed geometry: dims [B, S_max, H, D], strides {S*H*D, H*D, D, 1}.
// B=2: batch0 seq=2, batch1 seq=3, seqStride = H*D = 4.
const std::vector<int64_t> kDims = {2, 3, 2, 2};
const std::vector<int64_t> kStrides = {12, 4, 2, 1};
const std::vector<int64_t> kOffsets = {0, 8, 20}; // off[B] = 20

} // namespace

// ============================================================================
// Addressing math (parameterized over int32 / int64 aux)
// ============================================================================

template <typename IndexT>
class RaggedTensorTyped : public ::testing::Test
{
};

using IndexTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(RaggedTensorTyped, IndexTypes, );

TYPED_TEST(RaggedTensorTyped, Addressing)
{
    auto aux = makeOffsetAux<TypeParam>(kOffsets);
    RaggedTensor<float> tensor(kDims, kStrides, aux);
    tensor.fillWithValue(0.0f);

    checkAddressing(tensor, kDims, kStrides, kOffsets);
}

TYPED_TEST(RaggedTensorTyped, Iteration)
{
    auto aux = makeOffsetAux<TypeParam>(kOffsets);
    RaggedTensor<float> tensor(kDims, kStrides, aux);
    tensor.fillWithValue(0.0f);

    checkIteration(tensor, kOffsets);
}

TYPED_TEST(RaggedTensorTyped, Reporting)
{
    auto aux = makeOffsetAux<TypeParam>(kOffsets);
    const RaggedTensor<float> tensor(kDims, kStrides, aux);

    checkReporting(tensor, kOffsets.back());
}

// ============================================================================
// getIndex maps to readOffset(b) + within-batch offset
// ============================================================================

TEST(TestRaggedTensor, GetIndexUsesRaggedBase)
{
    auto aux = makeOffsetAux<int32_t>(kOffsets);
    const RaggedTensor<float> tensor(kDims, kStrides, aux);

    // batch 0 base 0: {0,1,1,1} -> 0 + 1*4 + 1*2 + 1 = 7
    EXPECT_EQ(tensor.getIndex(0, 1, 1, 1), 7);
    // batch 1 base 8: {1,2,1,1} -> 8 + 2*4 + 1*2 + 1 = 19
    EXPECT_EQ(tensor.getIndex(1, 2, 1, 1), 19);
    // bare batch index bases at ragged_offset[b]
    EXPECT_EQ(tensor.getIndex(1), 8);
}

// ============================================================================
// Empty batches are skipped during iteration
// ============================================================================

TEST(TestRaggedTensor, EmptyBatchSkipped)
{
    // B=3: batch0 seq=1, batch1 empty, batch2 seq=1. seqStride=4.
    const std::vector<int64_t> dims = {3, 3, 2, 2};
    const std::vector<int64_t> strides = {12, 4, 2, 1};
    const std::vector<int64_t> offsets = {0, 4, 4, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkIteration(tensor, offsets);
}

TEST(TestRaggedTensor, LeadingAndTrailingEmptyBatches)
{
    // B=4: batch0 empty, batch1 seq=1, batch2 seq=1, batch3 empty.
    const std::vector<int64_t> dims = {4, 2, 2, 2};
    const std::vector<int64_t> strides = {8, 4, 2, 1};
    const std::vector<int64_t> offsets = {0, 0, 4, 8, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkIteration(tensor, offsets);
}

TEST(TestRaggedTensor, AllEmptyBatchesBeginEqualsEnd)
{
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {8, 4, 2, 1};
    const std::vector<int64_t> offsets = {0, 0, 0};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, aux);

    EXPECT_EQ(tensor.elementCount(), 0u);
    EXPECT_EQ(tensor.begin(), tensor.end());
}

// ============================================================================
// BHSD logical order (sequence axis at logical index 2)
// ============================================================================

TEST(TestRaggedTensor, BhsdSequenceAxis)
{
    // Logical BHSD dims [B, H, S_max, D] over a physical-BSHD buffer.
    // Physical strides: B -> S*H*D, H -> D, S -> H*D, D -> 1.
    // B=2, H=2, S_max=3, D=2 => strides {12, 2, 4, 1}. seqStride = 4 (axis 2).
    const std::vector<int64_t> dims = {2, 2, 3, 2};
    const std::vector<int64_t> strides = {12, 2, 4, 1};
    const std::vector<int64_t> offsets = {0, 8, 20};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 20u);

    // Within-batch enumeration over a physical-BSHD buffer is still contiguous.
    checkIteration(tensor, offsets);

    // Addressing {b,h,s,d} -> readOffset(b) + h*D + s*(H*D) + d.
    // {1,1,2,1} -> 8 + 1*2 + 2*4 + 1 = 19
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 1), 19);
}

// ============================================================================
// physicalElementCount: optional (inferred) vs explicit
// ============================================================================

TEST(TestRaggedTensor, PhysicalElementCountInferredVsExplicit)
{
    auto auxInferred = makeOffsetAux<int32_t>(kOffsets);
    const RaggedTensor<float> inferred(kDims, kStrides, auxInferred);

    auto auxExplicit = makeOffsetAux<int32_t>(kOffsets);
    const RaggedTensor<float> explicitCount(kDims, kStrides, auxExplicit, static_cast<size_t>(20));

    EXPECT_EQ(inferred.elementSpace(), explicitCount.elementSpace());
    EXPECT_EQ(inferred.elementCount(), explicitCount.elementCount());
    EXPECT_EQ(inferred.dims(), explicitCount.dims());
    EXPECT_EQ(inferred.strides(), explicitCount.strides());
}

// ============================================================================
// raggedOffset() accessor
// ============================================================================

TEST(TestRaggedTensor, RaggedOffsetAccessor)
{
    auto aux = makeOffsetAux<int32_t>(kOffsets);
    const RaggedTensor<float> tensor(kDims, kStrides, aux);

    EXPECT_EQ(tensor.raggedOffset(), aux.get());
}

// ============================================================================
// Structural validation failures (RFC §4.5.5)
// ============================================================================

TEST(TestRaggedTensor, ValidationNullAuxThrows)
{
    EXPECT_THROW(const RaggedTensor<float> tensor(kDims, kStrides, nullptr),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationWrongElementCountThrows)
{
    // Aux with B (not B+1) entries.
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{2, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(kDims, kStrides, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationWrongRankThrows)
{
    // Rank-3 aux with elementCount B+1 == 3 (passes count check, fails rank check).
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{3, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(kDims, kStrides, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationBadElementSizeThrows)
{
    // int16_t aux -> elementSize 2, not in {4, 8}.
    auto aux16 = std::make_shared<Tensor<int16_t>>(std::vector<int64_t>{3, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(kDims, kStrides, aux16), std::invalid_argument);

    // int8_t aux -> elementSize 1.
    auto aux8 = std::make_shared<Tensor<int8_t>>(std::vector<int64_t>{3, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(kDims, kStrides, aux8), std::invalid_argument);
}

// ============================================================================
// Pinned allocator variant
// ============================================================================

TEST(TestRaggedTensor, PinnedVariantRoundTrips)
{
    auto aux = makeOffsetAux<int32_t>(kOffsets);
    PinnedRaggedTensor<float> tensor(kDims, kStrides, aux);
    tensor.fillWithValue(0.0f);

    tensor.setHostValue(42.0f, 1, 2, 1, 1);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 1, 1), 42.0f);

    checkIteration(tensor, kOffsets);
}

// ============================================================================
// fillWithData
// ============================================================================

TEST(TestRaggedTensor, FillWithData)
{
    auto aux = makeOffsetAux<int32_t>(kOffsets);
    RaggedTensor<int> tensor(kDims, kStrides, aux);

    std::vector<int> data(20);
    for(size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<int>(i);
    }
    tensor.fillWithData(data.data(), data.size() * sizeof(int));

    const auto* base = static_cast<const int*>(tensor.memory().hostData());
    for(size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_EQ(base[i], static_cast<int>(i));
    }
}
