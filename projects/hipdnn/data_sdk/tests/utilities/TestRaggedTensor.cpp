// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestRaggedTensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_ragged_test;

namespace
{

// BSHD-packed geometry: dims [B, S_max, H, D], strides {S*H*D, H*D, D, 1}.
// B=2: batch0 seq=2, batch1 seq=3, seqStride = H*D = 4.
const std::vector<int64_t> K_DIMS = {2, 3, 2, 2};
const std::vector<int64_t> K_STRIDES = {12, 4, 2, 1};
const std::vector<int64_t> K_OFFSETS = {0, 8, 20}; // off[B] = 20

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
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux);
    tensor.fillWithValue(0.0f);

    checkAddressing(tensor, K_DIMS, K_STRIDES, K_OFFSETS);
}

TYPED_TEST(RaggedTensorTyped, Iteration)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux);
    tensor.fillWithValue(0.0f);

    checkIteration(tensor, K_OFFSETS);
}

TYPED_TEST(RaggedTensorTyped, Reporting)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux);

    checkReporting(tensor, K_OFFSETS.back());
}

// ============================================================================
// getIndex maps to readOffset(b) + within-batch offset
// ============================================================================

TEST(TestRaggedTensor, GetIndexUsesRaggedBase)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux);

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
    auto auxInferred = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> inferred(K_DIMS, K_STRIDES, auxInferred);

    auto auxExplicit = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> explicitCount(
        K_DIMS, K_STRIDES, auxExplicit, static_cast<size_t>(20));

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
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux);

    EXPECT_EQ(tensor.raggedOffset(), aux.get());
}

// ============================================================================
// Structural validation failures (RFC §4.5.5)
// ============================================================================

TEST(TestRaggedTensor, ValidationNullAuxThrows)
{
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, nullptr),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationWrongElementCountThrows)
{
    // Aux with B (not B+1) entries.
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{2, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationWrongRankThrows)
{
    // Rank-3 aux with elementCount B+1 == 3 (passes count check, fails rank check).
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{3, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationBadElementSizeThrows)
{
    // int16_t aux -> elementSize 2, not in {4, 8}.
    auto aux16 = std::make_shared<Tensor<int16_t>>(std::vector<int64_t>{3, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux16), std::invalid_argument);

    // int8_t aux -> elementSize 1.
    auto aux8 = std::make_shared<Tensor<int8_t>>(std::vector<int64_t>{3, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux8), std::invalid_argument);
}

// ============================================================================
// Offset-content validation at construction (all build modes, ALMIOPEN-2124 §2.2)
// ============================================================================

TEST(TestRaggedTensor, ValidationOffsetZeroNotZeroThrows)
{
    // ragged_offset[0] must be 0.
    auto aux = makeOffsetAux<int32_t>({4, 8, 12});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationNonMonotonicThrows)
{
    // off[2] < off[1] -> negative block.
    auto aux = makeOffsetAux<int32_t>({0, 8, 4});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationBlockNotDivisibleThrows)
{
    // seqStride = H*D = 4; a per-batch block of 2 is not a whole number of rows.
    auto aux = makeOffsetAux<int32_t>({0, 2, 4});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationExtentExceedsSmaxThrows)
{
    // seqStride = 4, S_max = dims[1] = 3; block 16 -> extent 4 > 3.
    auto aux = makeOffsetAux<int32_t>({0, 16, 32});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationExplicitPhysicalElementCountMismatchThrows)
{
    // Explicit physicalElementCount must equal ragged_offset[B] (20).
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux, static_cast<size_t>(24)),
                 std::invalid_argument);
}

// ============================================================================
// elementCount() reports ragged_offset[B]; iteration is per-batch ascending (BSHD)
// ============================================================================

TEST(TestRaggedTensor, IterationIsPerBatchAscendingForBshd)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux);
    tensor.fillWithValue(0.0f);

    const auto* base = static_cast<const float*>(tensor.memory().hostData());
    std::vector<int64_t> visited;
    for(auto it = tensor.begin(); it != tensor.end(); ++it)
    {
        visited.push_back(static_cast<const float*>(*it) - base);
    }

    // Physical-BSHD buffer: visit order is the contiguous ascending [0, off[B]).
    std::vector<int64_t> expected(static_cast<size_t>(K_OFFSETS.back()));
    std::iota(expected.begin(), expected.end(), int64_t{0});
    EXPECT_EQ(visited, expected);
}

// ============================================================================
// Pinned allocator variant
// ============================================================================

TEST(TestRaggedTensor, PinnedVariantRoundTrips)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    PinnedRaggedTensor<float> tensor(K_DIMS, K_STRIDES, aux);
    tensor.fillWithValue(0.0f);

    tensor.setHostValue(42.0f, 1, 2, 1, 1);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 1, 1), 42.0f);

    checkIteration(tensor, K_OFFSETS);
}

// ============================================================================
// fillWithData
// ============================================================================

TEST(TestRaggedTensor, FillWithData)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<int> tensor(K_DIMS, K_STRIDES, aux);

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
