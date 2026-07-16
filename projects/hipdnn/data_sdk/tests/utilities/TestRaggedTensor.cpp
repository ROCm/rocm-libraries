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
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_ragged_test;

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
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux);
    tensor.fillWithValue(0.0f);

    checkAddressing(tensor, K_DIMS, K_STRIDES, K_OFFSETS);
}

TYPED_TEST(RaggedTensorTyped, Iteration)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux);
    tensor.fillWithValue(0.0f);

    checkIteration(tensor, K_OFFSETS);
}

TYPED_TEST(RaggedTensorTyped, Reporting)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux);

    checkReporting(tensor, K_OFFSETS.back());
}

// ============================================================================
// getIndex maps to readOffset(b) + within-batch offset
// ============================================================================

TEST(TestRaggedTensor, GetIndexUsesRaggedBase)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux);

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
    RaggedTensor<float> tensor(dims, strides, 1, aux);
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
    RaggedTensor<float> tensor(dims, strides, 1, aux);
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
    RaggedTensor<float> tensor(dims, strides, 1, aux);

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

    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(dims, strides, 2, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 20u);

    // Within-batch enumeration over a physical-BSHD buffer is still contiguous.
    checkIteration(tensor, K_OFFSETS);

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
    const RaggedTensor<float> inferred(K_DIMS, K_STRIDES, 1, auxInferred);

    auto auxExplicit = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> explicitCount(
        K_DIMS, K_STRIDES, 1, auxExplicit, static_cast<size_t>(20));

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
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux);

    EXPECT_EQ(tensor.raggedOffset(), aux.get());
}

// ============================================================================
// Structural validation failures (RFC 0014 §4.5.5)
// ============================================================================

TEST(TestRaggedTensor, ValidationNullAuxThrows)
{
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, nullptr),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationWrongElementCountThrows)
{
    // Aux with B (not B+1) entries.
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{2, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationWrongRankThrows)
{
    // Rank-3 aux with elementCount B+1 == 3 (passes count check, fails rank check).
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{3, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationBadElementSizeThrows)
{
    // int16_t aux -> elementSize 2, not in {4, 8}.
    auto aux16 = std::make_shared<Tensor<int16_t>>(std::vector<int64_t>{3, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux16),
                 std::invalid_argument);

    // int8_t aux -> elementSize 1.
    auto aux8 = std::make_shared<Tensor<int8_t>>(std::vector<int64_t>{3, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux8),
                 std::invalid_argument);
}

// ============================================================================
// Offset-content validation at construction (all build modes, RFC 0014 §4.5)
// ============================================================================

TEST(TestRaggedTensor, ValidationOffsetZeroNotZeroThrows)
{
    // ragged_offset[0] must be 0.
    auto aux = makeOffsetAux<int32_t>({4, 8, 12});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationNonMonotonicThrows)
{
    // off[2] < off[1] -> negative block.
    auto aux = makeOffsetAux<int32_t>({0, 8, 4});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationBlockNotDivisibleThrows)
{
    // seqStride = H*D = 4; a per-batch block of 2 is not a whole number of rows.
    auto aux = makeOffsetAux<int32_t>({0, 2, 4});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationExtentExceedsSmaxThrows)
{
    // seqStride = 4, S_max = dims[1] = 3; block 16 -> extent 4 > 3.
    auto aux = makeOffsetAux<int32_t>({0, 16, 32});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationExplicitPhysicalElementCountMismatchThrows)
{
    // Explicit physicalElementCount must equal ragged_offset[B] (20).
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    EXPECT_THROW(
        const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux, static_cast<size_t>(24)),
        std::invalid_argument);
}

// ============================================================================
// elementCount() reports ragged_offset[B]; iteration is per-batch ascending (BSHD)
// ============================================================================

TEST(TestRaggedTensor, IterationIsPerBatchAscendingForBshd)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux);
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

TEST(TestGpuRaggedTensor, PinnedVariantRoundTrips)
{
    SKIP_IF_NO_DEVICES();

    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    PinnedRaggedTensor<float> tensor(K_DIMS, K_STRIDES, 1, aux);
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
    RaggedTensor<int> tensor(K_DIMS, K_STRIDES, 1, aux);

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

// ============================================================================
// Offsets wider than INT32_MAX (int64 aux) survive the type-erased read
// ============================================================================

TEST(TestRaggedTensor, LargeOffsetExceedsInt32Max)
{
    const int64_t largeOffset = static_cast<int64_t>(INT32_MAX) + 1; // 2^31

    // B=1 with seqStride == off[B] so a single sequence row satisfies validation.
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const std::vector<int64_t> strides = {largeOffset, largeOffset, 1, 1};
    auto aux = makeOffsetAux<int64_t>({0, largeOffset});

    // Shallow (borrowed) so no buffer is allocated for the ~2^31 element span; only
    // getIndex is exercised, which reads the offset without touching the backing memory.
    float backing{};
    const ShallowRaggedTensor<float> tensor(&backing, dims, strides, 1, aux);

    EXPECT_EQ(tensor.getIndex(1), largeOffset);
    EXPECT_EQ(tensor.getIndex(0), 0);
}

// ============================================================================
// Two ragged tensors sharing one ragged_offset aux
// ============================================================================

TEST(TestRaggedTensor, SharedAuxBacksTwoTensors)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);

    const RaggedTensor<float> first(K_DIMS, K_STRIDES, 1, aux);
    const RaggedTensor<float> second(K_DIMS, K_STRIDES, 1, aux);

    EXPECT_EQ(first.raggedOffset(), aux.get());
    EXPECT_EQ(second.raggedOffset(), aux.get());
    EXPECT_GE(aux.use_count(), 3L); // caller + both tensors

    EXPECT_EQ(first.getIndex(1, 2, 1, 1), 19);
    EXPECT_EQ(second.getIndex(1, 2, 1, 1), 19);
}

// ============================================================================
// Single batch (B == 1): batch-carry in operator++ with rowOffsets of size 2
// ============================================================================

TEST(TestRaggedTensor, SingleBatch)
{
    const std::vector<int64_t> dims = {1, 3, 2, 2};
    const std::vector<int64_t> strides = {12, 4, 2, 1};
    const std::vector<int64_t> offsets = {0, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, 1, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkAddressing(tensor, dims, strides, offsets);
    checkIteration(tensor, offsets);
}

// ============================================================================
// Single-row sequences (S_max == 1): one sequence row per batch
// ============================================================================

TEST(TestRaggedTensor, SingleRowSequences)
{
    const std::vector<int64_t> dims = {2, 1, 2, 2};
    const std::vector<int64_t> strides = {4, 4, 2, 1};
    const std::vector<int64_t> offsets = {0, 4, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, 1, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkAddressing(tensor, dims, strides, offsets);
    checkIteration(tensor, offsets);
}

// ============================================================================
// Structural/offset validation: empty dims and non-positive sequence stride
// ============================================================================

TEST(TestRaggedTensor, ValidationEmptyPaddedDimsThrows)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    EXPECT_THROW(const RaggedTensor<float> tensor({}, K_STRIDES, 1, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationNonPositiveSequenceStrideThrows)
{
    // All non-batch strides negative -> the sequence axis stride is negative.
    const std::vector<int64_t> strides = {12, -1, -2, -3};
    auto aux = makeOffsetAux<int32_t>({0, 0, 0});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, strides, 1, aux), std::invalid_argument);
}

// ============================================================================
// Layout constructor: strides + sequence axis derived from the TensorLayout
// ============================================================================

namespace
{

// The sequence axis reported by the iterator is read from the tensor's RaggedCompositeIndex.
int seqAxisOf(RaggedTensor<float>& tensor)
{
    return std::get<ITensorIterator<false>::RaggedCompositeIndex>(tensor.begin().index()).seqAxis;
}

} // namespace

TEST(TestRaggedTensor, LayoutBshdDerivesStridesAndSeqAxis)
{
    // BSHD dims [B, S_max, H, D] -> sequence axis at logical index 2 over physical-BSHD memory.
    const std::vector<int64_t> dims = {2, 2, 3, 2};

    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(dims, TensorLayout::BSHD, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{12, 2, 4, 1}));
    EXPECT_EQ(seqAxisOf(tensor), 2);
    checkIteration(tensor, K_OFFSETS);
}

TEST(TestRaggedTensor, LayoutBhsdDerivesStridesAndSeqAxis)
{
    // BHSD dims [B, H, S_max, D] -> sequence axis at logical index 1.
    const std::vector<int64_t> dims = {2, 3, 2, 2};

    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(dims, TensorLayout::BHSD, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{12, 4, 2, 1}));
    EXPECT_EQ(seqAxisOf(tensor), 1);
    checkIteration(tensor, K_OFFSETS);
}

// ============================================================================
// Degenerate H=1: the sequence axis must come from the declared layout/axis, not
// from a stride scan (which ties H and S when H == 1 and mislabels the axis).
// ============================================================================

TEST(TestRaggedTensor, SingletonHeadLayoutSelectsSequenceAxis)
{
    // dims [B=2, S_max=3, H=1, D=2] under BSHD -> strides {6,2,2,1}, sequence axis 2.
    const std::vector<int64_t> dims = {2, 1, 3, 2};
    const std::vector<int64_t> offsets = {0, 4, 10};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, TensorLayout::BSHD, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{6, 2, 2, 1}));
    EXPECT_EQ(seqAxisOf(tensor), 2);
    EXPECT_EQ(tensor.elementCount(), 10u);
    checkIteration(tensor, offsets);
}

TEST(TestRaggedTensor, SingletonHeadExplicitSeqAxis)
{
    // Same geometry via the primary constructor with an explicit sequence axis.
    const std::vector<int64_t> dims = {2, 1, 3, 2};
    const std::vector<int64_t> strides = {6, 2, 2, 1};
    const std::vector<int64_t> offsets = {0, 4, 10};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, 2, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(seqAxisOf(tensor), 2);
    EXPECT_EQ(tensor.elementCount(), 10u);
    checkIteration(tensor, offsets);
}

// ============================================================================
// Structural validation: rank, strides/dims size, and sequence axis range
// ============================================================================

TEST(TestRaggedTensor, ValidationRankBelowTwoThrows)
{
    auto aux = makeOffsetAux<int32_t>({0, 0});
    EXPECT_THROW(const RaggedTensor<float> tensor({4}, {1}, 1, aux), std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationStridesDimsSizeMismatchThrows)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, {12, 4, 2}, 1, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationSeqAxisOutOfRangeThrows)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    // Batch axis (0) is not a valid sequence axis.
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 0, aux),
                 std::invalid_argument);
    // Sequence axis must be strictly less than the rank.
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, 4, aux),
                 std::invalid_argument);
}
