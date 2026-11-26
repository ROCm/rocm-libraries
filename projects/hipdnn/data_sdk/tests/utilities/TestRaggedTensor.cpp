// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestRaggedTensor.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
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
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    checkAddressing(tensor, K_DIMS, K_STRIDES, K_SEQ_AXIS, K_OFFSETS);
}

TYPED_TEST(RaggedTensorTyped, Iteration)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    checkIteration(tensor, K_OFFSETS);
}

TYPED_TEST(RaggedTensorTyped, Reporting)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);

    checkReporting(tensor, K_OFFSETS.back());
}

// ============================================================================
// getIndex maps to origin(b) + within-batch offset
// ============================================================================

TEST(TestRaggedTensor, GetIndexUsesRaggedBase)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);

    // batch 0 base 0: {b=0, h=1, s=1, d=1} -> 0 + 1*2 + 1*4 + 1 = 7
    EXPECT_EQ(tensor.getIndex(0, 1, 1, 1), 7);
    // batch 1 base 8: {b=1, h=1, s=2, d=1} -> 8 + 1*2 + 2*4 + 1 = 19
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 1), 19);
    // bare batch index bases at the batch's origin
    EXPECT_EQ(tensor.getIndex(1), 8);
}

// ============================================================================
// Empty batches are skipped during iteration
// ============================================================================

TEST(TestRaggedTensor, EmptyBatchSkipped)
{
    // B=3: batch0 seq=1, batch1 empty, batch2 seq=1. seqStride=4.
    const std::vector<int64_t> dims = {3, 2, 3, 2};
    const std::vector<int64_t> strides = {12, 2, 4, 1};
    const std::vector<int64_t> offsets = {0, 4, 4, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, SDPA_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkIteration(tensor, offsets);
}

TEST(TestRaggedTensor, LeadingAndTrailingEmptyBatches)
{
    // B=4: batch0 empty, batch1 seq=1, batch2 seq=1, batch3 empty.
    const std::vector<int64_t> dims = {4, 2, 2, 2};
    const std::vector<int64_t> strides = {8, 2, 4, 1};
    const std::vector<int64_t> offsets = {0, 0, 4, 8, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, SDPA_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkIteration(tensor, offsets);
}

TEST(TestRaggedTensor, AllEmptyBatchesBeginEqualsEnd)
{
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {8, 2, 4, 1};
    const std::vector<int64_t> offsets = {0, 0, 0};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, SDPA_SEQ_AXIS, aux);

    EXPECT_EQ(tensor.elementCount(), 0u);
    EXPECT_EQ(tensor.begin(), tensor.end());
}

// ============================================================================
// physicalElementCount: optional (inferred) vs explicit
// ============================================================================

TEST(TestRaggedTensor, PhysicalElementCountInferredVsExplicit)
{
    auto auxInferred = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> inferred(K_DIMS, K_STRIDES, K_SEQ_AXIS, auxInferred);

    auto auxExplicit = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> explicitCount(
        K_DIMS, K_STRIDES, K_SEQ_AXIS, auxExplicit, static_cast<size_t>(20));

    EXPECT_EQ(inferred.elementSpace(), 20u);
    EXPECT_EQ(inferred.elementSpace(), explicitCount.elementSpace());
    EXPECT_EQ(inferred.elementCount(), explicitCount.elementCount());
    EXPECT_EQ(inferred.dims(), explicitCount.dims());
    EXPECT_EQ(inferred.strides(), explicitCount.strides());
}

// ============================================================================
// Strided aux: value i sits at physical element i * strides()[0]
// ============================================================================

namespace
{

// INT32 [count, 1, 1, 1] aux with batch stride `stride`; the slots between values hold a poison
// no layout accepts.
std::shared_ptr<ITensor> makeStridedAux(const std::vector<int64_t>& values, int64_t stride)
{
    const auto count = static_cast<int64_t>(values.size());
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{count, 1, 1, 1},
                                                 std::vector<int64_t>{stride, 1, 1, 1});
    std::fill_n(aux->memory().hostData(), aux->memory().count(), int32_t{-1});
    for(int64_t i = 0; i < count; ++i)
    {
        aux->memory().hostData()[i * stride] = static_cast<int32_t>(values[static_cast<size_t>(i)]);
    }
    aux->memory().markHostModified();
    return aux;
}

// Writes `values` into the first slots of a malformed aux so that only the structural guard
// under test can reject it.
void fillLeading(Tensor<int32_t>& aux, const std::vector<int64_t>& values)
{
    for(size_t i = 0; i < values.size(); ++i)
    {
        aux.memory().hostData()[i] = static_cast<int32_t>(values[i]);
    }
    aux.memory().markHostModified();
}

} // namespace

TEST(TestRaggedTensor, StridedOffsetsMatchPacked)
{
    const RaggedTensor<float> packed(
        K_DIMS, K_STRIDES, K_SEQ_AXIS, makeOffsetAux<int32_t>(K_OFFSETS));
    const RaggedTensor<float> strided(K_DIMS, K_STRIDES, K_SEQ_AXIS, makeStridedAux(K_OFFSETS, 2));

    EXPECT_EQ(strided.raggedIterationInfo(), packed.raggedIterationInfo());
    EXPECT_EQ(strided.elementCount(), packed.elementCount());
    EXPECT_EQ(strided.elementSpace(), packed.elementSpace());
}

TEST(TestRaggedTensor, StridedLengthsMatchPacked)
{
    // Blocks of 8 rows' worth of elements with 3 and 2 live rows (seq stride 4).
    const std::vector<int64_t> blocks = {0, 12, 24};
    const std::vector<int64_t> lens = {3, 2};
    const RaggedTensor<float> packed(K_DIMS,
                                     K_STRIDES,
                                     K_SEQ_AXIS,
                                     makeOffsetAux<int32_t>(blocks),
                                     std::nullopt,
                                     1,
                                     makeStridedAux(lens, 1));
    const RaggedTensor<float> strided(K_DIMS,
                                      K_STRIDES,
                                      K_SEQ_AXIS,
                                      makeStridedAux(blocks, 3),
                                      std::nullopt,
                                      1,
                                      makeStridedAux(lens, 2));

    EXPECT_EQ(strided.raggedIterationInfo(), packed.raggedIterationInfo());
    EXPECT_EQ(strided.raggedIterationInfo()->lengths, lens);
    EXPECT_EQ(strided.elementCount(), packed.elementCount());
    EXPECT_EQ(strided.elementSpace(), packed.elementSpace());
}

TEST(TestRaggedTensor, ZeroAuxStrideThrowsWithMultipleBatches)
{
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{3, 1, 1, 1},
                                                 std::vector<int64_t>{0, 1, 1, 1});
    // Every batch reads slot 0; offsets {0, 0, 0} are valid, so only the stride guard rejects.
    fillLeading(*aux, {0});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux),
                 std::invalid_argument);
}

// ============================================================================
// raggedOffset() accessor
// ============================================================================

TEST(TestRaggedTensor, RaggedOffsetAccessor)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);

    EXPECT_EQ(tensor.raggedOffset(), aux.get());
}

// ============================================================================
// Structural validation failures (RFC 0014 §4.5 item 5)
// ============================================================================

TEST(TestRaggedTensor, ValidationNullAuxThrows)
{
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, nullptr),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationWrongElementCountThrows)
{
    // Aux with B (not B+1) entries.
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{2, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationWrongRankThrows)
{
    // Rank-3 aux with elementCount B+1 == 3 (passes count check, fails rank check).
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{3, 1, 1});
    fillLeading(*aux, K_OFFSETS);
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux),
                 std::invalid_argument);
}

// Rank 4 and B+1 elements, but along axis 1: reading at the axis-0 stride would run past the
// buffer.
TEST(TestRaggedTensor, ValidationOffsetsOffTheBatchAxisThrows)
{
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{1, 3, 1, 1});
    fillLeading(*aux, K_OFFSETS);
    EXPECT_THAT([&] { const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                    ::testing::HasSubstr("ragged_offset must have shape")));
}

TEST(TestRaggedTensor, ValidationBadElementSizeThrows)
{
    // int16_t aux -> elementSize 2, not in {4, 8}.
    auto aux16 = std::make_shared<Tensor<int16_t>>(std::vector<int64_t>{3, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux16),
                 std::invalid_argument);

    // int8_t aux -> elementSize 1.
    auto aux8 = std::make_shared<Tensor<int8_t>>(std::vector<int64_t>{3, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux8),
                 std::invalid_argument);
}

// ============================================================================
// Offset-content validation at construction (all build modes, RFC 0014 §4.5)
// ============================================================================

TEST(TestRaggedTensor, ValidationNonMonotonicThrows)
{
    // off[2] < off[1] -> negative block; without lengths the blocks define the batches.
    auto aux = makeOffsetAux<int32_t>({0, 8, 4});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux),
                 std::invalid_argument);

    // With lengths only the origins are used: batch 0 at 0, batch 1 at 8, offset[B] unused.
    auto lengths = makeOffsetAux<int32_t>({2, 3});
    const RaggedTensor<float> bound(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux, std::nullopt, 1, lengths);
    EXPECT_EQ(bound.elementCount(), 20u);
    EXPECT_EQ(bound.elementSpace(), 20u);
}

TEST(TestRaggedTensor, ValidationBlockNotDivisibleThrows)
{
    // seqStride = H*D = 4; a per-batch block of 2 is not a whole number of rows.
    auto aux = makeOffsetAux<int32_t>({0, 2, 4});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationExplicitPhysicalElementCountMismatchThrows)
{
    // Explicit physicalElementCount must equal elementSpace (20).
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    EXPECT_THROW(const RaggedTensor<float> tensor(
                     K_DIMS, K_STRIDES, K_SEQ_AXIS, aux, static_cast<size_t>(24)),
                 std::invalid_argument);
}

// ============================================================================
// elementCount() reports ragged_offset[B]; iteration is per-batch ascending (physical BSHD)
// ============================================================================

TEST(TestRaggedTensor, IterationIsPerBatchAscendingForPhysicalBshd)
{
    const std::vector<int64_t> physicalDims = {2, 3, 2, 2};
    const std::vector<int64_t> physicalStrides = {12, 4, 2, 1};

    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(physicalDims, physicalStrides, BSHD_SEQ_AXIS, aux);
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
    PinnedRaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    tensor.setHostValue(42.0f, 1, 1, 2, 1);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 1, 2, 1), 42.0f);

    checkIteration(tensor, K_OFFSETS);
}

// ============================================================================
// fillWithData
// ============================================================================

TEST(TestRaggedTensor, FillWithData)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<int> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);

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

    // B=2 so batch 1's origin (ragged_offset[1] == 2^31) is an INTERIOR, addressable index:
    // elementSpace == 2^31 + 1 > 2^31, so getIndex(1) stays in bounds and exercises the
    // full addressing path (getIndex -> getIndexImpl -> origin snapshot).
    const int64_t twoLargeOffsets = largeOffset * 2; // 2^32
    const std::vector<int64_t> dims = {2, 1, 1, 1};
    const std::vector<int64_t> strides = {largeOffset, largeOffset, 1, 1};
    auto aux = makeOffsetAux<int64_t>({0, largeOffset, twoLargeOffsets});

    // Shallow (borrowed) so no buffer is allocated for the ~2^32 element span; only the
    // addressing math is exercised, without touching the backing memory.
    float backing{};
    const ShallowRaggedTensor<float> tensor(&backing, dims, strides, BSHD_SEQ_AXIS, aux);

    // getIndex(1) bases at ragged_offset[1] == 2^31; the returned index only equals 2^31 if
    // the int64 offset survived the type-erased read AND the addressing return path without
    // truncating to int32.
    EXPECT_EQ(tensor.getIndex(1), largeOffset);
    EXPECT_EQ(tensor.getIndex(0), 0);
}

// ============================================================================
// Two ragged tensors sharing one ragged_offset aux
// ============================================================================

TEST(TestRaggedTensor, SharedAuxBacksTwoTensors)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);

    const RaggedTensor<float> first(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);
    const RaggedTensor<float> second(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);

    EXPECT_EQ(first.raggedOffset(), aux.get());
    EXPECT_EQ(second.raggedOffset(), aux.get());
    EXPECT_GE(aux.use_count(), 3L); // caller + both tensors

    EXPECT_EQ(first.getIndex(1, 1, 2, 1), 19);
    EXPECT_EQ(second.getIndex(1, 1, 2, 1), 19);
}

// ============================================================================
// ragged_offset_multiplier: stored (token) offsets recovered to element units
// ============================================================================

// Token-unit offsets {0,2,5} with multiplier == seqStride (H*D == 4) recover the
// canonical element offsets {0,8,20}, so addressing, sizing, and iteration must all
// match the multiplier==1 element form exactly.
TEST(TestRaggedTensor, MultiplierRecoversElementOffsets)
{
    const int64_t multiplier = K_STRIDES[K_SEQ_AXIS]; // seqStride = H*D = 4
    const std::vector<int64_t> tokenOffsets = {0, 2, 5};

    auto aux = makeOffsetAux<int32_t>(tokenOffsets);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux, std::nullopt, multiplier);
    tensor.fillWithValue(0.0f);

    checkReporting(tensor, K_OFFSETS.back()); // token off[B]=5 -> 20 elements
    checkAddressing(tensor, K_DIMS, K_STRIDES, K_SEQ_AXIS, K_OFFSETS);
    checkIteration(tensor, K_OFFSETS);
}

// getIndex bases at the element offset (stored * multiplier), not the raw token offset.
TEST(TestRaggedTensor, MultiplierScalesGetIndexBase)
{
    const std::vector<int64_t> tokenOffsets = {0, 2, 5};
    auto aux = makeOffsetAux<int32_t>(tokenOffsets);
    const RaggedTensor<float> tensor(
        K_DIMS, K_STRIDES, K_SEQ_AXIS, aux, std::nullopt, /*raggedOffsetMultiplier=*/4);

    EXPECT_EQ(tensor.getIndex(1), 8); // token 2 * multiplier 4
    EXPECT_EQ(tensor.getIndex(0), 0);
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 1), 19); // 8 + 1*2 + 2*4 + 1
}

TEST(TestRaggedTensor, MultiplierBelowOneThrows)
{
    auto aux = makeOffsetAux<int32_t>({0, 2, 5});
    EXPECT_THROW(
        const RaggedTensor<float> tensor(
            K_DIMS, K_STRIDES, K_SEQ_AXIS, aux, std::nullopt, /*raggedOffsetMultiplier=*/0),
        std::invalid_argument);
}

// Per-tensor multiplier: one shared token-offset aux addresses two tensors with distinct
// multipliers (the D_qk != D_v case where Q and O share a token offset).
TEST(TestRaggedTensor, SharedAuxDistinctMultipliers)
{
    auto aux = makeOffsetAux<int32_t>({0, 1, 2}); // one token per batch

    // Q-like: H*D_qk = 4.
    const std::vector<int64_t> dimsQ = {2, 2, 1, 2};
    const std::vector<int64_t> stridesQ = {4, 2, 4, 1};
    // O-like: H*D_v = 8.
    const std::vector<int64_t> dimsO = {2, 2, 1, 4};
    const std::vector<int64_t> stridesO = {8, 4, 8, 1};

    const RaggedTensor<float> q(
        dimsQ, stridesQ, SDPA_SEQ_AXIS, aux, std::nullopt, /*raggedOffsetMultiplier=*/4);
    const RaggedTensor<float> o(
        dimsO, stridesO, SDPA_SEQ_AXIS, aux, std::nullopt, /*raggedOffsetMultiplier=*/8);

    EXPECT_EQ(q.getIndex(1), 4); // token 1 * 4
    EXPECT_EQ(o.getIndex(1), 8); // token 1 * 8
}

// ============================================================================
// Single batch (B == 1): batch-carry in operator++ with one length entry
// ============================================================================

TEST(TestRaggedTensor, SingleBatch)
{
    const std::vector<int64_t> dims = {1, 2, 3, 2};
    const std::vector<int64_t> strides = {12, 2, 4, 1};
    const std::vector<int64_t> offsets = {0, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, SDPA_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkAddressing(tensor, dims, strides, SDPA_SEQ_AXIS, offsets);
    checkIteration(tensor, offsets);
}

// ============================================================================
// Single-row sequences (S_max == 1): one sequence row per batch
// ============================================================================

TEST(TestRaggedTensor, SingleRowSequences)
{
    const std::vector<int64_t> dims = {2, 2, 1, 2};
    const std::vector<int64_t> strides = {4, 2, 4, 1};
    const std::vector<int64_t> offsets = {0, 4, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, SDPA_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkAddressing(tensor, dims, strides, SDPA_SEQ_AXIS, offsets);
    checkIteration(tensor, offsets);
}

// ============================================================================
// Structural/offset validation: empty dims and non-positive sequence stride
// ============================================================================

TEST(TestRaggedTensor, ValidationEmptyPaddedDimsThrows)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    EXPECT_THROW(const RaggedTensor<float> tensor({}, K_STRIDES, K_SEQ_AXIS, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationNonPositiveSequenceStrideThrows)
{
    // All non-batch strides negative -> the sequence axis stride is negative.
    const std::vector<int64_t> strides = {12, -1, -2, -3};
    auto aux = makeOffsetAux<int32_t>({0, 0, 0});
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, strides, K_SEQ_AXIS, aux),
                 std::invalid_argument);
}

// ============================================================================
// Sequence axis reported by the iterator matches the caller-provided seqAxis
// ============================================================================

namespace
{

// The sequence axis reported by the iterator is read from the tensor's RaggedCompositeIndex.
int seqAxisOf(RaggedTensor<float>& tensor)
{
    return std::get<ITensorIterator<false>::RaggedCompositeIndex>(tensor.begin().index()).seqAxis;
}

void expectConstructionRejected(const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& strides,
                                int seqAxis,
                                const std::vector<int64_t>& offsets,
                                const std::string& expectedMessage)
{
    auto aux = makeOffsetAux<int32_t>(offsets);
    try
    {
        const RaggedTensor<float> tensor(dims, strides, seqAxis, aux);
        ADD_FAILURE() << "expected std::invalid_argument containing: " << expectedMessage;
    }
    catch(const std::invalid_argument& e)
    {
        EXPECT_THAT(e.what(), ::testing::HasSubstr(expectedMessage));
    }
}

} // namespace

// ============================================================================
// Degenerate H=1: H and S share a stride, so the sequence axis must come from the
// caller-provided axis rather than a stride scan that would tie the two.
// ============================================================================

TEST(TestRaggedTensor, SingletonHeadExplicitSeqAxis)
{
    // Logical dims [B=2, H=1, S_max=3, D=2] over BSHD strides {6, 2, 2, 1}; H and S tie at
    // stride 2.
    const std::vector<int64_t> dims = {2, 1, 3, 2};
    const std::vector<int64_t> strides = {6, 2, 2, 1};
    const std::vector<int64_t> offsets = {0, 4, 10};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, SDPA_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(seqAxisOf(tensor), SDPA_SEQ_AXIS);
    EXPECT_EQ(tensor.elementCount(), 10u);
    checkIteration(tensor, offsets);
}

// ============================================================================
// Structural validation: rank, strides/dims size, and sequence axis range
// ============================================================================

TEST(TestRaggedTensor, ValidationRankBelowTwoThrows)
{
    auto aux = makeOffsetAux<int32_t>({0, 0});
    EXPECT_THROW(const RaggedTensor<float> tensor({4}, {1}, K_SEQ_AXIS, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, ValidationStridesDimsSizeMismatchThrows)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    EXPECT_THROW(const RaggedTensor<float> tensor(K_DIMS, {12, 2, 4}, K_SEQ_AXIS, aux),
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

// ============================================================================
// Sequence-outermost layout: one sequence row must fit inside the sequence stride
// ============================================================================

TEST(TestRaggedTensor, TokenMajorSingletonHeadDimAccepted)
{
    // Token-major LSE shape: logical [B=2, H=2, S_max=3, D=1], strides {6, 1, 2, 1}.
    const std::vector<int64_t> dims = {2, 2, 3, 1};
    const std::vector<int64_t> strides = {6, 1, 2, 1};
    const std::vector<int64_t> offsets = {0, 4, 10};

    auto aux = makeOffsetAux<int32_t>(offsets);
    RaggedTensor<float> tensor(dims, strides, SDPA_SEQ_AXIS, aux);
    tensor.fillWithValue(0.0f);

    EXPECT_EQ(tensor.elementCount(), 10u);
    checkAddressing(tensor, dims, strides, SDPA_SEQ_AXIS, offsets);
    checkIteration(tensor, offsets);
}

TEST(TestRaggedTensor, ZeroHeadDimSkipsSpanRule)
{
    // BHSD strides would fail the sorted-stride rule if the empty head axis were measured.
    const std::vector<int64_t> dims = {2, 0, 3, 2};
    const std::vector<int64_t> strides = {12, 6, 2, 1};

    auto aux = makeOffsetAux<int32_t>({0, 0, 0});
    RaggedTensor<float> tensor(dims, strides, SDPA_SEQ_AXIS, aux);

    EXPECT_EQ(tensor.elementCount(), 0u);
    EXPECT_EQ(tensor.elementSpace(), 0u);
    EXPECT_EQ(tensor.begin(), tensor.end());
}

TEST(TestRaggedTensor, PhysicalBhsdStridesThrow)
{
    // Head-major (BHSD) strides alias the sequence rows of one batch with the heads of the next.
    auto aux = makeOffsetAux<int32_t>({0, 4, 10});
    EXPECT_THROW(const RaggedTensor<float> tensor({2, 2, 3, 2}, {12, 6, 2, 1}, SDPA_SEQ_AXIS, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, SingleRowBhsdStridesThrow)
{
    auto aux = makeOffsetAux<int32_t>({0, 2, 4});
    EXPECT_THROW(const RaggedTensor<float> tensor({2, 2, 1, 2}, {4, 2, 2, 1}, SDPA_SEQ_AXIS, aux),
                 std::invalid_argument);
}

TEST(TestRaggedTensor, NegativeNonBatchStrideThrows)
{
    expectConstructionRejected(K_DIMS,
                               {12, -2, 4, 1},
                               K_SEQ_AXIS,
                               K_OFFSETS,
                               "ragged tensor strides must be non-negative");
}

// ============================================================================
// RaggedIterationInfo: per-batch extents, out-of-block positions, equality
// ============================================================================

TEST(TestRaggedIterationInfo, SeqExtentPerBatch)
{
    const RaggedIterationInfo info{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, K_STRIDES[K_SEQ_AXIS]};

    EXPECT_EQ(info.seqExtent(0), 2);
    EXPECT_EQ(info.seqExtent(1), 3);
}

TEST(TestRaggedIterationInfo, SeqExtentOutsideBatchRangeIsZero)
{
    const RaggedIterationInfo info{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, K_STRIDES[K_SEQ_AXIS]};

    EXPECT_EQ(info.seqExtent(-1), 0);
    EXPECT_EQ(info.seqExtent(2), 0);
}

TEST(TestRaggedIterationInfo, IsOutOfBlockComparesSeqIndexToBatchExtent)
{
    const RaggedIterationInfo info{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, K_STRIDES[K_SEQ_AXIS]};

    EXPECT_FALSE(info.isOutOfBlock({0, 1, 1, 1}));
    EXPECT_TRUE(info.isOutOfBlock({0, 0, 2, 0}));
    EXPECT_FALSE(info.isOutOfBlock({1, 1, 2, 1}));
}

TEST(TestRaggedIterationInfo, EmptyBatchIsEntirelyOutOfBlock)
{
    const RaggedIterationInfo info{{0, 0}, {0, 2}, K_SEQ_AXIS, K_STRIDES[K_SEQ_AXIS]};

    EXPECT_EQ(info.seqExtent(0), 0);
    EXPECT_TRUE(info.isOutOfBlock({0, 0, 0, 0}));
    EXPECT_FALSE(info.isOutOfBlock({1, 0, 0, 0}));
}

TEST(TestRaggedIterationInfo, EqualityComparesEveryField)
{
    const RaggedIterationInfo info{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, 4};

    EXPECT_TRUE(info == (RaggedIterationInfo{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, 4}));
    EXPECT_FALSE(info != (RaggedIterationInfo{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, 4}));

    EXPECT_TRUE(info != (RaggedIterationInfo{{0, 12}, K_LENGTHS, K_SEQ_AXIS, 4}));
    EXPECT_TRUE(info != (RaggedIterationInfo{K_ORIGINS, {2, 2}, K_SEQ_AXIS, 4}));
    EXPECT_TRUE(info != (RaggedIterationInfo{K_ORIGINS, K_LENGTHS, BSHD_SEQ_AXIS, 4}));
    EXPECT_TRUE(info != (RaggedIterationInfo{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, 2}));
    EXPECT_FALSE(info == (RaggedIterationInfo{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, 2}));
}

TEST(TestRaggedIterationInfo, TensorReportsElementUnitOrigins)
{
    auto aux = makeOffsetAux<int32_t>({0, 2, 5});
    const RaggedTensor<float> tensor(
        K_DIMS, K_STRIDES, K_SEQ_AXIS, aux, std::nullopt, /*raggedOffsetMultiplier=*/4);

    const auto info = tensor.raggedIterationInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_TRUE(*info == (RaggedIterationInfo{K_ORIGINS, K_LENGTHS, K_SEQ_AXIS, 4}));
}

// Tensors that differ only in their origins report different layouts.
TEST(TestRaggedIterationInfo, OriginsDistinguishLayouts)
{
    auto packed = makeOffsetAux<int32_t>({0, 8, 20});
    auto shifted = makeOffsetAux<int32_t>({4, 12, 24});
    const RaggedTensor<float> a(K_DIMS, K_STRIDES, K_SEQ_AXIS, packed);
    const RaggedTensor<float> b(K_DIMS, K_STRIDES, K_SEQ_AXIS, shifted);

    EXPECT_EQ(a.raggedIterationInfo()->lengths, b.raggedIterationInfo()->lengths);
    EXPECT_NE(a.raggedIterationInfo(), b.raggedIterationInfo());
}

// ============================================================================
// Fill Tests
// ============================================================================

TEST(TestRaggedTensor, FillWithValuesHostGenerator)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);

    struct UniformCpuGenerator
    {
        explicit UniformCpuGenerator(float min, float max, unsigned int seed)
            : _min(min)
            , _max(max)
            , _seed(seed)
        {
        }

        void operator()(float* data, size_t count) const
        {
            std::mt19937 rng(_seed);
            std::uniform_real_distribution<float> dist(_min, _max);

            for(size_t i = 0; i < count; ++i)
            {
                data[i] = static_cast<float>(dist(rng));
            }
        }

    private:
        float _min;
        float _max;
        unsigned int _seed;
    };

    const float min = 2.0f;
    const float max = 5.0f;
    tensor.fillWithValues(UniformCpuGenerator(min, max, std::random_device{}()), true);

    for(auto it{tensor.cbegin()}; it != tensor.cend(); ++it)
    {
        auto val{(*static_cast<const float*>((*it)))};
        EXPECT_GE(val, min);
        EXPECT_LE(val, max);
    }
}

TEST(TestRaggedTensor, FillWithValuesDeviceGenerator)
{
    SKIP_IF_NO_DEVICES();

    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);

    struct DeviceGpuGenerator
    {
        void operator()(float* data, size_t count) const
        {
            std::vector<float> writeData(count);
            std::iota(writeData.begin(), writeData.end(), 0.0f);

            auto err = hipMemcpyWithStream(
                data, writeData.data(), count * sizeof(float), hipMemcpyHostToDevice, nullptr);
            if(err != hipSuccess)
            {
                throw std::runtime_error("hipMemcpyWithStream failed");
            }
        }
    };

    tensor.fillWithValues(DeviceGpuGenerator(), false);

    auto hostData = static_cast<float*>(tensor.rawHostData());
    for(size_t i = 0; i < tensor.elementSpace(); i++)
    {
        EXPECT_EQ(hostData[i], static_cast<float>(i));
    }
}

TEST(TestRaggedTensor, FillWithRandomValues)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, K_SEQ_AXIS, aux);

    tensor.fillWithRandomValues(1.0f, 3.0f);
    for(auto it{tensor.cbegin()}; it != tensor.cend(); ++it)
    {
        auto val{(*static_cast<const float*>((*it)))};
        EXPECT_GE(val, 1.0f);
        EXPECT_LE(val, 3.0f);
    }
}

// ============================================================================
// Layout table: every row (and every packed fixture) through RaggedTensor and
// ShallowRaggedTensor, with int32 and int64 offset and length auxes
// ============================================================================

namespace
{

struct LayoutCaseName
{
    std::string operator()(const ::testing::TestParamInfo<LayoutCase>& info) const
    {
        return info.param.name;
    }
};

template <typename OffsetT, typename LengthT>
void checkLayoutCase(const LayoutCase& layout)
{
    auto offsets = makeOffsetAux<OffsetT>(layout.offsets);
    std::shared_ptr<ITensor> lengths;
    if(layout.seqLen.has_value())
    {
        lengths = makeOffsetAux<LengthT>(*layout.seqLen);
    }

    if(!layout.rule.empty())
    {
        try
        {
            const RaggedTensor<float> tensor(layout.dims,
                                             layout.strides,
                                             layout.seqAxis,
                                             offsets,
                                             std::nullopt,
                                             layout.multiplier,
                                             lengths);
            ADD_FAILURE() << layout.name << ": expected rejection (" << layout.rule << ")";
        }
        catch(const std::invalid_argument& e)
        {
            EXPECT_THAT(e.what(), ::testing::HasSubstr(layout.rule)) << layout.name;
        }
        return;
    }

    RaggedTensor<float> tensor(layout.dims,
                               layout.strides,
                               layout.seqAxis,
                               offsets,
                               std::nullopt,
                               layout.multiplier,
                               lengths);

    const auto info = tensor.raggedIterationInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->lengths, layout.lengths);
    std::vector<int64_t> origins;
    origins.reserve(layout.lengths.size());
    for(size_t b = 0; b < layout.lengths.size(); ++b)
    {
        origins.push_back(layout.offsets[b] * layout.multiplier);
    }
    EXPECT_EQ(info->origins, origins);

    EXPECT_EQ(tensor.elementCount(), layout.count);
    EXPECT_EQ(tensor.elementSpace(), layout.space);
    EXPECT_FALSE(tensor.isPacked());
    checkBruteForce(tensor, layout);

    // An explicit physicalElementCount must equal elementSpace.
    const RaggedTensor<float> confirmed(layout.dims,
                                        layout.strides,
                                        layout.seqAxis,
                                        offsets,
                                        layout.space,
                                        layout.multiplier,
                                        lengths);
    EXPECT_EQ(confirmed.elementSpace(), layout.space);
    EXPECT_THROW(const RaggedTensor<float> tooBig(layout.dims,
                                                  layout.strides,
                                                  layout.seqAxis,
                                                  offsets,
                                                  layout.space + 1,
                                                  layout.multiplier,
                                                  lengths),
                 std::invalid_argument);
    if(layout.space > 0)
    {
        EXPECT_THROW(const RaggedTensor<float> tooSmall(layout.dims,
                                                        layout.strides,
                                                        layout.seqAxis,
                                                        offsets,
                                                        layout.space - 1,
                                                        layout.multiplier,
                                                        lengths),
                     std::invalid_argument);
    }

    // The borrowed-buffer type derives the same layout.
    std::vector<float> backing(std::max<size_t>(layout.space, 1), 0.0f);
    ShallowRaggedTensor<float> shallow(backing.data(),
                                       layout.dims,
                                       layout.strides,
                                       layout.seqAxis,
                                       offsets,
                                       std::nullopt,
                                       layout.multiplier,
                                       lengths);
    EXPECT_EQ(shallow.elementCount(), layout.count);
    EXPECT_EQ(shallow.elementSpace(), layout.space);
    EXPECT_TRUE(*shallow.raggedIterationInfo() == *info);
    checkBruteForce(shallow, layout);
}

} // namespace

class TestRaggedLayout : public ::testing::TestWithParam<LayoutCase>
{
};

TEST_P(TestRaggedLayout, FootprintAndBruteForceChecker)
{
    const auto& layout = GetParam();
    checkLayoutCase<int32_t, int32_t>(layout);
    checkLayoutCase<int64_t, int64_t>(layout);
    checkLayoutCase<int32_t, int64_t>(layout);
    checkLayoutCase<int64_t, int32_t>(layout);
}

INSTANTIATE_TEST_SUITE_P(TableRows,
                         TestRaggedLayout,
                         ::testing::ValuesIn(layoutTable()),
                         LayoutCaseName());
INSTANTIATE_TEST_SUITE_P(PackedFixtures,
                         TestRaggedLayout,
                         ::testing::ValuesIn(packedFixtures()),
                         LayoutCaseName());

// ============================================================================
// Row-specific behavior
// ============================================================================

namespace
{

const LayoutCase& tableRow(const std::string& name)
{
    for(const auto& row : layoutTable())
    {
        if(row.name == name)
        {
            return row;
        }
    }
    throw std::logic_error("no layout row " + name);
}

RaggedTensor<float> buildRow(const std::string& name, std::optional<size_t> physical = std::nullopt)
{
    const auto& row = tableRow(name);
    std::shared_ptr<ITensor> lengths;
    if(row.seqLen.has_value())
    {
        lengths = makeOffsetAux<int32_t>(*row.seqLen);
    }
    return {row.dims,
            row.strides,
            row.seqAxis,
            makeOffsetAux<int32_t>(row.offsets),
            physical,
            row.multiplier,
            lengths};
}

} // namespace

// L2: a fused-qkv V chunk spans its strided rows, not the offset table (old space 480).
TEST(TestRaggedLayoutRows, L2FusedQkvChunkSpansItsStridedRows)
{
    const auto tensor = buildRow("L2");
    EXPECT_EQ(tensor.elementSpace(), 416u);
    EXPECT_LT(tensor.elementSpace(), static_cast<size_t>(tableRow("L2").offsets.back()));
}

// L3: all-zero offsets give an empty tensor, not a wrapped span.
TEST(TestRaggedLayoutRows, L3EmptyTensorHasZeroSpace)
{
    auto tensor = buildRow("L3");
    EXPECT_EQ(tensor.elementCount(), 0u);
    EXPECT_EQ(tensor.elementSpace(), 0u);
    EXPECT_EQ(tensor.begin(), tensor.end());
}

// L5: a block longer than S_max iterates S_max rows; seq_len narrows it further (L5s).
TEST(TestRaggedLayoutRows, L5BlockLongerThanSmaxIsClamped)
{
    EXPECT_EQ(buildRow("L5").raggedIterationInfo()->lengths, (std::vector<int64_t>{5, 5}));
    EXPECT_EQ(buildRow("L5s").raggedIterationInfo()->lengths, (std::vector<int64_t>{3, 5}));
}

// L6: head-major primary; each batch's rows sit one head pitch apart.
TEST(TestRaggedLayoutRows, L6HeadMajorAddressing)
{
    const auto tensor = buildRow("L6");
    // batch 1 origin 4: {b=1, h=1, s=2, d=1} -> 4 + 1*10 + 2*2 + 1 = 19
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 1), 19);
}

// L7a/L7b/L7c: the [H,T] and [T,H] Stats layouts address the same logical tokens.
TEST(TestRaggedLayoutRows, L7StatsLayoutsShareLengthsAndCount)
{
    const auto a = buildRow("L7a");
    const auto b = buildRow("L7b");
    const auto c = buildRow("L7c");
    EXPECT_EQ(a.raggedIterationInfo()->lengths, b.raggedIterationInfo()->lengths);
    EXPECT_EQ(a.elementCount(), b.elementCount());
    EXPECT_EQ(a.elementSpace(), b.elementSpace());
    // Head pitch 10 instead of T = 5 widens the span only.
    EXPECT_EQ(c.elementCount(), a.elementCount());
    EXPECT_EQ(c.elementSpace(), 35u);
    // batch 1 (origin 2 / 8), head 2, s = 1: [H,T] -> 2 + 2*5 + 1, [T,H] -> 8 + 2*1 + 1*4
    EXPECT_EQ(a.getIndex(1, 2, 1, 0), 13);
    EXPECT_EQ(b.getIndex(1, 2, 1, 0), 14);
}

// L7d: gapped [H,T] with seq_len; the gap tokens belong to no batch.
TEST(TestRaggedLayoutRows, L7dGappedStatsAddressesLiveRowsOnly)
{
    const auto tensor = buildRow("L7d");
    EXPECT_EQ(tensor.elementCount(), 10u);
    EXPECT_EQ(tensor.elementSpace(), 14u);
    // batch 1 origin 4: head 1, s = 2 -> 4 + 2 + 7 = 13
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 0), 13);
}

// L8: a shifted origin is accepted (offset[0] != 0).
TEST(TestRaggedLayoutRows, L8ShiftedOriginIsAccepted)
{
    const auto tensor = buildRow("L8");
    EXPECT_EQ(tensor.getIndex(0), 4);
    EXPECT_EQ(tensor.getIndex(1), 8);
    EXPECT_EQ(tensor.elementSpace(), 14u);
}

// L9: non-monotonic origins with seq_len; offset[B] is not used.
TEST(TestRaggedLayoutRows, L9NonMonotonicOriginsAddressFromEachOrigin)
{
    const auto tensor = buildRow("L9");
    // batch 0 origin 10: {s=1, d=1} -> 10 + 2 + 1; batch 1 origin 0: {s=2, d=1} -> 0 + 4 + 1
    EXPECT_EQ(tensor.getIndex(0, 0, 1, 1), 13);
    EXPECT_EQ(tensor.getIndex(1, 0, 2, 1), 5);
}

// L9m / L7bm: a multiplier scales the origins and equals the element-offset form.
TEST(TestRaggedLayoutRows, MultiplierRowsEqualTheirElementOffsetForms)
{
    const auto scaled = buildRow("L9m");
    const auto direct = buildRow("L9");
    EXPECT_TRUE(scaled.raggedIterationInfo() == direct.raggedIterationInfo());
    EXPECT_EQ(scaled.elementSpace(), 14u);

    const auto tokenForm = buildRow("L7bm");
    const auto elementForm = buildRow("L7b");
    EXPECT_TRUE(tokenForm.raggedIterationInfo() == elementForm.raggedIterationInfo());

    // An explicit physicalElementCount is checked against the multiplier-scaled footprint.
    EXPECT_NO_THROW(buildRow("L9m", 14));
    EXPECT_THROW(buildRow("L9m", 5), std::invalid_argument);
}

// L10 / L11: aliasing layouts are rejected, naming the violated rule.
TEST(TestRaggedLayoutRows, AliasingRowsAreRejectedByRule)
{
    EXPECT_THAT([] { buildRow("L10"); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("overlap")));
    EXPECT_THAT(
        [] { buildRow("L11"); },
        ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("sorted-stride")));
}

// L12: a zero-stride (broadcast) axis is exempt; elements alias by design.
TEST(TestRaggedLayoutRows, L12BroadcastHeadAxisIsAccepted)
{
    const auto tensor = buildRow("L12");
    EXPECT_EQ(tensor.elementCount(), 20u);
    EXPECT_EQ(tensor.elementSpace(), 10u);
    EXPECT_EQ(tensor.getIndex(0, 0, 1, 1), tensor.getIndex(0, 1, 1, 1));
}

// L13: an empty batch beside a non-empty one.
TEST(TestRaggedLayoutRows, L13EmptyBatchBesideNonEmpty)
{
    EXPECT_EQ(buildRow("L13").raggedIterationInfo()->lengths, (std::vector<int64_t>{0, 3}));
}

// L14: the trailing gap past the last live row is not part of the buffer.
TEST(TestRaggedLayoutRows, L14SpaceExcludesTheTrailingGap)
{
    EXPECT_EQ(buildRow("L14").elementSpace(), 10u);
    // offset[B] == 12 is not a valid physical size.
    EXPECT_THROW(buildRow("L14", 12), std::invalid_argument);
}

// L15: a zero seq_len beside a non-zero origin leaves no live rows for that batch.
TEST(TestRaggedLayoutRows, L15ZeroLengthBesideNonZeroOrigin)
{
    const auto tensor = buildRow("L15");
    EXPECT_EQ(tensor.raggedIterationInfo()->origins, (std::vector<int64_t>{6, 0}));
    EXPECT_EQ(tensor.raggedIterationInfo()->lengths, (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(tensor.elementSpace(), 6u);
}

// rowOffsets(): the offset table of a packed, in-order layout, derived from origins/lengths.
TEST(TestRaggedLayoutRows, RowOffsetsOfPackedLayoutEqualsTheOffsetTable)
{
    const auto& row = tableRow("L1");
    EXPECT_EQ(buildRow("L1").raggedIterationInfo()->rowOffsets(), row.offsets);
}

// L14: origins and lengths are authoritative; the stored offset[B] (12) is not echoed.
TEST(TestRaggedLayoutRows, RowOffsetsEndAtTheLastLiveRow)
{
    EXPECT_EQ(buildRow("L14").raggedIterationInfo()->rowOffsets(),
              (std::vector<int64_t>{0, 4, 10}));
}

// L7d: batch 0 ends at 2 but batch 1 starts at 4, a gap the offset table cannot express.
TEST(TestRaggedLayoutRows, RowOffsetsThrowsForGappedBatches)
{
    EXPECT_THAT([] { buildRow("L7d").raggedIterationInfo()->rowOffsets(); },
                ::testing::ThrowsMessage<std::logic_error>(::testing::HasSubstr("batch 0")));
}

// L9: origins are not ascending, so no packed in-order table describes the batches.
TEST(TestRaggedLayoutRows, RowOffsetsThrowsForNonMonotonicOrigins)
{
    EXPECT_THROW(buildRow("L9").raggedIterationInfo()->rowOffsets(), std::logic_error);
}

// L15: the empty batch 0 owns no rows, so its stored origin (6) does not break contiguity; its
// entry is where the first live batch starts. Token stride is 2.
TEST(TestRaggedLayoutRows, RowOffsetsSkipsZeroLengthBatches)
{
    EXPECT_EQ(buildRow("L15").raggedIterationInfo()->rowOffsets(), (std::vector<int64_t>{0, 0, 6}));
}

// A trailing empty batch takes the running end; an empty batch between live batches does not
// hide a gap between them.
TEST(TestRaggedLayoutRows, RowOffsetsZeroLengthBatchesKeepGapDetection)
{
    const RaggedIterationInfo trailing{{0, 9}, {2, 0}, SDPA_SEQ_AXIS, 2};
    EXPECT_EQ(trailing.rowOffsets(), (std::vector<int64_t>{0, 4, 4}));

    const RaggedIterationInfo middleGap{{0, 7, 8}, {2, 0, 1}, SDPA_SEQ_AXIS, 2};
    EXPECT_THAT([&] { middleGap.rowOffsets(); },
                ::testing::ThrowsMessage<std::logic_error>(::testing::HasSubstr("batch 0")));
}

// A gap is reported against the last live batch before it, and a layout whose batches are all
// empty starts at the first stored origin.
TEST(TestRaggedLayoutRows, RowOffsetsNamesLastLiveBatchAndStartsAllEmptyAtFirstOrigin)
{
    const RaggedIterationInfo laterGap{{0, 4, 9}, {2, 2, 1}, SDPA_SEQ_AXIS, 2};
    EXPECT_THAT([&] { laterGap.rowOffsets(); },
                ::testing::ThrowsMessage<std::logic_error>(::testing::HasSubstr("batch 1")));

    const RaggedIterationInfo allEmpty{{5, 9}, {0, 0}, SDPA_SEQ_AXIS, 2};
    EXPECT_EQ(allEmpty.rowOffsets(), (std::vector<int64_t>{5, 5, 5}));
}

// A zero-batch layout has no batches to end: the table is empty, not {origins.back() + ...}.
TEST(TestRaggedLayoutRows, RowOffsetsOfZeroBatchLayoutIsEmpty)
{
    const RaggedIterationInfo info{{}, {}, SDPA_SEQ_AXIS, 4};
    EXPECT_TRUE(info.rowOffsets().empty());
}

namespace
{

// dims [2, H, 4, 2], strides [0, hStride, 2, 1], sequence axis 2 (token stride 2).
RaggedTensor<float> buildBoundaryTensor(int64_t heads,
                                        int64_t hStride,
                                        const std::vector<int64_t>& offsets,
                                        const std::optional<std::vector<int64_t>>& seqLen)
{
    std::shared_ptr<ITensor> lengths;
    if(seqLen.has_value())
    {
        lengths = makeOffsetAux<int32_t>(*seqLen);
    }
    return {{2, heads, 4, 2},
            {0, hStride, 2, 1},
            SDPA_SEQ_AXIS,
            makeOffsetAux<int32_t>(offsets),
            std::nullopt,
            1,
            lengths};
}

} // namespace

// The overlap rule rejects a single shared row and accepts touching intervals.
TEST(TestRaggedLayoutRows, OverlapRuleBoundaryIsOneSharedRow)
{
    // Batch 0 rows [0, 2); batch 1 origin 2 / token stride 2 = row 1, rows [1, 4): row 1 shared.
    EXPECT_THAT([] { buildBoundaryTensor(1, 2, {0, 2, 0}, std::vector<int64_t>{2, 3}); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("overlap")));
    // Batch 1 at row 2 touches batch 0 without sharing a row.
    EXPECT_NO_THROW(buildBoundaryTensor(1, 2, {0, 4, 0}, std::vector<int64_t>{2, 3}));
}

// A size-1 axis never contributes to an offset, so its stride is exempt from the sorted-stride
// rule; the same stride on an axis of extent 2 is rejected.
TEST(TestRaggedLayoutRows, SortedStrideRuleExemptsSizeOneAxes)
{
    // Head stride 3 lies inside the 10 elements spanned by the sequence and depth axes.
    const auto tensor = buildBoundaryTensor(1, 3, {0, 4, 10}, std::nullopt);
    EXPECT_EQ(tensor.elementCount(), 10u);
    EXPECT_EQ(tensor.elementSpace(), 10u);

    EXPECT_THAT(
        [] { buildBoundaryTensor(2, 3, {0, 4, 10}, std::nullopt); },
        ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("sorted-stride")));
}

// A zero-extent non-sequence axis leaves no elements even when the batch blocks are non-empty;
// iteration agrees with elementCount() instead of walking a zero-size buffer.
TEST(TestRaggedLayoutRows, ZeroExtentAxisWithNonEmptyBlocksIsEmpty)
{
    auto tensor = RaggedTensor<float>(
        {2, 0, 3, 2}, {12, 6, 2, 1}, SDPA_SEQ_AXIS, makeOffsetAux<int32_t>({0, 4, 10}));
    EXPECT_EQ(tensor.elementCount(), 0u);
    EXPECT_EQ(tensor.elementSpace(), 0u);
    EXPECT_EQ(tensor.begin(), tensor.end());
}

// ============================================================================
// Rejections beyond the table
// ============================================================================

namespace
{

// Builds the canonical K layout with the given offsets and optional lengths.
void constructK(const std::vector<int64_t>& offsets,
                const std::optional<std::vector<int64_t>>& seqLen = std::nullopt,
                const std::vector<int64_t>& strides = K_STRIDES)
{
    std::shared_ptr<ITensor> lengths;
    if(seqLen.has_value())
    {
        lengths = makeOffsetAux<int32_t>(*seqLen);
    }
    const RaggedTensor<float> tensor(
        K_DIMS, strides, K_SEQ_AXIS, makeOffsetAux<int32_t>(offsets), std::nullopt, 1, lengths);
}

} // namespace

TEST(TestRaggedLayoutRejections, LengthAboveSmaxOrNegative)
{
    // K_DIMS has S_max = 3.
    EXPECT_THAT(
        [] { constructK(K_OFFSETS, std::vector<int64_t>{2, 4}); },
        ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("length range")));
    EXPECT_THAT(
        [] { constructK(K_OFFSETS, std::vector<int64_t>{-1, 3}); },
        ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("length range")));
    EXPECT_NO_THROW(constructK(K_OFFSETS, std::vector<int64_t>{0, 3}));
}

TEST(TestRaggedLayoutRejections, OriginNotWholeTokenOrNegative)
{
    // seqStride = 4.
    EXPECT_THAT([] { constructK({2, 8, 20}, std::vector<int64_t>{2, 3}); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                    ::testing::HasSubstr("whole-token origin")));
    EXPECT_THAT([] { constructK({-4, 8, 20}, std::vector<int64_t>{2, 3}); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                    ::testing::HasSubstr("whole-token origin")));
    // Without lengths the first origin must be whole-token too.
    EXPECT_THAT([] { constructK({2, 6, 14}); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                    ::testing::HasSubstr("whole-token origin")));
}

TEST(TestRaggedLayoutRejections, NegativeStrideWithLengths)
{
    EXPECT_THROW(constructK(K_OFFSETS, std::vector<int64_t>{2, 3}, {12, -2, 4, 1}),
                 std::invalid_argument);
}

TEST(TestRaggedLayoutRejections, LengthTensorElementCountMustEqualB)
{
    auto offsets = makeOffsetAux<int32_t>(K_OFFSETS);
    auto tooMany = makeOffsetAux<int32_t>({2, 3, 0});
    EXPECT_THROW(const RaggedTensor<float> tensor(
                     K_DIMS, K_STRIDES, K_SEQ_AXIS, offsets, std::nullopt, 1, tooMany),
                 std::invalid_argument);
    auto tooFew = makeOffsetAux<int32_t>({2});
    EXPECT_THROW(const RaggedTensor<float> tensor(
                     K_DIMS, K_STRIDES, K_SEQ_AXIS, offsets, std::nullopt, 1, tooFew),
                 std::invalid_argument);
}

TEST(TestRaggedLayoutRejections, LengthTensorElementSizeMustBe4Or8)
{
    auto offsets = makeOffsetAux<int32_t>(K_OFFSETS);
    auto lengths16 = std::make_shared<Tensor<int16_t>>(std::vector<int64_t>{2, 1, 1, 1});
    EXPECT_THROW(const RaggedTensor<float> tensor(
                     K_DIMS, K_STRIDES, K_SEQ_AXIS, offsets, std::nullopt, 1, lengths16),
                 std::invalid_argument);
}

TEST(TestRaggedLayoutRejections, LengthTensorMustBeRank4)
{
    auto offsets = makeOffsetAux<int32_t>(K_OFFSETS);
    auto rank2 = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{2, 1});
    fillLeading(*rank2, K_LENGTHS);
    EXPECT_THROW(const RaggedTensor<float> tensor(
                     K_DIMS, K_STRIDES, K_SEQ_AXIS, offsets, std::nullopt, 1, rank2),
                 std::invalid_argument);
}

TEST(TestRaggedLayoutRejections, LengthTensorOffTheBatchAxisThrows)
{
    auto offsets = makeOffsetAux<int32_t>(K_OFFSETS);
    auto lengths = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{1, 2, 1, 1});
    fillLeading(*lengths, K_LENGTHS);
    EXPECT_THROW(const RaggedTensor<float> tensor(
                     K_DIMS, K_STRIDES, K_SEQ_AXIS, offsets, std::nullopt, 1, lengths),
                 std::invalid_argument);
}

TEST(TestRaggedLayoutRejections, ShallowTensorValidatesLengthTensorToo)
{
    auto offsets = makeOffsetAux<int32_t>(K_OFFSETS);
    auto tooMany = makeOffsetAux<int32_t>({2, 3, 0});
    std::vector<float> backing(20, 0.0f);
    EXPECT_THROW(
        const ShallowRaggedTensor<float> tensor(
            backing.data(), K_DIMS, K_STRIDES, K_SEQ_AXIS, offsets, std::nullopt, 1, tooMany),
        std::invalid_argument);
}

// ============================================================================
// Offsets and lengths are snapshotted at construction
// ============================================================================

TEST(TestRaggedTensor, OffsetsAndLengthsAreSnapshotted)
{
    auto offsets = makeOffsetAux<int32_t>({0, 8, 20});
    auto lengths = makeOffsetAux<int32_t>({2, 3});
    const RaggedTensor<float> tensor(
        K_DIMS, K_STRIDES, K_SEQ_AXIS, offsets, std::nullopt, 1, lengths);

    *static_cast<int32_t*>(offsets->hostDataOffsetFromIndex(1)) = 100;
    *static_cast<int32_t*>(lengths->hostDataOffsetFromIndex(0)) = 0;

    EXPECT_EQ(tensor.getIndex(1), 8);
    EXPECT_EQ(tensor.raggedIterationInfo()->lengths, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensor.elementCount(), 20u);
}
