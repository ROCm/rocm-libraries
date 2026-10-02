// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestGraphSize.cpp
 * @brief Covers the benchmarking ceiling's arithmetic.
 *
 * An undersized charge admits problems too large to time, silently. Checks dtype width,
 * stride span rather than element count, and saturation instead of wrap.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/GraphSize.hpp>

#include <limits>
#include <utility>
#include <vector>

namespace hipdnn_corpus_gen
{
namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

/// One tensor of @p dims in @p type, laid out with @p strides (packed row-major if empty), as a
/// serialized graph.
builders::GraphBytes
    graphOf(const std::vector<int64_t>& dims, fb::DataType type, std::vector<int64_t> strides = {})
{
    if(strides.empty())
    {
        strides.assign(dims.size(), 1);
        for(size_t i = dims.size(); i > 1; --i)
        {
            strides[i - 2] = strides[i - 1] * dims[i - 1];
        }
    }
    builders::TensorSpec tensor;
    tensor.uid = 1;
    tensor.name = "x";
    tensor.dims = dims;
    tensor.strides = std::move(strides);
    tensor.dataType = type;
    return builders::reduction(tensor,
                               tensor,
                               fb::ReductionMode::ADD,
                               /*deterministic=*/false,
                               builders::GraphTypes::uniform(type));
}

} // namespace

TEST(TestGraphSize, WidthFollowsTheDataType)
{
    // Charged as four bytes, an fp64 problem twice the ceiling would be admitted.
    EXPECT_EQ(elementBytes(fb::DataType::DOUBLE), 8);
    EXPECT_EQ(elementBytes(fb::DataType::INT64), 8);
    EXPECT_EQ(elementBytes(fb::DataType::FLOAT), 4);
    EXPECT_EQ(elementBytes(fb::DataType::HALF), 2);
    EXPECT_EQ(elementBytes(fb::DataType::BFLOAT16), 2);
    EXPECT_EQ(elementBytes(fb::DataType::FP8_E4M3), 1);
    EXPECT_EQ(elementBytes(fb::DataType::INT8), 1);
}

TEST(TestGraphSize, SubByteTypesRoundUpRatherThanToZero)
{
    // Packing belongs to the tensor; rounding to zero would make block-scaled problems free.
    EXPECT_EQ(elementBytes(fb::DataType::FP4_E2M1), 1);
    EXPECT_EQ(elementBytes(fb::DataType::FP6_E2M3), 1);
    EXPECT_EQ(elementBytes(fb::DataType::INT4), 1);
}

TEST(TestGraphSize, AnUnknownTypeIsChargedTheWidest)
{
    // Fails safe: an unknown type must not slip a large problem past the ceiling.
    EXPECT_EQ(elementBytes(fb::DataType::UNSET), 8);
}

TEST(TestGraphSize, ATensorCostsItsElementsTimesItsWidth)
{
    const std::vector<int64_t> dims{2, 3, 4};
    // reduction() writes the input and the output, so both tensors are counted.
    EXPECT_EQ(graphBytes(graphOf(dims, fb::DataType::FLOAT)), 2 * 24 * 4);
    EXPECT_EQ(graphBytes(graphOf(dims, fb::DataType::HALF)), 2 * 24 * 2);
    EXPECT_EQ(graphBytes(graphOf(dims, fb::DataType::DOUBLE)), 2 * 24 * 8);
}

TEST(TestGraphSize, APaddedTensorIsChargedTheSpanTheBenchAllocates)
{
    // The bench allocates up to the furthest addressable element: 2x2 with row stride 4096 is
    // a 4098-element allocation.
    const int64_t span = 1 + ((2 - 1) * 4096) + ((2 - 1) * 1);
    EXPECT_EQ(graphBytes(graphOf({2, 2}, fb::DataType::FLOAT, {4096, 1})), 2 * span * 4);
}

TEST(TestGraphSize, ATensorTheBenchCannotSizeIsOverEveryCeiling)
{
    // Mismatched-rank or negative strides have no allocatable span.
    EXPECT_EQ(graphBytes(graphOf({2, 2}, fb::DataType::FLOAT, {1})),
              std::numeric_limits<int64_t>::max());
    EXPECT_EQ(graphBytes(graphOf({2, 2}, fb::DataType::FLOAT, {-2, 1})),
              std::numeric_limits<int64_t>::max());
}

TEST(TestGraphSize, AStrideThatWrapsTheSpanSaturates)
{
    // Two elements are cheap; a stride near int64's range puts the second one past it.
    EXPECT_EQ(
        graphBytes(graphOf({2, 1}, fb::DataType::INT8, {std::numeric_limits<int64_t>::max(), 1})),
        std::numeric_limits<int64_t>::max());
}

TEST(TestGraphSize, AnEnormousProblemSaturatesRatherThanWrapping)
{
    // A wrapped total would read as small and be admitted.
    const int64_t huge = std::numeric_limits<int64_t>::max() / 4;
    EXPECT_EQ(graphBytes(graphOf({huge, huge}, fb::DataType::FLOAT)),
              std::numeric_limits<int64_t>::max());
    EXPECT_EQ(graphBytes(graphOf({huge}, fb::DataType::DOUBLE)),
              std::numeric_limits<int64_t>::max());
}

} // namespace hipdnn_corpus_gen
