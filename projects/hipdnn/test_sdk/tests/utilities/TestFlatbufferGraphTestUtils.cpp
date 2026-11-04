// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

namespace
{

std::vector<int64_t> tensorStrides(const flatbuffers::FlatBufferBuilder& builder,
                                   const std::string& name)
{
    const auto* graph = GetGraph(builder.GetBufferPointer());
    for(const auto* tensor : *graph->tensors())
    {
        if(tensor->name()->str() == name)
        {
            return {tensor->strides()->begin(), tensor->strides()->end()};
        }
    }
    ADD_FAILURE() << "no tensor named " << name;
    return {};
}

// Builds the default MX matmul graph (scale A {32, 4}, scale B {4, 32}) with the given
// scale strides; an empty vector keeps the helper's default.
flatbuffers::FlatBufferBuilder
    createMxMatmulGraphWithScaleStrides(const std::vector<int64_t>& scaleAStrides,
                                        const std::vector<int64_t>& scaleBStrides)
{
    return createValidMxMatmulGraph({32, 128},
                                    {1, 32},
                                    {128, 32},
                                    {32, 1},
                                    {32, 32},
                                    {32, 1},
                                    {32, 4},
                                    {4, 32},
                                    DataType::FP8_E4M3,
                                    DataType::HALF,
                                    DataType::FP8_E8M0,
                                    DataType::FP8_E8M0,
                                    DataType::FLOAT,
                                    32,
                                    false,
                                    false,
                                    DataType::UNSET,
                                    scaleAStrides,
                                    scaleBStrides);
}

} // namespace

TEST(TestCreateValidMxMatmulGraph, DefaultsToRowMajorScaleStrides)
{
    const auto builder = createValidMxMatmulGraph();

    EXPECT_EQ(tensorStrides(builder, "scale_a"), (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(tensorStrides(builder, "scale_b"), (std::vector<int64_t>{32, 1}));
}

TEST(TestCreateValidMxMatmulGraph, UsesGivenScaleStrides)
{
    const auto builder = createMxMatmulGraphWithScaleStrides({1, 32}, {1, 4});

    EXPECT_EQ(tensorStrides(builder, "scale_a"), (std::vector<int64_t>{1, 32}));
    EXPECT_EQ(tensorStrides(builder, "scale_b"), (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(tensorStrides(builder, "x_a"), (std::vector<int64_t>{1, 32}));
    EXPECT_EQ(tensorStrides(builder, "x_b"), (std::vector<int64_t>{32, 1}));
}

TEST(TestCreateValidMxMatmulGraph, OverridesEachScaleSeparately)
{
    const auto onlyA = createMxMatmulGraphWithScaleStrides({1, 32}, {});
    EXPECT_EQ(tensorStrides(onlyA, "scale_a"), (std::vector<int64_t>{1, 32}));
    EXPECT_EQ(tensorStrides(onlyA, "scale_b"), (std::vector<int64_t>{32, 1}));

    const auto onlyB = createMxMatmulGraphWithScaleStrides({}, {1, 4});
    EXPECT_EQ(tensorStrides(onlyB, "scale_a"), (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(tensorStrides(onlyB, "scale_b"), (std::vector<int64_t>{1, 4}));
}
