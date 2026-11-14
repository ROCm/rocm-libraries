// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "engines/plans/MiopenUnaryActivationChecks.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

bool isSupported(const flatbuffers::FlatBufferBuilder& builder)
{
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    return unary_activation_applicability::isSupported(graph);
}

flatbuffers::FlatBufferBuilder reluGraph(const std::vector<int64_t>& dims,
                                         std::optional<std::vector<int64_t>> inputStrides,
                                         std::optional<std::vector<int64_t>> outputStrides)
{
    return createPointwiseGraph(PointwiseGraphSpec::unary(
        PointwiseMode::RELU_FWD, dims, std::move(inputStrides), dims, std::move(outputStrides)));
}

} // namespace

TEST(TestMiopenUnaryActivationChecks, AcceptsMatchingPackedLayouts)
{
    EXPECT_TRUE(isSupported(reluGraph(
        {1, 3, 4, 4}, std::vector<int64_t>{48, 16, 4, 1}, std::vector<int64_t>{48, 16, 4, 1})));
    // Channels-last on both sides preserves every logical index.
    EXPECT_TRUE(isSupported(reluGraph(
        {1, 3, 4, 4}, std::vector<int64_t>{48, 1, 12, 3}, std::vector<int64_t>{48, 1, 12, 3})));
}

TEST(TestMiopenUnaryActivationChecks, DeclinesInputAndOutputWithDifferentLayouts)
{
    // Same element count and both packed, but channel-first input against channel-last output:
    // MIOpen walks both flat and would pair different logical elements.
    EXPECT_FALSE(isSupported(reluGraph(
        {1, 3, 4, 4}, std::vector<int64_t>{48, 16, 4, 1}, std::vector<int64_t>{48, 1, 12, 3})));
}

TEST(TestMiopenUnaryActivationChecks, AcceptsNonPackedTensorsWithDifferentStrides)
{
    // Padded rows on both sides: MIOpen's strided activation solver takes both stride sets.
    EXPECT_TRUE(isSupported(reluGraph(
        {2, 3, 4, 4}, std::vector<int64_t>{240, 80, 16, 2}, std::vector<int64_t>{192, 64, 16, 2})));
    // One packed and one padded side.
    EXPECT_TRUE(isSupported(reluGraph(
        {2, 3, 4, 4}, std::vector<int64_t>{48, 16, 4, 1}, std::vector<int64_t>{192, 64, 16, 2})));
}

TEST(TestMiopenUnaryActivationChecks, AcceptsPackedTensorsWhoseUnitDimStridesDiffer)
{
    EXPECT_TRUE(isSupported(reluGraph(
        {1, 3, 4, 4}, std::vector<int64_t>{48, 16, 4, 1}, std::vector<int64_t>{999, 16, 4, 1})));
}

TEST(TestMiopenUnaryActivationChecks, DeclinesInputAndOutputWithDifferentDims)
{
    const auto builder
        = createPointwiseGraph(PointwiseGraphSpec::unary(PointwiseMode::RELU_FWD,
                                                         {1, 3, 4, 4},
                                                         std::vector<int64_t>{48, 16, 4, 1},
                                                         {1, 3, 16, 1},
                                                         std::vector<int64_t>{48, 16, 1, 1}));
    EXPECT_FALSE(isSupported(builder));
}

TEST(TestMiopenUnaryActivationChecks, AcceptsPackedRank1)
{
    EXPECT_TRUE(isSupported(reluGraph({16}, std::vector<int64_t>{1}, std::vector<int64_t>{1})));
}

TEST(TestMiopenUnaryActivationChecks, DeclinesNonPackedRank1)
{
    // The strided side differs from the packed one, so this is also a layout mismatch; the
    // equal-strided pair below isolates the rank-1 packing rule.
    EXPECT_FALSE(isSupported(reluGraph({16}, std::vector<int64_t>{2}, std::vector<int64_t>{1})));
    EXPECT_FALSE(isSupported(reluGraph({16}, std::vector<int64_t>{1}, std::vector<int64_t>{2})));
    EXPECT_FALSE(isSupported(reluGraph({16}, std::vector<int64_t>{2}, std::vector<int64_t>{2})));
}

TEST(TestMiopenUnaryActivationChecks, DeclinesPassByValueBufferTensors)
{
    EXPECT_TRUE(isSupported(reluGraph({8}, std::vector<int64_t>{1}, std::vector<int64_t>{1})));

    for(const bool markInput : {true, false})
    {
        const auto builder = mutateGraph(
            reluGraph({8}, std::vector<int64_t>{1}, std::vector<int64_t>{1}),
            [&](fb::GraphT& graph) {
                const auto* attrs = graph.nodes.at(0)->attributes.AsPointwiseAttributes();
                const auto uid = markInput ? attrs->in_0_tensor_uid : attrs->out_0_tensor_uid;
                for(auto& tensor : graph.tensors)
                {
                    if(tensor->uid == uid)
                    {
                        tensor->is_runtime_pass_by_value = true;
                    }
                }
            });
        EXPECT_FALSE(isSupported(builder)) << (markInput ? "input" : "output");
    }
}
