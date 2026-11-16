// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "engines/plans/MiopenConvApplicabilityChecks.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

using TensorEdit = std::function<void(fb::TensorAttributesT&)>;

// Applies `edit` to the tensor named `name`.
flatbuffers::FlatBufferBuilder editTensor(const flatbuffers::FlatBufferBuilder& source,
                                          const std::string& name,
                                          const TensorEdit& edit)
{
    return mutateGraph(source, [&](fb::GraphT& graph) {
        for(auto& tensor : graph.tensors)
        {
            if(tensor->name == name)
            {
                edit(*tensor);
            }
        }
    });
}

// Whether the layout checks accept the conv -> [bias] -> activation graph.
bool layoutsSupported(const flatbuffers::FlatBufferBuilder& builder)
{
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto& convAttr = graph.getNodeWrapper(0).attributesAs<fb::ConvolutionFwdAttributes>();
    const bool hasBias = graph.nodeCount() == 3;
    const auto* biasAttr
        = hasBias ? &graph.getNodeWrapper(1).attributesAs<fb::PointwiseAttributes>() : nullptr;
    const auto& activAttr
        = graph.getNodeWrapper(hasBias ? 2 : 1).attributesAs<fb::PointwiseAttributes>();

    try
    {
        conv_applicability::checkConvFwdBiasActivLayouts(
            convAttr, biasAttr, activAttr, graph.getTensorMap());
        return true;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return false;
    }
}

// NCHW conv, x/w 4x4x... : x [4,4,4,4], y [4,4,4,4], per-channel bias [1,4,1,1].
flatbuffers::FlatBufferBuilder cbaGraph()
{
    return createValidConvFwdBiasActivGraph();
}

std::vector<int64_t> nhwcStrides()
{
    return {64, 1, 16, 4};
}

} // namespace

TEST(TestMiopenConvApplicabilityChecks, AcceptsPerChannelBiasAndMatchingOutput)
{
    EXPECT_TRUE(layoutsSupported(cbaGraph()));
    EXPECT_TRUE(layoutsSupported(createValidConvFwdActivGraph()));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsChannelsLastEndToEnd)
{
    const auto builder = createValidConvFwdBiasActivGraph(
        {4, 4, 4, 4}, nhwcStrides(), {4, 4, 1, 1}, {4, 1, 4, 4}, {4, 4, 4, 4}, nhwcStrides());
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsChannelsLastWithSingleInputChannel)
{
    // The unit-extent input channel ties with W at stride 1; MIOpen orders ties by extent, so
    // the layout is channels-last.
    const auto builder = createValidConvFwdBiasActivGraph(
        {2, 1, 8, 8}, {64, 1, 8, 1}, {4, 1, 1, 1}, {1, 1, 1, 1}, {2, 4, 8, 8}, {256, 1, 32, 4});
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesChannelsFirstOutputForSingleChannelChannelsLast)
{
    const auto builder = createValidConvFwdBiasActivGraph(
        {2, 1, 8, 8}, {64, 1, 8, 1}, {4, 1, 1, 1}, {1, 1, 1, 1}, {2, 4, 8, 8}, {256, 64, 8, 1});
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsSingleBatchSinglePixelInputs)
{
    // Every dimension but C is a unit; the tied unit strides resolve by extent.
    EXPECT_TRUE(layoutsSupported(createValidConvFwdBiasActivGraph(
        {1, 3, 1, 1}, {3, 1, 3, 3}, {4, 3, 1, 1}, {3, 1, 1, 1}, {1, 4, 1, 1}, {4, 1, 4, 4})));
    EXPECT_TRUE(layoutsSupported(createValidConvFwdBiasActivGraph(
        {1, 3, 1, 1}, {3, 1, 1, 1}, {4, 3, 1, 1}, {3, 1, 1, 1}, {1, 4, 1, 1}, {4, 1, 1, 1})));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesSingleBatchOutputWithNonCanonicalBatchStride)
{
    // MIOpen's fused execute compares whole descriptors, unit-dimension strides included.
    const auto builder = editTensor(
        createValidConvFwdBiasActivGraph(
            {1, 4, 4, 4}, {64, 16, 4, 1}, {4, 4, 1, 1}, {4, 1, 1, 1}, {1, 4, 4, 4}, {64, 16, 4, 1}),
        "y",
        [](fb::TensorAttributesT& t) { t.strides = {999, 16, 4, 1}; });
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsSingleBatchOutputWithCanonicalStrides)
{
    EXPECT_TRUE(layoutsSupported(createValidConvFwdBiasActivGraph(
        {1, 4, 4, 4}, {64, 16, 4, 1}, {4, 4, 1, 1}, {4, 1, 1, 1}, {1, 4, 4, 4}, {64, 16, 4, 1})));
    EXPECT_TRUE(layoutsSupported(createValidConvFwdBiasActivGraph(
        {1, 4, 4, 4}, {64, 1, 16, 4}, {4, 4, 1, 1}, {4, 1, 4, 4}, {1, 4, 4, 4}, {64, 1, 16, 4})));
}

TEST(TestMiopenConvApplicabilityChecks,
     DeclinesChannelsLastSingleOutputChannelOverChannelsFirstInput)
{
    // K = 1: the unit channel dimension still carries a stride MIOpen compares.
    const auto builder = createValidConvFwdBiasActivGraph(
        {2, 4, 8, 8}, {256, 64, 8, 1}, {1, 4, 1, 1}, {1, 1, 1, 1}, {2, 1, 8, 8}, {64, 1, 8, 1});
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsSingleChannelBiasWithAnyChannelStride)
{
    const auto builder = editTensor(
        createValidConvFwdBiasActivGraph(
            {4, 4, 4, 4}, {64, 16, 4, 1}, {1, 4, 1, 1}, {4, 1, 1, 1}, {4, 1, 4, 4}, {16, 16, 4, 1}),
        "bias",
        [](fb::TensorAttributesT& t) {
            t.dims = {1, 1, 1, 1};
            t.strides = {7, 5, 3, 1};
        });
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesFullTensorBias)
{
    // A packed [N,C,H,W] residual is broadcast-compatible for ADD but is not a per-channel bias.
    const auto builder = editTensor(cbaGraph(), "bias", [](fb::TensorAttributesT& t) {
        t.dims = {4, 4, 4, 4};
        t.strides = {64, 16, 4, 1};
    });
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesBiasWithWrongChannelCount)
{
    const auto builder = editTensor(cbaGraph(), "bias", [](fb::TensorAttributesT& t) {
        t.dims = {1, 2, 1, 1};
        t.strides = {2, 1, 1, 1};
    });
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesBiasWithStridedChannels)
{
    const auto builder = editTensor(
        cbaGraph(), "bias", [](fb::TensorAttributesT& t) { t.strides = {8, 2, 2, 2}; });
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsLowerRankPerChannelBias)
{
    // [4,1,1] right-aligns against the rank-4 output as [1,4,1,1].
    const auto builder = editTensor(cbaGraph(), "bias", [](fb::TensorAttributesT& t) {
        t.dims = {4, 1, 1};
        t.strides = {1, 1, 1};
    });
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesLowerRankBiasOnWidth)
{
    // Right-aligned, a rank-1 bias lies on W, not on the channel axis.
    EXPECT_FALSE(layoutsSupported(editTensor(cbaGraph(), "bias", [](fb::TensorAttributesT& t) {
        t.dims = {4};
        t.strides = {1};
    })));
    EXPECT_FALSE(layoutsSupported(editTensor(cbaGraph(), "bias", [](fb::TensorAttributesT& t) {
        t.dims = {1};
        t.strides = {1};
    })));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesBiasWithDifferentRank)
{
    // Rank 3 puts the channel extent on H once right-aligned; rank 5 exceeds the output's rank.
    EXPECT_FALSE(layoutsSupported(editTensor(cbaGraph(), "bias", [](fb::TensorAttributesT& t) {
        t.dims = {1, 4, 1};
        t.strides = {4, 1, 1};
    })));
    EXPECT_FALSE(layoutsSupported(editTensor(cbaGraph(), "bias", [](fb::TensorAttributesT& t) {
        t.dims = {1, 4, 1, 1, 1};
        t.strides = {4, 1, 1, 1, 1};
    })));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesOutputInDifferentLayoutThanInput)
{
    // x is NCHW; MIOpen derives an NCHW output, but the graph asks for NHWC.
    const auto builder
        = editTensor(cbaGraph(), "y", [](fb::TensorAttributesT& t) { t.strides = nhwcStrides(); });
    EXPECT_FALSE(layoutsSupported(builder));
    EXPECT_FALSE(layoutsSupported(
        editTensor(createValidConvFwdActivGraph(), "y", [](fb::TensorAttributesT& t) {
            t.strides = nhwcStrides();
        })));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsFusedOutputInNclLayout)
{
    const auto builder = createValidConvFwdBiasActivGraph(
        {2, 4, 8}, {32, 8, 1}, {4, 4, 3}, {12, 3, 1}, {2, 4, 6}, {24, 6, 1}, {0}, {0}, {1}, {1});
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsFusedOutputInNlcLayout)
{
    const auto builder = createValidConvFwdBiasActivGraph(
        {2, 4, 8}, {32, 1, 4}, {4, 4, 3}, {12, 1, 4}, {2, 4, 6}, {24, 1, 4}, {0}, {0}, {1}, {1});
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsFusedOutputInNlcLayoutWithUnitOutputChannels)
{
    // K = 1: the unit channel dimension ties with L at stride 1 in the channels-last output.
    const auto builder = createValidConvFwdBiasActivGraph(
        {2, 4, 8}, {32, 1, 4}, {1, 4, 3}, {12, 1, 4}, {2, 1, 6}, {6, 1, 1}, {0}, {0}, {1}, {1});
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsFusedOutputInNclLayoutWithUnitOutputLength)
{
    const auto builder = createValidConvFwdBiasActivGraph(
        {2, 4, 3}, {12, 3, 1}, {4, 4, 3}, {12, 3, 1}, {2, 4, 1}, {4, 1, 1}, {0}, {0}, {1}, {1});
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesNlcInputWithNclOutput)
{
    const auto builder = createValidConvFwdBiasActivGraph(
        {2, 4, 8}, {32, 1, 4}, {4, 4, 3}, {12, 1, 4}, {2, 4, 6}, {24, 6, 1}, {0}, {0}, {1}, {1});
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsFusedOutputInNcdhwLayout)
{
    const auto builder = createValidConvFwdBiasActivGraph({2, 4, 4, 4, 4},
                                                          {256, 64, 16, 4, 1},
                                                          {4, 4, 1, 1, 1},
                                                          {4, 1, 1, 1, 1},
                                                          {2, 4, 4, 4, 4},
                                                          {256, 64, 16, 4, 1},
                                                          {0, 0, 0},
                                                          {0, 0, 0},
                                                          {1, 1, 1},
                                                          {1, 1, 1});
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsFusedOutputInNdhwcLayout)
{
    const auto builder = createValidConvFwdBiasActivGraph({2, 4, 4, 4, 4},
                                                          {256, 1, 64, 16, 4},
                                                          {4, 4, 1, 1, 1},
                                                          {4, 1, 4, 4, 4},
                                                          {2, 4, 4, 4, 4},
                                                          {256, 1, 64, 16, 4},
                                                          {0, 0, 0},
                                                          {0, 0, 0},
                                                          {1, 1, 1},
                                                          {1, 1, 1});
    EXPECT_TRUE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesOutputWithDifferentDataTypeThanInput)
{
    const auto builder = editTensor(
        cbaGraph(), "y", [](fb::TensorAttributesT& t) { t.data_type = fb::DataType::HALF; });
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesOutputWithDifferentDimsThanConvolution)
{
    const auto builder = editTensor(cbaGraph(), "y", [](fb::TensorAttributesT& t) {
        t.dims = {4, 4, 2, 8};
        t.strides = {64, 16, 8, 1};
    });
    EXPECT_FALSE(layoutsSupported(builder));
}

TEST(TestMiopenConvApplicabilityChecks, DetectsPassByValueTensors)
{
    const auto legal = createValidConvFwdBiasActivGraph();
    const GraphWrapper legalGraph(legal.GetBufferPointer(), legal.GetSize());
    EXPECT_FALSE(conv_applicability::hasPassByValueTensor(legalGraph.getTensorMap()));

    for(const char* name : {"x", "w", "bias", "y"})
    {
        const auto runtime = editTensor(
            legal, name, [](fb::TensorAttributesT& t) { t.is_runtime_pass_by_value = true; });
        const GraphWrapper runtimeGraph(runtime.GetBufferPointer(), runtime.GetSize());
        EXPECT_TRUE(conv_applicability::hasPassByValueTensor(runtimeGraph.getTensorMap())) << name;

        const auto baked = editTensor(
            legal, name, [](fb::TensorAttributesT& t) { t.value.Set(fb::Float64Value(1.0)); });
        const GraphWrapper bakedGraph(baked.GetBufferPointer(), baked.GetSize());
        EXPECT_TRUE(conv_applicability::hasPassByValueTensor(bakedGraph.getTensorMap())) << name;
    }
}
