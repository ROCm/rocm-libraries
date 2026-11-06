// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "harness/tolerance/ConvFwdDynamicTolerance.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
namespace tol = hipdnn_integration_tests::tolerance;
namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

constexpr int64_t Y_UID = 0;
constexpr int64_t W_UID = 1;
constexpr int64_t X_UID = 2;
constexpr int64_t SCALE_UID = 3;
constexpr int64_t OUT_UID = 4;

struct GraphSpec
{
    std::vector<int64_t> wDims = {32, 8, 3, 3};
    DataType convCompute = DataType::FLOAT;
    bool withScale = false;
    bool withConv = true;
};

// x [2, 16, 8, 8], w as given, y [2, 32, 6, 6]; optionally y * scale -> out.
fb::GraphWrapper buildGraph(flatbuffers::FlatBufferBuilder& b, const GraphSpec& spec)
{
    b.Clear();
    const std::vector<int64_t> xDims = {2, 16, 8, 8};
    const std::vector<int64_t> yDims = {2, 32, 6, 6};
    const std::vector<int64_t> scaleDims = {1, 1, 1, 1};
    const std::vector<int64_t> ones = {1, 1, 1, 1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, Y_UID, "y", DataType::HALF, &ones, &yDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, W_UID, "w", DataType::HALF, &ones, &spec.wDims));
    tensors.push_back(CreateTensorAttributesDirect(b, X_UID, "x", DataType::HALF, &ones, &xDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, SCALE_UID, "scale", DataType::HALF, &ones, &scaleDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, OUT_UID, "out", DataType::HALF, &ones, &yDims));

    std::vector<flatbuffers::Offset<Node>> nodes;
    if(spec.withConv)
    {
        const std::vector<int64_t> zeros = {0, 0};
        const std::vector<int64_t> unit = {1, 1};
        const auto conv = CreateConvolutionFwdAttributesDirect(
            b, X_UID, W_UID, Y_UID, &zeros, &zeros, &unit, &unit, ConvMode::CROSS_CORRELATION);
        nodes.push_back(CreateNodeDirect(
            b, "conv", spec.convCompute, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));
    }
    if(spec.withScale)
    {
        const auto mul = CreatePointwiseAttributes(b,
                                                   PointwiseMode::MUL,
                                                   flatbuffers::nullopt,
                                                   flatbuffers::nullopt,
                                                   flatbuffers::nullopt,
                                                   flatbuffers::nullopt,
                                                   Y_UID,
                                                   SCALE_UID,
                                                   flatbuffers::nullopt,
                                                   OUT_UID);
        nodes.push_back(CreateNodeDirect(
            b, "mul", DataType::FLOAT, NodeAttributes::PointwiseAttributes, mul.Union()));
    }

    auto graph = CreateGraphDirect(
        b, "conv", DataType::FLOAT, DataType::HALF, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    return fb::GraphWrapper(b.GetBufferPointer(), b.GetSize());
}

std::shared_ptr<ITensor> filled(const std::vector<int64_t>& dims, float value)
{
    auto t = std::make_shared<Tensor<half>>(dims);
    t->fillWithValue(static_cast<half>(value));
    return t;
}

struct Inputs
{
    std::unordered_map<int64_t, std::shared_ptr<ITensor>> byUid;

    tol::InputMomentsLookup lookup()
    {
        return [this](int64_t uid) -> std::optional<tol::InputMoments> {
            const auto it = byUid.find(uid);
            if(it == byUid.end())
            {
                return std::nullopt;
            }
            return tol::hostMoments(*it->second, DataType::HALF);
        };
    }
};

Inputs defaultInputs(const GraphSpec& spec)
{
    Inputs in;
    in.byUid[X_UID] = filled({2, 16, 8, 8}, 2.0f);
    in.byUid[W_UID] = filled(spec.wDims, 0.5f);
    in.byUid[SCALE_UID] = filled({1, 1, 1, 1}, -3.0f);
    return in;
}

} // namespace

// Grouped weights hold C/groups channels, so K is read from the weights, not the input.
TEST(TestConvFwdDynamicTolerance, AccumulationLengthIsWeightChannelsTimesFilterVolume)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.wDims = {32, 4, 3, 3};
    auto in = defaultInputs(spec);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_EQ(scale->accumulationLength, 36u);
}

TEST(TestConvFwdDynamicTolerance, MeanSquaresComeFromTheInputValues)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    auto in = defaultInputs(spec);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->inputMeanSquare, 4.0);
    EXPECT_DOUBLE_EQ(scale->weightMeanSquare, 0.25);
    EXPECT_DOUBLE_EQ(scale->outputScale, 1.0);
}

TEST(TestConvFwdDynamicTolerance, FusedMultiplyScalesTheOutputByItsLargestMagnitude)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.withScale = true;
    auto in = defaultInputs(spec);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 3.0);
}

TEST(TestConvFwdDynamicTolerance, GraphWithoutConvolutionIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.withConv = false;
    spec.withScale = true;
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// The model assumes float accumulation; a convolution computing in half falls back to the
// fixed default.
TEST(TestConvFwdDynamicTolerance, NonFloatAccumulatorIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.convCompute = DataType::HALF;
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

TEST(TestConvFwdDynamicTolerance, MissingWeightsAreNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    auto in = defaultInputs(spec);
    in.byUid.erase(W_UID);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

TEST(TestConvFwdDynamicTolerance, OnlyLowPrecisionOutputsGetADynamicTolerance)
{
    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = 144;
    scale.inputMeanSquare = 1.0 / 3.0;
    scale.weightMeanSquare = 1.0 / 3.0;
    EXPECT_TRUE(tol::convFwdOutputTolerance(scale, DataType::HALF).has_value());
    EXPECT_TRUE(tol::convFwdOutputTolerance(scale, DataType::BFLOAT16).has_value());
    EXPECT_FALSE(tol::convFwdOutputTolerance(scale, DataType::FLOAT).has_value());
}

// Recipe-filled inputs are never read: their statistics come from the recipe.
TEST(TestConvFwdDynamicTolerance, RecipeMomentsOfAUniformDraw)
{
    const auto m = tol::recipeMoments(hipdnn_integration_tests::FillRecipe::free(-1.0f, 3.0f));
    EXPECT_DOUBLE_EQ(m.meanSquare, 7.0 / 3.0);
    EXPECT_DOUBLE_EQ(m.maxAbs, 3.0);
}

TEST(TestConvFwdDynamicTolerance, RecipeMomentsOfAFixedValue)
{
    const auto m = tol::recipeMoments(hipdnn_integration_tests::FillRecipe::fixed(-0.5f));
    EXPECT_DOUBLE_EQ(m.meanSquare, 0.25);
    EXPECT_DOUBLE_EQ(m.maxAbs, 0.5);
}

// A strided tensor's buffer has gaps between rows; only the logical elements count.
TEST(TestConvFwdDynamicTolerance, HostMomentsSkipTheGapsOfAStridedTensor)
{
    Tensor<half> t({2, 3}, {4, 1});
    t.fillWithValue(static_cast<half>(10.0f));
    for(auto it = t.begin(); it != t.end(); ++it)
    {
        *static_cast<half*>(*it) = static_cast<half>(1.0f);
    }
    const auto m = tol::hostMoments(t, DataType::HALF);
    ASSERT_TRUE(m.has_value());
    EXPECT_DOUBLE_EQ(m->meanSquare, 1.0);
    EXPECT_DOUBLE_EQ(m->maxAbs, 1.0);
}
