// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp>

#include "harness/tolerance/ConvFwdDynamicTolerance.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
namespace tol = hipdnn_integration_tests::tolerance;
namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using hipdnn_data_sdk::types::bfloat16;
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
constexpr int64_t BIAS_UID = 5;
constexpr int64_t MUL_OUT_UID = 6;
constexpr int64_t Y2_UID = 7;
constexpr int64_t W2_UID = 8;
constexpr int64_t RAW_X_UID = 9;
constexpr int64_t MUL2_OUT_UID = 10;

// How a multiply is wired into the graph.
enum class Mul
{
    NONE,
    SCALE_SECOND, ///< MUL(y, scale), as the bundle generator emits it
    SCALE_FIRST, ///< MUL(scale, y)
    SQUARE, ///< MUL(y, y): not a scale
    UNRELATED, ///< MUL(scale, bias): neither operand comes from the convolution
    TWO_SCALES, ///< MUL(y, scale), then MUL(that, bias): two scales
};

struct GraphSpec
{
    std::vector<int64_t> wDims = {32, 8, 3, 3};
    DataType convCompute = DataType::FLOAT;
    bool withConv = true;
    bool secondConv = false; ///< a second convolution of the same x with its own weights
    bool convOutputVirtual = true; ///< as the bundle graphs have it, unless the conv result is read
    bool preConvRelu = false; ///< x is relu(raw x): x is not a graph input
    bool producedXIsVirtual = true; ///< with preConvRelu: false leaves x written but not virtual
    bool virtualXWithoutProducer = false; ///< x is virtual and no node writes it
    std::optional<float> bakedX; ///< x is a constant stored in the graph
    Mul mul = Mul::NONE;
    std::optional<float> bakedScale; ///< the scale is a constant stored in the graph
    bool withBias = false; ///< ADD(previous result, bias) after the multiply
    bool bakeScaleAsFloat32 = false; ///< a baked value of the wrong type for a half tensor
};

// x [2, 16, 8, 8], w as given, y [2, 32, 6, 6]; scale [1, 3, 1, 1].
fb::GraphWrapper buildGraph(flatbuffers::FlatBufferBuilder& b, const GraphSpec& spec)
{
    b.Clear();
    const std::vector<int64_t> xDims = {2, 16, 8, 8};
    const std::vector<int64_t> yDims = {2, 32, 6, 6};
    const std::vector<int64_t> scaleDims = {1, 3, 1, 1};
    const std::vector<int64_t> biasDims = {1, 32, 1, 1};
    const std::vector<int64_t> ones = {1, 1, 1, 1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const auto addTensor
        = [&](int64_t uid, const char* name, const std::vector<int64_t>& dims, bool isVirtual) {
              tensors.push_back(CreateTensorAttributesDirect(
                  b, uid, name, DataType::HALF, &ones, &dims, isVirtual));
          };
    addTensor(Y_UID, "y", yDims, spec.convOutputVirtual);
    addTensor(W_UID, "w", spec.wDims, false);
    if(spec.bakedX.has_value())
    {
        tensors.push_back(
            CreateTensorAttributesDirect(b,
                                         X_UID,
                                         "x",
                                         DataType::HALF,
                                         &ones,
                                         &ones,
                                         false,
                                         TensorValue::Float16Value,
                                         b.CreateStruct(Float16Value(*spec.bakedX)).Union()));
    }
    else
    {
        addTensor(X_UID,
                  "x",
                  xDims,
                  spec.preConvRelu ? spec.producedXIsVirtual : spec.virtualXWithoutProducer);
    }
    if(spec.preConvRelu)
    {
        addTensor(RAW_X_UID, "raw_x", xDims, false);
    }
    if(spec.secondConv)
    {
        addTensor(Y2_UID, "y2", yDims, false);
        addTensor(W2_UID, "w2", spec.wDims, false);
    }
    if(spec.bakedScale.has_value())
    {
        const flatbuffers::Offset<void> value
            = spec.bakeScaleAsFloat32 ? b.CreateStruct(Float32Value(*spec.bakedScale)).Union()
                                      : b.CreateStruct(Float16Value(*spec.bakedScale)).Union();
        tensors.push_back(CreateTensorAttributesDirect(
            b,
            SCALE_UID,
            "scale",
            DataType::HALF,
            &ones,
            &scaleDims,
            false,
            spec.bakeScaleAsFloat32 ? TensorValue::Float32Value : TensorValue::Float16Value,
            value));
    }
    else
    {
        addTensor(SCALE_UID, "scale", scaleDims, false);
    }
    addTensor(OUT_UID, "out", yDims, false);
    addTensor(BIAS_UID, "bias", biasDims, false);
    addTensor(MUL_OUT_UID, "mul_out", yDims, true);
    addTensor(MUL2_OUT_UID, "mul2_out", yDims, true);

    std::vector<flatbuffers::Offset<Node>> nodes;
    const auto addPointwise
        = [&](PointwiseMode mode, int64_t in0, flatbuffers::Optional<int64_t> in1, int64_t out) {
              const auto pointwise = CreatePointwiseAttributes(b,
                                                               mode,
                                                               flatbuffers::nullopt,
                                                               flatbuffers::nullopt,
                                                               flatbuffers::nullopt,
                                                               flatbuffers::nullopt,
                                                               in0,
                                                               in1,
                                                               flatbuffers::nullopt,
                                                               out);
              nodes.push_back(CreateNodeDirect(b,
                                               "pointwise",
                                               DataType::FLOAT,
                                               NodeAttributes::PointwiseAttributes,
                                               pointwise.Union()));
          };
    const auto addConv = [&](int64_t x, int64_t w, int64_t y) {
        const std::vector<int64_t> zeros = {0, 0};
        const std::vector<int64_t> unit = {1, 1};
        const auto conv = CreateConvolutionFwdAttributesDirect(
            b, x, w, y, &zeros, &zeros, &unit, &unit, ConvMode::CROSS_CORRELATION);
        nodes.push_back(CreateNodeDirect(
            b, "conv", spec.convCompute, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));
    };

    if(spec.preConvRelu)
    {
        addPointwise(PointwiseMode::RELU_FWD, RAW_X_UID, flatbuffers::nullopt, X_UID);
    }
    if(spec.withConv)
    {
        addConv(X_UID, W_UID, Y_UID);
    }
    if(spec.secondConv)
    {
        addConv(X_UID, W2_UID, Y2_UID);
    }
    int64_t last = Y_UID;
    switch(spec.mul)
    {
    case Mul::NONE:
        break;
    case Mul::SCALE_SECOND:
        addPointwise(PointwiseMode::MUL, Y_UID, SCALE_UID, MUL_OUT_UID);
        last = MUL_OUT_UID;
        break;
    case Mul::SCALE_FIRST:
        addPointwise(PointwiseMode::MUL, SCALE_UID, Y_UID, MUL_OUT_UID);
        last = MUL_OUT_UID;
        break;
    case Mul::SQUARE:
        addPointwise(PointwiseMode::MUL, Y_UID, Y_UID, MUL_OUT_UID);
        last = MUL_OUT_UID;
        break;
    case Mul::UNRELATED:
        addPointwise(PointwiseMode::MUL, SCALE_UID, BIAS_UID, MUL_OUT_UID);
        last = MUL_OUT_UID;
        break;
    case Mul::TWO_SCALES:
        addPointwise(PointwiseMode::MUL, Y_UID, SCALE_UID, MUL_OUT_UID);
        addPointwise(PointwiseMode::MUL, MUL_OUT_UID, BIAS_UID, MUL2_OUT_UID);
        last = MUL2_OUT_UID;
        break;
    default:
        break;
    }
    if(spec.withBias)
    {
        addPointwise(PointwiseMode::ADD, last, BIAS_UID, OUT_UID);
    }

    const auto graph = CreateGraphDirect(
        b, "conv", DataType::FLOAT, DataType::HALF, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    return fb::GraphWrapper(b.GetBufferPointer(), b.GetSize());
}

template <typename T>
std::shared_ptr<ITensor> filledWith(const std::vector<int64_t>& dims,
                                    const std::vector<float>& cycle)
{
    auto t = std::make_shared<Tensor<T>>(dims);
    size_t i = 0;
    for(auto it = t->begin(); it != t->end(); ++it)
    {
        *static_cast<T*>(*it) = static_cast<T>(cycle[i++ % cycle.size()]);
    }
    return t;
}

std::shared_ptr<ITensor> filled(const std::vector<int64_t>& dims, float value)
{
    return filledWith<half>(dims, {value});
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

// x is 2.0 everywhere and w 0.5; the scale cycles through -3, 1, 2 (largest magnitude 3, root
// mean square 2.2, mean square 14/3), and the bias is small.
Inputs defaultInputs(const GraphSpec& spec)
{
    Inputs in;
    in.byUid[X_UID] = filled({2, 16, 8, 8}, 2.0f);
    in.byUid[RAW_X_UID] = filled({2, 16, 8, 8}, 2.0f);
    in.byUid[W_UID] = filled(spec.wDims, 0.5f);
    in.byUid[SCALE_UID] = filledWith<half>({1, 3, 1, 1}, {-3.0f, 1.0f, 2.0f});
    in.byUid[BIAS_UID] = filled({1, 32, 1, 1}, 0.01f);
    return in;
}

} // namespace

// ── Accumulation length ─────────────────────────────────────────────────────

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

// 1D, 2D and 3D weights: every dimension after the first counts.
TEST(TestConvFwdDynamicTolerance, AccumulationLengthCoversEveryWeightRank)
{
    const std::vector<std::pair<std::vector<int64_t>, uint64_t>> cases
        = {{{32, 4, 3}, 12u}, {{32, 4, 3, 3}, 36u}, {{32, 4, 3, 3, 5}, 180u}};
    for(const auto& [wDims, expectedK] : cases)
    {
        flatbuffers::FlatBufferBuilder b;
        GraphSpec spec;
        spec.wDims = wDims;
        auto in = defaultInputs(spec);
        const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
        ASSERT_TRUE(scale.has_value()) << "weight rank " << wDims.size();
        EXPECT_EQ(scale->accumulationLength, expectedK) << "weight rank " << wDims.size();
    }
}

TEST(TestConvFwdDynamicTolerance, WeightsWithoutAChannelDimensionAreNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.wDims = {32};
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

TEST(TestConvFwdDynamicTolerance, WeightsWithAZeroDimensionAreNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.wDims = {32, 0, 3, 3};
    auto in = defaultInputs(GraphSpec{}); // a tensor cannot have a zero dimension
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// ── Input statistics ────────────────────────────────────────────────────────

TEST(TestConvFwdDynamicTolerance, MeanSquaresComeFromTheInputValues)
{
    flatbuffers::FlatBufferBuilder b;
    const GraphSpec spec;
    auto in = defaultInputs(spec);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->inputMeanSquare, 4.0);
    EXPECT_DOUBLE_EQ(scale->weightMeanSquare, 0.25);
    EXPECT_DOUBLE_EQ(scale->outputScale, 1.0);
}

TEST(TestConvFwdDynamicTolerance, MissingWeightsAreNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    const GraphSpec spec;
    auto in = defaultInputs(spec);
    in.byUid.erase(W_UID);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// x must be a graph input: when a pointwise node produces it (it is virtual), the recipe or blob
// for the raw input says nothing about what the convolution reads.
TEST(TestConvFwdDynamicTolerance, InputProducedByAnotherNodeIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.preConvRelu = true;
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// Each of the two conditions that make x not a graph input rejects it on its own: an x that a
// node writes but that is not virtual, and a virtual x that no node writes.
TEST(TestConvFwdDynamicTolerance, InputThatANodeWritesButIsNotVirtualIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.preConvRelu = true;
    spec.producedXIsVirtual = false;
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

TEST(TestConvFwdDynamicTolerance, VirtualInputThatNoNodeWritesIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.virtualXWithoutProducer = true;
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// A baked x is read from the graph, so its mean square is the square of its value, whatever the
// lookup holds for that uid; w, filled with 0.5, gives 0.25.
TEST(TestConvFwdDynamicTolerance, BakedInputGivesItsOwnMeanSquare)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.bakedX = 2.0f;
    auto in = defaultInputs(spec);
    in.byUid[X_UID] = filled({2, 16, 8, 8}, 7.0f);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->inputMeanSquare, 4.0);
    EXPECT_DOUBLE_EQ(scale->weightMeanSquare, 0.25);
}

// ── Graph shapes ────────────────────────────────────────────────────────────

TEST(TestConvFwdDynamicTolerance, GraphWithoutConvolutionIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.withConv = false;
    spec.mul = Mul::SCALE_SECOND;
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

TEST(TestConvFwdDynamicTolerance, SeveralConvolutionsAreNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.secondConv = true;
    auto in = defaultInputs(spec);
    in.byUid[W2_UID] = filled(spec.wDims, 0.5f);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// The model assumes float accumulation; a convolution computing in any other type falls back to
// the fixed default.
TEST(TestConvFwdDynamicTolerance, NonFloatAccumulatorIsNotModeled)
{
    for(const auto compute : {DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE})
    {
        flatbuffers::FlatBufferBuilder b;
        GraphSpec spec;
        spec.convCompute = compute;
        auto in = defaultInputs(spec);
        EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value())
            << EnumNameDataType(compute);
    }
}

// ── Fused pointwise nodes ───────────────────────────────────────────────────

// The scale cycles through -3, 1, 2: its largest magnitude (3) differs from its root mean
// square (2.2) and from its mean (0), so only the largest magnitude passes.
TEST(TestConvFwdDynamicTolerance, FusedMultiplyScalesTheOutputByItsLargestMagnitude)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    auto in = defaultInputs(spec);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 3.0);
}

// MUL is commutative: the scale may be the first operand.
TEST(TestConvFwdDynamicTolerance, FusedMultiplyWithTheScaleFirstScalesToo)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.mul = Mul::SCALE_FIRST;
    auto in = defaultInputs(spec);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 3.0);
}

// The conv result may be a graph output as well (not virtual) and still is the convolution's
// operand of the multiply, not a scale.
TEST(TestConvFwdDynamicTolerance, FusedMultiplyOfANonVirtualConvolutionResultStillScales)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.convOutputVirtual = false;
    auto in = defaultInputs(spec);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 3.0);
}

// The shipped Variant3 half and bfloat16 graphs bake the scale into the graph. The lookup does
// not know that tensor (the harness holds no recipe for it), so the value must come from the
// graph.
TEST(TestConvFwdDynamicTolerance, BakedScaleIsReadFromTheGraphNotTheLookup)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.bakedScale = 0.5f;
    auto in = defaultInputs(spec);
    in.byUid.erase(SCALE_UID);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 0.5);
}

TEST(TestConvFwdDynamicTolerance, BakedScaleAboveOneWidensAndANegativeOneCountsByMagnitude)
{
    for(const float baked : {4.0f, -2.0f})
    {
        flatbuffers::FlatBufferBuilder b;
        GraphSpec spec;
        spec.mul = Mul::SCALE_SECOND;
        spec.bakedScale = baked;
        auto in = defaultInputs(spec);
        const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
        ASSERT_TRUE(scale.has_value());
        EXPECT_DOUBLE_EQ(scale->outputScale, std::abs(static_cast<double>(baked)));
    }
}

TEST(TestConvFwdDynamicTolerance, BakedScaleThatIsNotAFiniteNumberIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.bakedScale = std::numeric_limits<float>::quiet_NaN();
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// A baked value stored as a float32 constant in a half tensor does not describe that tensor.
TEST(TestConvFwdDynamicTolerance, BakedScaleOfTheWrongTypeIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.bakedScale = 0.5f;
    spec.bakeScaleAsFloat32 = true;
    auto in = defaultInputs(spec);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// A multiply whose scale the lookup cannot describe leaves the graph outside the model.
TEST(TestConvFwdDynamicTolerance, FusedMultiplyWithAnUnknownScaleIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    auto in = defaultInputs(spec);
    in.byUid.erase(SCALE_UID);
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// Squaring the result, or multiplying two unrelated tensors, is not a scale of the output.
TEST(TestConvFwdDynamicTolerance, MultiplyThatIsNotAScaleIsNotModeled)
{
    for(const Mul mul : {Mul::SQUARE, Mul::UNRELATED})
    {
        flatbuffers::FlatBufferBuilder b;
        GraphSpec spec;
        spec.mul = mul;
        auto in = defaultInputs(spec);
        EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
    }
}

// A bias add takes a graph input as its second operand too, but is no scale: the bias (0.01
// at most) must not shrink the tolerance, and the bias add after a multiply keeps the multiply's.
TEST(TestConvFwdDynamicTolerance, BiasAddLeavesTheOutputScaleAlone)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.withBias = true;
    auto in = defaultInputs(spec);
    const auto plain = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(plain.has_value());
    EXPECT_DOUBLE_EQ(plain->outputScale, 1.0);

    GraphSpec scaled;
    scaled.mul = Mul::SCALE_SECOND;
    scaled.withBias = true;
    const auto both = tol::convFwdErrorScale(buildGraph(b, scaled), in.lookup());
    ASSERT_TRUE(both.has_value());
    EXPECT_DOUBLE_EQ(both->outputScale, 3.0);
}

// Scales multiply: two fused multiplies, by a tensor of magnitude 3 and by one of 0.01, scale the
// output by the product of their largest magnitudes.
TEST(TestConvFwdDynamicTolerance, SeveralFusedMultipliesScaleByTheProduct)
{
    flatbuffers::FlatBufferBuilder b;
    GraphSpec spec;
    spec.mul = Mul::TWO_SCALES;
    auto in = defaultInputs(spec);
    const auto scale = tol::convFwdErrorScale(buildGraph(b, spec), in.lookup());
    ASSERT_TRUE(scale.has_value());
    EXPECT_NEAR(scale->outputScale, 3.0 * static_cast<double>(static_cast<half>(0.01f)), 1e-12);
}

// ── Output dispatch ─────────────────────────────────────────────────────────

// The two types differ by a factor of about 16,000 in atol (half carries the lambda term, bfloat16
// does not) and by 8 the other way in rtol, so a swap of the dispatch shows in the values.
TEST(TestConvFwdDynamicTolerance, OutputTypeSelectsItsOwnTolerance)
{
    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = 144;
    scale.inputMeanSquare = 1.0 / 3.0;
    scale.weightMeanSquare = 1.0 / 3.0;
    scale.outputScale = 2.0;

    const auto expectedHalf
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropOutputTolerance<half>(scale);
    const auto expectedBfloat16
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropOutputTolerance<bfloat16>(scale);
    ASSERT_NE(expectedHalf.atol, expectedBfloat16.atol);
    ASSERT_NE(expectedHalf.rtol, expectedBfloat16.rtol);

    const auto halfTolerance = tol::convFwdOutputTolerance(scale, DataType::HALF);
    ASSERT_TRUE(halfTolerance.has_value());
    EXPECT_DOUBLE_EQ(halfTolerance->atol, expectedHalf.atol);
    EXPECT_DOUBLE_EQ(halfTolerance->rtol, expectedHalf.rtol);

    const auto bfloat16Tolerance = tol::convFwdOutputTolerance(scale, DataType::BFLOAT16);
    ASSERT_TRUE(bfloat16Tolerance.has_value());
    EXPECT_DOUBLE_EQ(bfloat16Tolerance->atol, expectedBfloat16.atol);
    EXPECT_DOUBLE_EQ(bfloat16Tolerance->rtol, expectedBfloat16.rtol);
}

TEST(TestConvFwdDynamicTolerance, OnlyLowPrecisionOutputsGetADynamicTolerance)
{
    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = 144;
    scale.inputMeanSquare = 1.0 / 3.0;
    scale.weightMeanSquare = 1.0 / 3.0;
    EXPECT_FALSE(tol::convFwdOutputTolerance(scale, DataType::FLOAT).has_value());
    EXPECT_FALSE(tol::convFwdOutputTolerance(scale, DataType::DOUBLE).has_value());
}

// The harness asks hasDynamicToleranceFor() before it scans any input, and convFwdOutputTolerance()
// answers for the same types: the two agree for every element type.
TEST(TestConvFwdDynamicTolerance, TheOutputTypePredicateAgreesWithTheToleranceFunction)
{
    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = 144;
    scale.inputMeanSquare = 1.0 / 3.0;
    scale.weightMeanSquare = 1.0 / 3.0;
    for(const auto type : EnumValuesDataType())
    {
        EXPECT_EQ(tol::hasDynamicToleranceFor(type),
                  tol::convFwdOutputTolerance(scale, type).has_value())
            << EnumNameDataType(type);
    }
    EXPECT_TRUE(tol::hasDynamicToleranceFor(DataType::HALF));
    EXPECT_TRUE(tol::hasDynamicToleranceFor(DataType::BFLOAT16));
    EXPECT_FALSE(tol::hasDynamicToleranceFor(DataType::FLOAT));
}

// ── Recipe and host moments ─────────────────────────────────────────────────

// Recipe-filled inputs are never read: their statistics come from the recipe.
TEST(TestConvFwdDynamicTolerance, RecipeMomentsOfAUniformDraw)
{
    const auto m = tol::recipeMoments(hipdnn_integration_tests::FillRecipe::free(-1.0f, 3.0f));
    EXPECT_DOUBLE_EQ(m.meanSquare, 7.0 / 3.0);
    EXPECT_DOUBLE_EQ(m.maxAbs, 3.0);
}

// The negative end is the larger one: the largest magnitude is |lo|, not |hi|.
TEST(TestConvFwdDynamicTolerance, RecipeMomentsOfADrawWithTheLargerNegativeEnd)
{
    const auto m = tol::recipeMoments(hipdnn_integration_tests::FillRecipe::free(-2.0f, 1.0f));
    EXPECT_DOUBLE_EQ(m.meanSquare, 1.0);
    EXPECT_DOUBLE_EQ(m.maxAbs, 2.0);
}

TEST(TestConvFwdDynamicTolerance, RecipeMomentsOfAFixedValue)
{
    const auto m = tol::recipeMoments(hipdnn_integration_tests::FillRecipe::fixed(-0.5f));
    EXPECT_DOUBLE_EQ(m.meanSquare, 0.25);
    EXPECT_DOUBLE_EQ(m.maxAbs, 0.5);
}

// A ramp 1, 2, 3, 4 has mean square 7.5 and largest magnitude 4: neither equals a constant's,
// and the packed linear walk must visit every element.
TEST(TestConvFwdDynamicTolerance, HostMomentsOfAPackedRampInEveryElementType)
{
    const std::vector<float> ramp = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<std::pair<std::shared_ptr<ITensor>, DataType>> tensors = {
        {filledWith<half>({8}, ramp), DataType::HALF},
        {filledWith<bfloat16>({8}, ramp), DataType::BFLOAT16},
        {filledWith<float>({8}, ramp), DataType::FLOAT},
        {filledWith<double>({8}, ramp), DataType::DOUBLE},
    };
    for(const auto& [tensor, dataType] : tensors)
    {
        const auto m = tol::hostMoments(*tensor, dataType);
        ASSERT_TRUE(m.has_value()) << EnumNameDataType(dataType);
        EXPECT_DOUBLE_EQ(m->meanSquare, 7.5) << EnumNameDataType(dataType);
        EXPECT_DOUBLE_EQ(m->maxAbs, 4.0) << EnumNameDataType(dataType);
    }
}

TEST(TestConvFwdDynamicTolerance, HostMomentsOfAnUnmodeledElementTypeAreAbsent)
{
    Tensor<int32_t> t({4});
    EXPECT_FALSE(tol::hostMoments(t, DataType::INT32).has_value());
}

// A NaN or an infinity has no finite statistic: the graph falls back to the default tolerance
// instead of throwing from the tolerance function.
TEST(TestConvFwdDynamicTolerance, HostMomentsOfDataWithANonFiniteValueAreAbsent)
{
    for(const float bad :
        {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
    {
        const auto tensor = filledWith<float>({4}, {1.0f, 2.0f, bad, 3.0f});
        EXPECT_FALSE(tol::hostMoments(*tensor, DataType::FLOAT).has_value());
    }
}

TEST(TestConvFwdDynamicTolerance, GraphWithANonFiniteInputIsNotModeled)
{
    flatbuffers::FlatBufferBuilder b;
    const GraphSpec spec;
    auto in = defaultInputs(spec);
    in.byUid[X_UID]
        = filledWith<half>({2, 16, 8, 8}, {1.0f, std::numeric_limits<float>::quiet_NaN()});
    EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), in.lookup()).has_value());
}

// A recipe's moments come from the lookup unchecked, so a fill of an infinity must be stopped
// before it reaches the tolerance function, which throws on it: the graph falls back to the
// default tolerance, as it does for a non-finite golden blob.
TEST(TestConvFwdDynamicTolerance, GraphWithANonFiniteRecipeIsNotModeled)
{
    using hipdnn_integration_tests::FillRecipe;
    constexpr float K_INFINITY = std::numeric_limits<float>::infinity();
    const std::vector<FillRecipe> bad = {FillRecipe::fixed(K_INFINITY),
                                         FillRecipe::free(-K_INFINITY, K_INFINITY),
                                         FillRecipe::free(0.0f, K_INFINITY)};
    for(const auto& badRecipe : bad)
    {
        for(const int64_t badUid : {X_UID, W_UID})
        {
            flatbuffers::FlatBufferBuilder b;
            const GraphSpec spec;
            const auto lookup = [&](int64_t uid) -> std::optional<tol::InputMoments> {
                return tol::recipeMoments(uid == badUid ? badRecipe
                                                        : FillRecipe::free(-1.0f, 1.0f));
            };
            EXPECT_FALSE(tol::convFwdErrorScale(buildGraph(b, spec), lookup).has_value())
                << "bad uid " << badUid;
        }
    }
}

// A strided tensor's buffer has gaps between rows; only the logical elements count.
TEST(TestConvFwdDynamicTolerance, HostMomentsSkipTheGapsOfAStridedTensor)
{
    Tensor<half> t({2, 3}, {4, 1});
    t.fillWithValue(static_cast<half>(10.0f));
    float next = 1.0f;
    for(auto it = t.begin(); it != t.end(); ++it)
    {
        *static_cast<half*>(*it) = static_cast<half>(next);
        next += 1.0f;
    }
    const auto m = tol::hostMoments(t, DataType::HALF);
    ASSERT_TRUE(m.has_value());
    EXPECT_DOUBLE_EQ(m->meanSquare, 91.0 / 6.0); // 1, 4, 9, 16, 25, 36
    EXPECT_DOUBLE_EQ(m->maxAbs, 6.0);
}
