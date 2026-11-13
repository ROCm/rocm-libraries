// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp>

#include "harness/tolerance/ConvDynamicTolerance.hpp"

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
constexpr int64_t ADD_OUT_UID = 11;
constexpr int64_t BRANCH_OUT_UID = 12;
constexpr int64_t CYCLE_A_UID = 13;
constexpr int64_t CYCLE_B_UID = 14;

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

// Nodes that read a tensor twice, tensors that meet again, or a third operand; each ends in the
// graph's output through a relu or directly.
enum class Wiring
{
    NONE,
    MUL_THEN_ADD_Y, ///< ADD(last, y) after a multiply: y is read by two nodes whose results meet
    ADD_Y_TWICE, ///< ADD(y, y)
    SELECT_IN2_PRODUCED, ///< z = ADD(y, bias); BINARY_SELECT(bias, bias, z): in_2 is produced
    SELECT_IN2_INPUT, ///< z = ADD(y, bias); BINARY_SELECT(z, bias, bias): in_2 is a graph input
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
    DataType convOutputType = DataType::FLOAT; ///< the type of the convolution's result
    DataType intermediateType = DataType::FLOAT; ///< the type of the multiplies' results
    DataType pointwiseCompute = DataType::FLOAT; ///< the compute type of every pointwise node
    std::optional<DataType> addCompute; ///< the compute type of the bias add, when it differs
    DataType biasType = DataType::HALF; ///< the type of the bias, a graph input
    DataType outputType = DataType::HALF; ///< the type of the graph's output
    bool withRelu = false; ///< RELU_FWD after the bias add, as the Variant2 bundles have it
    bool branch
        = false; ///< a second output, ADD(y, bias): the convolution's result has two readers
    bool branchFirst = false; ///< with branch: that add is listed before the multiply
    bool reverseNodes = false; ///< the nodes are listed from the last to the first
    bool cycle = false; ///< two more nodes, each reading the other's result
    Wiring wiring = Wiring::NONE; ///< more nodes, after the multiply
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
    // Graph inputs and outputs are half. A tensor one node writes and another reads is float
    // unless a test says otherwise.
    const auto addTensor = [&](int64_t uid,
                               const char* name,
                               const std::vector<int64_t>& dims,
                               bool isVirtual,
                               DataType type = DataType::HALF) {
        tensors.push_back(
            CreateTensorAttributesDirect(b, uid, name, type, &ones, &dims, isVirtual));
    };
    addTensor(Y_UID, "y", yDims, spec.convOutputVirtual, spec.convOutputType);
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
                  spec.preConvRelu ? spec.producedXIsVirtual : spec.virtualXWithoutProducer,
                  spec.preConvRelu || spec.virtualXWithoutProducer ? DataType::FLOAT
                                                                   : DataType::HALF);
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
    addTensor(OUT_UID, "out", yDims, false, spec.outputType);
    addTensor(BIAS_UID, "bias", biasDims, false, spec.biasType);
    addTensor(ADD_OUT_UID, "add_out", yDims, true, spec.intermediateType);
    addTensor(MUL_OUT_UID, "mul_out", yDims, true, spec.intermediateType);
    addTensor(MUL2_OUT_UID, "mul2_out", yDims, true, spec.intermediateType);
    addTensor(BRANCH_OUT_UID, "branch_out", yDims, false);
    addTensor(CYCLE_A_UID, "cycle_a", yDims, true, spec.intermediateType);
    addTensor(CYCLE_B_UID, "cycle_b", yDims, true, spec.intermediateType);

    std::vector<flatbuffers::Offset<Node>> nodes;
    const auto addPointwise = [&](PointwiseMode mode,
                                  int64_t in0,
                                  flatbuffers::Optional<int64_t> in1,
                                  int64_t out,
                                  std::optional<DataType> compute = std::nullopt,
                                  flatbuffers::Optional<int64_t> in2 = flatbuffers::nullopt) {
        const auto pointwise = CreatePointwiseAttributes(b,
                                                         mode,
                                                         flatbuffers::nullopt,
                                                         flatbuffers::nullopt,
                                                         flatbuffers::nullopt,
                                                         flatbuffers::nullopt,
                                                         in0,
                                                         in1,
                                                         in2,
                                                         out);
        nodes.push_back(CreateNodeDirect(b,
                                         "pointwise",
                                         compute.value_or(spec.pointwiseCompute),
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
    if(spec.branch && spec.branchFirst)
    {
        addPointwise(PointwiseMode::ADD, Y_UID, BIAS_UID, BRANCH_OUT_UID);
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
    if(spec.branch && !spec.branchFirst)
    {
        addPointwise(PointwiseMode::ADD, Y_UID, BIAS_UID, BRANCH_OUT_UID);
    }
    if(spec.cycle)
    {
        addPointwise(PointwiseMode::ADD, CYCLE_A_UID, BIAS_UID, CYCLE_B_UID);
        addPointwise(PointwiseMode::ADD, CYCLE_B_UID, BIAS_UID, CYCLE_A_UID);
    }
    switch(spec.wiring)
    {
    case Wiring::NONE:
        break;
    case Wiring::MUL_THEN_ADD_Y:
        addPointwise(PointwiseMode::ADD, last, Y_UID, ADD_OUT_UID);
        addPointwise(PointwiseMode::RELU_FWD, ADD_OUT_UID, flatbuffers::nullopt, OUT_UID);
        break;
    case Wiring::ADD_Y_TWICE:
        addPointwise(PointwiseMode::ADD, Y_UID, Y_UID, ADD_OUT_UID);
        addPointwise(PointwiseMode::RELU_FWD, ADD_OUT_UID, flatbuffers::nullopt, OUT_UID);
        break;
    case Wiring::SELECT_IN2_PRODUCED:
        addPointwise(PointwiseMode::ADD, Y_UID, BIAS_UID, ADD_OUT_UID);
        addPointwise(PointwiseMode::BINARY_SELECT,
                     BIAS_UID,
                     BIAS_UID,
                     MUL_OUT_UID,
                     std::nullopt,
                     ADD_OUT_UID);
        addPointwise(PointwiseMode::RELU_FWD, MUL_OUT_UID, flatbuffers::nullopt, OUT_UID);
        break;
    case Wiring::SELECT_IN2_INPUT:
        addPointwise(PointwiseMode::ADD, Y_UID, BIAS_UID, ADD_OUT_UID);
        addPointwise(
            PointwiseMode::BINARY_SELECT, ADD_OUT_UID, BIAS_UID, OUT_UID, std::nullopt, BIAS_UID);
        break;
    default:
        break;
    }
    if(spec.withBias)
    {
        const int64_t addOut = spec.withRelu ? ADD_OUT_UID : OUT_UID;
        addPointwise(PointwiseMode::ADD, last, BIAS_UID, addOut, spec.addCompute);
        if(spec.withRelu)
        {
            addPointwise(PointwiseMode::RELU_FWD, ADD_OUT_UID, flatbuffers::nullopt, OUT_UID);
        }
    }

    if(spec.reverseNodes)
    {
        std::reverse(nodes.begin(), nodes.end());
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

    tol::InputMomentsLookup lookup() const
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

// x is +-2 and w +-0.5, alternating, so their means are zero and sigma is sqrt(K * 4 * 0.25);
// the scale cycles through -3, 1, 2 (largest magnitude 3, root mean square 2.2, mean square
// 14/3), and the bias is small.
Inputs defaultInputs(const GraphSpec& spec)
{
    Inputs in;
    in.byUid[X_UID] = filledWith<half>({2, 16, 8, 8}, {2.0f, -2.0f});
    in.byUid[RAW_X_UID] = filledWith<half>({2, 16, 8, 8}, {2.0f, -2.0f});
    in.byUid[W_UID] = filledWith<half>(spec.wDims, {0.5f, -0.5f});
    in.byUid[SCALE_UID] = filledWith<half>({1, 3, 1, 1}, {-3.0f, 1.0f, 2.0f});
    in.byUid[BIAS_UID] = filled({1, 32, 1, 1}, 0.01f);
    return in;
}

// The error scale of the graph `spec` describes, given the statistics of `in`. The graph's one
// convolution is a forward one, so a scale of another direction fails the test.
std::optional<hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale>
    errorScale(const GraphSpec& spec, const Inputs& in)
{
    using hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale;
    flatbuffers::FlatBufferBuilder b;
    const auto scale = tol::convErrorScale(buildGraph(b, spec), in.lookup());
    if(!scale.has_value())
    {
        return std::nullopt;
    }
    EXPECT_TRUE(std::holds_alternative<ConvFpropErrorScale>(*scale));
    return std::get<ConvFpropErrorScale>(*scale);
}

// The same, with the default inputs of `spec`.
std::optional<hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale>
    errorScale(const GraphSpec& spec)
{
    return errorScale(spec, defaultInputs(spec));
}

} // namespace

// ── Accumulation length ─────────────────────────────────────────────────────

// Grouped weights hold C/groups channels, so K is read from the weights, not the input.
TEST(TestConvFwdDynamicTolerance, AccumulationLengthIsWeightChannelsTimesFilterVolume)
{
    GraphSpec spec;
    spec.wDims = {32, 4, 3, 3};
    const auto scale = errorScale(spec);
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
        GraphSpec spec;
        spec.wDims = wDims;
        const auto scale = errorScale(spec);
        ASSERT_TRUE(scale.has_value()) << "weight rank " << wDims.size();
        EXPECT_EQ(scale->accumulationLength, expectedK) << "weight rank " << wDims.size();
    }
}

TEST(TestConvFwdDynamicTolerance, WeightsWithoutAChannelDimensionAreNotModeled)
{
    GraphSpec spec;
    spec.wDims = {32};
    EXPECT_FALSE(errorScale(spec).has_value());
}

TEST(TestConvFwdDynamicTolerance, WeightsWithAZeroDimensionAreNotModeled)
{
    GraphSpec spec;
    spec.wDims = {32, 0, 3, 3};
    auto in = defaultInputs(GraphSpec{}); // a tensor cannot have a zero dimension
    EXPECT_FALSE(errorScale(spec, in).has_value());
}

// ── Input statistics ────────────────────────────────────────────────────────

TEST(TestConvFwdDynamicTolerance, MeanSquaresComeFromTheInputValues)
{
    const GraphSpec spec;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->inputMeanSquare, 4.0);
    EXPECT_DOUBLE_EQ(scale->weightMeanSquare, 0.25);
    EXPECT_DOUBLE_EQ(scale->outputScale, 1.0);
}

TEST(TestConvFwdDynamicTolerance, MissingWeightsAreNotModeled)
{
    const GraphSpec spec;
    auto in = defaultInputs(spec);
    in.byUid.erase(W_UID);
    EXPECT_FALSE(errorScale(spec, in).has_value());
}

// x must be a graph input: when a pointwise node produces it (it is virtual), the recipe or blob
// for the raw input says nothing about what the convolution reads.
TEST(TestConvFwdDynamicTolerance, InputProducedByAnotherNodeIsNotModeled)
{
    GraphSpec spec;
    spec.preConvRelu = true;
    EXPECT_FALSE(errorScale(spec).has_value());
}

// Each of the two conditions that make x not a graph input rejects it on its own: an x that a
// node writes but that is not virtual, and a virtual x that no node writes.
TEST(TestConvFwdDynamicTolerance, InputThatANodeWritesButIsNotVirtualIsNotModeled)
{
    GraphSpec spec;
    spec.preConvRelu = true;
    spec.producedXIsVirtual = false;
    EXPECT_FALSE(errorScale(spec).has_value());
}

TEST(TestConvFwdDynamicTolerance, VirtualInputThatNoNodeWritesIsNotModeled)
{
    GraphSpec spec;
    spec.virtualXWithoutProducer = true;
    EXPECT_FALSE(errorScale(spec).has_value());
}

// A baked x is read from the graph, so its mean square is the square of its value, whatever the
// lookup holds for that uid; w, filled with 0.5, gives 0.25.
TEST(TestConvFwdDynamicTolerance, BakedInputGivesItsOwnMeanSquare)
{
    GraphSpec spec;
    spec.bakedX = 2.0f;
    auto in = defaultInputs(spec);
    in.byUid[X_UID] = filled({2, 16, 8, 8}, 7.0f);
    const auto scale = errorScale(spec, in);
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->inputMeanSquare, 4.0);
    EXPECT_DOUBLE_EQ(scale->weightMeanSquare, 0.25);
}

// ── Graph shapes ────────────────────────────────────────────────────────────

TEST(TestConvFwdDynamicTolerance, GraphWithoutConvolutionIsNotModeled)
{
    GraphSpec spec;
    spec.withConv = false;
    spec.mul = Mul::SCALE_SECOND;
    EXPECT_FALSE(errorScale(spec).has_value());
}

TEST(TestConvFwdDynamicTolerance, SeveralConvolutionsAreNotModeled)
{
    GraphSpec spec;
    spec.secondConv = true;
    auto in = defaultInputs(spec);
    in.byUid[W2_UID] = filled(spec.wDims, 0.5f);
    EXPECT_FALSE(errorScale(spec, in).has_value());
}

// The model assumes float accumulation; a convolution computing in any other type falls back to
// the fixed default.
TEST(TestConvFwdDynamicTolerance, NonFloatAccumulatorIsNotModeled)
{
    for(const auto compute : {DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE})
    {
        GraphSpec spec;
        spec.convCompute = compute;
        EXPECT_FALSE(errorScale(spec).has_value()) << EnumNameDataType(compute);
    }
}

// ── Rounding points ─────────────────────────────────────────────────────────

constexpr double HALF_EPS = 0.0009765625; // 2^-10
constexpr double BFLOAT16_EPS = 0.0078125; // 2^-7

// sqrt(K * mean(x^2) * mean(w^2)) for the default graph: K = 8 * 3 * 3, x = 2, w = 0.5.
const double SIGMA = std::sqrt(72.0);

// The statistics the default inputs give the bias: half(0.01) is a little below 0.01.
const double BIAS_RMS = static_cast<double>(static_cast<half>(0.01f));

// A convolution result that another node reads and that is kept in half or bfloat16 is rounded
// there, whether or not the tensor is virtual: bfloat16 results 1 and 1.0078125 differ by one
// step, and after adding -1 they are 0 and 0.0078125, which no relative term covers. The point
// sits at the size of the convolution's output, and an add passes the error on as it is.
TEST(TestConvFwdDynamicTolerance, ConvolutionResultReadByANodeAndStoredBelowFloatIsARoundingPoint)
{
    for(const bool isVirtual : {true, false})
    {
        for(const auto& [type, eps] :
            {std::pair{DataType::HALF, HALF_EPS}, std::pair{DataType::BFLOAT16, BFLOAT16_EPS}})
        {
            GraphSpec spec;
            spec.withBias = true;
            spec.convOutputVirtual = isVirtual;
            spec.convOutputType = type;
            const auto scale = errorScale(spec);
            SCOPED_TRACE(std::string(EnumNameDataType(type)) + (isVirtual ? " virtual" : ""));
            ASSERT_TRUE(scale.has_value());
            ASSERT_EQ(scale->roundingPoints.size(), 1u);
            EXPECT_DOUBLE_EQ(scale->roundingPoints[0].epsilon, eps);
            EXPECT_NEAR(scale->roundingPoints[0].magnitude, SIGMA, 1e-12);
            EXPECT_DOUBLE_EQ(scale->roundingPoints[0].slope, 1.0);
        }
    }
}

// float and double hold the result exactly as the model's float accumulation has it.
TEST(TestConvFwdDynamicTolerance, IntermediatesStoredInFloatOrDoubleAreNoRoundingPoint)
{
    for(const auto type : {DataType::FLOAT, DataType::DOUBLE})
    {
        GraphSpec spec;
        spec.withBias = true;
        spec.withRelu = true;
        spec.convOutputType = type;
        spec.intermediateType = type;
        const auto scale = errorScale(spec);
        ASSERT_TRUE(scale.has_value()) << EnumNameDataType(type);
        EXPECT_TRUE(scale->roundingPoints.empty()) << EnumNameDataType(type);
    }
}

// A convolution result that no node reads is the graph's output: rounded once, to the output
// type, which is the model's rtol.
TEST(TestConvFwdDynamicTolerance, ConvolutionResultNoNodeReadsIsNoRoundingPoint)
{
    for(const auto type : {DataType::HALF, DataType::BFLOAT16})
    {
        GraphSpec spec;
        spec.convOutputVirtual = false;
        spec.convOutputType = type;
        const auto scale = errorScale(spec);
        ASSERT_TRUE(scale.has_value()) << EnumNameDataType(type);
        EXPECT_TRUE(scale->roundingPoints.empty()) << EnumNameDataType(type);
    }
}

// The model knows the rounding of half, bfloat16, float and double. An intermediate of any other
// type, or of none, is outside it.
TEST(TestConvFwdDynamicTolerance, IntermediateOfAnUnknownTypeIsNotModeled)
{
    for(const auto type : {DataType::UNSET, DataType::INT32, DataType::UINT8, DataType::BOOLEAN})
    {
        GraphSpec convResult;
        convResult.withBias = true;
        convResult.convOutputType = type;
        EXPECT_FALSE(errorScale(convResult).has_value()) << EnumNameDataType(type);

        GraphSpec pointwiseResult;
        pointwiseResult.mul = Mul::SCALE_SECOND;
        pointwiseResult.withBias = true;
        pointwiseResult.intermediateType = type;
        EXPECT_FALSE(errorScale(pointwiseResult).has_value()) << EnumNameDataType(type);
    }
}

// A pointwise result stored below float that another node reads is a point at the size of that
// result: after a multiply the convolution's output is scaled by the largest magnitude of the
// scale (3).
TEST(TestConvFwdDynamicTolerance, PointwiseResultStoredBelowFloatIsARoundingPointAtItsOwnSize)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.withBias = true;
    spec.intermediateType = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 1u);
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].epsilon, BFLOAT16_EPS);
    EXPECT_NEAR(scale->roundingPoints[0].magnitude, 3.0 * SIGMA, 1e-12);
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].slope, 1.0);
}

// An error in the convolution's result is multiplied by the scale on its way to the output:
// the slope of a point before a multiply is the largest magnitude of the scale, and the slopes of
// several multiplies multiply.
TEST(TestConvFwdDynamicTolerance, SlopeOfARoundingPointIsTheProductOfTheScalesAfterIt)
{
    GraphSpec one;
    one.mul = Mul::SCALE_SECOND;
    one.convOutputType = DataType::BFLOAT16;
    const auto single = errorScale(one);
    ASSERT_TRUE(single.has_value());
    ASSERT_EQ(single->roundingPoints.size(), 1u);
    EXPECT_NEAR(single->roundingPoints[0].magnitude, SIGMA, 1e-12);
    EXPECT_DOUBLE_EQ(single->roundingPoints[0].slope, 3.0);

    GraphSpec two;
    two.mul = Mul::TWO_SCALES;
    two.convOutputType = DataType::BFLOAT16;
    const auto both = errorScale(two);
    ASSERT_TRUE(both.has_value());
    ASSERT_EQ(both->roundingPoints.size(), 1u);
    EXPECT_NEAR(both->roundingPoints[0].slope, 3.0 * BIAS_RMS, 1e-12);
}

// A bias add computed in bfloat16 over a float convolution result (the develop Variant2
// graph): the add converts the result to bfloat16 and rounds its own result to bfloat16. The
// bias is stored in bfloat16 already, so it takes no rounding. The first point is the
// convolution's output, the second is the output of the add: the bias adds in quadrature.
TEST(TestConvFwdDynamicTolerance, PointwiseComputingBelowFloatRoundsItsOperandsAndItsResult)
{
    GraphSpec spec;
    spec.withBias = true;
    spec.withRelu = true;
    spec.addCompute = DataType::BFLOAT16;
    spec.biasType = DataType::BFLOAT16;
    spec.outputType = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 2u);
    for(const auto& point : scale->roundingPoints)
    {
        EXPECT_DOUBLE_EQ(point.epsilon, BFLOAT16_EPS);
        EXPECT_DOUBLE_EQ(point.slope, 1.0);
    }
    EXPECT_NEAR(scale->roundingPoints[0].magnitude, SIGMA, 1e-12);
    EXPECT_NEAR(
        scale->roundingPoints[1].magnitude, std::sqrt(SIGMA * SIGMA + BIAS_RMS * BIAS_RMS), 1e-12);
}

// An operand stored as another type than the node computes in is converted, and rounded: the
// bias too, when it is stored in half and the add computes in bfloat16.
TEST(TestConvFwdDynamicTolerance, OperandStoredAsAnotherTypeThanTheComputeTypeIsRounded)
{
    GraphSpec spec;
    spec.withBias = true;
    spec.addCompute = DataType::BFLOAT16;
    spec.biasType = DataType::HALF;
    spec.outputType = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    // the convolution's result, the bias; the add's result is the graph output, stored in the
    // type the add computes in: that rounding is the model's rtol.
    ASSERT_EQ(scale->roundingPoints.size(), 2u);
    EXPECT_NEAR(scale->roundingPoints[0].magnitude, SIGMA, 1e-12);
    EXPECT_NEAR(scale->roundingPoints[1].magnitude, BIAS_RMS, 1e-12);
}

// The same type stored and computed is one rounding, not two: a bfloat16 graph throughout has
// a point at every intermediate and none for the nodes' own conversions.
TEST(TestConvFwdDynamicTolerance, StoredAndComputedInTheSameTypeIsOneRoundingPerIntermediate)
{
    GraphSpec spec;
    spec.withBias = true;
    spec.withRelu = true;
    spec.pointwiseCompute = DataType::BFLOAT16;
    spec.convOutputType = DataType::BFLOAT16;
    spec.intermediateType = DataType::BFLOAT16;
    spec.biasType = DataType::BFLOAT16;
    spec.outputType = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 2u);
    EXPECT_NEAR(scale->roundingPoints[0].magnitude, SIGMA, 1e-12);
    EXPECT_NEAR(
        scale->roundingPoints[1].magnitude, std::sqrt(SIGMA * SIGMA + BIAS_RMS * BIAS_RMS), 1e-12);
}

// A multiply that computes below float rounds the scale too when the scale is stored in another
// type, and an error in the scale reaches the output multiplied by the size of the
// convolution's result: that point's slope is the result's magnitude, not the scale's largest
// magnitude, whichever operand the scale is.
TEST(TestConvFwdDynamicTolerance, ScaleOfAMultiplyComputingBelowFloatIsRoundedWithTheResultAsSlope)
{
    const double scaleRms = std::sqrt(14.0 / 3.0);
    for(const auto order : {Mul::SCALE_SECOND, Mul::SCALE_FIRST})
    {
        GraphSpec spec;
        spec.mul = order;
        spec.pointwiseCompute = DataType::BFLOAT16;
        const auto scale = errorScale(spec);
        ASSERT_TRUE(scale.has_value());
        // the convolution's result and the scale, in the multiply's operand order, then the
        // product
        ASSERT_EQ(scale->roundingPoints.size(), 3u);
        const bool scaleFirst = order == Mul::SCALE_FIRST;
        const auto& result = scale->roundingPoints[scaleFirst ? 1 : 0];
        const auto& scaleOperand = scale->roundingPoints[scaleFirst ? 0 : 1];
        EXPECT_NEAR(result.magnitude, SIGMA, 1e-12);
        EXPECT_DOUBLE_EQ(result.slope, 3.0);
        EXPECT_NEAR(scaleOperand.magnitude, scaleRms, 1e-12);
        EXPECT_NEAR(scaleOperand.slope, SIGMA, 1e-12);
        EXPECT_NEAR(scale->roundingPoints[2].magnitude, 3.0 * SIGMA, 1e-12);
        EXPECT_DOUBLE_EQ(scale->roundingPoints[2].slope, 1.0);
    }
}

// A tensor that two nodes read has the largest of their slopes: the convolution's result feeds a
// multiply (slope 3) and an add (slope 1), in whichever order they are listed.
TEST(TestConvFwdDynamicTolerance, TensorTwoNodesReadHasTheLargestSlopeOfThem)
{
    for(const bool branchFirst : {true, false})
    {
        GraphSpec spec;
        spec.mul = Mul::SCALE_SECOND;
        spec.convOutputType = DataType::BFLOAT16;
        spec.branch = true;
        spec.branchFirst = branchFirst;
        const auto scale = errorScale(spec);
        ASSERT_TRUE(scale.has_value()) << branchFirst;
        ASSERT_EQ(scale->roundingPoints.size(), 1u) << branchFirst;
        EXPECT_DOUBLE_EQ(scale->roundingPoints[0].slope, 3.0) << branchFirst;
    }
}

// The order in which the graph lists its nodes does not matter: each node is walked once the
// tensors it reads have a magnitude.
TEST(TestConvFwdDynamicTolerance, NodesListedAgainstTheDependencyOrderGiveTheSameScale)
{
    GraphSpec spec;
    spec.withBias = true;
    spec.withRelu = true;
    spec.pointwiseCompute = DataType::BFLOAT16;
    spec.biasType = DataType::BFLOAT16;
    spec.outputType = DataType::BFLOAT16;
    const auto forward = errorScale(spec);
    spec.reverseNodes = true;
    const auto reversed = errorScale(spec);
    ASSERT_TRUE(forward.has_value());
    ASSERT_TRUE(reversed.has_value());
    ASSERT_EQ(forward->roundingPoints.size(), 3u);
    ASSERT_EQ(reversed->roundingPoints.size(), forward->roundingPoints.size());
    for(size_t i = 0; i < forward->roundingPoints.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(reversed->roundingPoints[i].epsilon, forward->roundingPoints[i].epsilon);
        EXPECT_DOUBLE_EQ(reversed->roundingPoints[i].magnitude,
                         forward->roundingPoints[i].magnitude);
        EXPECT_DOUBLE_EQ(reversed->roundingPoints[i].slope, forward->roundingPoints[i].slope);
    }
    EXPECT_DOUBLE_EQ(reversed->outputScale, forward->outputScale);
}

// Nodes that read each other's results have no order to walk them in: the graph is outside the
// model.
TEST(TestConvFwdDynamicTolerance, NodesThatReadEachOthersResultsAreNotModeled)
{
    GraphSpec spec;
    EXPECT_TRUE(errorScale(spec).has_value());
    spec.cycle = true;
    EXPECT_FALSE(errorScale(spec).has_value());
}

// A node that computes below float and stores in a type with more bits than it computes in
// still rounds its result in the compute type, and the output of the graph is no exception: the
// rtol covers only the rounding to the stored type.
TEST(TestConvFwdDynamicTolerance, ResultOfANodeComputingBelowFloatIsRoundedEvenWhenStoredInFloat)
{
    GraphSpec spec;
    spec.withBias = true;
    spec.addCompute = DataType::HALF;
    spec.outputType = DataType::FLOAT;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    // the convolution's result (float to half) and the result of the add; the bias is half
    ASSERT_EQ(scale->roundingPoints.size(), 2u);
    for(const auto& point : scale->roundingPoints)
    {
        EXPECT_DOUBLE_EQ(point.epsilon, HALF_EPS);
    }
}

// A compute type the model does not know (an unset one included) leaves the graph outside it;
// double is as exact as float.
TEST(TestConvFwdDynamicTolerance, PointwiseComputeTypeTheModelDoesNotKnowIsNotModeled)
{
    for(const auto compute : {DataType::UNSET, DataType::INT32, DataType::UINT8})
    {
        GraphSpec spec;
        spec.mul = Mul::SCALE_SECOND;
        spec.withBias = true;
        spec.pointwiseCompute = compute;
        EXPECT_FALSE(errorScale(spec).has_value()) << EnumNameDataType(compute);
    }
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.withBias = true;
    spec.pointwiseCompute = DataType::DOUBLE;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
}

// The statistics of an operand are asked for when a multiply or a point needs them, and not
// otherwise: a graph whose bias has none is modelled while no point depends on it, and is not
// once a point after the add does.
TEST(TestConvFwdDynamicTolerance, OperandWithoutStatisticsOnlyMattersToThePointsThatNeedIt)
{
    GraphSpec floatOnly;
    floatOnly.withBias = true;
    auto in = defaultInputs(floatOnly);
    in.byUid.erase(BIAS_UID);
    EXPECT_TRUE(errorScale(floatOnly, in).has_value());

    GraphSpec pointAfterTheAdd;
    pointAfterTheAdd.withBias = true;
    pointAfterTheAdd.withRelu = true;
    pointAfterTheAdd.intermediateType = DataType::BFLOAT16;
    EXPECT_TRUE(errorScale(pointAfterTheAdd).has_value());
    EXPECT_FALSE(errorScale(pointAfterTheAdd, in).has_value());

    // A point before the add does not look at the bias.
    GraphSpec pointBeforeTheAdd;
    pointBeforeTheAdd.withBias = true;
    pointBeforeTheAdd.convOutputType = DataType::BFLOAT16;
    EXPECT_TRUE(errorScale(pointBeforeTheAdd, in).has_value());
}

// ── Slopes and magnitudes of the walk ───────────────────────────────────────

namespace
{

void expectPoint(const hipdnn_test_sdk::utilities::conv::RoundingPoint& point,
                 double epsilon,
                 double magnitude,
                 double slope)
{
    EXPECT_DOUBLE_EQ(point.epsilon, epsilon);
    EXPECT_NEAR(point.magnitude, magnitude, 1e-12 * std::max(magnitude, 1.0));
    EXPECT_NEAR(point.slope, slope, 1e-12 * std::max(slope, 1.0));
}

// The root mean square of the scale the default inputs give: -3, 1, 2.
const double SCALE_RMS = std::sqrt(14.0 / 3.0);

} // namespace

// y is read by MUL(y, s) and by the ADD that follows it, and the two results meet again: an error
// in y travels both paths and the contributions add, 3 + 1 = 4, where the largest of the readers'
// slopes would give 3. The values add the same way: 3 sigma from the multiply and sigma from the
// other path make 4 sigma at the add, not sqrt(10) sigma.
TEST(TestConvFwdDynamicTolerance, TensorReadByTwoNodesWhoseResultsMeetAgainHasTheSumOfTheirSlopes)
{
    for(const bool reverseNodes : {false, true})
    {
        GraphSpec spec;
        spec.mul = Mul::SCALE_SECOND;
        spec.wiring = Wiring::MUL_THEN_ADD_Y;
        spec.convOutputType = DataType::BFLOAT16;
        spec.intermediateType = DataType::BFLOAT16;
        spec.reverseNodes = reverseNodes;
        SCOPED_TRACE(reverseNodes ? "reversed" : "listed in order");
        const auto scale = errorScale(spec);
        ASSERT_TRUE(scale.has_value());
        // y, the multiply's result and the add's result, in the order of their uids
        ASSERT_EQ(scale->roundingPoints.size(), 3u);
        expectPoint(scale->roundingPoints[0], BFLOAT16_EPS, SIGMA, 4.0);
        expectPoint(scale->roundingPoints[1], BFLOAT16_EPS, 3.0 * SIGMA, 1.0);
        expectPoint(scale->roundingPoints[2], BFLOAT16_EPS, 4.0 * SIGMA, 1.0);
    }
}

// A node that reads a tensor in both of its operands counts it twice: ADD(y, y) doubles an error
// in y and the value, where independent operands would give sqrt(2).
TEST(TestConvFwdDynamicTolerance, NodeThatReadsATensorTwiceCountsItTwice)
{
    GraphSpec spec;
    spec.wiring = Wiring::ADD_Y_TWICE;
    spec.convOutputType = DataType::BFLOAT16;
    spec.intermediateType = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 2u);
    expectPoint(scale->roundingPoints[0], BFLOAT16_EPS, SIGMA, 2.0);
    expectPoint(scale->roundingPoints[1], BFLOAT16_EPS, 2.0 * SIGMA, 1.0);
}

// A convolution result that a multiply reads as its second operand is read all the same: stored
// in bfloat16 it is a point, and the multiply's scale is its slope.
TEST(TestConvFwdDynamicTolerance, ConvolutionResultReadOnlyAsTheSecondOperandIsRead)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_FIRST;
    spec.convOutputType = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 1u);
    expectPoint(scale->roundingPoints[0], BFLOAT16_EPS, SIGMA, 3.0);
}

// Two multiplies computing in bfloat16. An error in a scale reaches the output multiplied by the
// size of the result it scales and by the scales after it: the first multiply's scale by the
// second scale (the bias, 0.0099) times sigma, the second multiply's scale (the bias again) by
// the size of the first result, 3 sigma, as the output has slope 1.
TEST(TestConvFwdDynamicTolerance, ScalesOfSeveralMultipliesAreRoundedWithTheSlopeAfterThem)
{
    GraphSpec spec;
    spec.mul = Mul::TWO_SCALES;
    spec.pointwiseCompute = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    const double second = BIAS_RMS;
    const auto& points = scale->roundingPoints;
    ASSERT_EQ(points.size(), 6u);
    expectPoint(points[0], BFLOAT16_EPS, SIGMA, 3.0 * second); // y into the first multiply
    expectPoint(points[1], BFLOAT16_EPS, SCALE_RMS, second * SIGMA); // the first scale
    expectPoint(points[2], BFLOAT16_EPS, 3.0 * SIGMA, second); // the first result
    expectPoint(points[3], BFLOAT16_EPS, 3.0 * SIGMA, second); // that result into the second
    expectPoint(points[4], BFLOAT16_EPS, BIAS_RMS, 3.0 * SIGMA); // the second scale
    expectPoint(points[5], BFLOAT16_EPS, 3.0 * SIGMA * second, 1.0); // the second result
}

// y goes to a multiply (slope 3) and to an add that is a second output (slope 1), both computing
// in bfloat16: each node's rounding of y is the same error in the same tensor, so both points
// take the slope of the tensor, the largest of the two, not the slope of the node's own path.
TEST(TestConvFwdDynamicTolerance, EveryReaderOfATensorRoundsItWithTheSlopeOfTheTensor)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.branch = true;
    spec.pointwiseCompute = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    const auto& points = scale->roundingPoints;
    ASSERT_EQ(points.size(), 6u);
    expectPoint(points[0], BFLOAT16_EPS, SIGMA, 3.0); // y into the multiply
    expectPoint(points[1], BFLOAT16_EPS, SCALE_RMS, SIGMA); // the scale
    expectPoint(points[2], BFLOAT16_EPS, 3.0 * SIGMA, 1.0); // the multiply's result
    expectPoint(points[3], BFLOAT16_EPS, SIGMA, 3.0); // y into the add
    expectPoint(points[4], BFLOAT16_EPS, BIAS_RMS, 1.0); // the bias
    expectPoint(points[5], BFLOAT16_EPS, std::sqrt(SIGMA * SIGMA + BIAS_RMS * BIAS_RMS), 1.0);
}

// A graph input of a type the model does not know, read by a node computing below float, leaves
// the graph outside it: the node would convert and round it, and nothing says by how much.
TEST(TestConvFwdDynamicTolerance, GraphInputOfAnUnknownTypeIntoANodeComputingBelowFloatIsNotModeled)
{
    GraphSpec known;
    known.withBias = true;
    known.addCompute = DataType::BFLOAT16;
    EXPECT_TRUE(errorScale(known).has_value());
    for(const auto type : {DataType::UNSET, DataType::INT32, DataType::UINT8})
    {
        GraphSpec spec = known;
        spec.biasType = type;
        EXPECT_FALSE(errorScale(spec).has_value()) << EnumNameDataType(type);
    }
}

// A third operand is read like the other two. z = ADD(y, bias) is stored in bfloat16 and read only
// as the in_2 of BINARY_SELECT(bias, bias, z), whose result is stored in bfloat16 and read by a
// relu: z is a point (it is read, though only as in_2) at sqrt(sigma^2 + bias^2), and the result
// one at sqrt(z^2 + 2 bias^2), in whichever order the nodes are listed.
TEST(TestConvFwdDynamicTolerance, ThirdOperandIsReadForConsumptionAndMagnitude)
{
    for(const bool reverseNodes : {false, true})
    {
        GraphSpec spec;
        spec.wiring = Wiring::SELECT_IN2_PRODUCED;
        spec.intermediateType = DataType::BFLOAT16;
        spec.reverseNodes = reverseNodes;
        SCOPED_TRACE(reverseNodes ? "reversed" : "listed in order");
        const auto scale = errorScale(spec);
        ASSERT_TRUE(scale.has_value());
        // the select's result, then z, in the order of their uids
        ASSERT_EQ(scale->roundingPoints.size(), 2u);
        expectPoint(scale->roundingPoints[0],
                    BFLOAT16_EPS,
                    std::sqrt(SIGMA * SIGMA + 3.0 * BIAS_RMS * BIAS_RMS),
                    1.0);
        expectPoint(scale->roundingPoints[1],
                    BFLOAT16_EPS,
                    std::sqrt(SIGMA * SIGMA + BIAS_RMS * BIAS_RMS),
                    1.0);
    }
}

// The third operand of a node computing below float is converted and rounded like the others when
// it is a graph input stored in another type: BINARY_SELECT(z, bias, bias) in bfloat16 over a
// half bias has a point at each of the two bias operands.
TEST(TestConvFwdDynamicTolerance, ThirdOperandThatIsAGraphInputIsRoundedLikeTheOthers)
{
    GraphSpec spec;
    spec.wiring = Wiring::SELECT_IN2_INPUT;
    spec.pointwiseCompute = DataType::BFLOAT16;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    const double z = std::sqrt(SIGMA * SIGMA + BIAS_RMS * BIAS_RMS);
    const auto& points = scale->roundingPoints;
    // the add: y, bias, its result; the select: z, bias, bias (in_2), its result
    ASSERT_EQ(points.size(), 7u);
    expectPoint(points[0], BFLOAT16_EPS, SIGMA, 1.0);
    expectPoint(points[1], BFLOAT16_EPS, BIAS_RMS, 1.0);
    expectPoint(points[2], BFLOAT16_EPS, z, 1.0);
    expectPoint(points[3], BFLOAT16_EPS, z, 1.0);
    expectPoint(points[4], BFLOAT16_EPS, BIAS_RMS, 1.0);
    expectPoint(points[5], BFLOAT16_EPS, BIAS_RMS, 1.0);
    expectPoint(points[6], BFLOAT16_EPS, std::sqrt(SIGMA * SIGMA + 3.0 * BIAS_RMS * BIAS_RMS), 1.0);
}

// The statistics of an input are looked up when a multiply or a rounding point needs them, and
// not before: a float graph with a bias never asks for the bias (a golden blob would be scanned
// on the host for nothing), a point after the add does, and a point before it does not.
TEST(TestConvFwdDynamicTolerance, InputsAreLookedUpOnlyWhenSomethingNeedsThem)
{
    const auto lookedUp = [](const GraphSpec& spec) {
        std::set<int64_t> asked;
        const auto inputs = defaultInputs(spec);
        const auto lookup = [&](int64_t uid) {
            asked.insert(uid);
            return inputs.lookup()(uid);
        };
        flatbuffers::FlatBufferBuilder b;
        EXPECT_TRUE(tol::convErrorScale(buildGraph(b, spec), lookup).has_value());
        return asked;
    };
    const std::set<int64_t> convolutionInputs = {X_UID, W_UID};

    GraphSpec floatOnly;
    floatOnly.withBias = true;
    floatOnly.withRelu = true;
    EXPECT_EQ(lookedUp(floatOnly), convolutionInputs);

    GraphSpec pointAfterTheAdd = floatOnly;
    pointAfterTheAdd.intermediateType = DataType::BFLOAT16;
    EXPECT_EQ(lookedUp(pointAfterTheAdd), (std::set<int64_t>{X_UID, W_UID, BIAS_UID}));

    GraphSpec pointBeforeTheAdd;
    pointBeforeTheAdd.withBias = true;
    pointBeforeTheAdd.convOutputType = DataType::BFLOAT16;
    EXPECT_EQ(lookedUp(pointBeforeTheAdd), convolutionInputs);

    GraphSpec multiply;
    multiply.mul = Mul::SCALE_SECOND;
    EXPECT_EQ(lookedUp(multiply), (std::set<int64_t>{X_UID, W_UID, SCALE_UID}));
}

// ── Fused pointwise nodes ───────────────────────────────────────────────────

// The scale cycles through -3, 1, 2: its largest magnitude (3) differs from its root mean
// square (2.2) and from its mean (0), so only the largest magnitude passes.
TEST(TestConvFwdDynamicTolerance, FusedMultiplyScalesTheOutputByItsLargestMagnitude)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 3.0);
}

// MUL is commutative: the scale may be the first operand.
TEST(TestConvFwdDynamicTolerance, FusedMultiplyWithTheScaleFirstScalesToo)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_FIRST;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 3.0);
}

// The conv result may be a graph output as well (not virtual) and still is the convolution's
// operand of the multiply, not a scale.
TEST(TestConvFwdDynamicTolerance, FusedMultiplyOfANonVirtualConvolutionResultStillScales)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.convOutputVirtual = false;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 3.0);
}

// The conv-fwd test-plan Variant3 half and bfloat16 graphs bake the scale into the graph. The lookup does
// not know that tensor (the harness holds no recipe for it), so the value must come from the
// graph.
TEST(TestConvFwdDynamicTolerance, BakedScaleIsReadFromTheGraphNotTheLookup)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.bakedScale = 0.5f;
    auto in = defaultInputs(spec);
    in.byUid.erase(SCALE_UID);
    const auto scale = errorScale(spec, in);
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 0.5);
}

TEST(TestConvFwdDynamicTolerance, BakedScaleAboveOneWidensAndANegativeOneCountsByMagnitude)
{
    for(const float baked : {4.0f, -2.0f})
    {
        GraphSpec spec;
        spec.mul = Mul::SCALE_SECOND;
        spec.bakedScale = baked;
        const auto scale = errorScale(spec);
        ASSERT_TRUE(scale.has_value());
        EXPECT_DOUBLE_EQ(scale->outputScale, std::abs(static_cast<double>(baked)));
    }
}

TEST(TestConvFwdDynamicTolerance, BakedScaleThatIsNotAFiniteNumberIsNotModeled)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.bakedScale = std::numeric_limits<float>::quiet_NaN();
    auto in = defaultInputs(spec);
    EXPECT_FALSE(errorScale(spec, in).has_value());
}

// A baked value stored as a float32 constant in a half tensor does not describe that tensor.
TEST(TestConvFwdDynamicTolerance, BakedScaleOfTheWrongTypeIsNotModeled)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    spec.bakedScale = 0.5f;
    spec.bakeScaleAsFloat32 = true;
    EXPECT_FALSE(errorScale(spec).has_value());
}

// A multiply whose scale the lookup cannot describe leaves the graph outside the model.
TEST(TestConvFwdDynamicTolerance, FusedMultiplyWithAnUnknownScaleIsNotModeled)
{
    GraphSpec spec;
    spec.mul = Mul::SCALE_SECOND;
    auto in = defaultInputs(spec);
    in.byUid.erase(SCALE_UID);
    EXPECT_FALSE(errorScale(spec, in).has_value());
}

// Squaring the result, or multiplying two unrelated tensors, is not a scale of the output.
TEST(TestConvFwdDynamicTolerance, MultiplyThatIsNotAScaleIsNotModeled)
{
    for(const Mul mul : {Mul::SQUARE, Mul::UNRELATED})
    {
        GraphSpec spec;
        spec.mul = mul;
        EXPECT_FALSE(errorScale(spec).has_value());
    }
}

// A bias add takes a graph input as its second operand too, but is no scale: the bias (0.01
// at most) must not shrink the tolerance, and the bias add after a multiply keeps the multiply's.
TEST(TestConvFwdDynamicTolerance, BiasAddLeavesTheOutputScaleAlone)
{
    GraphSpec spec;
    spec.withBias = true;
    const auto plain = errorScale(spec);
    ASSERT_TRUE(plain.has_value());
    EXPECT_DOUBLE_EQ(plain->outputScale, 1.0);

    GraphSpec scaled;
    scaled.mul = Mul::SCALE_SECOND;
    scaled.withBias = true;
    const auto both = errorScale(scaled);
    ASSERT_TRUE(both.has_value());
    EXPECT_DOUBLE_EQ(both->outputScale, 3.0);
}

// Scales multiply: two fused multiplies, by a tensor of magnitude 3 and by one of 0.01, scale the
// output by the product of their largest magnitudes.
TEST(TestConvFwdDynamicTolerance, SeveralFusedMultipliesScaleByTheProduct)
{
    GraphSpec spec;
    spec.mul = Mul::TWO_SCALES;
    const auto scale = errorScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_NEAR(scale->outputScale, 3.0 * static_cast<double>(static_cast<half>(0.01f)), 1e-12);
}

// ── Output dispatch ─────────────────────────────────────────────────────────

// The three types differ in atol (half carries the lambda term, bfloat16 and float do not) and in
// rtol (2^-9, 2^-6 and, at K = 144 and c = 4, 13 * 2^-22: 2^-22 and the summation-order term), so
// a swap of the dispatch shows in the values.
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
    const auto expectedFloat
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropOutputTolerance<float>(scale);
    ASSERT_NE(expectedHalf.atol, expectedBfloat16.atol);
    ASSERT_NE(expectedHalf.rtol, expectedBfloat16.rtol);
    ASSERT_NE(expectedFloat.rtol, expectedBfloat16.rtol);
    ASSERT_NE(expectedFloat.atol, expectedBfloat16.atol);

    const auto halfTolerance = tol::convOutputTolerance(scale, DataType::HALF);
    ASSERT_TRUE(halfTolerance.has_value());
    EXPECT_DOUBLE_EQ(halfTolerance->atol, expectedHalf.atol);
    EXPECT_DOUBLE_EQ(halfTolerance->rtol, expectedHalf.rtol);

    const auto bfloat16Tolerance = tol::convOutputTolerance(scale, DataType::BFLOAT16);
    ASSERT_TRUE(bfloat16Tolerance.has_value());
    EXPECT_DOUBLE_EQ(bfloat16Tolerance->atol, expectedBfloat16.atol);
    EXPECT_DOUBLE_EQ(bfloat16Tolerance->rtol, expectedBfloat16.rtol);

    const auto floatTolerance = tol::convOutputTolerance(scale, DataType::FLOAT);
    ASSERT_TRUE(floatTolerance.has_value());
    EXPECT_DOUBLE_EQ(floatTolerance->atol, expectedFloat.atol);
    EXPECT_DOUBLE_EQ(floatTolerance->rtol, expectedFloat.rtol);
}

// The coefficients an engine sets reach the tolerance of the type they are passed with.
TEST(TestConvFwdDynamicTolerance, EngineCoefficientsReachTheToleranceOfTheirOutputType)
{
    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = 144;
    scale.inputMeanSquare = 1.0 / 3.0;
    scale.weightMeanSquare = 1.0 / 3.0;
    hipdnn_test_sdk::utilities::conv::ConvFpropCoefficients coefficients;
    coefficients.summationOrder = 11.0;

    const auto withDefault = tol::convOutputTolerance(scale, DataType::FLOAT);
    const auto withEngine = tol::convOutputTolerance(scale, DataType::FLOAT, coefficients);
    const auto expected
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropOutputTolerance<float>(scale,
                                                                                     coefficients);
    ASSERT_TRUE(withDefault.has_value());
    ASSERT_TRUE(withEngine.has_value());
    EXPECT_DOUBLE_EQ(withEngine->atol, expected.atol);
    EXPECT_DOUBLE_EQ(withEngine->rtol, expected.rtol);
    EXPECT_GT(withEngine->atol, withDefault->atol);
}

TEST(TestConvFwdDynamicTolerance, OnlyHalfBfloat16AndFloatOutputsGetADynamicTolerance)
{
    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = 144;
    scale.inputMeanSquare = 1.0 / 3.0;
    scale.weightMeanSquare = 1.0 / 3.0;
    EXPECT_FALSE(tol::convOutputTolerance(scale, DataType::DOUBLE).has_value());
    EXPECT_FALSE(tol::convOutputTolerance(scale, DataType::INT32).has_value());
}

// The harness asks hasDynamicToleranceFor() before it scans any input, and convOutputTolerance()
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
                  tol::convOutputTolerance(scale, type).has_value())
            << EnumNameDataType(type);
    }
    EXPECT_TRUE(tol::hasDynamicToleranceFor(DataType::HALF));
    EXPECT_TRUE(tol::hasDynamicToleranceFor(DataType::BFLOAT16));
    EXPECT_TRUE(tol::hasDynamicToleranceFor(DataType::FLOAT));
    EXPECT_FALSE(tol::hasDynamicToleranceFor(DataType::DOUBLE));
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

// A value list repeats over the elements, so its moments are the list's own: 3, -4 has mean
// square 12.5 and largest magnitude 4, not the default lo/hi of a free draw.
TEST(TestConvFwdDynamicTolerance, RecipeMomentsOfAValueList)
{
    const auto ints = tol::recipeMoments(
        hipdnn_integration_tests::FillRecipe::values(std::vector<int64_t>{3, -4}));
    EXPECT_DOUBLE_EQ(ints.meanSquare, 12.5);
    EXPECT_DOUBLE_EQ(ints.maxAbs, 4.0);

    const auto doubles = tol::recipeMoments(
        hipdnn_integration_tests::FillRecipe::values(std::vector<double>{0.5, -1.5}));
    EXPECT_DOUBLE_EQ(doubles.meanSquare, 1.25);
    EXPECT_DOUBLE_EQ(doubles.maxAbs, 1.5);
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
    const GraphSpec spec;
    auto in = defaultInputs(spec);
    in.byUid[X_UID]
        = filledWith<half>({2, 16, 8, 8}, {1.0f, std::numeric_limits<float>::quiet_NaN()});
    EXPECT_FALSE(errorScale(spec, in).has_value());
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
            EXPECT_FALSE(tol::convErrorScale(buildGraph(b, spec), lookup).has_value())
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
