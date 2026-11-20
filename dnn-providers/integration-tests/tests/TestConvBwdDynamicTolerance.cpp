// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
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

// The backward-data (dgrad) and backward-weights (wrw) graphs of the convolution dynamic
// tolerance. The graph rules every direction shares (intermediates, multiplies, input
// statistics) are covered on forward graphs in TestConvFwdDynamicTolerance.cpp.

using namespace hipdnn_flatbuffers_sdk::data_objects;
namespace tol = hipdnn_integration_tests::tolerance;
namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace sdk = hipdnn_test_sdk::utilities::conv;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

constexpr int64_t X_UID = 0; ///< x, or dx for dgrad
constexpr int64_t DY_UID = 1;
constexpr int64_t W_UID = 2; ///< w, or dw for wrw
constexpr int64_t OUT_UID = 3; ///< the result of the fused activation
constexpr int64_t ACT_X_UID = 4; ///< the forward input the backward activation reads
constexpr int64_t Y_UID = 5; ///< the result of a forward convolution added next to the dgrad

enum class Direction
{
    DGRAD,
    WRW,
};

struct GraphSpec
{
    Direction direction = Direction::DGRAD;
    std::vector<int64_t> xDims = {2, 16, 8, 8};
    std::vector<int64_t> dyDims = {2, 32, 6, 6};
    std::vector<int64_t> wDims = {32, 8, 3, 3}; ///< 2 groups of 8 input channels
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    bool fusedBackwardActivation = false; ///< backwardMode(conv result, act_x) after the conv
    DataType resultType = DataType::FLOAT; ///< the dgrad result's type when the activation reads it
    bool dyWrittenByANode = false; ///< dy is RELU_FWD of another input: not a graph input
    bool addForwardConvolution = false; ///< a second, forward, convolution in the same graph
    PointwiseMode backwardMode = PointwiseMode::RELU_BWD; ///< the mode of that activation
    DataType activationCompute = DataType::FLOAT; ///< the compute type of that activation
    flatbuffers::Optional<float> reluSlope; ///< its lower-clip slope
    flatbuffers::Optional<float> eluAlpha; ///< its alpha
    std::optional<int64_t> graded; ///< the graded output, when not the default (gradedOutput)
    bool activationOperandsSwapped = false; ///< the conv result as the forward input (in_1)
    DataType outType = DataType::HALF; ///< the stored type of the activation's result
};

fb::GraphWrapper buildGraph(flatbuffers::FlatBufferBuilder& b, const GraphSpec& spec)
{
    b.Clear();
    const std::vector<int64_t> ones(spec.xDims.size(), 1);
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const auto addTensor = [&](int64_t uid,
                               const char* name,
                               const std::vector<int64_t>& dims,
                               bool isVirtual,
                               DataType type = DataType::HALF) {
        tensors.push_back(
            CreateTensorAttributesDirect(b, uid, name, type, &ones, &dims, isVirtual));
    };
    const int64_t resultUid = spec.direction == Direction::DGRAD ? X_UID : W_UID;
    // The convolution's result is a virtual intermediate when the activation reads it.
    const bool dgradResultIsRead = spec.fusedBackwardActivation && resultUid == X_UID;
    const bool wrwResultIsRead = spec.fusedBackwardActivation && resultUid == W_UID;
    addTensor(X_UID,
              "x",
              spec.xDims,
              dgradResultIsRead,
              dgradResultIsRead ? spec.resultType : DataType::HALF);
    addTensor(DY_UID,
              "dy",
              spec.dyDims,
              spec.dyWrittenByANode,
              spec.dyWrittenByANode ? DataType::FLOAT : DataType::HALF);
    addTensor(W_UID,
              "w",
              spec.wDims,
              wrwResultIsRead,
              wrwResultIsRead ? spec.resultType : DataType::HALF);
    addTensor(OUT_UID, "out", spec.xDims, false, spec.outType);
    addTensor(ACT_X_UID, "act_x", spec.xDims, false);
    addTensor(Y_UID, "y", spec.dyDims, false);

    std::vector<flatbuffers::Offset<Node>> nodes;
    const std::vector<int64_t> zeros(spec.stride.size(), 0);
    if(spec.dyWrittenByANode)
    {
        const auto relu = CreatePointwiseAttributes(b,
                                                    PointwiseMode::RELU_FWD,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    ACT_X_UID,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    DY_UID);
        nodes.push_back(CreateNodeDirect(
            b, "relu", DataType::FLOAT, NodeAttributes::PointwiseAttributes, relu.Union()));
    }
    if(spec.direction == Direction::DGRAD)
    {
        const auto conv = CreateConvolutionBwdAttributesDirect(b,
                                                               DY_UID,
                                                               W_UID,
                                                               X_UID,
                                                               &zeros,
                                                               &zeros,
                                                               &spec.stride,
                                                               &spec.dilation,
                                                               ConvMode::CROSS_CORRELATION);
        nodes.push_back(CreateNodeDirect(
            b, "dgrad", DataType::FLOAT, NodeAttributes::ConvolutionBwdAttributes, conv.Union()));
    }
    else
    {
        const auto conv = CreateConvolutionWrwAttributesDirect(b,
                                                               X_UID,
                                                               DY_UID,
                                                               W_UID,
                                                               &zeros,
                                                               &zeros,
                                                               &spec.stride,
                                                               &spec.dilation,
                                                               ConvMode::CROSS_CORRELATION);
        nodes.push_back(CreateNodeDirect(
            b, "wrw", DataType::FLOAT, NodeAttributes::ConvolutionWrwAttributes, conv.Union()));
    }
    if(spec.fusedBackwardActivation)
    {
        const auto activation
            = CreatePointwiseAttributes(b,
                                        spec.backwardMode,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        spec.reluSlope,
                                        flatbuffers::nullopt,
                                        spec.activationOperandsSwapped ? ACT_X_UID : resultUid,
                                        spec.activationOperandsSwapped ? resultUid : ACT_X_UID,
                                        flatbuffers::nullopt,
                                        OUT_UID,
                                        flatbuffers::nullopt,
                                        spec.eluAlpha);
        nodes.push_back(CreateNodeDirect(b,
                                         "activation_bwd",
                                         spec.activationCompute,
                                         NodeAttributes::PointwiseAttributes,
                                         activation.Union()));
    }
    if(spec.addForwardConvolution)
    {
        const auto conv = CreateConvolutionFwdAttributesDirect(b,
                                                               ACT_X_UID,
                                                               W_UID,
                                                               Y_UID,
                                                               &zeros,
                                                               &zeros,
                                                               &spec.stride,
                                                               &spec.dilation,
                                                               ConvMode::CROSS_CORRELATION);
        nodes.push_back(CreateNodeDirect(
            b, "fwd", DataType::FLOAT, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));
    }

    const auto graph = CreateGraphDirect(
        b, "conv_bwd", DataType::FLOAT, DataType::HALF, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    return fb::GraphWrapper(b.GetBufferPointer(), b.GetSize());
}

std::shared_ptr<ITensor> filled(const std::vector<int64_t>& dims, float value)
{
    auto t = std::make_shared<Tensor<half>>(dims);
    t->fillWithValue(static_cast<half>(value));
    return t;
}

// The graded output: the one the spec names, else the activation's result when there is one,
// else the convolution's.
int64_t gradedOutput(const GraphSpec& spec)
{
    if(spec.graded.has_value())
    {
        return *spec.graded;
    }
    if(spec.fusedBackwardActivation)
    {
        return OUT_UID;
    }
    return spec.direction == Direction::DGRAD ? X_UID : W_UID;
}

// x is 2 everywhere, dy 3 and w 0.5: mean squares 4, 9 and 0.25. The tensors take the default
// shapes whatever the graph says: only their values count, and a tensor cannot have a zero
// dimension.
std::optional<tol::ConvErrorScale> errorScale(const GraphSpec& spec)
{
    const GraphSpec shapes;
    std::unordered_map<int64_t, std::shared_ptr<ITensor>> byUid;
    byUid[X_UID] = filled(shapes.xDims, 2.0f);
    byUid[DY_UID] = filled(shapes.dyDims, 3.0f);
    byUid[W_UID] = filled(shapes.wDims, 0.5f);
    byUid[ACT_X_UID] = filled(shapes.xDims, 1.0f);
    const auto lookup = [&byUid](int64_t uid) -> std::optional<tol::InputMoments> {
        const auto it = byUid.find(uid);
        if(it == byUid.end())
        {
            return std::nullopt;
        }
        return tol::hostMoments(*it->second, DataType::HALF);
    };
    flatbuffers::FlatBufferBuilder b;
    return tol::convErrorScale(buildGraph(b, spec), lookup, gradedOutput(spec));
}

std::optional<sdk::ConvDgradErrorScale> dgradScale(const GraphSpec& spec)
{
    const auto scale = errorScale(spec);
    if(!scale.has_value())
    {
        return std::nullopt;
    }
    EXPECT_TRUE(std::holds_alternative<sdk::ConvDgradErrorScale>(*scale));
    return std::get<sdk::ConvDgradErrorScale>(*scale);
}

std::optional<sdk::ConvWrwErrorScale> wrwScale(const GraphSpec& spec)
{
    const auto scale = errorScale(spec);
    if(!scale.has_value())
    {
        return std::nullopt;
    }
    EXPECT_TRUE(std::holds_alternative<sdk::ConvWrwErrorScale>(*scale));
    return std::get<sdk::ConvWrwErrorScale>(*scale);
}

// Whether the graph is modeled when its two convolution operands (dy and w for dgrad, x and dy
// for wrw) are filled from these recipes and every other input from a free one.
bool modeledWithRecipes(const GraphSpec& spec,
                        const hipdnn_integration_tests::FillRecipe& first,
                        const hipdnn_integration_tests::FillRecipe& second)
{
    const bool dgrad = spec.direction == Direction::DGRAD;
    const int64_t firstUid = dgrad ? DY_UID : X_UID;
    const int64_t secondUid = dgrad ? W_UID : DY_UID;
    const auto lookup = [&](int64_t uid) -> std::optional<tol::InputMoments> {
        if(uid == firstUid)
        {
            return tol::recipeMoments(first);
        }
        if(uid == secondUid)
        {
            return tol::recipeMoments(second);
        }
        return tol::recipeMoments(hipdnn_integration_tests::FillRecipe::free(-1.0f, 1.0f));
    };
    flatbuffers::FlatBufferBuilder b;
    return tol::convErrorScale(buildGraph(b, spec), lookup, gradedOutput(spec)).has_value();
}

// One FIXED operand still changes sign with the other; two sum products that round alike.
void expectFixedOnBothOperandsToFallBack(Direction direction)
{
    using hipdnn_integration_tests::FillRecipe;
    GraphSpec spec;
    spec.direction = direction;
    const auto free = FillRecipe::free(-1.0f, 1.0f);
    const auto fixed = FillRecipe::fixed(0.5f);
    const auto list = FillRecipe::values(std::vector<int64_t>{3, -4});
    EXPECT_TRUE(modeledWithRecipes(spec, free, free));
    EXPECT_TRUE(modeledWithRecipes(spec, fixed, free));
    EXPECT_TRUE(modeledWithRecipes(spec, free, fixed));
    EXPECT_TRUE(modeledWithRecipes(spec, fixed, list));
    EXPECT_FALSE(modeledWithRecipes(spec, fixed, fixed));
}

// The backward activations with a hipDNN reference are priced behind a dgrad; the other five have
// none yet and no output of them was calibrated, so they keep the per-op default.
const std::vector<PointwiseMode> PRICED_BACKWARD_ACTIVATIONS
    = {PointwiseMode::RELU_BWD, PointwiseMode::SIGMOID_BWD, PointwiseMode::TANH_BWD};

const std::vector<PointwiseMode> UNPRICED_BACKWARD_ACTIVATIONS
    = {PointwiseMode::SOFTPLUS_BWD,
       PointwiseMode::ELU_BWD,
       PointwiseMode::SWISH_BWD,
       PointwiseMode::GELU_BWD,
       PointwiseMode::GELU_APPROX_TANH_BWD};

} // namespace

// ── Backward data ───────────────────────────────────────────────────────────

// dx has 16 channels and w 8 per group, so 2 groups of 16 output channels. With stride 2 along
// the first dimension only every other one of its 3 taps reaches a dx element: at most 2.
TEST(TestConvBwdDynamicTolerance, DgradCountsTheOutputChannelsOfAGroupTimesTheTapsThatReach)
{
    GraphSpec spec;
    spec.stride = {2, 1};
    const auto scale = dgradScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_EQ(scale->accumulationLength, 16u * 2u * 3u);
}

// A dilation that shares the stride's factor walks the taps onto every residue: all 3 reach.
TEST(TestConvBwdDynamicTolerance, DgradStrideSharingTheDilationsFactorReachesEveryTap)
{
    GraphSpec spec;
    spec.stride = {2, 2};
    spec.dilation = {2, 2};
    const auto scale = dgradScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_EQ(scale->accumulationLength, 16u * 3u * 3u);
}

TEST(TestConvBwdDynamicTolerance, DgradStatisticsComeFromDyAndTheWeights)
{
    const auto scale = dgradScale(GraphSpec{});
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->dyMeanSquare, 9.0);
    EXPECT_DOUBLE_EQ(scale->weightMeanSquare, 0.25);
    EXPECT_DOUBLE_EQ(scale->dyMean, 3.0);
    EXPECT_DOUBLE_EQ(scale->weightMean, 0.5);
    EXPECT_DOUBLE_EQ(scale->outputScale, 1.0);
}

TEST(TestConvBwdDynamicTolerance, DgradWithDxChannelsNotAMultipleOfTheWeightsIsNotModeled)
{
    GraphSpec spec;
    spec.xDims = {2, 12, 8, 8};
    EXPECT_FALSE(errorScale(spec).has_value());
}

TEST(TestConvBwdDynamicTolerance, DgradWithGroupsThatDoNotDivideTheOutputChannelsIsNotModeled)
{
    GraphSpec spec;
    spec.xDims = {2, 24, 8, 8}; // 3 groups of 8 input channels, but 32 output channels
    EXPECT_FALSE(errorScale(spec).has_value());
}

TEST(TestConvBwdDynamicTolerance, DgradWithAStrideOfTheWrongRankIsNotModeled)
{
    GraphSpec spec;
    spec.stride = {1, 1, 1};
    EXPECT_FALSE(errorScale(spec).has_value());
}

TEST(TestConvBwdDynamicTolerance, DgradWithAZeroStrideIsNotModeled)
{
    GraphSpec spec;
    spec.stride = {0, 1};
    EXPECT_FALSE(errorScale(spec).has_value());
}

// A backward activation reads the dgrad result and the forward input; its slope is at most 1,
// so it leaves the scale alone. The result it reads is float: no rounding point.
TEST(TestConvBwdDynamicTolerance, DgradFollowedByABackwardActivationIsModeled)
{
    GraphSpec spec;
    spec.fusedBackwardActivation = true;
    const auto scale = dgradScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->outputScale, 1.0);
    EXPECT_TRUE(scale->roundingPoints.empty());
}

// The dgrad result stored in half before the activation reads it is rounded there: a rounding
// point at the size of that result. dy is 3 and w 0.5 everywhere, so with K = 16 * 3 * 3 = 144
// products of 1.5 the result is 216, its root mean square, and relu passes an error on with
// slope 1.
TEST(TestConvBwdDynamicTolerance, DgradResultStoredInHalfAndReadByAnActivationIsARoundingPoint)
{
    GraphSpec spec;
    spec.fusedBackwardActivation = true;
    spec.resultType = DataType::HALF;
    const auto scale = dgradScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 1u);
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].epsilon, std::ldexp(1.0, -10));
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].magnitude, 216.0);
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].slope, 1.0);
}

// Catches a priced backward mode dropped from the priced set: each has a derivative of at most 1.
TEST(TestConvBwdDynamicTolerance, DgradFollowedByEachPricedBackwardActivationIsModeled)
{
    for(const auto mode : PRICED_BACKWARD_ACTIVATIONS)
    {
        GraphSpec spec;
        spec.fusedBackwardActivation = true;
        spec.backwardMode = mode;
        const auto scale = dgradScale(spec);
        ASSERT_TRUE(scale.has_value()) << EnumNamePointwiseMode(mode);
        EXPECT_DOUBLE_EQ(scale->outputScale, 1.0) << EnumNamePointwiseMode(mode);
        EXPECT_TRUE(scale->roundingPoints.empty()) << EnumNamePointwiseMode(mode);
    }
}

// Catches a backward activation without a hipDNN reference being priced again, behind a dgrad or
// a wrw: its output keeps the per-op default.
TEST(TestConvBwdDynamicTolerance, BackwardActivationWithoutAReferenceIsNotModeled)
{
    for(const auto direction : {Direction::DGRAD, Direction::WRW})
    {
        for(const auto mode : UNPRICED_BACKWARD_ACTIVATIONS)
        {
            GraphSpec spec;
            spec.direction = direction;
            spec.fusedBackwardActivation = true;
            spec.backwardMode = mode;
            EXPECT_FALSE(errorScale(spec).has_value()) << EnumNamePointwiseMode(mode);
        }
    }
}

// Catches a leaky relu slope above 1 being priced as a slope of 1 behind a dgrad, and an elu
// backward being priced at any alpha: it has no reference, so its alpha never makes it modeled.
TEST(TestConvBwdDynamicTolerance, BackwardActivationThatSteepensTheGradientIsNotModeled)
{
    for(const auto& [slope, modeled] : {std::pair{0.01f, true},
                                        std::pair{1.0f, true},
                                        std::pair{-1.0f, true},
                                        std::pair{2.0f, false},
                                        std::pair{-2.0f, false}})
    {
        GraphSpec spec;
        spec.fusedBackwardActivation = true;
        spec.reluSlope = slope;
        EXPECT_EQ(errorScale(spec).has_value(), modeled) << "slope " << slope;
    }
    for(const float alpha : {0.5f, 1.0f, 2.0f, -2.0f})
    {
        GraphSpec spec;
        spec.fusedBackwardActivation = true;
        spec.backwardMode = PointwiseMode::ELU_BWD;
        spec.eluAlpha = alpha;
        EXPECT_FALSE(errorScale(spec).has_value()) << "alpha " << alpha;
    }
}

// Catches a backward activation priced as a slope of the convolution's result when that result
// is its forward input (in_1): the output is then g * f'(dx), not a gradient scaled by f'.
TEST(TestConvBwdDynamicTolerance,
     BackwardActivationReadingTheDgradResultAsItsForwardInputIsNotModeled)
{
    GraphSpec spec;
    spec.fusedBackwardActivation = true;
    spec.activationOperandsSwapped = true;
    EXPECT_FALSE(errorScale(spec).has_value());
}

// Catches the forward input of a backward activation entering the magnitude of its result. The
// activation computes in half and stores its result as float, so the result is a rounding point
// at its own size: that of the dgrad result, 216, with act_x (mean square 1) adding nothing; the
// float dgrad result converted to half is a second point at the same size.
TEST(TestConvBwdDynamicTolerance, ForwardInputDoesNotEnterTheMagnitudeOfABackwardActivation)
{
    GraphSpec spec;
    spec.fusedBackwardActivation = true;
    spec.activationCompute = DataType::HALF;
    spec.outType = DataType::FLOAT;
    const auto scale = dgradScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 2u);
    for(const auto& point : scale->roundingPoints)
    {
        EXPECT_DOUBLE_EQ(point.magnitude, 216.0);
    }
}

// Catches a half-compute activation not rounding the float dgrad result it reads (at 216).
TEST(TestConvBwdDynamicTolerance, BackwardActivationComputingInHalfRoundsTheDgradResult)
{
    GraphSpec spec;
    spec.fusedBackwardActivation = true;
    spec.activationCompute = DataType::HALF;
    const auto scale = dgradScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 1u);
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].epsilon, std::ldexp(1.0, -10));
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].magnitude, 216.0);
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].slope, 1.0);
}

// Catches the graded output being ignored: the result graded itself has no point of its own.
TEST(TestConvBwdDynamicTolerance, DgradResultGradedDirectlyHasTheScaleOfTheActivationButNoPoint)
{
    GraphSpec through;
    through.fusedBackwardActivation = true;
    through.resultType = DataType::HALF;
    GraphSpec direct = through;
    direct.graded = X_UID;
    const auto viaActivation = dgradScale(through);
    const auto asResult = dgradScale(direct);
    ASSERT_TRUE(viaActivation.has_value());
    ASSERT_TRUE(asResult.has_value());
    EXPECT_EQ(asResult->accumulationLength, viaActivation->accumulationLength);
    EXPECT_DOUBLE_EQ(asResult->outputScale, viaActivation->outputScale);
    EXPECT_EQ(viaActivation->roundingPoints.size(), 1u);
    EXPECT_TRUE(asResult->roundingPoints.empty());
}

// Catches an output that does not depend on the convolution (an input, or one no node writes).
TEST(TestConvBwdDynamicTolerance, GradedOutputThatDoesNotDependOnTheConvolutionIsNotModeled)
{
    GraphSpec input;
    input.fusedBackwardActivation = true;
    input.graded = ACT_X_UID;
    EXPECT_FALSE(errorScale(input).has_value());

    GraphSpec unwritten;
    unwritten.graded = OUT_UID;
    EXPECT_FALSE(errorScale(unwritten).has_value());
}

// Catches the FIXED-on-both fallback being lost on dgrad (dy and w).
TEST(TestConvBwdDynamicTolerance, DgradWithFixedFillOfDyAndWeightsIsNotModeled)
{
    expectFixedOnBothOperandsToFallBack(Direction::DGRAD);
}

TEST(TestConvBwdDynamicTolerance, DyWrittenByAnotherNodeIsNotModeled)
{
    GraphSpec spec;
    spec.dyWrittenByANode = true;
    EXPECT_FALSE(errorScale(spec).has_value());
}

TEST(TestConvBwdDynamicTolerance, ForwardAndBackwardConvolutionsInOneGraphAreNotModeled)
{
    GraphSpec spec;
    spec.addForwardConvolution = true;
    EXPECT_FALSE(errorScale(spec).has_value());
}

// ── Backward weights ────────────────────────────────────────────────────────

// Every dw element sums over the batch and every dy position: 2 * 6 * 6.
TEST(TestConvBwdDynamicTolerance, WrwCountsTheBatchTimesTheDyVolume)
{
    GraphSpec spec;
    spec.direction = Direction::WRW;
    const auto scale = wrwScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_EQ(scale->accumulationLength, 72u);
}

TEST(TestConvBwdDynamicTolerance, WrwStatisticsComeFromTheInputAndDy)
{
    GraphSpec spec;
    spec.direction = Direction::WRW;
    const auto scale = wrwScale(spec);
    ASSERT_TRUE(scale.has_value());
    EXPECT_DOUBLE_EQ(scale->inputMeanSquare, 4.0);
    EXPECT_DOUBLE_EQ(scale->dyMeanSquare, 9.0);
    EXPECT_DOUBLE_EQ(scale->inputMean, 2.0);
    EXPECT_DOUBLE_EQ(scale->dyMean, 3.0);
}

TEST(TestConvBwdDynamicTolerance, WrwWithAnEmptyBatchIsNotModeled)
{
    GraphSpec spec;
    spec.direction = Direction::WRW;
    spec.dyDims = {0, 32, 6, 6};
    EXPECT_FALSE(errorScale(spec).has_value());
}

// Catches the FIXED-on-both fallback being lost on wrw (x and dy).
TEST(TestConvBwdDynamicTolerance, WrwWithFixedFillOfTheInputAndDyIsNotModeled)
{
    expectFixedOnBothOperandsToFallBack(Direction::WRW);
}

// Catches a backward activation priced behind a wrw; its control, the same graph behind a dgrad.
TEST(TestConvBwdDynamicTolerance, BackwardActivationBehindWrwIsNotModeled)
{
    for(const auto mode : PRICED_BACKWARD_ACTIVATIONS)
    {
        GraphSpec spec;
        spec.fusedBackwardActivation = true;
        spec.backwardMode = mode;
        EXPECT_TRUE(errorScale(spec).has_value()) << EnumNamePointwiseMode(mode);
        spec.direction = Direction::WRW;
        EXPECT_FALSE(errorScale(spec).has_value()) << EnumNamePointwiseMode(mode);
    }
}

// ── Output dispatch ─────────────────────────────────────────────────────────

// Each direction is graded with its own test SDK function, for every modeled output type.
TEST(TestConvBwdDynamicTolerance, EachDirectionUsesItsOwnToleranceFunction)
{
    sdk::ConvDgradErrorScale dgrad;
    dgrad.accumulationLength = 144;
    dgrad.dyMeanSquare = 1.0 / 3.0;
    dgrad.weightMeanSquare = 1.0 / 3.0;
    sdk::ConvWrwErrorScale wrw;
    wrw.accumulationLength = 4096;
    wrw.inputMeanSquare = 1.0 / 3.0;
    wrw.dyMeanSquare = 1.0 / 3.0;

    const auto expectSame = [](const std::optional<sdk::AbsRelTolerance>& got,
                               const sdk::AbsRelTolerance& want,
                               const char* what) {
        ASSERT_TRUE(got.has_value()) << what;
        EXPECT_DOUBLE_EQ(got->atol, want.atol) << what;
        EXPECT_DOUBLE_EQ(got->rtol, want.rtol) << what;
    };
    expectSame(tol::convOutputTolerance(dgrad, DataType::FLOAT),
               sdk::calculateConvDgradOutputTolerance<float>(dgrad),
               "dgrad float");
    expectSame(tol::convOutputTolerance(dgrad, DataType::HALF),
               sdk::calculateConvDgradOutputTolerance<half>(dgrad),
               "dgrad half");
    expectSame(tol::convOutputTolerance(dgrad, DataType::BFLOAT16),
               sdk::calculateConvDgradOutputTolerance<bfloat16>(dgrad),
               "dgrad bfloat16");
    expectSame(tol::convOutputTolerance(wrw, DataType::FLOAT),
               sdk::calculateConvWrwOutputTolerance<float>(wrw),
               "wrw float");
    expectSame(tol::convOutputTolerance(wrw, DataType::HALF),
               sdk::calculateConvWrwOutputTolerance<half>(wrw),
               "wrw half");
    expectSame(tol::convOutputTolerance(wrw, DataType::BFLOAT16),
               sdk::calculateConvWrwOutputTolerance<bfloat16>(wrw),
               "wrw bfloat16");
    EXPECT_FALSE(tol::convOutputTolerance(dgrad, DataType::DOUBLE).has_value());
    EXPECT_FALSE(tol::convOutputTolerance(wrw, DataType::DOUBLE).has_value());
}
