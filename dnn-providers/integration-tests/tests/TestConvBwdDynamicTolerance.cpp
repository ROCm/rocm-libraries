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
    bool fusedReluBackward = false; ///< RELU_BWD(dgrad result, x) after the dgrad
    DataType resultType = DataType::FLOAT; ///< the dgrad result's type when the activation reads it
    bool dyWrittenByANode = false; ///< dy is RELU_FWD of another input: not a graph input
    bool addForwardConvolution = false; ///< a second, forward, convolution in the same graph
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
    const bool dgradResultIsRead = spec.direction == Direction::DGRAD && spec.fusedReluBackward;
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
    addTensor(W_UID, "w", spec.wDims, false);
    addTensor(OUT_UID, "out", spec.xDims, false);
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
    if(spec.fusedReluBackward)
    {
        const auto relu = CreatePointwiseAttributes(b,
                                                    PointwiseMode::RELU_BWD,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    X_UID,
                                                    ACT_X_UID,
                                                    flatbuffers::nullopt,
                                                    OUT_UID);
        nodes.push_back(CreateNodeDirect(
            b, "relu_bwd", DataType::FLOAT, NodeAttributes::PointwiseAttributes, relu.Union()));
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
    return tol::convErrorScale(buildGraph(b, spec), lookup);
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
    spec.fusedReluBackward = true;
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
    spec.fusedReluBackward = true;
    spec.resultType = DataType::HALF;
    const auto scale = dgradScale(spec);
    ASSERT_TRUE(scale.has_value());
    ASSERT_EQ(scale->roundingPoints.size(), 1u);
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].epsilon, std::ldexp(1.0, -10));
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].magnitude, 216.0);
    EXPECT_DOUBLE_EQ(scale->roundingPoints[0].slope, 1.0);
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
