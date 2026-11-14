// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenConvTensorChecks.hpp"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::strideOrderNhwc;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using miopen_plugin::conv_applicability::validateConvFwdIoTensors;

namespace
{

// Each negative below breaks exactly one rule, and is matched to the wording of that rule's
// reason, so removing a check turns its tests red instead of being masked by another check.
constexpr const char* DATA_TYPE_REASON = "data type";
constexpr const char* LAYOUT_REASON = "channels-first or all channels-last";
constexpr const char* PACKED_REASON = "is not packed";

struct TensorSpec
{
    DataType dataType = DataType::FLOAT;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    bool nullDims = false;
    bool nullStrides = false;
};

struct FwdCase
{
    const char* name;
    TensorSpec x;
    TensorSpec w;
    TensorSpec y;
};

TensorSpec withStrides(const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& strides,
                       DataType dataType = DataType::FLOAT)
{
    TensorSpec spec;
    spec.dataType = dataType;
    spec.dims = dims;
    spec.strides = strides;
    return spec;
}

TensorSpec channelsFirst(const std::vector<int64_t>& dims, DataType dataType = DataType::FLOAT)
{
    return withStrides(dims, generateStrides(dims), dataType);
}

TensorSpec channelsLast(const std::vector<int64_t>& dims, DataType dataType = DataType::FLOAT)
{
    return withStrides(dims, generateStrides(dims, strideOrderNhwc(dims.size())), dataType);
}

flatbuffers::FlatBufferBuilder buildTensor(int64_t uid, const TensorSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        uid,
        "t",
        spec.dataType,
        spec.nullStrides ? nullptr : &spec.strides,
        spec.nullDims ? nullptr : &spec.dims);
    builder.Finish(attrOffset);
    return builder;
}

// Empty when the tensors are accepted, otherwise the reason they are declined.
std::string declineReason(const TensorSpec& x, const TensorSpec& w, const TensorSpec& y)
{
    const auto xBuffer = buildTensor(1, x);
    const auto wBuffer = buildTensor(2, w);
    const auto yBuffer = buildTensor(3, y);

    try
    {
        validateConvFwdIoTensors(
            *flatbuffers::GetRoot<TensorAttributes>(xBuffer.GetBufferPointer()),
            *flatbuffers::GetRoot<TensorAttributes>(wBuffer.GetBufferPointer()),
            *flatbuffers::GetRoot<TensorAttributes>(yBuffer.GetBufferPointer()));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
        return e.what();
    }
    return {};
}

void expectAccepted(const TensorSpec& x, const TensorSpec& w, const TensorSpec& y)
{
    EXPECT_EQ(declineReason(x, w, y), "");
}

void expectDeclined(const TensorSpec& x,
                    const TensorSpec& w,
                    const TensorSpec& y,
                    const char* reason)
{
    EXPECT_THAT(declineReason(x, w, y), ::testing::HasSubstr(reason));
}

void expectDeclined(const FwdCase& fwdCase, const char* reason)
{
    SCOPED_TRACE(fwdCase.name);
    expectDeclined(fwdCase.x, fwdCase.w, fwdCase.y, reason);
}

// Convolution shapes of rank 3, 4 and 5 (N, C, spatial..): x, w, y dims for a 1D, 2D and 3D
// convolution with C = 8, K = 4, a size-3 filter and no padding.
struct RankShapes
{
    std::vector<int64_t> x;
    std::vector<int64_t> w;
    std::vector<int64_t> y;
};

std::vector<RankShapes> rankShapes()
{
    return {RankShapes{{2, 8, 16}, {4, 8, 3}, {2, 4, 14}},
            RankShapes{{2, 8, 16, 16}, {4, 8, 3, 3}, {2, 4, 14, 14}},
            RankShapes{{2, 8, 8, 8, 8}, {4, 8, 3, 3, 3}, {2, 4, 6, 6, 6}}};
}

// Shapes shared by the 2D layout, dtype and packing cases.
const std::vector<int64_t> X_DIMS = {2, 4, 8, 8};
const std::vector<int64_t> W_DIMS = {4, 4, 3, 3};
const std::vector<int64_t> Y_DIMS = {2, 4, 6, 6};

} // namespace

// ---------------------------------------------------------------------------------------------
// Accepted
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvTensorChecks, AcceptsUniformPackedChannelsFirstAtEveryRank)
{
    for(const auto& shapes : rankShapes())
    {
        SCOPED_TRACE("rank " + std::to_string(shapes.x.size()));
        expectAccepted(channelsFirst(shapes.x), channelsFirst(shapes.w), channelsFirst(shapes.y));
    }
}

TEST(TestMiopenConvTensorChecks, AcceptsUniformPackedChannelsLastAtEveryRank)
{
    for(const auto& shapes : rankShapes())
    {
        SCOPED_TRACE("rank " + std::to_string(shapes.x.size()));
        expectAccepted(channelsLast(shapes.x), channelsLast(shapes.w), channelsLast(shapes.y));
    }
}

TEST(TestMiopenConvTensorChecks, AcceptsEverySupportedDataType)
{
    for(const auto dataType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType));
        expectAccepted(channelsFirst(X_DIMS, dataType),
                       channelsFirst(W_DIMS, dataType),
                       channelsFirst(Y_DIMS, dataType));
        expectAccepted(channelsLast(X_DIMS, dataType),
                       channelsLast(W_DIMS, dataType),
                       channelsLast(Y_DIMS, dataType));
    }
}

// Degenerate shapes that run today and carry support claims. A single output channel gives y
// strides in which the extent-1 channel dimension ties with its neighbour.
TEST(TestMiopenConvTensorChecks, AcceptsOneDimensionalChannelsLastWithSingleOutputChannel)
{
    expectAccepted(withStrides({2, 16, 16}, {256, 1, 16}),
                   withStrides({1, 16, 3}, {48, 1, 16}),
                   withStrides({2, 1, 16}, {16, 1, 1}));
    expectAccepted(withStrides({1, 16, 16}, {256, 1, 16}),
                   withStrides({1, 16, 3}, {48, 1, 16}),
                   withStrides({1, 1, 16}, {16, 1, 1}));
}

// One input channel with an Rx1 filter: x, w and y all carry extent-1 dimensions whose strides tie.
TEST(TestMiopenConvTensorChecks, AcceptsSingleInputChannelWithRx1Filter)
{
    expectAccepted(channelsLast({2, 1, 16, 1}),
                   withStrides({16, 1, 3, 1}, {3, 1, 1, 1}),
                   channelsLast({2, 16, 16, 1}));
}

// A global convolution reduces the spatial extent to 1, so y has extent-1 spatial dimensions.
TEST(TestMiopenConvTensorChecks, AcceptsGlobalConvolutionWithUnitSpatialOutput)
{
    expectAccepted(
        channelsFirst({2, 16, 8, 8}), channelsFirst({32, 16, 8, 8}), channelsFirst({2, 32, 1, 1}));
    expectAccepted(
        channelsLast({2, 16, 8, 8}), channelsLast({32, 16, 8, 8}), channelsLast({2, 32, 1, 1}));
}

// With C = 1 and every spatial extent 1 the two canonical stride vectors coincide: such a tensor is
// both channels-first and channels-last and takes the class of the other tensors.
TEST(TestMiopenConvTensorChecks, AcceptsTensorsThatAreBothLayoutClasses)
{
    const std::vector<int64_t> xDims = {2, 1, 1, 1};
    const std::vector<int64_t> wDims = {4, 1, 1, 1};
    const std::vector<int64_t> yDims = {2, 4, 1, 1};

    ASSERT_EQ(generateStrides(xDims), generateStrides(xDims, strideOrderNhwc(xDims.size())));
    ASSERT_EQ(generateStrides(wDims), generateStrides(wDims, strideOrderNhwc(wDims.size())));
    ASSERT_NE(generateStrides(yDims), generateStrides(yDims, strideOrderNhwc(yDims.size())));

    expectAccepted(channelsFirst(xDims), channelsFirst(wDims), channelsFirst(yDims));
    expectAccepted(channelsFirst(xDims), channelsFirst(wDims), channelsLast(yDims));
}

// ---------------------------------------------------------------------------------------------
// Declined: layout
// ---------------------------------------------------------------------------------------------

// Every tensor in turn is the odd one out.
TEST(TestMiopenConvTensorChecks, DeclinesMixedLayoutWhicheverTensorDiffers)
{
    const std::vector<FwdCase> cases = {
        {"x cl, w cf, y cf", channelsLast(X_DIMS), channelsFirst(W_DIMS), channelsFirst(Y_DIMS)},
        {"x cf, w cl, y cf", channelsFirst(X_DIMS), channelsLast(W_DIMS), channelsFirst(Y_DIMS)},
        {"x cf, w cf, y cl", channelsFirst(X_DIMS), channelsFirst(W_DIMS), channelsLast(Y_DIMS)},
        {"x cf, w cl, y cl", channelsFirst(X_DIMS), channelsLast(W_DIMS), channelsLast(Y_DIMS)},
        {"x cl, w cf, y cl", channelsLast(X_DIMS), channelsFirst(W_DIMS), channelsLast(Y_DIMS)},
        {"x cl, w cl, y cf", channelsLast(X_DIMS), channelsLast(W_DIMS), channelsFirst(Y_DIMS)},
    };

    for(const auto& fwdCase : cases)
    {
        expectDeclined(fwdCase, LAYOUT_REASON);
    }
}

TEST(TestMiopenConvTensorChecks, DeclinesMixedLayoutAtEveryRank)
{
    for(const auto& shapes : rankShapes())
    {
        SCOPED_TRACE("rank " + std::to_string(shapes.x.size()));
        expectDeclined(channelsFirst(shapes.x),
                       channelsLast(shapes.w),
                       channelsFirst(shapes.y),
                       LAYOUT_REASON);
    }
}

// The convolution-forward layout-mix bundle shapes that fail on MIOpen today, with their exact
// dims and strides including the extent-1 dimensions.
TEST(TestMiopenConvTensorChecks, DeclinesMixedLayoutBundleShapes)
{
    const std::vector<FwdCase> cases = {
        {"1d bf16 x cf w cl y cf",
         withStrides({8, 16, 16}, {256, 16, 1}, DataType::BFLOAT16),
         withStrides({32, 16, 1}, {16, 1, 16}, DataType::BFLOAT16),
         withStrides({8, 32, 16}, {512, 16, 1}, DataType::BFLOAT16)},
        {"2d bf16 grouped x cf w cl y cf",
         withStrides({2, 16, 16, 16}, {4096, 256, 16, 1}, DataType::BFLOAT16),
         withStrides({16, 1, 3, 3}, {9, 1, 3, 1}, DataType::BFLOAT16),
         withStrides({2, 16, 16, 16}, {4096, 256, 16, 1}, DataType::BFLOAT16)},
        {"1d fp32 single input channel x cf w cf y cl",
         withStrides({2, 1, 16}, {16, 16, 1}),
         withStrides({16, 1, 3}, {3, 3, 1}),
         withStrides({2, 16, 16}, {256, 1, 16})},
        {"2d bf16 single input channel x cl w cf y cf",
         withStrides({2, 1, 16, 16}, {256, 1, 16, 1}, DataType::BFLOAT16),
         withStrides({16, 1, 3, 3}, {9, 9, 3, 1}, DataType::BFLOAT16),
         withStrides({2, 16, 16, 16}, {4096, 256, 16, 1}, DataType::BFLOAT16)},
        {"3d fp16 single input channel x cf w cl y cl",
         withStrides({2, 1, 16, 16, 16}, {4096, 4096, 256, 16, 1}, DataType::HALF),
         withStrides({16, 1, 3, 3, 3}, {27, 1, 9, 3, 1}, DataType::HALF),
         withStrides({2, 16, 16, 16, 16}, {65536, 1, 4096, 256, 16}, DataType::HALF)},
    };

    for(const auto& fwdCase : cases)
    {
        expectDeclined(fwdCase, LAYOUT_REASON);
    }
}

// Strides that are packed but neither channels-first nor channels-last (here dimension 0 is the
// innermost one) belong to no class, so a graph made only of them has no common class.
TEST(TestMiopenConvTensorChecks, DeclinesPackedTensorsWithNeitherCanonicalStrides)
{
    const std::vector<int64_t> order = {0, 1, 2, 3};
    const auto other = [&order](const std::vector<int64_t>& dims) {
        return withStrides(dims, generateStrides(dims, order));
    };

    expectDeclined(other(X_DIMS), other(W_DIMS), other(Y_DIMS), LAYOUT_REASON);
    expectDeclined(channelsFirst(X_DIMS), channelsFirst(W_DIMS), other(Y_DIMS), LAYOUT_REASON);
    expectDeclined(channelsLast(X_DIMS), other(W_DIMS), channelsLast(Y_DIMS), LAYOUT_REASON);
}

// Layout is judged before packing: a non-packed tensor does not rescue a mixed-layout graph.
TEST(TestMiopenConvTensorChecks, DeclinesMixedLayoutWhenOneTensorIsAlsoNotPacked)
{
    expectDeclined(withStrides(X_DIMS, {512, 128, 16, 1}),
                   channelsLast(W_DIMS),
                   channelsFirst(Y_DIMS),
                   LAYOUT_REASON);
}

// ---------------------------------------------------------------------------------------------
// Declined: data type
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvTensorChecks, DeclinesMixedDataTypeWhicheverTensorDiffers)
{
    const std::vector<FwdCase> cases = {
        {"y differs",
         channelsFirst(X_DIMS, DataType::HALF),
         channelsFirst(W_DIMS, DataType::HALF),
         channelsFirst(Y_DIMS, DataType::FLOAT)},
        {"w differs",
         channelsFirst(X_DIMS, DataType::FLOAT),
         channelsFirst(W_DIMS, DataType::BFLOAT16),
         channelsFirst(Y_DIMS, DataType::FLOAT)},
        {"x differs",
         channelsFirst(X_DIMS, DataType::HALF),
         channelsFirst(W_DIMS, DataType::FLOAT),
         channelsFirst(Y_DIMS, DataType::FLOAT)},
    };

    for(const auto& fwdCase : cases)
    {
        expectDeclined(fwdCase, DATA_TYPE_REASON);
    }
}

TEST(TestMiopenConvTensorChecks, DeclinesUnsupportedDataTypeShared)
{
    expectDeclined(channelsFirst(X_DIMS, DataType::DOUBLE),
                   channelsFirst(W_DIMS, DataType::DOUBLE),
                   channelsFirst(Y_DIMS, DataType::DOUBLE),
                   DATA_TYPE_REASON);
}

// ---------------------------------------------------------------------------------------------
// Declined: non-packed
// ---------------------------------------------------------------------------------------------

// The padded tensor keeps the stride order of its layout, so the three tensors still share one
// layout class and only the packing rule declines the graph.
TEST(TestMiopenConvTensorChecks, DeclinesNonPackedChannelsFirstInput)
{
    const std::vector<FwdCase> cases = {
        {"padded rows",
         withStrides(X_DIMS, {512, 128, 16, 1}),
         channelsFirst(W_DIMS),
         channelsFirst(Y_DIMS)},
        {"padded channels",
         withStrides(X_DIMS, {512, 128, 8, 1}),
         channelsFirst(W_DIMS),
         channelsFirst(Y_DIMS)},
        {"padded planes",
         withStrides(X_DIMS, {384, 64, 8, 1}),
         channelsFirst(W_DIMS),
         channelsFirst(Y_DIMS)},
    };

    for(const auto& fwdCase : cases)
    {
        expectDeclined(fwdCase, PACKED_REASON);
    }
}

TEST(TestMiopenConvTensorChecks, DeclinesNonPackedChannelsLastInput)
{
    const std::vector<FwdCase> cases = {
        {"padded rows",
         withStrides(X_DIMS, {320, 1, 40, 4}),
         channelsLast(W_DIMS),
         channelsLast(Y_DIMS)},
        {"padded columns",
         withStrides(X_DIMS, {512, 1, 64, 8}),
         channelsLast(W_DIMS),
         channelsLast(Y_DIMS)},
        {"padded batch",
         withStrides(X_DIMS, {384, 1, 32, 4}),
         channelsLast(W_DIMS),
         channelsLast(Y_DIMS)},
    };

    for(const auto& fwdCase : cases)
    {
        expectDeclined(fwdCase, PACKED_REASON);
    }
}

TEST(TestMiopenConvTensorChecks, DeclinesNonPackedWeights)
{
    expectDeclined(channelsFirst(X_DIMS),
                   withStrides(W_DIMS, {48, 12, 4, 1}),
                   channelsFirst(Y_DIMS),
                   PACKED_REASON);
}

TEST(TestMiopenConvTensorChecks, DeclinesNonPackedOutput)
{
    expectDeclined(channelsFirst(X_DIMS),
                   channelsFirst(W_DIMS),
                   withStrides(Y_DIMS, {192, 48, 8, 1}),
                   PACKED_REASON);
}

// ---------------------------------------------------------------------------------------------
// Declined: malformed tensors
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvTensorChecks, DeclinesNullDims)
{
    auto x = channelsFirst(X_DIMS);
    x.nullDims = true;

    expectDeclined(x, channelsFirst(W_DIMS), channelsFirst(Y_DIMS), "null dims");
}

TEST(TestMiopenConvTensorChecks, DeclinesNullStrides)
{
    auto w = channelsFirst(W_DIMS);
    w.nullStrides = true;

    expectDeclined(channelsFirst(X_DIMS), w, channelsFirst(Y_DIMS), "null dims or strides");
}

TEST(TestMiopenConvTensorChecks, DeclinesDimsAndStridesOfDifferentSize)
{
    expectDeclined(channelsFirst(X_DIMS),
                   channelsFirst(W_DIMS),
                   withStrides(Y_DIMS, {144, 36, 6}),
                   "different dims and strides sizes");
}

TEST(TestMiopenConvTensorChecks, DeclinesRankBelowThree)
{
    expectDeclined(
        channelsFirst({2, 8}), channelsFirst({4, 8}), channelsFirst({2, 4}), "rank below 3");
}
