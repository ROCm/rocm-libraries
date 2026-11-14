// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenConvTensorChecks.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::isTensorPacked;
using hipdnn_data_sdk::utilities::strideOrderNhwc;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using miopen_plugin::conv_applicability::validateConvBwdDataIoTensors;
using miopen_plugin::conv_applicability::validateConvBwdDataSize;
using miopen_plugin::conv_applicability::validateConvFwdIoTensors;
using miopen_plugin::conv_applicability::validateConvWrwIoTensors;
using miopen_plugin::conv_applicability::validateFusedConvBiasOperand;

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

const TensorAttributes& attributesOf(const flatbuffers::FlatBufferBuilder& buffer)
{
    return *flatbuffers::GetRoot<TensorAttributes>(buffer.GetBufferPointer());
}

// Empty when `validate` does not throw, otherwise the reason it throws with.
std::string declineReasonOf(const std::function<void()>& validate)
{
    try
    {
        validate();
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
        return e.what();
    }
    return {};
}

// Empty when the tensors are accepted, otherwise the reason they are declined.
std::string declineReason(const TensorSpec& x, const TensorSpec& w, const TensorSpec& y)
{
    const auto xBuffer = buildTensor(1, x);
    const auto wBuffer = buildTensor(2, w);
    const auto yBuffer = buildTensor(3, y);

    return declineReasonOf([&]() {
        validateConvFwdIoTensors(
            attributesOf(xBuffer), attributesOf(wBuffer), attributesOf(yBuffer));
    });
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

// ---------------------------------------------------------------------------------------------
// Backward data and backward weights
// ---------------------------------------------------------------------------------------------

// The two backward operations take the same three tensors in different roles. A case is written in
// backward-data roles (dy, w, dx); backward weights poses the same problem with x the shape of dx
// and dw the shape of w, so every case runs against both public validators.
enum class BackwardOp
{
    DATA,
    WEIGHTS
};

enum class Role
{
    DY,
    W,
    DX
};

const std::vector<BackwardOp> BACKWARD_OPS = {BackwardOp::DATA, BackwardOp::WEIGHTS};
const std::vector<Role> ROLES = {Role::DY, Role::W, Role::DX};

const char* operationName(BackwardOp op)
{
    return op == BackwardOp::DATA ? "Convolution backward data" : "Convolution backward weights";
}

// The name of a tensor role in the reasons of the given operation.
const char* roleName(BackwardOp op, Role role)
{
    const bool data = op == BackwardOp::DATA;
    if(role == Role::DY)
    {
        return "dy";
    }
    if(role == Role::W)
    {
        return data ? "w" : "dw";
    }
    return data ? "dx" : "x";
}

struct BackwardCase
{
    const char* name;
    TensorSpec dy;
    TensorSpec w;
    TensorSpec dx;
};

const TensorSpec& tensorAt(const BackwardCase& tensors, Role role)
{
    if(role == Role::DY)
    {
        return tensors.dy;
    }
    if(role == Role::W)
    {
        return tensors.w;
    }
    return tensors.dx;
}

BackwardCase withTensorAt(BackwardCase tensors, Role role, TensorSpec spec)
{
    if(role == Role::DY)
    {
        tensors.dy = std::move(spec);
    }
    else if(role == Role::W)
    {
        tensors.w = std::move(spec);
    }
    else
    {
        tensors.dx = std::move(spec);
    }
    return tensors;
}

// dy, w, dx packed in one layout (and one data type): dx has the shape of x, dy that of y.
BackwardCase uniformBackwardCase(const RankShapes& shapes,
                                 bool channelsLastLayout,
                                 DataType dataType = DataType::FLOAT)
{
    const auto make = [channelsLastLayout, dataType](const std::vector<int64_t>& dims) {
        return channelsLastLayout ? channelsLast(dims, dataType) : channelsFirst(dims, dataType);
    };
    return {"uniform", make(shapes.y), make(shapes.w), make(shapes.x)};
}

// Strides of `dims` with the padding of one dimension: in the physical order of the layout
// (outermost first: N, C, spatial.. for channels-first; N, spatial.., C for channels-last) the
// extent of the dimension `level` positions from the innermost one counts one element more when
// the strides are laid out. Level 0 pads the innermost dimension (channels in channels-last, row
// length in channels-first), the highest useful level pads the planes. The strides keep the
// strict order of the layout, so the stride-order class is unchanged; every dim must have an
// extent above 1 for the tensor to be non-packed.
TensorSpec paddedTensor(const std::vector<int64_t>& dims,
                        bool channelsLastLayout,
                        size_t level,
                        DataType dataType = DataType::FLOAT)
{
    const size_t rank = dims.size();

    std::vector<size_t> physical;
    physical.push_back(0);
    if(channelsLastLayout)
    {
        for(size_t i = 2; i < rank; ++i)
        {
            physical.push_back(i);
        }
        physical.push_back(1);
    }
    else
    {
        for(size_t i = 1; i < rank; ++i)
        {
            physical.push_back(i);
        }
    }

    std::vector<int64_t> strides(rank, 0);
    int64_t stride = 1;
    for(size_t position = 0; position < rank; ++position)
    {
        const size_t dim = physical[rank - 1 - position];
        strides[dim] = stride;
        stride *= dims[dim] + (position == level ? 1 : 0);
    }
    return withStrides(dims, strides, dataType);
}

// Empty when both operations accept the tensors; the operation name is part of a decline reason.
std::string backwardDeclineReason(BackwardOp op, const BackwardCase& tensors)
{
    const auto dyBuffer = buildTensor(1, tensors.dy);
    const auto wBuffer = buildTensor(2, tensors.w);
    const auto dxBuffer = buildTensor(3, tensors.dx);

    return declineReasonOf([&]() {
        if(op == BackwardOp::DATA)
        {
            validateConvBwdDataIoTensors(
                attributesOf(dyBuffer), attributesOf(wBuffer), attributesOf(dxBuffer));
        }
        else
        {
            validateConvWrwIoTensors(
                attributesOf(dxBuffer), attributesOf(dyBuffer), attributesOf(wBuffer));
        }
    });
}

// Both backward operations accept the tensors.
void expectBackwardAccepted(const BackwardCase& tensors)
{
    SCOPED_TRACE(tensors.name);
    for(const BackwardOp op : BACKWARD_OPS)
    {
        SCOPED_TRACE(operationName(op));
        EXPECT_EQ(backwardDeclineReason(op, tensors), "");
    }
}

// Both backward operations decline the tensors with a reason that contains `reason` and names the
// operation.
void expectBackwardDeclined(const BackwardCase& tensors, const char* reason)
{
    SCOPED_TRACE(tensors.name);
    for(const BackwardOp op : BACKWARD_OPS)
    {
        SCOPED_TRACE(operationName(op));
        const std::string actual = backwardDeclineReason(op, tensors);
        EXPECT_THAT(actual, ::testing::HasSubstr(reason));
        EXPECT_THAT(actual, ::testing::HasSubstr(operationName(op)));
    }
}

// Both backward operations decline the tensors because the tensor at `offender` is channels-last
// and not packed, and say so under that tensor's name in the operation.
void expectChannelsLastNonPackedDeclined(const BackwardCase& tensors, Role offender)
{
    SCOPED_TRACE(tensors.name);
    for(const BackwardOp op : BACKWARD_OPS)
    {
        SCOPED_TRACE(operationName(op));
        const std::string reason = std::string(operationName(op)) + ": tensor "
                                   + roleName(op, offender) + " is channels-last and not packed";
        EXPECT_THAT(backwardDeclineReason(op, tensors), ::testing::HasSubstr(reason));
    }
}

// Backward data size rule. Literal values, so a wrong limit in the implementation is caught.
constexpr int64_t MAX_ELEMENTS = 2147483647;
constexpr const char* SIZE_REASON_DY = "tensor dy has more than 2147483647 elements";
constexpr const char* SIZE_REASON_DX = "tensor dx has more than 2147483647 elements";

std::string sizeDeclineReason(const TensorSpec& dy, const TensorSpec& dx)
{
    const auto dyBuffer = buildTensor(1, dy);
    const auto dxBuffer = buildTensor(2, dx);

    return declineReasonOf(
        [&]() { validateConvBwdDataSize(attributesOf(dyBuffer), attributesOf(dxBuffer)); });
}

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

// ---------------------------------------------------------------------------------------------
// Backward data and backward weights: accepted
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvTensorChecks, BackwardAcceptsUniformPackedAtEveryRank)
{
    for(const auto& shapes : rankShapes())
    {
        for(const bool channelsLastLayout : {false, true})
        {
            SCOPED_TRACE("rank " + std::to_string(shapes.x.size())
                         + (channelsLastLayout ? " channels-last" : " channels-first"));
            expectBackwardAccepted(uniformBackwardCase(shapes, channelsLastLayout));
        }
    }
}

TEST(TestMiopenConvTensorChecks, BackwardAcceptsEverySupportedDataType)
{
    const RankShapes shapes = {X_DIMS, W_DIMS, Y_DIMS};

    for(const auto dataType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const bool channelsLastLayout : {false, true})
        {
            SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType));
            expectBackwardAccepted(uniformBackwardCase(shapes, channelsLastLayout, dataType));
        }
    }
}

// Degenerate shapes that are uniform: they keep their canonical strides and are accepted.
TEST(TestMiopenConvTensorChecks, BackwardAcceptsDegenerateUniformShapes)
{
    // 1D channels-last with a single channel on dy (the 1D NLC bundle pattern).
    expectBackwardAccepted({"1d single output channel",
                            withStrides({2, 1, 16}, {16, 1, 1}),
                            withStrides({1, 16, 3}, {48, 1, 16}),
                            withStrides({2, 16, 16}, {256, 1, 16})});
    expectBackwardAccepted({"1d single output channel and batch",
                            withStrides({1, 1, 16}, {16, 1, 1}),
                            withStrides({1, 16, 3}, {48, 1, 16}),
                            withStrides({1, 16, 16}, {256, 1, 16})});

    // One input channel with an Rx1 filter.
    expectBackwardAccepted({"single input channel with Rx1 filter",
                            channelsLast({2, 16, 16, 1}),
                            withStrides({16, 1, 3, 1}, {3, 1, 1, 1}),
                            channelsLast({2, 1, 16, 1})});

    // A global convolution makes dy's spatial extent 1: its canonical channels-first strides are
    // accepted next to channels-first w and dx.
    expectBackwardAccepted({"1d global convolution channels-first",
                            channelsFirst({2, 32, 1}),
                            channelsFirst({32, 16, 16}),
                            channelsFirst({2, 16, 16})});
    expectBackwardAccepted({"2d global convolution channels-first",
                            channelsFirst({2, 32, 1, 1}),
                            channelsFirst({32, 16, 8, 8}),
                            channelsFirst({2, 16, 8, 8})});
    expectBackwardAccepted({"2d global convolution channels-last",
                            channelsLast({2, 32, 1, 1}),
                            channelsLast({32, 16, 8, 8}),
                            channelsLast({2, 16, 8, 8})});
}

// With C = 1 and every spatial extent 1 the two canonical stride vectors coincide: such a tensor is
// both channels-first and channels-last and takes the class of the other tensors.
TEST(TestMiopenConvTensorChecks, BackwardAcceptsTensorsThatAreBothLayoutClasses)
{
    const std::vector<int64_t> dxDims = {2, 1, 1, 1};
    const std::vector<int64_t> wDims = {4, 1, 1, 1};
    const std::vector<int64_t> dyDims = {2, 4, 1, 1};

    ASSERT_EQ(generateStrides(dxDims), generateStrides(dxDims, strideOrderNhwc(dxDims.size())));
    ASSERT_EQ(generateStrides(wDims), generateStrides(wDims, strideOrderNhwc(wDims.size())));

    expectBackwardAccepted(
        {"dy channels-first", channelsFirst(dyDims), channelsFirst(wDims), channelsFirst(dxDims)});
    expectBackwardAccepted(
        {"dy channels-last", channelsLast(dyDims), channelsFirst(wDims), channelsFirst(dxDims)});
}

// Unlike forward, a channels-first tensor with padded strides is accepted next to packed
// channels-first tensors: row-, channel- and plane-padded, at every position and rank.
TEST(TestMiopenConvTensorChecks, BackwardAcceptsNonPackedChannelsFirst)
{
    for(const auto& shapes : rankShapes())
    {
        const BackwardCase base = uniformBackwardCase(shapes, /*channelsLastLayout=*/false);

        for(const Role role : ROLES)
        {
            const auto& dims = tensorAt(base, role).dims;

            for(size_t level = 0; level + 1 < dims.size(); ++level)
            {
                SCOPED_TRACE("rank " + std::to_string(dims.size()) + ", tensor "
                             + std::to_string(static_cast<int>(role)) + ", padded level "
                             + std::to_string(level));
                const TensorSpec padded = paddedTensor(dims, /*channelsLastLayout=*/false, level);
                ASSERT_FALSE(isTensorPacked(padded.dims, padded.strides));

                expectBackwardAccepted(withTensorAt(base, role, padded));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Backward data and backward weights: declined, layout
// ---------------------------------------------------------------------------------------------

// Every tensor in turn is the odd one out.
TEST(TestMiopenConvTensorChecks, BackwardDeclinesMixedLayoutWhicheverTensorDiffers)
{
    const std::vector<BackwardCase> cases = {
        {"dy cl, w cf, dx cf", channelsLast(Y_DIMS), channelsFirst(W_DIMS), channelsFirst(X_DIMS)},
        {"dy cf, w cl, dx cf", channelsFirst(Y_DIMS), channelsLast(W_DIMS), channelsFirst(X_DIMS)},
        {"dy cf, w cf, dx cl", channelsFirst(Y_DIMS), channelsFirst(W_DIMS), channelsLast(X_DIMS)},
        {"dy cf, w cl, dx cl", channelsFirst(Y_DIMS), channelsLast(W_DIMS), channelsLast(X_DIMS)},
        {"dy cl, w cf, dx cl", channelsLast(Y_DIMS), channelsFirst(W_DIMS), channelsLast(X_DIMS)},
        {"dy cl, w cl, dx cf", channelsLast(Y_DIMS), channelsLast(W_DIMS), channelsFirst(X_DIMS)},
    };

    for(const auto& backwardCase : cases)
    {
        expectBackwardDeclined(backwardCase, LAYOUT_REASON);
    }
}

TEST(TestMiopenConvTensorChecks, BackwardDeclinesMixedLayoutAtEveryRank)
{
    for(const auto& shapes : rankShapes())
    {
        SCOPED_TRACE("rank " + std::to_string(shapes.x.size()));
        expectBackwardDeclined({"dy cf, w cl, dx cf",
                                channelsFirst(shapes.y),
                                channelsLast(shapes.w),
                                channelsFirst(shapes.x)},
                               LAYOUT_REASON);
    }
}

// The backward-data layout-mix bundle shapes that fail on MIOpen today (1D, 2D and 3D), with their
// exact dims and strides including the extent-1 dimensions. The first one is the degenerate global
// convolution: dy [2,32,1] has only the channels-first canonical strides.
TEST(TestMiopenConvTensorChecks, BackwardDeclinesMixedLayoutBundleShapes)
{
    const std::vector<BackwardCase> cases = {
        {"1d half global convolution, dy cf, w cl, dx cl",
         withStrides({2, 32, 1}, {32, 1, 1}, DataType::HALF),
         withStrides({32, 16, 16}, {256, 1, 16}, DataType::HALF),
         withStrides({2, 16, 16}, {256, 1, 16}, DataType::HALF)},
        {"2d bf16, dy cf, w cl, dx cf",
         withStrides({2, 32, 14, 14}, {6272, 196, 14, 1}, DataType::BFLOAT16),
         withStrides({32, 16, 3, 3}, {144, 1, 48, 16}, DataType::BFLOAT16),
         withStrides({2, 16, 16, 16}, {4096, 256, 16, 1}, DataType::BFLOAT16)},
        {"2d half, dy cf, w cl, dx cl",
         withStrides({2, 32, 8, 8}, {2048, 64, 8, 1}, DataType::HALF),
         withStrides({32, 16, 2, 2}, {64, 1, 32, 16}, DataType::HALF),
         withStrides({2, 16, 16, 16}, {4096, 1, 256, 16}, DataType::HALF)},
        {"2d float, dy cf, w cf, dx cl",
         withStrides({2, 32, 2, 2}, {128, 4, 2, 1}),
         withStrides({32, 3, 16, 16}, {768, 256, 16, 1}),
         withStrides({2, 3, 32, 32}, {3072, 1, 96, 3})},
        {"3d bf16, dy cf, w cl, dx cf",
         withStrides({2, 32, 14, 14, 14}, {87808, 2744, 196, 14, 1}, DataType::BFLOAT16),
         withStrides({32, 16, 3, 3, 3}, {432, 1, 144, 48, 16}, DataType::BFLOAT16),
         withStrides({2, 16, 16, 16, 16}, {65536, 4096, 256, 16, 1}, DataType::BFLOAT16)},
        {"3d float, dy cf, w cf, dx cl",
         withStrides({2, 32, 16, 16, 16}, {131072, 4096, 256, 16, 1}),
         withStrides({32, 16, 3, 3, 3}, {432, 27, 9, 3, 1}),
         withStrides({2, 16, 16, 16, 16}, {65536, 1, 4096, 256, 16})},
    };

    for(const auto& backwardCase : cases)
    {
        expectBackwardDeclined(backwardCase, LAYOUT_REASON);
    }
}

// Strides that are packed but neither channels-first nor channels-last (here dimension 0 is the
// innermost one) belong to no class, so a graph made only of them has no common class.
TEST(TestMiopenConvTensorChecks, BackwardDeclinesPackedTensorsWithNeitherCanonicalStrides)
{
    const std::vector<int64_t> order = {0, 1, 2, 3};
    const auto other = [&order](const std::vector<int64_t>& dims) {
        return withStrides(dims, generateStrides(dims, order));
    };

    expectBackwardDeclined({"all other", other(Y_DIMS), other(W_DIMS), other(X_DIMS)},
                           LAYOUT_REASON);
    expectBackwardDeclined(
        {"dx other", channelsFirst(Y_DIMS), channelsFirst(W_DIMS), other(X_DIMS)}, LAYOUT_REASON);
    expectBackwardDeclined({"w other", channelsLast(Y_DIMS), other(W_DIMS), channelsLast(X_DIMS)},
                           LAYOUT_REASON);
}

// Layout is judged before the channels-last packing rule, and a non-packed tensor is classified by
// its stride order: a padded channels-last tensor does not rescue a mixed-layout graph, and a
// padded channels-first one next to channels-last tensors is a mixed-layout graph.
TEST(TestMiopenConvTensorChecks, BackwardJudgesLayoutOfNonPackedTensorsByStrideOrder)
{
    expectBackwardDeclined({"dx cl padded, dy cf, w cl",
                            channelsFirst(Y_DIMS),
                            channelsLast(W_DIMS),
                            paddedTensor(X_DIMS, /*channelsLastLayout=*/true, 1)},
                           LAYOUT_REASON);
    expectBackwardDeclined({"dy cf padded, w cl, dx cl",
                            paddedTensor(Y_DIMS, /*channelsLastLayout=*/false, 1),
                            channelsLast(W_DIMS),
                            channelsLast(X_DIMS)},
                           LAYOUT_REASON);
}

// ---------------------------------------------------------------------------------------------
// Backward data and backward weights: declined, data type
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvTensorChecks, BackwardDeclinesMixedDataTypeWhicheverTensorDiffers)
{
    const std::vector<BackwardCase> cases = {
        {"dx differs",
         channelsFirst(Y_DIMS, DataType::HALF),
         channelsFirst(W_DIMS, DataType::HALF),
         channelsFirst(X_DIMS, DataType::FLOAT)},
        {"w differs",
         channelsFirst(Y_DIMS, DataType::FLOAT),
         channelsFirst(W_DIMS, DataType::BFLOAT16),
         channelsFirst(X_DIMS, DataType::FLOAT)},
        {"dy differs",
         channelsFirst(Y_DIMS, DataType::HALF),
         channelsFirst(W_DIMS, DataType::FLOAT),
         channelsFirst(X_DIMS, DataType::FLOAT)},
    };

    for(const auto& backwardCase : cases)
    {
        expectBackwardDeclined(backwardCase, DATA_TYPE_REASON);
    }
}

TEST(TestMiopenConvTensorChecks, BackwardDeclinesUnsupportedDataTypeShared)
{
    expectBackwardDeclined({"double",
                            channelsFirst(Y_DIMS, DataType::DOUBLE),
                            channelsFirst(W_DIMS, DataType::DOUBLE),
                            channelsFirst(X_DIMS, DataType::DOUBLE)},
                           DATA_TYPE_REASON);
}

// ---------------------------------------------------------------------------------------------
// Backward data and backward weights: declined, channels-last non-packed
// ---------------------------------------------------------------------------------------------

// The padded tensor keeps the channels-last stride order, so the three tensors still share one
// layout class and only the channels-last packing rule declines the graph. Row-, channel- and
// plane-padded, at every position and rank, for both operations.
TEST(TestMiopenConvTensorChecks, BackwardDeclinesNonPackedChannelsLast)
{
    for(const auto& shapes : rankShapes())
    {
        const BackwardCase base = uniformBackwardCase(shapes, /*channelsLastLayout=*/true);

        for(const Role role : ROLES)
        {
            const auto& dims = tensorAt(base, role).dims;

            for(size_t level = 0; level + 1 < dims.size(); ++level)
            {
                SCOPED_TRACE("rank " + std::to_string(dims.size()) + ", padded level "
                             + std::to_string(level));
                const TensorSpec padded = paddedTensor(dims, /*channelsLastLayout=*/true, level);
                ASSERT_FALSE(isTensorPacked(padded.dims, padded.strides));

                expectChannelsLastNonPackedDeclined(withTensorAt(base, role, padded), role);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Backward data and backward weights: declined, malformed tensors
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvTensorChecks, BackwardDeclinesMalformedTensors)
{
    auto nullDims = uniformBackwardCase({X_DIMS, W_DIMS, Y_DIMS}, false);
    nullDims.dy.nullDims = true;
    expectBackwardDeclined(nullDims, "null dims");

    auto nullStrides = uniformBackwardCase({X_DIMS, W_DIMS, Y_DIMS}, false);
    nullStrides.w.nullStrides = true;
    expectBackwardDeclined(nullStrides, "null dims or strides");

    expectBackwardDeclined({"size mismatch",
                            channelsFirst(Y_DIMS),
                            channelsFirst(W_DIMS),
                            withStrides(X_DIMS, {256, 64, 8})},
                           "different dims and strides sizes");

    expectBackwardDeclined(
        {"rank 2", channelsFirst({2, 4}), channelsFirst({4, 8}), channelsFirst({2, 8})},
        "rank below 3");
}

// ---------------------------------------------------------------------------------------------
// Backward data: size
// ---------------------------------------------------------------------------------------------

// Exactly 2^31 - 1 elements, and just below, on dy and on dx separately.
TEST(TestMiopenConvTensorChecks, BackwardDataSizeAcceptsUpToTheLimit)
{
    const TensorSpec small = channelsFirst(Y_DIMS);

    const std::vector<std::vector<int64_t>> atOrBelowLimit = {
        {1, MAX_ELEMENTS, 1, 1},
        {2, MAX_ELEMENTS / 2, 1, 1},
        {1, 1, 46340, 46340},
        {MAX_ELEMENTS, 1, 1, 1},
    };

    for(const auto& dims : atOrBelowLimit)
    {
        SCOPED_TRACE(std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x"
                     + std::to_string(dims[2]) + "x" + std::to_string(dims[3]));
        EXPECT_EQ(sizeDeclineReason(channelsFirst(dims), small), "");
        EXPECT_EQ(sizeDeclineReason(small, channelsFirst(dims)), "");
    }
}

TEST(TestMiopenConvTensorChecks, BackwardDataSizeDeclinesAboveTheLimit)
{
    const TensorSpec small = channelsFirst(Y_DIMS);

    const std::vector<std::vector<int64_t>> aboveLimit = {
        {1, MAX_ELEMENTS + 1, 1, 1},
        {2, (MAX_ELEMENTS + 1) / 2, 1, 1},
        {1, 1, 46341, 46341},
        {MAX_ELEMENTS + 1, 1, 1, 1},
    };

    for(const auto& dims : aboveLimit)
    {
        SCOPED_TRACE(std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x"
                     + std::to_string(dims[2]) + "x" + std::to_string(dims[3]));
        EXPECT_THAT(sizeDeclineReason(channelsFirst(dims), small),
                    ::testing::HasSubstr(SIZE_REASON_DY));
        EXPECT_THAT(sizeDeclineReason(small, channelsFirst(dims)),
                    ::testing::HasSubstr(SIZE_REASON_DX));
    }
}

// The two real-data-type cases the ticket names: a 46341 x 46341 single-channel image has
// 2,147,488,281 elements, in half precision channels-first and bfloat16 channels-last.
TEST(TestMiopenConvTensorChecks, BackwardDataSizeDeclinesTheSingleChannelImageShapes)
{
    const std::vector<int64_t> dims = {1, 1, 46341, 46341};

    EXPECT_THAT(
        sizeDeclineReason(channelsFirst(dims, DataType::HALF), channelsFirst(dims, DataType::HALF)),
        ::testing::HasSubstr(SIZE_REASON_DY));
    EXPECT_THAT(sizeDeclineReason(channelsLast(dims, DataType::BFLOAT16),
                                  channelsLast(dims, DataType::BFLOAT16)),
                ::testing::HasSubstr(SIZE_REASON_DY));
}

// A product that overflows a 64-bit integer must decline, not wrap around to a small count.
TEST(TestMiopenConvTensorChecks, BackwardDataSizeDeclinesDimsWhoseProductOverflows)
{
    const TensorSpec small = channelsFirst(Y_DIMS);

    const std::vector<std::vector<int64_t>> overflowing = {
        {std::numeric_limits<int64_t>::max(), 2, 1, 1},
        {2, 1, 1, std::numeric_limits<int64_t>::max()},
        {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 2, 2},
        {4294967296, 4294967296, 1, 1},
    };

    for(const auto& dims : overflowing)
    {
        // The strides are never read, and computing canonical ones would overflow.
        const TensorSpec spec = withStrides(dims, std::vector<int64_t>(dims.size(), 1));

        SCOPED_TRACE(std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x"
                     + std::to_string(dims[2]) + "x" + std::to_string(dims[3]));
        EXPECT_THAT(sizeDeclineReason(spec, small), ::testing::HasSubstr(SIZE_REASON_DY));
        EXPECT_THAT(sizeDeclineReason(small, spec), ::testing::HasSubstr(SIZE_REASON_DX));
    }
}

TEST(TestMiopenConvTensorChecks, BackwardDataSizeDeclinesNullAndNegativeDims)
{
    auto nullDy = channelsFirst(Y_DIMS);
    nullDy.nullDims = true;
    EXPECT_THAT(sizeDeclineReason(nullDy, channelsFirst(X_DIMS)),
                ::testing::HasSubstr("tensor dy has null dims"));

    auto nullDx = channelsFirst(X_DIMS);
    nullDx.nullDims = true;
    EXPECT_THAT(sizeDeclineReason(channelsFirst(Y_DIMS), nullDx),
                ::testing::HasSubstr("tensor dx has null dims"));

    EXPECT_THAT(sizeDeclineReason(withStrides({4, -2, 3, 3}, {1, 1, 1, 1}), channelsFirst(X_DIMS)),
                ::testing::HasSubstr("tensor dy has a negative dim"));
}

// The size rule reads only the dims: strides and data type play no part.
TEST(TestMiopenConvTensorChecks, BackwardDataSizeReadsOnlyTheDims)
{
    auto dy = channelsFirst(Y_DIMS, DataType::DOUBLE);
    dy.nullStrides = true;

    EXPECT_EQ(sizeDeclineReason(dy, withStrides(X_DIMS, {7, 7, 7, 7}, DataType::UNSET)), "");
}

// Only validateConvBwdDataSize carries the limit: tensors of more than 2^31 - 1 elements pass the
// io-tensor rules of both backward operations when those rules hold, so backward weights (and
// forward) are not affected by the backward-data ticket.
TEST(TestMiopenConvTensorChecks, SizeLimitIsNotPartOfTheIoTensorRules)
{
    const std::vector<int64_t> hugeDims = {1, MAX_ELEMENTS + 1, 1, 1};
    const std::vector<int64_t> unitDims = {1, 1, 1, 1};

    expectBackwardAccepted(
        {"dx huge", channelsFirst(unitDims), channelsFirst(unitDims), channelsFirst(hugeDims)});
    expectBackwardAccepted(
        {"dy huge", channelsFirst(hugeDims), channelsFirst(unitDims), channelsFirst(unitDims)});
    expectAccepted(channelsFirst(hugeDims), channelsFirst(unitDims), channelsFirst(unitDims));
}

// ---------------------------------------------------------------------------------------------
// Fused convolution + bias + activation: the bias operand
// ---------------------------------------------------------------------------------------------

namespace
{

constexpr const char* BIAS_DIMS_REASON = "bias operand must have dims";
constexpr const char* BIAS_DATA_TYPE_REASON = "must equal the io data type";

// Empty when the bias operand is accepted, otherwise the reason it is declined. `yConv` is the
// virtual convolution output (float for a half or bfloat16 graph) and carries no strides check.
std::string fusedDeclineReason(const TensorSpec& x, const TensorSpec& yConv, const TensorSpec& bias)
{
    const auto xBuffer = buildTensor(1, x);
    const auto yBuffer = buildTensor(2, yConv);
    const auto biasBuffer = buildTensor(3, bias);

    return declineReasonOf([&]() {
        validateFusedConvBiasOperand(
            attributesOf(xBuffer), attributesOf(yBuffer), attributesOf(biasBuffer));
    });
}

// Per-channel bias dims of a convolution output: 1 everywhere but the channel dim.
std::vector<int64_t> perChannelDims(const std::vector<int64_t>& yDims)
{
    std::vector<int64_t> dims(yDims.size(), 1);
    dims[1] = yDims[1];
    return dims;
}

const std::vector<DataType> FUSED_IO_DATA_TYPES
    = {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16};

void expectBiasDimsDeclined(const char* name,
                            const std::vector<int64_t>& biasDims,
                            const std::vector<int64_t>& yDims = Y_DIMS)
{
    SCOPED_TRACE(name);
    EXPECT_THAT(
        fusedDeclineReason(channelsFirst(X_DIMS), channelsFirst(yDims), channelsFirst(biasDims)),
        ::testing::HasSubstr(BIAS_DIMS_REASON));
}

} // namespace

// The control of every negative below: a [1, K, 1, ..] bias of the io data type at rank 3, 4 and 5,
// with the float convolution intermediate that a half or bfloat16 graph carries.
TEST(TestMiopenConvTensorChecks, FusedBiasAcceptsPerChannelBiasOfTheIoDataType)
{
    for(const auto& shapes : rankShapes())
    {
        for(const DataType dataType : FUSED_IO_DATA_TYPES)
        {
            SCOPED_TRACE("rank " + std::to_string(shapes.y.size()) + ", "
                         + hipdnn_flatbuffers_sdk::data_objects::toString(dataType));

            EXPECT_EQ(fusedDeclineReason(channelsFirst(shapes.x, dataType),
                                         channelsFirst(shapes.y, DataType::FLOAT),
                                         channelsFirst(perChannelDims(shapes.y), dataType)),
                      "");
            EXPECT_EQ(fusedDeclineReason(channelsLast(shapes.x, dataType),
                                         channelsLast(shapes.y, dataType),
                                         channelsFirst(perChannelDims(shapes.y), dataType)),
                      "");
        }
    }
}

// With one output channel the per-channel bias and the scalar coincide, so [1, 1, 1, 1] is a valid
// bias there (and only there).
TEST(TestMiopenConvTensorChecks, FusedBiasAcceptsSingleValueBiasForOneOutputChannel)
{
    const std::vector<int64_t> yDims = {2, 1, 6, 6};

    EXPECT_EQ(fusedDeclineReason(
                  channelsFirst(X_DIMS), channelsFirst(yDims), channelsFirst({1, 1, 1, 1})),
              "");
}

// ALMIOPEN-3014: MIOpen reads the operand as a per-channel vector whatever its shape is.
TEST(TestMiopenConvTensorChecks, FusedBiasDeclinesOperandThatIsNotPerChannel)
{
    // Y_DIMS is [2, 4, 6, 6]: K == 4, rank 4.
    expectBiasDimsDeclined("full-size residual", Y_DIMS);
    expectBiasDimsDeclined("scalar with K > 1", {1, 1, 1, 1});
    expectBiasDimsDeclined("batch extent above 1", {2, 4, 1, 1});
    expectBiasDimsDeclined("wrong channel count", {1, 3, 1, 1});
    expectBiasDimsDeclined("channel count at the wrong position", {1, 1, 1, 4});
    expectBiasDimsDeclined("spatial extent above 1", {1, 4, 6, 1});
    expectBiasDimsDeclined("rank 3 operand against a rank 4 output", {1, 4, 1});
    expectBiasDimsDeclined("rank 5 operand against a rank 4 output", {1, 4, 1, 1, 1});
    expectBiasDimsDeclined("rank 3 full-size residual", {2, 4, 14}, {2, 4, 14});
    expectBiasDimsDeclined("scalar against a rank 5 output", {1, 1, 1, 1, 1}, {2, 4, 6, 6, 6});
}

// The reason names the dims the operand must have and the dims it has.
TEST(TestMiopenConvTensorChecks, FusedBiasDimsReasonNamesExpectedAndActualDims)
{
    EXPECT_THAT(fusedDeclineReason(
                    channelsFirst(X_DIMS), channelsFirst(Y_DIMS), channelsFirst({1, 1, 1, 1})),
                ::testing::AllOf(::testing::HasSubstr("[1, 4, 1, 1]"),
                                 ::testing::HasSubstr("got [1, 1, 1, 1]")));
}

// ALMIOPEN-3016: the dims are right and only the data type differs from x.
TEST(TestMiopenConvTensorChecks, FusedBiasDeclinesDataTypeDifferentFromTheIoDataType)
{
    const std::vector<int64_t> biasDims = perChannelDims(Y_DIMS);

    // fp32 bias on a half graph slips through the node checks: the intermediate is float too.
    EXPECT_THAT(fusedDeclineReason(channelsFirst(X_DIMS, DataType::HALF),
                                   channelsFirst(Y_DIMS, DataType::FLOAT),
                                   channelsFirst(biasDims, DataType::FLOAT)),
                ::testing::HasSubstr(BIAS_DATA_TYPE_REASON));
    EXPECT_THAT(fusedDeclineReason(channelsFirst(X_DIMS, DataType::BFLOAT16),
                                   channelsFirst(Y_DIMS, DataType::FLOAT),
                                   channelsFirst(biasDims, DataType::FLOAT)),
                ::testing::HasSubstr(BIAS_DATA_TYPE_REASON));
    EXPECT_THAT(fusedDeclineReason(channelsFirst(X_DIMS, DataType::FLOAT),
                                   channelsFirst(Y_DIMS, DataType::FLOAT),
                                   channelsFirst(biasDims, DataType::HALF)),
                ::testing::HasSubstr(BIAS_DATA_TYPE_REASON));
    EXPECT_THAT(fusedDeclineReason(channelsFirst(X_DIMS, DataType::HALF),
                                   channelsFirst(Y_DIMS, DataType::FLOAT),
                                   channelsFirst(biasDims, DataType::BFLOAT16)),
                ::testing::HasSubstr(BIAS_DATA_TYPE_REASON));
}

TEST(TestMiopenConvTensorChecks, FusedBiasDataTypeReasonNamesBothTypes)
{
    EXPECT_THAT(fusedDeclineReason(channelsFirst(X_DIMS, DataType::HALF),
                                   channelsFirst(Y_DIMS, DataType::FLOAT),
                                   channelsFirst(perChannelDims(Y_DIMS), DataType::FLOAT)),
                ::testing::AllOf(::testing::HasSubstr("FLOAT"), ::testing::HasSubstr("HALF")));
}

// Both rules are independent: a wrong shape with the right type is declined for the shape, and a
// right shape with the wrong type is declined for the type (the cases above), so neither check can
// stand in for the other.
TEST(TestMiopenConvTensorChecks, FusedBiasShapeIsCheckedWithTheRightDataType)
{
    EXPECT_THAT(fusedDeclineReason(channelsFirst(X_DIMS, DataType::HALF),
                                   channelsFirst(Y_DIMS, DataType::FLOAT),
                                   channelsFirst(Y_DIMS, DataType::HALF)),
                ::testing::HasSubstr(BIAS_DIMS_REASON));
}

TEST(TestMiopenConvTensorChecks, FusedBiasDeclinesMalformedTensors)
{
    const TensorSpec x = channelsFirst(X_DIMS);
    const TensorSpec y = channelsFirst(Y_DIMS);
    const TensorSpec bias = channelsFirst(perChannelDims(Y_DIMS));

    TensorSpec nullX = x;
    nullX.nullDims = true;
    EXPECT_THAT(fusedDeclineReason(nullX, y, bias), ::testing::HasSubstr("null dims"));

    TensorSpec nullY = y;
    nullY.nullDims = true;
    EXPECT_THAT(fusedDeclineReason(x, nullY, bias), ::testing::HasSubstr("null dims"));

    TensorSpec nullBias = bias;
    nullBias.nullDims = true;
    EXPECT_THAT(fusedDeclineReason(x, y, nullBias), ::testing::HasSubstr("null dims"));

    EXPECT_THAT(fusedDeclineReason(x, channelsFirst({2, 4}), channelsFirst({1, 4})),
                ::testing::HasSubstr("rank below 3"));
}
