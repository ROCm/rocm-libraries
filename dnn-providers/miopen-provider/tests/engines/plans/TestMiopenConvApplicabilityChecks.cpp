// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "common/ConvolutionCommon.hpp"
#include "engines/plans/MiopenConvApplicabilityChecks.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::isTensorPacked;
using hipdnn_data_sdk::utilities::strideOrderNhwc;
using hipdnn_data_sdk::utilities::vecToString;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using miopen_plugin::conv_applicability::validateConvBwdDataIoTensors;
using miopen_plugin::conv_applicability::validateConvBwdDataSize;
using miopen_plugin::conv_applicability::validateConvFwdIoTensors;
using miopen_plugin::conv_applicability::validateConvWrwIoTensors;
using test_conv_common::packedStrides;

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

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
    return withStrides(dims, packedStrides(dims, false), dataType);
}

TensorSpec channelsLast(const std::vector<int64_t>& dims, DataType dataType = DataType::FLOAT)
{
    return withStrides(dims, packedStrides(dims, true), dataType);
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

// Shapes shared by the 2D layout, dtype and packing cases (and by the plan builder tests).
const std::vector<int64_t>& convXDims = test_conv_common::CONV_2D_X_DIMS;
const std::vector<int64_t>& convWDims = test_conv_common::CONV_2D_W_DIMS;
const std::vector<int64_t>& convYDims = test_conv_common::CONV_2D_Y_DIMS;

// ---------------------------------------------------------------------------------------------
// Forward, backward data and backward weights
// ---------------------------------------------------------------------------------------------

// The three operations take the same three tensors in different roles. A case is written in
// backward-data roles (dy, w, dx); forward poses the same problem with x the shape of dx and y the
// shape of dy, backward weights with x the shape of dx and dw the shape of w, so every case runs
// against all three public validators unless a test names the operations it is about.
enum class ConvOp
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHTS
};

enum class Role
{
    DY,
    W,
    DX
};

const std::vector<ConvOp> ALL_OPS
    = {ConvOp::FORWARD, ConvOp::BACKWARD_DATA, ConvOp::BACKWARD_WEIGHTS};
const std::vector<ConvOp> FORWARD_OPS = {ConvOp::FORWARD};
const std::vector<ConvOp> BACKWARD_OPS = {ConvOp::BACKWARD_DATA, ConvOp::BACKWARD_WEIGHTS};
const std::vector<Role> ROLES = {Role::DY, Role::W, Role::DX};

const char* operationName(ConvOp op)
{
    if(op == ConvOp::FORWARD)
    {
        return "Convolution forward";
    }
    return op == ConvOp::BACKWARD_DATA ? "Convolution backward data"
                                       : "Convolution backward weights";
}

// The name of a tensor role in the reasons of the given operation.
const char* roleName(ConvOp op, Role role)
{
    if(role == Role::DY)
    {
        return op == ConvOp::FORWARD ? "y" : "dy";
    }
    if(role == Role::W)
    {
        return op == ConvOp::BACKWARD_WEIGHTS ? "dw" : "w";
    }
    return op == ConvOp::BACKWARD_DATA ? "dx" : "x";
}

struct ConvCase
{
    const char* name;
    TensorSpec dy;
    TensorSpec w;
    TensorSpec dx;
};

// A case written in forward roles, for the tests that are about forward alone: x is the dx role and
// y the dy role.
ConvCase
    forwardCase(const char* name, const TensorSpec& x, const TensorSpec& w, const TensorSpec& y)
{
    return {name, y, w, x};
}

const TensorSpec& tensorAt(const ConvCase& tensors, Role role)
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

ConvCase withTensorAt(ConvCase tensors, Role role, TensorSpec spec)
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
ConvCase uniformCase(const RankShapes& shapes,
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

// Empty when the operation accepts the tensors; the operation name is part of a decline reason.
std::string declineReason(ConvOp op, const ConvCase& tensors)
{
    const auto dyBuffer = buildTensor(1, tensors.dy);
    const auto wBuffer = buildTensor(2, tensors.w);
    const auto dxBuffer = buildTensor(3, tensors.dx);

    return declineReasonOf([&]() {
        if(op == ConvOp::FORWARD)
        {
            validateConvFwdIoTensors(
                attributesOf(dxBuffer), attributesOf(wBuffer), attributesOf(dyBuffer));
        }
        else if(op == ConvOp::BACKWARD_DATA)
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

// Every operation of `ops` accepts the tensors.
void expectAccepted(const ConvCase& tensors, const std::vector<ConvOp>& ops = ALL_OPS)
{
    SCOPED_TRACE(tensors.name);
    for(const ConvOp op : ops)
    {
        SCOPED_TRACE(operationName(op));
        EXPECT_EQ(declineReason(op, tensors), "");
    }
}

// Every operation of `ops` declines the tensors with a reason that contains `reason` and names the
// operation.
void expectDeclined(const ConvCase& tensors,
                    const char* reason,
                    const std::vector<ConvOp>& ops = ALL_OPS)
{
    SCOPED_TRACE(tensors.name);
    for(const ConvOp op : ops)
    {
        SCOPED_TRACE(operationName(op));
        const std::string actual = declineReason(op, tensors);
        EXPECT_THAT(actual, ::testing::HasSubstr(reason));
        EXPECT_THAT(actual, ::testing::HasSubstr(operationName(op)));
    }
}

void expectForwardDeclined(const TensorSpec& x,
                           const TensorSpec& w,
                           const TensorSpec& y,
                           const char* reason)
{
    expectDeclined(forwardCase("forward", x, w, y), reason, FORWARD_OPS);
}

// Both backward operations decline the tensors because the tensor at `offender` is channels-last
// and not packed, and say so under that tensor's name in the operation.
void expectChannelsLastNonPackedDeclined(const ConvCase& tensors, Role offender)
{
    SCOPED_TRACE(tensors.name);
    for(const ConvOp op : BACKWARD_OPS)
    {
        SCOPED_TRACE(operationName(op));
        const std::string reason = std::string(operationName(op)) + ": tensor "
                                   + roleName(op, offender) + " is channels-last and not packed";
        EXPECT_THAT(declineReason(op, tensors), ::testing::HasSubstr(reason));
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

// Dims whose element count reaches INT64_MAX: the packing extent and the canonical strides of such
// dims overflow a signed 64-bit integer, and so does an element count formed over them.
std::vector<std::vector<int64_t>> overflowingDims()
{
    return {
        {std::numeric_limits<int64_t>::max(), 2, 1, 1},
        {2, 1, 1, std::numeric_limits<int64_t>::max()},
        {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 2, 2},
        {4294967296, 4294967296, 1, 1},
        {std::numeric_limits<int64_t>::max(), 1, 1, 1},
    };
}

// Dims whose product, counting a zero dim as 1, reaches INT64_MAX although the element count is
// zero: overflowing in either dim order, so the io validators decline them, while the size check
// (which counts the zero) accepts both orders.
std::vector<std::vector<int64_t>> zeroExtentOverflowingDims()
{
    return {
        {0, 4294967296, 4294967296, 1},
        {4294967296, 4294967296, 0, 1},
    };
}

// All dims the io validators decline for overflow.
std::vector<std::vector<int64_t>> ioOverflowingDims()
{
    auto dims = overflowingDims();
    const auto zeroExtent = zeroExtentOverflowingDims();
    dims.insert(dims.end(), zeroExtent.begin(), zeroExtent.end());
    return dims;
}

std::string overflowReason(const char* name)
{
    return std::string("tensor ") + name + " has dims whose element count reaches the 64-bit limit";
}

// Strides that no canonical computation could produce for the dims: the validators must decline
// on the dims alone.
TensorSpec withUnitStrides(const std::vector<int64_t>& dims)
{
    return withStrides(dims, std::vector<int64_t>(dims.size(), 1));
}

std::string strideOverflowReason(const char* name)
{
    return std::string("tensor ") + name
           + " has strides whose packing arithmetic overflows a 64-bit integer";
}

struct OverflowingStrides
{
    const char* name;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

// Dims that are fine on their own (the element count is small) with strides for which the SDK
// packing check, the sum of (dim - 1) * stride plus 1 in int64_t, overflows in one of its steps.
std::vector<OverflowingStrides> overflowingStrides()
{
    constexpr int64_t MAX_VALUE = std::numeric_limits<int64_t>::max();
    constexpr int64_t MIN_VALUE = std::numeric_limits<int64_t>::min();
    constexpr int64_t TWO_POW_62 = int64_t{1} << 62;

    return {
        {"one term overflows", {1, 4, 1}, {4, TWO_POW_62, 1}},
        {"one negative term overflows", {1, 3, 1}, {1, -MAX_VALUE, 1}},
        {"term overflows next to a maximal stride", {1, 5, 1}, {MAX_VALUE, TWO_POW_62 + 1, 1}},
        {"every term fits, their sum overflows", {2, 2, 1}, {MAX_VALUE, MAX_VALUE, 1}},
        {"every term fits, their negative sum overflows", {2, 2, 1}, {-MAX_VALUE, -MAX_VALUE, 1}},
        {"the sum fits, adding one overflows", {2, 1, 1}, {MAX_VALUE, 1, 1}},
        {"a zero dim negates the minimum stride", {0, 2, 1}, {MIN_VALUE, 1, 1}},
    };
}

std::string describe(const OverflowingStrides& overflowing)
{
    return std::string(overflowing.name) + ": dims " + vecToString(overflowing.dims) + ", strides "
           + vecToString(overflowing.strides);
}

// ---------------------------------------------------------------------------------------------
// Fused convolution + bias + activation graphs
// ---------------------------------------------------------------------------------------------

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

// Empty when the layout checks accept the conv -> [bias] -> activation graph, otherwise the
// reason they decline it.
std::string layoutDeclineReason(const flatbuffers::FlatBufferBuilder& builder)
{
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto& convAttr = graph.getNodeWrapper(0).attributesAs<fb::ConvolutionFwdAttributes>();
    const bool hasBias = graph.nodeCount() == 3;
    const auto* biasAttr
        = hasBias ? &graph.getNodeWrapper(1).attributesAs<fb::PointwiseAttributes>() : nullptr;
    const auto& activAttr
        = graph.getNodeWrapper(hasBias ? 2 : 1).attributesAs<fb::PointwiseAttributes>();

    return declineReasonOf([&]() {
        conv_applicability::checkConvFwdBiasActivLayouts(
            convAttr, biasAttr, activAttr, graph.getTensorMap());
    });
}

// Whether the layout checks accept the conv -> [bias] -> activation graph.
bool layoutsSupported(const flatbuffers::FlatBufferBuilder& builder)
{
    return layoutDeclineReason(builder).empty();
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

// The conv -> bias -> activation graph of `cbaGraph` with every tensor of data type `dataType`.
flatbuffers::FlatBufferBuilder cbaGraphOfDataType(DataType dataType)
{
    return mutateGraph(cbaGraph(), [dataType](fb::GraphT& graph) {
        for(auto& tensor : graph.tensors)
        {
            tensor->data_type = dataType;
        }
    });
}

// Passes the bias as the first operand of the ADD (bias + y_conv) instead of the second.
flatbuffers::FlatBufferBuilder biasAsFirstAddOperand(const flatbuffers::FlatBufferBuilder& source)
{
    return mutateGraph(source, [](fb::GraphT& graph) {
        auto* add = graph.nodes.at(1)->attributes.AsPointwiseAttributes();
        const auto convOutputUid = add->in_0_tensor_uid;
        add->in_0_tensor_uid = *add->in_1_tensor_uid;
        add->in_1_tensor_uid = convOutputUid;
    });
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

TEST(TestMiopenConvApplicabilityChecks, AcceptsPerChannelBiasAsFirstAddOperand)
{
    EXPECT_TRUE(layoutsSupported(biasAsFirstAddOperand(cbaGraph())));
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesFullTensorBiasAsFirstAddOperand)
{
    const auto builder = biasAsFirstAddOperand(editTensor(cbaGraph(), "bias", [](auto& t) {
        t.dims = {4, 4, 4, 4};
        t.strides = {64, 16, 4, 1};
    }));
    EXPECT_THAT(layoutDeclineReason(builder), ::testing::HasSubstr("per-channel"));
}

// The control of the data type rule below: a bias of the input data type, for every supported
// data type and in both operand orders.
TEST(TestMiopenConvApplicabilityChecks, AcceptsBiasOfTheInputDataType)
{
    for(const auto dataType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        SCOPED_TRACE(fb::EnumNameDataType(dataType));
        EXPECT_EQ(layoutDeclineReason(cbaGraphOfDataType(dataType)), "");
        EXPECT_EQ(layoutDeclineReason(biasAsFirstAddOperand(cbaGraphOfDataType(dataType))), "");
    }
}

// ALMIOPEN-3016: the dims are right and only the bias data type differs from the input's. The
// fused solver does not validate it, and the node checks of the fused builder accept an fp32 bias
// on a half or bfloat16 convolution when the intermediates are float too.
TEST(TestMiopenConvApplicabilityChecks, DeclinesBiasDataTypeDifferentFromInput)
{
    const std::vector<DataType> dataTypes = {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16};

    for(const auto inputType : dataTypes)
    {
        for(const auto biasType : dataTypes)
        {
            if(biasType == inputType)
            {
                continue;
            }

            SCOPED_TRACE(std::string(fb::EnumNameDataType(inputType)) + " input, "
                         + fb::EnumNameDataType(biasType) + " bias");
            const auto builder = editTensor(cbaGraphOfDataType(inputType),
                                            "bias",
                                            [biasType](auto& t) { t.data_type = biasType; });
            const std::string reason = std::string("Bias tensor data type ")
                                       + fb::EnumNameDataType(biasType)
                                       + " must equal the convolution input data type "
                                       + fb::EnumNameDataType(inputType);

            EXPECT_THAT(layoutDeclineReason(builder), ::testing::HasSubstr(reason));
            EXPECT_THAT(layoutDeclineReason(biasAsFirstAddOperand(builder)),
                        ::testing::HasSubstr(reason));
        }
    }
}

// The float bias on a half graph that the node checks let through: the convolution and bias
// intermediates and the bias are float, the input and output are half.
TEST(TestMiopenConvApplicabilityChecks, DeclinesFloatBiasOnHalfGraphWithFloatIntermediates)
{
    auto builder = cbaGraphOfDataType(DataType::HALF);
    for(const char* name : {"bias", "y_conv", "y_bias"})
    {
        builder = editTensor(builder, name, [](auto& t) { t.data_type = DataType::FLOAT; });
    }

    EXPECT_THAT(layoutDeclineReason(builder),
                ::testing::HasSubstr("Bias tensor data type FLOAT must equal the convolution "
                                     "input data type HALF"));
}

// The bias data type is compared with the convolution input and with nothing else. With the input
// and the weights in different data types, a bias of the input type is accepted and a bias of the
// weights type is declined, in both operand orders. The case is a legal input to the check, not a
// claim that the convolution of such an input and weights is supported.
TEST(TestMiopenConvApplicabilityChecks, BiasDataTypeIsComparedWithTheInputNotTheWeights)
{
    const std::vector<DataType> dataTypes = {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16};

    for(const auto inputType : dataTypes)
    {
        for(const auto weightsType : dataTypes)
        {
            if(weightsType == inputType)
            {
                continue;
            }

            SCOPED_TRACE(std::string(fb::EnumNameDataType(inputType)) + " input, "
                         + fb::EnumNameDataType(weightsType) + " weights");
            const auto mixed = editTensor(cbaGraphOfDataType(inputType),
                                          "w",
                                          [weightsType](auto& t) { t.data_type = weightsType; });

            // The bias has the type of the input.
            EXPECT_EQ(layoutDeclineReason(mixed), "");
            EXPECT_EQ(layoutDeclineReason(biasAsFirstAddOperand(mixed)), "");

            // The bias has the type of the weights.
            const auto weightsTypedBias
                = editTensor(mixed, "bias", [weightsType](auto& t) { t.data_type = weightsType; });
            const std::string reason = std::string("Bias tensor data type ")
                                       + fb::EnumNameDataType(weightsType)
                                       + " must equal the convolution input data type "
                                       + fb::EnumNameDataType(inputType);

            EXPECT_THAT(layoutDeclineReason(weightsTypedBias), ::testing::HasSubstr(reason));
            EXPECT_THAT(layoutDeclineReason(biasAsFirstAddOperand(weightsTypedBias)),
                        ::testing::HasSubstr(reason));
        }
    }
}

// A bias with both wrong dims and wrong data type is declined for its dims: the shape rule comes
// first, so the reason is the same whichever rule a graph also breaks.
TEST(TestMiopenConvApplicabilityChecks, ReportsBiasShapeBeforeBiasDataType)
{
    const auto builder = editTensor(cbaGraphOfDataType(DataType::FLOAT), "bias", [](auto& t) {
        t.dims = {4, 4, 4, 4};
        t.strides = {64, 16, 4, 1};
        t.data_type = DataType::HALF;
    });
    EXPECT_THAT(layoutDeclineReason(builder), ::testing::HasSubstr("per-channel"));
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

// ---------------------------------------------------------------------------------------------
// Accepted
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvApplicabilityChecks, AcceptsUniformPackedAtEveryRank)
{
    for(const auto& shapes : rankShapes())
    {
        for(const bool channelsLastLayout : {false, true})
        {
            SCOPED_TRACE("rank " + std::to_string(shapes.x.size())
                         + (channelsLastLayout ? " channels-last" : " channels-first"));
            expectAccepted(uniformCase(shapes, channelsLastLayout));
        }
    }
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsEverySupportedDataType)
{
    const RankShapes shapes = {convXDims, convWDims, convYDims};

    for(const auto dataType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const bool channelsLastLayout : {false, true})
        {
            SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType));
            expectAccepted(uniformCase(shapes, channelsLastLayout, dataType));
        }
    }
}

// Degenerate shapes that run today, carry support claims and are uniform: they keep their
// canonical strides and are accepted.
TEST(TestMiopenConvApplicabilityChecks, AcceptsDegenerateUniformShapes)
{
    // 1D channels-last with a single channel on dy (the 1D NLC bundle pattern): a single output
    // channel gives dy strides in which the extent-1 channel dimension ties with its neighbour.
    expectAccepted({"1d single output channel",
                    withStrides({2, 1, 16}, {16, 1, 1}),
                    withStrides({1, 16, 3}, {48, 1, 16}),
                    withStrides({2, 16, 16}, {256, 1, 16})});
    expectAccepted({"1d single output channel and batch",
                    withStrides({1, 1, 16}, {16, 1, 1}),
                    withStrides({1, 16, 3}, {48, 1, 16}),
                    withStrides({1, 16, 16}, {256, 1, 16})});

    // One input channel with an Rx1 filter: all three tensors carry extent-1 dimensions whose
    // strides tie.
    expectAccepted({"single input channel with Rx1 filter",
                    channelsLast({2, 16, 16, 1}),
                    withStrides({16, 1, 3, 1}, {3, 1, 1, 1}),
                    channelsLast({2, 1, 16, 1})});

    // A global convolution makes dy's spatial extent 1: its canonical channels-first strides are
    // accepted next to channels-first w and dx.
    expectAccepted({"1d global convolution channels-first",
                    channelsFirst({2, 32, 1}),
                    channelsFirst({32, 16, 16}),
                    channelsFirst({2, 16, 16})});
    expectAccepted({"2d global convolution channels-first",
                    channelsFirst({2, 32, 1, 1}),
                    channelsFirst({32, 16, 8, 8}),
                    channelsFirst({2, 16, 8, 8})});
    expectAccepted({"2d global convolution channels-last",
                    channelsLast({2, 32, 1, 1}),
                    channelsLast({32, 16, 8, 8}),
                    channelsLast({2, 16, 8, 8})});
}

// With C = 1 and every spatial extent 1 the two canonical stride vectors coincide: such a tensor is
// both channels-first and channels-last and takes the class of the other tensors.
TEST(TestMiopenConvApplicabilityChecks, AcceptsTensorsThatAreBothLayoutClasses)
{
    const std::vector<int64_t> dxDims = {2, 1, 1, 1};
    const std::vector<int64_t> wDims = {4, 1, 1, 1};
    const std::vector<int64_t> dyDims = {2, 4, 1, 1};

    ASSERT_EQ(generateStrides(dxDims), generateStrides(dxDims, strideOrderNhwc(dxDims.size())));
    ASSERT_EQ(generateStrides(wDims), generateStrides(wDims, strideOrderNhwc(wDims.size())));
    ASSERT_NE(generateStrides(dyDims), generateStrides(dyDims, strideOrderNhwc(dyDims.size())));

    expectAccepted(
        {"dy channels-first", channelsFirst(dyDims), channelsFirst(wDims), channelsFirst(dxDims)});
    expectAccepted(
        {"dy channels-last", channelsLast(dyDims), channelsFirst(wDims), channelsFirst(dxDims)});
}

// Unlike forward, a channels-first tensor with padded strides is accepted next to packed
// channels-first tensors: row-, channel- and plane-padded, at every position and rank.
TEST(TestMiopenConvApplicabilityChecks, BackwardAcceptsNonPackedChannelsFirst)
{
    for(const auto& shapes : rankShapes())
    {
        const ConvCase base = uniformCase(shapes, /*channelsLastLayout=*/false);

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

                expectAccepted(withTensorAt(base, role, padded), BACKWARD_OPS);
            }
        }
    }
}

// A tensor with a zero dim has no elements, and the io validators do not decline it for that: only
// dims whose product would overflow are declined (below). The zero is the leading dim, where the
// canonical strides of both layouts still describe a packed tensor, so the tensor takes its layout
// class from them like any other. Each tensor in turn is empty, and the input and output of an
// empty batch together.
TEST(TestMiopenConvApplicabilityChecks, AcceptsEmptyTensorsWithSaneDims)
{
    for(const auto& shapes : rankShapes())
    {
        for(const bool channelsLastLayout : {false, true})
        {
            const std::string layout = channelsLastLayout ? " channels-last" : " channels-first";
            const ConvCase base = uniformCase(shapes, channelsLastLayout);
            const auto empty = [channelsLastLayout](std::vector<int64_t> dims) {
                dims[0] = 0;
                return channelsLastLayout ? channelsLast(dims) : channelsFirst(dims);
            };

            for(const Role role : ROLES)
            {
                SCOPED_TRACE("rank " + std::to_string(shapes.x.size()) + layout + ", tensor "
                             + std::to_string(static_cast<int>(role)) + " empty");
                expectAccepted(withTensorAt(base, role, empty(tensorAt(base, role).dims)));
            }

            SCOPED_TRACE("rank " + std::to_string(shapes.x.size()) + layout + ", empty batch");
            expectAccepted({"empty batch", empty(shapes.y), base.w, empty(shapes.x)});
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Declined: layout
// ---------------------------------------------------------------------------------------------

// Every tensor in turn is the odd one out.
TEST(TestMiopenConvApplicabilityChecks, DeclinesMixedLayoutWhicheverTensorDiffers)
{
    const std::vector<ConvCase> cases = {
        {"dy cl, w cf, dx cf",
         channelsLast(convYDims),
         channelsFirst(convWDims),
         channelsFirst(convXDims)},
        {"dy cf, w cl, dx cf",
         channelsFirst(convYDims),
         channelsLast(convWDims),
         channelsFirst(convXDims)},
        {"dy cf, w cf, dx cl",
         channelsFirst(convYDims),
         channelsFirst(convWDims),
         channelsLast(convXDims)},
        {"dy cf, w cl, dx cl",
         channelsFirst(convYDims),
         channelsLast(convWDims),
         channelsLast(convXDims)},
        {"dy cl, w cf, dx cl",
         channelsLast(convYDims),
         channelsFirst(convWDims),
         channelsLast(convXDims)},
        {"dy cl, w cl, dx cf",
         channelsLast(convYDims),
         channelsLast(convWDims),
         channelsFirst(convXDims)},
    };

    for(const auto& mixedCase : cases)
    {
        expectDeclined(mixedCase, LAYOUT_REASON);
    }
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesMixedLayoutAtEveryRank)
{
    for(const auto& shapes : rankShapes())
    {
        SCOPED_TRACE("rank " + std::to_string(shapes.x.size()));
        expectDeclined({"dy cf, w cl, dx cf",
                        channelsFirst(shapes.y),
                        channelsLast(shapes.w),
                        channelsFirst(shapes.x)},
                       LAYOUT_REASON);
    }
}

// The convolution-forward layout-mix bundle shapes that fail on MIOpen today, with their exact
// dims and strides including the extent-1 dimensions.
TEST(TestMiopenConvApplicabilityChecks, DeclinesMixedLayoutBundleShapes)
{
    const std::vector<ConvCase> cases = {
        forwardCase("1d bf16 x cf w cl y cf",
                    withStrides({8, 16, 16}, {256, 16, 1}, DataType::BFLOAT16),
                    withStrides({32, 16, 1}, {16, 1, 16}, DataType::BFLOAT16),
                    withStrides({8, 32, 16}, {512, 16, 1}, DataType::BFLOAT16)),
        forwardCase("2d bf16 grouped x cf w cl y cf",
                    withStrides({2, 16, 16, 16}, {4096, 256, 16, 1}, DataType::BFLOAT16),
                    withStrides({16, 1, 3, 3}, {9, 1, 3, 1}, DataType::BFLOAT16),
                    withStrides({2, 16, 16, 16}, {4096, 256, 16, 1}, DataType::BFLOAT16)),
        forwardCase("1d fp32 single input channel x cf w cf y cl",
                    withStrides({2, 1, 16}, {16, 16, 1}),
                    withStrides({16, 1, 3}, {3, 3, 1}),
                    withStrides({2, 16, 16}, {256, 1, 16})),
        forwardCase("2d bf16 single input channel x cl w cf y cf",
                    withStrides({2, 1, 16, 16}, {256, 1, 16, 1}, DataType::BFLOAT16),
                    withStrides({16, 1, 3, 3}, {9, 9, 3, 1}, DataType::BFLOAT16),
                    withStrides({2, 16, 16, 16}, {4096, 256, 16, 1}, DataType::BFLOAT16)),
        forwardCase("3d fp16 single input channel x cf w cl y cl",
                    withStrides({2, 1, 16, 16, 16}, {4096, 4096, 256, 16, 1}, DataType::HALF),
                    withStrides({16, 1, 3, 3, 3}, {27, 1, 9, 3, 1}, DataType::HALF),
                    withStrides({2, 16, 16, 16, 16}, {65536, 1, 4096, 256, 16}, DataType::HALF)),
    };

    for(const auto& bundleCase : cases)
    {
        expectDeclined(bundleCase, LAYOUT_REASON, FORWARD_OPS);
    }
}

// The backward-data layout-mix bundle shapes that fail on MIOpen today (1D, 2D and 3D), with their
// exact dims and strides including the extent-1 dimensions. The first one is the degenerate global
// convolution: dy [2,32,1] has only the channels-first canonical strides.
TEST(TestMiopenConvApplicabilityChecks, BackwardDeclinesMixedLayoutBundleShapes)
{
    const std::vector<ConvCase> cases = {
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

    for(const auto& bundleCase : cases)
    {
        expectDeclined(bundleCase, LAYOUT_REASON, BACKWARD_OPS);
    }
}

// Strides that are packed but neither channels-first nor channels-last (here dimension 0 is the
// innermost one) belong to no class, so a graph made only of them has no common class.
TEST(TestMiopenConvApplicabilityChecks, DeclinesPackedTensorsWithNeitherCanonicalStrides)
{
    const std::vector<int64_t> order = {0, 1, 2, 3};
    const auto other = [&order](const std::vector<int64_t>& dims) {
        return withStrides(dims, generateStrides(dims, order));
    };

    expectDeclined({"all other", other(convYDims), other(convWDims), other(convXDims)},
                   LAYOUT_REASON);
    expectDeclined(
        {"dx other", channelsFirst(convYDims), channelsFirst(convWDims), other(convXDims)},
        LAYOUT_REASON);
    expectDeclined({"w other", channelsLast(convYDims), other(convWDims), channelsLast(convXDims)},
                   LAYOUT_REASON);
}

// Layout is judged before packing: a non-packed tensor does not rescue a mixed-layout graph.
TEST(TestMiopenConvApplicabilityChecks, DeclinesMixedLayoutWhenOneTensorIsAlsoNotPacked)
{
    expectForwardDeclined(withStrides(convXDims, {512, 128, 16, 1}),
                          channelsLast(convWDims),
                          channelsFirst(convYDims),
                          LAYOUT_REASON);
}

// Layout is judged before the channels-last packing rule, and a non-packed tensor is classified by
// its stride order: a padded channels-last tensor does not rescue a mixed-layout graph, and a
// padded channels-first one next to channels-last tensors is a mixed-layout graph.
TEST(TestMiopenConvApplicabilityChecks, BackwardJudgesLayoutOfNonPackedTensorsByStrideOrder)
{
    expectDeclined({"dx cl padded, dy cf, w cl",
                    channelsFirst(convYDims),
                    channelsLast(convWDims),
                    paddedTensor(convXDims, /*channelsLastLayout=*/true, 1)},
                   LAYOUT_REASON,
                   BACKWARD_OPS);
    expectDeclined({"dy cf padded, w cl, dx cl",
                    paddedTensor(convYDims, /*channelsLastLayout=*/false, 1),
                    channelsLast(convWDims),
                    channelsLast(convXDims)},
                   LAYOUT_REASON,
                   BACKWARD_OPS);
}

// ---------------------------------------------------------------------------------------------
// Declined: data type
// ---------------------------------------------------------------------------------------------

// Every tensor in turn is the odd one out, with another data type in each of the other two.
TEST(TestMiopenConvApplicabilityChecks, DeclinesMixedDataTypeWhicheverTensorDiffers)
{
    const std::vector<ConvCase> cases = {
        {"dx differs",
         channelsFirst(convYDims, DataType::HALF),
         channelsFirst(convWDims, DataType::HALF),
         channelsFirst(convXDims, DataType::FLOAT)},
        {"w differs",
         channelsFirst(convYDims, DataType::FLOAT),
         channelsFirst(convWDims, DataType::BFLOAT16),
         channelsFirst(convXDims, DataType::FLOAT)},
        {"dy differs",
         channelsFirst(convYDims, DataType::HALF),
         channelsFirst(convWDims, DataType::FLOAT),
         channelsFirst(convXDims, DataType::FLOAT)},
        {"dy differs, dx and w share a type",
         channelsFirst(convYDims, DataType::FLOAT),
         channelsFirst(convWDims, DataType::HALF),
         channelsFirst(convXDims, DataType::HALF)},
        {"dx differs, dy and w share a type",
         channelsFirst(convYDims, DataType::FLOAT),
         channelsFirst(convWDims, DataType::FLOAT),
         channelsFirst(convXDims, DataType::HALF)},
    };

    for(const auto& mixedCase : cases)
    {
        expectDeclined(mixedCase, DATA_TYPE_REASON);
    }
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesUnsupportedDataTypeShared)
{
    expectDeclined({"double",
                    channelsFirst(convYDims, DataType::DOUBLE),
                    channelsFirst(convWDims, DataType::DOUBLE),
                    channelsFirst(convXDims, DataType::DOUBLE)},
                   DATA_TYPE_REASON);
}

// ---------------------------------------------------------------------------------------------
// Declined: non-packed
// ---------------------------------------------------------------------------------------------

// Forward declines every non-packed tensor. The padded tensor keeps the stride order of its
// layout, so the three tensors still share one layout class and only the packing rule declines the
// graph.
TEST(TestMiopenConvApplicabilityChecks, DeclinesNonPackedChannelsFirstInput)
{
    const std::vector<ConvCase> cases = {
        forwardCase("padded rows",
                    withStrides(convXDims, {512, 128, 16, 1}),
                    channelsFirst(convWDims),
                    channelsFirst(convYDims)),
        forwardCase("padded channels",
                    withStrides(convXDims, {512, 128, 8, 1}),
                    channelsFirst(convWDims),
                    channelsFirst(convYDims)),
        forwardCase("padded planes",
                    withStrides(convXDims, {384, 64, 8, 1}),
                    channelsFirst(convWDims),
                    channelsFirst(convYDims)),
    };

    for(const auto& paddedCase : cases)
    {
        expectDeclined(paddedCase, PACKED_REASON, FORWARD_OPS);
    }
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesNonPackedChannelsLastInput)
{
    const std::vector<ConvCase> cases = {
        forwardCase("padded rows",
                    withStrides(convXDims, {320, 1, 40, 4}),
                    channelsLast(convWDims),
                    channelsLast(convYDims)),
        forwardCase("padded columns",
                    withStrides(convXDims, {512, 1, 64, 8}),
                    channelsLast(convWDims),
                    channelsLast(convYDims)),
        forwardCase("padded batch",
                    withStrides(convXDims, {384, 1, 32, 4}),
                    channelsLast(convWDims),
                    channelsLast(convYDims)),
    };

    for(const auto& paddedCase : cases)
    {
        expectDeclined(paddedCase, PACKED_REASON, FORWARD_OPS);
    }
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesNonPackedWeights)
{
    expectForwardDeclined(channelsFirst(convXDims),
                          withStrides(convWDims, {48, 12, 4, 1}),
                          channelsFirst(convYDims),
                          PACKED_REASON);
}

TEST(TestMiopenConvApplicabilityChecks, DeclinesNonPackedOutput)
{
    expectForwardDeclined(channelsFirst(convXDims),
                          channelsFirst(convWDims),
                          withStrides(convYDims, {192, 48, 8, 1}),
                          PACKED_REASON);
}

// Backward declines only a non-packed channels-last tensor. The padded tensor keeps the
// channels-last stride order, so the three tensors still share one layout class and only the
// channels-last packing rule declines the graph. Row-, channel- and plane-padded, at every
// position and rank, for both operations.
TEST(TestMiopenConvApplicabilityChecks, BackwardDeclinesNonPackedChannelsLast)
{
    for(const auto& shapes : rankShapes())
    {
        const ConvCase base = uniformCase(shapes, /*channelsLastLayout=*/true);

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

// A tensor with a broadcast axis (an extent above 1 with stride 0) is not packed (isTensorPacked),
// and is judged like any other non-packed tensor: forward declines it with every non-packed
// tensor, backward classifies it by its stride order and accepts it next to packed channels-first
// tensors. This pins the behaviour that follows from that classification, unchanged from the
// base; it is not a rule of its own for broadcast tensors.
TEST(TestMiopenConvApplicabilityChecks, BroadcastAxisIsJudgedLikeAnyNonPackedTensor)
{
    const ConvCase base
        = uniformCase({convXDims, convWDims, convYDims}, /*channelsLastLayout=*/false);

    for(const Role role : ROLES)
    {
        auto strides = tensorAt(base, role).strides;
        strides.back() = 0;
        const TensorSpec broadcast = withStrides(tensorAt(base, role).dims, strides);
        ASSERT_FALSE(isTensorPacked(broadcast.dims, broadcast.strides));

        SCOPED_TRACE("tensor " + std::to_string(static_cast<int>(role)) + " broadcast");
        const ConvCase tensors = withTensorAt(base, role, broadcast);
        expectDeclined(tensors, PACKED_REASON, FORWARD_OPS);
        expectAccepted(tensors, BACKWARD_OPS);
    }
}

// ---------------------------------------------------------------------------------------------
// Declined: malformed tensors
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvApplicabilityChecks, DeclinesMalformedTensors)
{
    auto nullDyDims = uniformCase({convXDims, convWDims, convYDims}, false);
    nullDyDims.dy.nullDims = true;
    expectDeclined(nullDyDims, "null dims");

    auto nullDxDims = uniformCase({convXDims, convWDims, convYDims}, false);
    nullDxDims.dx.nullDims = true;
    expectDeclined(nullDxDims, "null dims");

    auto nullStrides = uniformCase({convXDims, convWDims, convYDims}, false);
    nullStrides.w.nullStrides = true;
    expectDeclined(nullStrides, "null dims or strides");

    expectDeclined({"dx dims and strides of different size",
                    channelsFirst(convYDims),
                    channelsFirst(convWDims),
                    withStrides(convXDims, {256, 64, 8})},
                   "different dims and strides sizes");
    expectDeclined({"dy dims and strides of different size",
                    withStrides(convYDims, {144, 36, 6}),
                    channelsFirst(convWDims),
                    channelsFirst(convXDims)},
                   "different dims and strides sizes");

    expectDeclined({"rank 2", channelsFirst({2, 4}), channelsFirst({4, 8}), channelsFirst({2, 8})},
                   "rank below 3");
}

// ---------------------------------------------------------------------------------------------
// Backward data: size
// ---------------------------------------------------------------------------------------------

// Exactly 2^31 - 1 elements, and just below, on dy and on dx separately.
TEST(TestMiopenConvApplicabilityChecks, BackwardDataSizeAcceptsUpToTheLimit)
{
    const TensorSpec small = channelsFirst(convYDims);

    const std::vector<std::vector<int64_t>> atOrBelowLimit = {
        {1, MAX_ELEMENTS, 1, 1},
        {2, MAX_ELEMENTS / 2, 1, 1},
        {1, 1, 46340, 46340},
        {MAX_ELEMENTS, 1, 1, 1},
    };

    for(const auto& dims : atOrBelowLimit)
    {
        SCOPED_TRACE(vecToString(dims));
        EXPECT_EQ(sizeDeclineReason(channelsFirst(dims), small), "");
        EXPECT_EQ(sizeDeclineReason(small, channelsFirst(dims)), "");
    }
}

TEST(TestMiopenConvApplicabilityChecks, BackwardDataSizeDeclinesAboveTheLimit)
{
    const TensorSpec small = channelsFirst(convYDims);

    const std::vector<std::vector<int64_t>> aboveLimit = {
        {1, MAX_ELEMENTS + 1, 1, 1},
        {2, (MAX_ELEMENTS + 1) / 2, 1, 1},
        {1, 1, 46341, 46341},
        {MAX_ELEMENTS + 1, 1, 1, 1},
    };

    for(const auto& dims : aboveLimit)
    {
        SCOPED_TRACE(vecToString(dims));
        EXPECT_THAT(sizeDeclineReason(channelsFirst(dims), small),
                    ::testing::HasSubstr(SIZE_REASON_DY));
        EXPECT_THAT(sizeDeclineReason(small, channelsFirst(dims)),
                    ::testing::HasSubstr(SIZE_REASON_DX));
    }
}

// The two real-data-type cases the ticket names: a 46341 x 46341 single-channel image has
// 2,147,488,281 elements, in half precision channels-first and bfloat16 channels-last.
TEST(TestMiopenConvApplicabilityChecks, BackwardDataSizeDeclinesTheSingleChannelImageShapes)
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
TEST(TestMiopenConvApplicabilityChecks, BackwardDataSizeDeclinesDimsWhoseProductOverflows)
{
    const TensorSpec small = channelsFirst(convYDims);

    for(const auto& dims : overflowingDims())
    {
        // The strides are never read, and computing canonical ones would overflow.
        const TensorSpec spec = withUnitStrides(dims);

        SCOPED_TRACE(vecToString(dims));
        EXPECT_THAT(sizeDeclineReason(spec, small), ::testing::HasSubstr(SIZE_REASON_DY));
        EXPECT_THAT(sizeDeclineReason(small, spec), ::testing::HasSubstr(SIZE_REASON_DX));
    }
}

// The element count of a tensor with a zero dim is zero whatever the order of the other dims, so
// the size check accepts both orders of the overflowing dims, although the io validators decline
// them.
TEST(TestMiopenConvApplicabilityChecks, BackwardDataSizeAcceptsZeroExtentDimsInAnyOrder)
{
    const TensorSpec small = channelsFirst(convYDims);

    for(const auto& dims : zeroExtentOverflowingDims())
    {
        SCOPED_TRACE(vecToString(dims));
        const TensorSpec spec = withUnitStrides(dims);

        EXPECT_EQ(sizeDeclineReason(spec, small), "");
        EXPECT_EQ(sizeDeclineReason(small, spec), "");
    }
}

// A zero dim makes the element count zero, whatever the other dims are, so an empty tensor is
// within the limit. A zero counted as 1 would put the first two of these above it.
TEST(TestMiopenConvApplicabilityChecks, BackwardDataSizeCountsAZeroDimAsZeroElements)
{
    const TensorSpec small = channelsFirst(convYDims);

    const std::vector<std::vector<int64_t>> emptyDims = {
        {0, 3000000000, 1, 1},
        {3000000000, 0, 1, 1},
        {0, 5, 5, 5},
        {2, 0, 3, 3},
    };

    for(const auto& dims : emptyDims)
    {
        SCOPED_TRACE(vecToString(dims));
        EXPECT_EQ(sizeDeclineReason(channelsFirst(dims), small), "");
        EXPECT_EQ(sizeDeclineReason(small, channelsFirst(dims)), "");
    }
}

TEST(TestMiopenConvApplicabilityChecks, BackwardDataSizeDeclinesNullAndNegativeDims)
{
    auto nullDy = channelsFirst(convYDims);
    nullDy.nullDims = true;
    EXPECT_THAT(sizeDeclineReason(nullDy, channelsFirst(convXDims)),
                ::testing::HasSubstr("tensor dy has null dims"));

    auto nullDx = channelsFirst(convXDims);
    nullDx.nullDims = true;
    EXPECT_THAT(sizeDeclineReason(channelsFirst(convYDims), nullDx),
                ::testing::HasSubstr("tensor dx has null dims"));

    EXPECT_THAT(
        sizeDeclineReason(withStrides({4, -2, 3, 3}, {1, 1, 1, 1}), channelsFirst(convXDims)),
        ::testing::HasSubstr("tensor dy has a negative dim"));
}

// The size rule reads only the dims: strides and data type play no part.
TEST(TestMiopenConvApplicabilityChecks, BackwardDataSizeReadsOnlyTheDims)
{
    auto dy = channelsFirst(convYDims, DataType::DOUBLE);
    dy.nullStrides = true;

    EXPECT_EQ(sizeDeclineReason(dy, withStrides(convXDims, {7, 7, 7, 7}, DataType::UNSET)), "");
}

// Only validateConvBwdDataSize carries the limit: tensors of more than 2^31 - 1 elements pass the
// io-tensor rules of every operation when those rules hold, so backward weights (and forward) are
// not affected by the backward-data ticket.
TEST(TestMiopenConvApplicabilityChecks, SizeLimitIsNotPartOfTheIoTensorRules)
{
    const std::vector<int64_t> hugeDims = {1, MAX_ELEMENTS + 1, 1, 1};
    const std::vector<int64_t> unitDims = {1, 1, 1, 1};

    expectAccepted(
        {"dx huge", channelsFirst(unitDims), channelsFirst(unitDims), channelsFirst(hugeDims)});
    expectAccepted(
        {"dy huge", channelsFirst(hugeDims), channelsFirst(unitDims), channelsFirst(unitDims)});
}

// ---------------------------------------------------------------------------------------------
// Dims that overflow: every io validator declines them before packing or strides are computed
// ---------------------------------------------------------------------------------------------

// Every operation, every tensor in turn, under the tensor's name in the operation.
TEST(TestMiopenConvApplicabilityChecks, DeclinesDimsWhoseElementCountOverflows)
{
    const ConvCase packed = uniformCase({convXDims, convWDims, convYDims}, false);

    for(const auto& dims : ioOverflowingDims())
    {
        SCOPED_TRACE(vecToString(dims));
        const TensorSpec huge = withUnitStrides(dims);

        for(const Role role : ROLES)
        {
            for(const ConvOp op : ALL_OPS)
            {
                SCOPED_TRACE(operationName(op));
                SCOPED_TRACE(roleName(op, role));
                const std::string actual = declineReason(op, withTensorAt(packed, role, huge));
                EXPECT_THAT(actual, ::testing::HasSubstr(overflowReason(roleName(op, role))));
                EXPECT_THAT(actual, ::testing::HasSubstr(operationName(op)));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Strides that overflow: every io validator declines them before the packing check runs
// ---------------------------------------------------------------------------------------------

// Every operation, every tensor in turn, under the tensor's name in the operation.
TEST(TestMiopenConvApplicabilityChecks, DeclinesStridesWhoseArithmeticOverflows)
{
    const ConvCase packed = uniformCase({convXDims, convWDims, convYDims}, false);

    for(const auto& overflowing : overflowingStrides())
    {
        SCOPED_TRACE(describe(overflowing));
        const TensorSpec huge = withStrides(overflowing.dims, overflowing.strides);

        for(const Role role : ROLES)
        {
            for(const ConvOp op : ALL_OPS)
            {
                SCOPED_TRACE(operationName(op));
                SCOPED_TRACE(roleName(op, role));
                const std::string actual = declineReason(op, withTensorAt(packed, role, huge));
                EXPECT_THAT(actual, ::testing::HasSubstr(strideOverflowReason(roleName(op, role))));
                EXPECT_THAT(actual, ::testing::HasSubstr(operationName(op)));
            }
        }
    }
}

// Large strides whose arithmetic fits are judged as before, by layout and packing alone: a packed
// tensor of 2^40 elements is accepted, a non-packed channels-first tensor with large strides is
// declined by forward for not being packed and accepted by backward, and a tensor with a negative
// stride is declined by the layout rule, not for overflow.
TEST(TestMiopenConvApplicabilityChecks, StridesThatDoNotOverflowAreJudgedByLayoutAndPacking)
{
    const std::vector<int64_t> largeDims = {1, int64_t{1} << 40, 1};
    const TensorSpec largePacked = channelsFirst(largeDims);
    expectAccepted({"large packed", largePacked, largePacked, largePacked});

    const TensorSpec largePadded = withStrides({1, 4, 1}, {int64_t{1} << 60, int64_t{1} << 58, 1});
    ASSERT_FALSE(isTensorPacked(largePadded.dims, largePadded.strides));
    expectForwardDeclined(
        largePadded, channelsFirst(convWDims), channelsFirst(convYDims), PACKED_REASON);
    expectAccepted(
        {"large padded dy", largePadded, channelsFirst(convWDims), channelsFirst(convXDims)},
        BACKWARD_OPS);

    const TensorSpec reversed = withStrides({1, 3, 1}, {3, -1, 1});
    expectForwardDeclined(
        reversed, channelsFirst(convWDims), channelsFirst(convYDims), LAYOUT_REASON);
    expectDeclined({"reversed dy", reversed, channelsFirst(convWDims), channelsFirst(convXDims)},
                   LAYOUT_REASON,
                   BACKWARD_OPS);
}

// A negative dim has no element count: declined by name, not left to the stride arithmetic.
TEST(TestMiopenConvApplicabilityChecks, DeclinesNegativeDims)
{
    const TensorSpec negative = withUnitStrides({4, -2, 3, 3});
    const ConvCase packed = uniformCase({convXDims, convWDims, convYDims}, false);

    for(const Role role : ROLES)
    {
        for(const ConvOp op : ALL_OPS)
        {
            SCOPED_TRACE(operationName(op));
            EXPECT_THAT(declineReason(op, withTensorAt(packed, role, negative)),
                        ::testing::HasSubstr(std::string("tensor ") + roleName(op, role)
                                             + " has a negative dim"));
        }
    }
}
