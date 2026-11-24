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
using test_conv_common::MIXED_LAYOUT_2D_DATA_TYPE;
using test_conv_common::MIXED_LAYOUT_2D_DX_DIMS;
using test_conv_common::MIXED_LAYOUT_2D_DX_STRIDES;
using test_conv_common::MIXED_LAYOUT_2D_DY_DIMS;
using test_conv_common::MIXED_LAYOUT_2D_DY_STRIDES;
using test_conv_common::MIXED_LAYOUT_2D_W_DIMS;
using test_conv_common::MIXED_LAYOUT_2D_W_STRIDES;
using test_conv_common::packedStrides;

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

// Each negative below breaks exactly one rule, and is matched to the wording of that rule's
// reason, so removing a check turns its tests red instead of being masked by another check.
constexpr const char* DATA_TYPE_REASON = "data type";
constexpr const char* LAYOUT_REASON = "channels-first or all channels-last";
constexpr const char* PACKED_REASON = "is not packed";
constexpr const char* GROUPED_REASON
    = "is not packed; a grouped convolution needs x, w and y packed";
constexpr const char* CHANNELS_LAST_3D_REASON
    = "is not packed; a 3D channels-last convolution needs x, w and y packed";
constexpr const char* SHARED_ADDRESS_REASON = "may have elements that share an address";

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

// Packed `dims` in channels-last or channels-first order.
TensorSpec packedTensor(const std::vector<int64_t>& dims,
                        bool channelsLastLayout,
                        DataType dataType = DataType::FLOAT)
{
    return channelsLastLayout ? channelsLast(dims, dataType) : channelsFirst(dims, dataType);
}

// The layout in the words of a failure trace.
const char* layoutLabel(bool channelsLastLayout)
{
    return channelsLastLayout ? "channels-last" : "channels-first";
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

// The shapes of `shapes` as a grouped convolution: w keeps K and takes half of x's channels, so
// there are 2 groups.
RankShapes groupedShapes(const RankShapes& shapes, int64_t groups = 2)
{
    RankShapes grouped = shapes;
    grouped.w[1] = shapes.x[1] / groups;
    return grouped;
}

// Shapes shared by the 2D layout, dtype and packing cases (and by the plan builder tests).
const std::vector<int64_t>& convXDims = test_conv_common::CONV_2D_X_DIMS;
const std::vector<int64_t>& convWDims = test_conv_common::CONV_2D_W_DIMS;
const std::vector<int64_t>& convYDims = test_conv_common::CONV_2D_Y_DIMS;

// The three shapes in the roles of uniformCase. Built on first use: the dims are inline variables
// of the shared header, whose initialization order against a namespace-scope copy is unspecified.
const RankShapes& convShapes()
{
    static const RankShapes s_SHAPES = {convXDims, convWDims, convYDims};
    return s_SHAPES;
}

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

// A tensor role in a failure trace, named as in backward data.
std::string roleTrace(Role role)
{
    return std::string("tensor ") + roleName(ConvOp::BACKWARD_DATA, role);
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
    return {"uniform",
            packedTensor(shapes.y, channelsLastLayout, dataType),
            packedTensor(shapes.w, channelsLastLayout, dataType),
            packedTensor(shapes.x, channelsLastLayout, dataType)};
}

// Strides of `dims` with the padding of one dimension: in the physical order of the layout
// (outermost first: N, C, spatial.. for channels-first; N, spatial.., C for channels-last) the
// extent of the dimension `level` positions from the innermost one counts `padding` elements more
// when the strides are laid out. Level 0 pads the innermost dimension (channels in channels-last,
// row length in channels-first), the highest useful level pads the planes. The strides keep the
// strict order of the layout, so the stride-order class is unchanged; every dim must have an
// extent above 1 for the tensor to be non-packed.
TensorSpec paddedTensor(const std::vector<int64_t>& dims,
                        bool channelsLastLayout,
                        size_t level,
                        int64_t padding = 1,
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
        stride *= dims[dim] + (position == level ? padding : 0);
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

// The reason a validator gives, as a function of the name that the offending tensor has in the
// operation, which differs by operation.
using ReasonOf = std::function<std::string(const char* name)>;

// Every operation of `ops` declines the tensors with the reason `reasonOf` gives for the name that
// the tensor at `offender` has in that operation. Reasons open with the operation name, so the
// match is on "<operation>: <reason>".
void expectDeclinedAtRole(const ConvCase& tensors,
                          Role offender,
                          const ReasonOf& reasonOf,
                          const std::vector<ConvOp>& ops = ALL_OPS)
{
    SCOPED_TRACE(tensors.name);
    for(const ConvOp op : ops)
    {
        const char* name = roleName(op, offender);
        SCOPED_TRACE(operationName(op));
        SCOPED_TRACE(name);
        EXPECT_THAT(declineReason(op, tensors),
                    ::testing::HasSubstr(std::string(operationName(op)) + ": " + reasonOf(name)));
    }
}

// The reason for a forward tensor `name` that is non-packed in a convolution that needs packed
// tensors, with the decline reason `rule` of that convolution.
ReasonOf packedRuleReason(const char* rule)
{
    return [rule](const char* name) { return std::string("tensor ") + name + " " + rule; };
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

std::string channelsLastNonPackedReason(const char* name)
{
    return std::string("tensor ") + name + " is channels-last and not packed";
}

std::string negativeDimReason(const char* name)
{
    return std::string("tensor ") + name + " has a negative dim";
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
            const std::string reason = test_conv_common::biasDataTypeReason(biasType, inputType);

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
    EXPECT_THAT(layoutDeclineReason(test_conv_common::cbaGraphWithFloatBiasOnHalfGraph()),
                ::testing::HasSubstr(test_conv_common::FLOAT_BIAS_ON_HALF_GRAPH_REASON));
    EXPECT_EQ(std::string(test_conv_common::FLOAT_BIAS_ON_HALF_GRAPH_REASON),
              test_conv_common::biasDataTypeReason(DataType::FLOAT, DataType::HALF));
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
            const std::string reason = test_conv_common::biasDataTypeReason(weightsType, inputType);

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
            SCOPED_TRACE("rank " + std::to_string(shapes.x.size()) + " "
                         + layoutLabel(channelsLastLayout));
            expectAccepted(uniformCase(shapes, channelsLastLayout));
        }
    }
}

TEST(TestMiopenConvApplicabilityChecks, AcceptsEverySupportedDataType)
{
    for(const auto dataType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const bool channelsLastLayout : {false, true})
        {
            SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType));
            expectAccepted(uniformCase(convShapes(), channelsLastLayout, dataType));
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
                    channelsLast({2, 1, 16}),
                    channelsLast({1, 16, 3}),
                    channelsLast({2, 16, 16})});
    expectAccepted({"1d single output channel and batch",
                    channelsLast({1, 1, 16}),
                    channelsLast({1, 16, 3}),
                    channelsLast({1, 16, 16})});

    // One input channel with an Rx1 filter: all three tensors carry extent-1 dimensions whose
    // strides tie.
    expectAccepted({"single input channel with Rx1 filter",
                    channelsLast({2, 16, 16, 1}),
                    channelsLast({16, 1, 3, 1}),
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

// A channels-first tensor with padded strides is accepted next to packed channels-first tensors:
// row-, channel- and plane-padded, at every position and rank.
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
                SCOPED_TRACE("rank " + std::to_string(dims.size()) + ", " + roleTrace(role)
                             + ", padded level " + std::to_string(level));
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
            const std::string layout = std::string(" ") + layoutLabel(channelsLastLayout);
            const ConvCase base = uniformCase(shapes, channelsLastLayout);
            const auto empty = [channelsLastLayout](std::vector<int64_t> dims) {
                dims[0] = 0;
                return packedTensor(dims, channelsLastLayout);
            };

            for(const Role role : ROLES)
            {
                SCOPED_TRACE("rank " + std::to_string(shapes.x.size()) + layout + ", "
                             + roleTrace(role) + " empty");
                expectAccepted(withTensorAt(base, role, empty(tensorAt(base, role).dims)));
            }

            SCOPED_TRACE("rank " + std::to_string(shapes.x.size()) + layout + ", empty batch");
            expectAccepted({"empty batch", empty(shapes.y), base.w, empty(shapes.x)});
        }
    }
}

// A zero dim is not declined for being zero but for what it does to the canonical strides:
// generateStrides gives stride 0 to every dim that is outer to the zero dim in the layout's memory
// order. For channels-first those are the dims before the zero dim; for channels-last they are the
// batch dim, plus the spatial dims when the channel dim is zero, or the earlier spatial dims when a
// spatial dim is zero. A dim of extent above 1 at stride 0 is a broadcast axis: the tensor is not
// packed and its stride order matches neither layout, so every operation declines it by the layout
// rule, whichever of the three tensors it is and in either layout. Without such a dim the tensor is
// accepted: a zero in the leading dim (above) and a zero whose outer dims allow it (next test).
// The first rows have a batch of extent above 1, which is always a broadcast axis. The explicit
// rows have a batch of 1: by hand, channels-last {1, 0, 8, 8} has strides {0, 1, 0, 0}, so its
// spatial dims of extent 8 are broadcast axes, and channels-first {1, 4, 0, 8} has {0, 0, 8, 1},
// so its channel dim of extent 4 is. This pins the current behaviour; it does not say that an
// empty channel or spatial extent has to be declined.
TEST(TestMiopenConvApplicabilityChecks, DeclinesEmptyTensorsWhoseZeroExtentIsNotLeading)
{
    for(const bool channelsLastLayout : {false, true})
    {
        const ConvCase base = uniformCase(convShapes(), channelsLastLayout);

        for(size_t zeroIndex = 1; zeroIndex < convXDims.size(); ++zeroIndex)
        {
            auto dims = convXDims;
            dims[zeroIndex] = 0;
            const TensorSpec empty = packedTensor(dims, channelsLastLayout);
            ASSERT_FALSE(isTensorPacked(empty.dims, empty.strides));

            for(const Role role : ROLES)
            {
                SCOPED_TRACE(std::string(layoutLabel(channelsLastLayout)) + " " + vecToString(dims)
                             + ", " + roleTrace(role));
                expectDeclined(withTensorAt(base, role, empty), LAYOUT_REASON);
            }
        }
    }

    struct ZeroStrideAxisCase
    {
        bool channelsLastLayout;
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
    };
    const std::vector<ZeroStrideAxisCase> unitBatchCases
        = {{true, {1, 0, 8, 8}, {0, 1, 0, 0}}, {false, {1, 4, 0, 8}, {0, 0, 8, 1}}};

    for(const auto& unitBatch : unitBatchCases)
    {
        const TensorSpec empty = withStrides(unitBatch.dims, unitBatch.strides);
        ASSERT_EQ(empty.strides, packedStrides(empty.dims, unitBatch.channelsLastLayout));
        ASSERT_FALSE(isTensorPacked(empty.dims, empty.strides));

        const ConvCase base = uniformCase(convShapes(), unitBatch.channelsLastLayout);

        for(const Role role : ROLES)
        {
            SCOPED_TRACE(std::string(layoutLabel(unitBatch.channelsLastLayout)) + " "
                         + vecToString(unitBatch.dims) + ", " + roleTrace(role));
            expectDeclined(withTensorAt(base, role, empty), LAYOUT_REASON);
        }
    }
}

// A zero dim whose outer dims in the layout's memory order all have extent 1 is accepted: the zero
// strides it leaves sit on extent-1 dims, so there is no broadcast axis, and the tensor is packed
// (element count 0 equals the span -1 plus 1) with the canonical strides of its layout. By hand,
// {1, 0, 8, 8} channels-first has strides {0, 64, 8, 1} and {1, 4, 0, 8} channels-last has
// {0, 1, 32, 4}. Every operation accepts such a tensor in each of the three roles. This pins the
// current behaviour.
TEST(TestMiopenConvApplicabilityChecks, AcceptsEmptyTensorsWithoutAZeroStrideAxis)
{
    struct UnitBatchCase
    {
        bool channelsLastLayout;
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
    };
    const std::vector<UnitBatchCase> cases
        = {{false, {1, 0, 8, 8}, {0, 64, 8, 1}}, {true, {1, 4, 0, 8}, {0, 1, 32, 4}}};

    for(const auto& unitBatch : cases)
    {
        const TensorSpec empty = withStrides(unitBatch.dims, unitBatch.strides);
        ASSERT_EQ(empty.strides, packedStrides(empty.dims, unitBatch.channelsLastLayout));
        ASSERT_TRUE(isTensorPacked(empty.dims, empty.strides));

        const ConvCase base = uniformCase(convShapes(), unitBatch.channelsLastLayout);

        for(const Role role : ROLES)
        {
            SCOPED_TRACE(std::string(layoutLabel(unitBatch.channelsLastLayout)) + " "
                         + vecToString(unitBatch.dims) + ", " + roleTrace(role));
            expectAccepted(withTensorAt(base, role, empty));
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

// The layout-mix bundle shapes that fail on MIOpen today, with their exact dims and strides
// including the extent-1 dimensions. The convolution-forward bundles (1D, 2D and 3D) are written
// in forward roles and run through forward; the backward-data bundles (1D, 2D and 3D) are written
// in backward-data roles and run through both backward operations. The first of those is the
// degenerate global convolution: dy [2,32,1] has only the channels-first canonical strides.
TEST(TestMiopenConvApplicabilityChecks, DeclinesMixedLayoutBundleShapes)
{
    struct BundleShapes
    {
        std::vector<ConvCase> cases;
        std::vector<ConvOp> ops;
    };

    const std::vector<BundleShapes> bundles = {
        {{
             forwardCase("1d bf16 x cf w cl y cf",
                         withStrides(test_conv_common::MIXED_LAYOUT_1D_DX_DIMS,
                                     test_conv_common::MIXED_LAYOUT_1D_DX_STRIDES,
                                     DataType::BFLOAT16),
                         withStrides(test_conv_common::MIXED_LAYOUT_1D_W_DIMS,
                                     test_conv_common::MIXED_LAYOUT_1D_W_STRIDES,
                                     DataType::BFLOAT16),
                         withStrides(test_conv_common::MIXED_LAYOUT_1D_DY_DIMS,
                                     test_conv_common::MIXED_LAYOUT_1D_DY_STRIDES,
                                     DataType::BFLOAT16)),
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
             forwardCase(
                 "3d fp16 single input channel x cf w cl y cl",
                 withStrides({2, 1, 16, 16, 16}, {4096, 4096, 256, 16, 1}, DataType::HALF),
                 withStrides({16, 1, 3, 3, 3}, {27, 1, 9, 3, 1}, DataType::HALF),
                 withStrides({2, 16, 16, 16, 16}, {65536, 1, 4096, 256, 16}, DataType::HALF)),
         },
         FORWARD_OPS},
        {{
             {"1d half global convolution, dy cf, w cl, dx cl",
              withStrides({2, 32, 1}, {32, 1, 1}, DataType::HALF),
              withStrides({32, 16, 16}, {256, 1, 16}, DataType::HALF),
              withStrides({2, 16, 16}, {256, 1, 16}, DataType::HALF)},
             {"2d bf16, dy cf, w cl, dx cf",
              withStrides(
                  MIXED_LAYOUT_2D_DY_DIMS, MIXED_LAYOUT_2D_DY_STRIDES, MIXED_LAYOUT_2D_DATA_TYPE),
              withStrides(
                  MIXED_LAYOUT_2D_W_DIMS, MIXED_LAYOUT_2D_W_STRIDES, MIXED_LAYOUT_2D_DATA_TYPE),
              withStrides(
                  MIXED_LAYOUT_2D_DX_DIMS, MIXED_LAYOUT_2D_DX_STRIDES, MIXED_LAYOUT_2D_DATA_TYPE)},
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
         },
         BACKWARD_OPS},
    };

    for(const auto& bundle : bundles)
    {
        for(const auto& bundleCase : bundle.cases)
        {
            expectDeclined(bundleCase, LAYOUT_REASON, bundle.ops);
        }
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

                expectDeclinedAtRole(withTensorAt(base, role, padded),
                                     role,
                                     channelsLastNonPackedReason,
                                     BACKWARD_OPS);
            }
        }
    }
}

// A tensor with a broadcast axis (an extent above 1 with stride 0) is never packed
// (isTensorPacked). Forward declines it because two of its elements share an address. Backward
// classifies it by its stride order and accepts it, as a non-packed channels-first tensor, next
// to packed channels-first tensors; that pins how a zero-stride axis is classified through
// isTensorPacked and the backward rules and does not say that MIOpen computes such a backward
// convolution correctly.
TEST(TestMiopenConvApplicabilityChecks, ForwardDeclinesBroadcastAxisBackwardAcceptsIt)
{
    const ConvCase base = uniformCase(convShapes(), /*channelsLastLayout=*/false);

    for(const Role role : ROLES)
    {
        auto strides = tensorAt(base, role).strides;
        strides.back() = 0;
        const TensorSpec broadcast = withStrides(tensorAt(base, role).dims, strides);
        ASSERT_FALSE(isTensorPacked(broadcast.dims, broadcast.strides));

        SCOPED_TRACE(roleTrace(role) + " broadcast");
        expectDeclined(withTensorAt(base, role, broadcast), SHARED_ADDRESS_REASON, FORWARD_OPS);
        expectAccepted(withTensorAt(base, role, broadcast), BACKWARD_OPS);
    }
}

// Forward accepts a non-packed tensor of an ungrouped convolution that is not 3D channels-last:
// row-, channel- and plane-padded, every tensor in turn, at ranks 3 and 4 in both layouts and at
// rank 5 channels-first.
TEST(TestMiopenConvApplicabilityChecks, ForwardAcceptsNonPackedTensorsOfUngroupedConvolutions)
{
    for(const auto& shapes : rankShapes())
    {
        for(const bool channelsLastLayout : {false, true})
        {
            if(channelsLastLayout && shapes.x.size() == 5)
            {
                continue;
            }
            const ConvCase base = uniformCase(shapes, channelsLastLayout);

            for(const Role role : ROLES)
            {
                const auto& dims = tensorAt(base, role).dims;

                for(size_t level = 0; level + 1 < dims.size(); ++level)
                {
                    SCOPED_TRACE("rank " + std::to_string(dims.size()) + ", "
                                 + layoutLabel(channelsLastLayout) + ", " + roleTrace(role)
                                 + ", padded level " + std::to_string(level));
                    const TensorSpec padded = paddedTensor(dims, channelsLastLayout, level);
                    ASSERT_FALSE(isTensorPacked(padded.dims, padded.strides));

                    expectAccepted(withTensorAt(base, role, padded), FORWARD_OPS);
                }
            }
        }
    }
}

// A non-packed 3D channels-last forward convolution is declined, whichever tensor is padded and
// wherever: MIOpen reuses the invoker of a same-shape packed problem in either order
// (ALMIOPEN-3356). The packed one stays accepted.
TEST(TestMiopenConvApplicabilityChecks, ForwardDeclinesNonPacked3dChannelsLast)
{
    const RankShapes shapes = rankShapes().back();
    ASSERT_EQ(shapes.x.size(), 5U);
    const ConvCase base = uniformCase(shapes, /*channelsLastLayout=*/true);
    expectAccepted(base, FORWARD_OPS);

    for(const Role role : ROLES)
    {
        const auto& dims = tensorAt(base, role).dims;

        for(size_t level = 0; level + 1 < dims.size(); ++level)
        {
            SCOPED_TRACE(roleTrace(role) + ", padded level " + std::to_string(level));
            const TensorSpec padded = paddedTensor(dims, /*channelsLastLayout=*/true, level);
            ASSERT_FALSE(isTensorPacked(padded.dims, padded.strides));

            expectDeclinedAtRole(withTensorAt(base, role, padded),
                                 role,
                                 packedRuleReason(CHANNELS_LAST_3D_REASON),
                                 FORWARD_OPS);
        }
    }
}

// A non-packed tensor of a grouped forward convolution is declined at every rank and in both
// layouts, whichever tensor is padded and wherever (ALMIOPEN-3013): MIOpen gets it wrong even run
// alone. The packed grouped convolution stays accepted, and so does a depthwise one; a depthwise
// one with a padded input is declined.
TEST(TestMiopenConvApplicabilityChecks, ForwardDeclinesNonPackedTensorsOfGroupedConvolutions)
{
    for(const auto& shapes : rankShapes())
    {
        for(const bool channelsLastLayout : {false, true})
        {
            const ConvCase base = uniformCase(groupedShapes(shapes), channelsLastLayout);
            SCOPED_TRACE("rank " + std::to_string(shapes.x.size()) + ", "
                         + layoutLabel(channelsLastLayout));
            expectAccepted(base, FORWARD_OPS);

            for(const Role role : ROLES)
            {
                const auto& dims = tensorAt(base, role).dims;

                for(size_t level = 0; level + 1 < dims.size(); ++level)
                {
                    SCOPED_TRACE(roleTrace(role) + ", padded level " + std::to_string(level));
                    const TensorSpec padded = paddedTensor(dims, channelsLastLayout, level);
                    ASSERT_FALSE(isTensorPacked(padded.dims, padded.strides));

                    expectDeclinedAtRole(withTensorAt(base, role, padded),
                                         role,
                                         packedRuleReason(GROUPED_REASON),
                                         FORWARD_OPS);
                }
            }

            // One channel per group, and as many output channels as input channels.
            RankShapes depthwiseShapes = shapes;
            depthwiseShapes.w[0] = shapes.x[1];
            depthwiseShapes.w[1] = 1;
            depthwiseShapes.y[1] = shapes.x[1];
            const ConvCase depthwise = uniformCase(depthwiseShapes, channelsLastLayout);
            expectAccepted(depthwise, FORWARD_OPS);
            const TensorSpec paddedInput = paddedTensor(depthwiseShapes.x, channelsLastLayout, 0);
            expectDeclinedAtRole(withTensorAt(depthwise, Role::DX, paddedInput),
                                 Role::DX,
                                 packedRuleReason(GROUPED_REASON),
                                 FORWARD_OPS);
        }
    }
}

// Forward declines a non-packed tensor two of whose elements share an address, and accepts one
// whose elements are disjoint although no stride is 1 or the batch stride leaves a gap.
TEST(TestMiopenConvApplicabilityChecks, ForwardDeclinesNonPackedTensorsWhoseElementsShareAnAddress)
{
    const ConvCase base = uniformCase(convShapes(), /*channelsLastLayout=*/false);

    // H steps by 4 over rows 8 long: rows overlap. The stride order stays channels-first.
    const TensorSpec overlappingRows = withStrides(convXDims, {256, 64, 4, 1});
    ASSERT_FALSE(isTensorPacked(overlappingRows.dims, overlappingRows.strides));
    expectDeclinedAtRole(withTensorAt(base, Role::DX, overlappingRows),
                         Role::DX,
                         packedRuleReason(SHARED_ADDRESS_REASON),
                         FORWARD_OPS);

    // W steps by 4 over 8 columns while H steps by 8: rows interleave. The stride order stays
    // channels-first.
    const TensorSpec interleaved = withStrides(convXDims, {512, 64, 8, 4});
    ASSERT_FALSE(isTensorPacked(interleaved.dims, interleaved.strides));
    expectDeclinedAtRole(withTensorAt(base, Role::DX, interleaved),
                         Role::DX,
                         packedRuleReason(SHARED_ADDRESS_REASON),
                         FORWARD_OPS);

    const TensorSpec innermostStride2 = withStrides(convXDims, {512, 128, 16, 2});
    const TensorSpec batchGap = withStrides(convXDims, {600, 64, 8, 1});
    for(const auto& disjoint : {innermostStride2, batchGap})
    {
        SCOPED_TRACE(vecToString(disjoint.strides));
        ASSERT_FALSE(isTensorPacked(disjoint.dims, disjoint.strides));
        expectAccepted(withTensorAt(base, Role::DX, disjoint), FORWARD_OPS);
    }
}

// A packed tensor can still have elements that share an address: isTensorPacked compares only the
// span of the strides with the element count. These strides decrease strictly, so the strict
// layout path accepts their layout, and forward must decline them for the shared address, in the
// x role and in the y role. The canonical packed graph stays accepted.
TEST(TestMiopenConvApplicabilityChecks, ForwardDeclinesPackedTensorsWhoseElementsShareAnAddress)
{
    const ConvCase base = uniformCase(convShapes(), /*channelsLastLayout=*/false);
    expectAccepted(base, FORWARD_OPS);

    // W steps by 4 over 8 columns while H steps by 5: rows overlap. Span 512 = 2 * 4 * 8 * 8.
    const TensorSpec overlappingInput = withStrides(convXDims, {256, 64, 5, 4});
    ASSERT_TRUE(isTensorPacked(overlappingInput.dims, overlappingInput.strides));
    expectDeclinedAtRole(withTensorAt(base, Role::DX, overlappingInput),
                         Role::DX,
                         packedRuleReason(SHARED_ADDRESS_REASON),
                         FORWARD_OPS);

    // W steps by 3 over 6 columns while H steps by 4. Span 288 = 2 * 4 * 6 * 6.
    const TensorSpec overlappingOutput = withStrides(convYDims, {144, 36, 4, 3});
    ASSERT_TRUE(isTensorPacked(overlappingOutput.dims, overlappingOutput.strides));
    expectDeclinedAtRole(withTensorAt(base, Role::DY, overlappingOutput),
                         Role::DY,
                         packedRuleReason(SHARED_ADDRESS_REASON),
                         FORWARD_OPS);
}

// MIOpen names no layout above rank 5, so the strict layout path stops there: a batch-1 channel
// slice of rank 5 is accepted and the same slice of rank 6 is declined by the layout rule.
TEST(TestMiopenConvApplicabilityChecks, ForwardStrictLayoutStopsAtRankFive)
{
    const TensorSpec slice5 = withStrides({1, 4, 2, 2, 2}, {64, 8, 4, 2, 1});
    ASSERT_TRUE(isTensorPacked(slice5.dims, slice5.strides));
    expectAccepted(forwardCase("rank 5 batch-1 slice",
                               slice5,
                               channelsFirst({4, 4, 1, 1, 1}),
                               channelsFirst({1, 4, 2, 2, 2})),
                   FORWARD_OPS);

    const TensorSpec slice6 = withStrides({1, 4, 2, 2, 2, 2}, {128, 16, 8, 4, 2, 1});
    ASSERT_TRUE(isTensorPacked(slice6.dims, slice6.strides));
    expectForwardDeclined(slice6,
                          channelsFirst({4, 4, 1, 1, 1, 1}),
                          channelsFirst({1, 4, 2, 2, 2, 2}),
                          LAYOUT_REASON);
}

// The span of the dimension with the largest stride may overflow a 64-bit integer without any
// element sharing an address: no other dimension has to fit beyond it. A non-packed channels-last
// output whose length span is 14 * 7e17 is accepted by forward, not declined for overflow or for a
// shared address.
TEST(TestMiopenConvApplicabilityChecks, ForwardAcceptsDisjointTensorWhoseOutermostSpanOverflows)
{
    constexpr int64_t LENGTH_STRIDE = 700000000000000000;
    const ConvCase base = uniformCase(rankShapes().front(), /*channelsLastLayout=*/true);
    const TensorSpec output = withStrides({1, 4, 14}, {LENGTH_STRIDE + 1, 1, LENGTH_STRIDE});
    ASSERT_FALSE(isTensorPacked(output.dims, output.strides));
    ASSERT_GT(LENGTH_STRIDE, std::numeric_limits<int64_t>::max() / 14);

    expectAccepted(withTensorAt(base, Role::DY, output), FORWARD_OPS);
}

// Packed tensors whose strides differ from the canonical ones only on extent-1 dimensions, and
// which MIOpen's strict layout check places in one layout, are accepted by forward: a 1x1
// convolution on a [N, C, 1, 1] channels-first input with the channels-last output PyTorch
// allocates for it (1D, 2D and 3D), and a batch-1 input sliced from a wider channel dimension.
// Backward has no such acceptance and declines them as a mixed layout.
TEST(TestMiopenConvApplicabilityChecks, ForwardAcceptsPackedTensorsStrictlyInOneLayout)
{
    const std::vector<ConvCase> cases = {
        forwardCase("2d squeeze-excitation, y channels-last",
                    withStrides({8, 16, 1, 1}, {16, 1, 1, 1}),
                    withStrides({4, 16, 1, 1}, {16, 1, 1, 1}),
                    withStrides({8, 4, 1, 1}, {4, 1, 4, 4})),
        forwardCase("1d squeeze-excitation, y channels-last",
                    withStrides({8, 16, 1}, {16, 1, 1}),
                    withStrides({4, 16, 1}, {16, 1, 1}),
                    withStrides({8, 4, 1}, {4, 1, 4})),
        forwardCase("3d squeeze-excitation, y channels-last",
                    withStrides({8, 16, 1, 1, 1}, {16, 1, 1, 1, 1}),
                    withStrides({4, 16, 1, 1, 1}, {16, 1, 1, 1, 1}),
                    withStrides({8, 4, 1, 1, 1}, {4, 1, 4, 4, 4})),
        forwardCase("batch-1 slice of 8 channels",
                    withStrides({1, 4, 8, 8}, {512, 64, 8, 1}),
                    channelsFirst(convWDims),
                    channelsFirst({1, 4, 6, 6})),
    };

    for(const auto& strictCase : cases)
    {
        for(const auto& tensor : {strictCase.dy, strictCase.w, strictCase.dx})
        {
            ASSERT_TRUE(isTensorPacked(tensor.dims, tensor.strides));
        }
        expectAccepted(strictCase, FORWARD_OPS);
        expectDeclined(strictCase, LAYOUT_REASON, BACKWARD_OPS);
    }
}

// Graphs whose tensors share a layout only once the strides of extent-1 dimensions are ignored
// stay declined by forward, as do strict layouts with a non-packed tensor. MIOpen returned wrong
// output (0.4-0.7% of the elements) for the depthwise one and ran for over 10 minutes on a
// [4, 1, 224, 224] input of the single-channel one.
TEST(TestMiopenConvApplicabilityChecks, ForwardDeclinesLayoutsThatOnlyIgnoringExtentOneStridesUnify)
{
    expectForwardDeclined(withStrides({8, 16, 1, 1}, {16, 1, 1, 1}),
                          withStrides({16, 1, 3, 3}, {9, 9, 3, 1}),
                          withStrides({8, 16, 1, 1}, {16, 1, 16, 16}),
                          LAYOUT_REASON);
    expectForwardDeclined(withStrides({8, 1, 28, 28}, {784, 784, 28, 1}),
                          withStrides({32, 1, 3, 3}, {9, 9, 3, 1}),
                          channelsLast({8, 32, 26, 26}),
                          LAYOUT_REASON);

    // The squeeze-excitation shape above with a gap after each batch: strictly channels-last, but
    // not packed.
    const TensorSpec gappedInput = withStrides({8, 16, 1, 1}, {32, 1, 1, 1});
    ASSERT_FALSE(isTensorPacked(gappedInput.dims, gappedInput.strides));
    expectForwardDeclined(gappedInput,
                          withStrides({4, 16, 1, 1}, {16, 1, 1, 1}),
                          withStrides({8, 4, 1, 1}, {4, 1, 4, 4}),
                          LAYOUT_REASON);
}

// ---------------------------------------------------------------------------------------------
// Declined: malformed tensors
// ---------------------------------------------------------------------------------------------

TEST(TestMiopenConvApplicabilityChecks, DeclinesMalformedTensors)
{
    auto nullDyDims = uniformCase(convShapes(), false);
    nullDyDims.dy.nullDims = true;
    expectDeclined(nullDyDims, "null dims");

    auto nullDxDims = uniformCase(convShapes(), false);
    nullDxDims.dx.nullDims = true;
    expectDeclined(nullDxDims, "null dims");

    auto nullStrides = uniformCase(convShapes(), false);
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
    const ConvCase packed = uniformCase(convShapes(), false);

    for(const auto& dims : ioOverflowingDims())
    {
        SCOPED_TRACE(vecToString(dims));
        const TensorSpec huge = withUnitStrides(dims);

        for(const Role role : ROLES)
        {
            expectDeclinedAtRole(withTensorAt(packed, role, huge), role, overflowReason);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Strides that overflow: every io validator declines them before the packing check runs
// ---------------------------------------------------------------------------------------------

// Every operation, every tensor in turn, under the tensor's name in the operation.
TEST(TestMiopenConvApplicabilityChecks, DeclinesStridesWhoseArithmeticOverflows)
{
    const ConvCase packed = uniformCase(convShapes(), false);

    for(const auto& overflowing : overflowingStrides())
    {
        SCOPED_TRACE(describe(overflowing));
        const TensorSpec huge = withStrides(overflowing.dims, overflowing.strides);

        for(const Role role : ROLES)
        {
            expectDeclinedAtRole(withTensorAt(packed, role, huge), role, strideOverflowReason);
        }
    }
}

// The overflow check forms the partial sums of (dim - 1) * stride in dim order, as isTensorPacked
// does, so which strides overflow depends on the order. With dims [2, 2, 2] every term is the
// stride itself.
// - [INT64_MAX, 1, -INT64_MAX]: the partial sums are INT64_MAX, then INT64_MAX + 1, which
//   overflows, so every operation declines it for overflow although the full sum is 1 and a
//   reverse-order replay would accept it. isTensorPacked is not asked for it, as it would overflow.
// - [-INT64_MAX, 1, INT64_MAX]: the partial sums are -INT64_MAX, -INT64_MAX + 1, then 1, and the
//   final sum plus 1 is 2, so nothing overflows and a reverse-order replay (INT64_MAX, then
//   INT64_MAX + 1) would decline it for overflow. The element count 8 is not the span 2, so the
//   tensor is not packed, and its stride order [2, 1, 0] is neither layout, so every operation
//   declines it by the layout rule instead.
TEST(TestMiopenConvApplicabilityChecks, StrideOverflowFollowsThePackingSumOrder)
{
    constexpr int64_t MAX_VALUE = std::numeric_limits<int64_t>::max();

    const ConvCase packed = uniformCase(convShapes(), false);
    const TensorSpec partialSumOverflows = withStrides({2, 2, 2}, {MAX_VALUE, 1, -MAX_VALUE});
    const TensorSpec partialSumsFit = withStrides({2, 2, 2}, {-MAX_VALUE, 1, MAX_VALUE});
    ASSERT_FALSE(isTensorPacked(partialSumsFit.dims, partialSumsFit.strides));

    for(const Role role : ROLES)
    {
        expectDeclinedAtRole(
            withTensorAt(packed, role, partialSumOverflows), role, strideOverflowReason);

        for(const ConvOp op : ALL_OPS)
        {
            SCOPED_TRACE(operationName(op));
            SCOPED_TRACE(roleName(op, role));

            const std::string fitting
                = declineReason(op, withTensorAt(packed, role, partialSumsFit));
            EXPECT_THAT(fitting, ::testing::HasSubstr(LAYOUT_REASON));
            EXPECT_THAT(
                fitting,
                ::testing::Not(::testing::HasSubstr(strideOverflowReason(roleName(op, role)))));
        }
    }
}

// Large strides whose arithmetic fits are judged as before, by layout and packing alone: a packed
// tensor of 2^40 elements is accepted, a non-packed channels-first tensor with large strides is
// declined by forward for not being packed in a grouped convolution and accepted by backward, and
// a tensor with a negative stride is declined by the layout rule, not for overflow.
TEST(TestMiopenConvApplicabilityChecks, StridesThatDoNotOverflowAreJudgedByLayoutAndPacking)
{
    const std::vector<int64_t> largeDims = {1, int64_t{1} << 40, 1};
    const TensorSpec largePacked = channelsFirst(largeDims);
    expectAccepted({"large packed", largePacked, largePacked, largePacked});

    const TensorSpec largePadded = withStrides({1, 4, 1}, {int64_t{1} << 60, int64_t{1} << 58, 1});
    ASSERT_FALSE(isTensorPacked(largePadded.dims, largePadded.strides));
    expectForwardDeclined(largePadded,
                          channelsFirst(groupedShapes(convShapes()).w),
                          channelsFirst(convYDims),
                          GROUPED_REASON);
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

// Strides on the last values the packing arithmetic can represent are not declined for overflow:
// the validators judge them by layout and packing, after the overflow bound. Each sits one step
// from a declined row of overflowingStrides. The other tensors are those of a grouped
// convolution, where forward needs every tensor packed.
// - [2, 1, 1] with [INT64_MAX - 1, 1, 1]: the factor (2 - 1) is exactly INT64_MAX / (INT64_MAX - 1)
//   and the final sum (INT64_MAX - 1) + 1 is exactly INT64_MAX. The span is not the element count
//   2, so the tensor is not packed; its stride order is channels-first, so forward declines it as
//   not packed and backward accepts it.
// - [2, 1, 1] with [INT64_MIN, 1, 1]: the term (2 - 1) * INT64_MIN is exactly INT64_MIN, as is
//   the sum 0 + INT64_MIN. The tensor is not packed and its stride order (the two unit strides
//   before INT64_MIN) is neither layout, so every operation declines it by the layout rule, not
//   for overflow.
TEST(TestMiopenConvApplicabilityChecks, StridesOnTheRepresentableLimitAreJudgedByLayoutAndPacking)
{
    constexpr int64_t MAX_VALUE = std::numeric_limits<int64_t>::max();
    constexpr int64_t MIN_VALUE = std::numeric_limits<int64_t>::min();

    const ConvCase base = uniformCase(groupedShapes(convShapes()), false);
    const TensorSpec spanReachesMax = withStrides({2, 1, 1}, {MAX_VALUE - 1, 1, 1});
    const TensorSpec termIsMin = withStrides({2, 1, 1}, {MIN_VALUE, 1, 1});
    ASSERT_FALSE(isTensorPacked(spanReachesMax.dims, spanReachesMax.strides));
    ASSERT_FALSE(isTensorPacked(termIsMin.dims, termIsMin.strides));

    for(const Role role : ROLES)
    {
        SCOPED_TRACE(roleTrace(role));

        expectDeclined(withTensorAt(base, role, spanReachesMax), GROUPED_REASON, FORWARD_OPS);
        expectAccepted(withTensorAt(base, role, spanReachesMax), BACKWARD_OPS);

        expectDeclined(withTensorAt(base, role, termIsMin), LAYOUT_REASON);
    }
}

// The overflow check before isTensorPacked (the replay of its arithmetic) must not decline strides
// whose arithmetic fits, and the packed verdict is isTensorPacked's own, which the validators read
// from it directly. This pins that for rows none of which is near a limit: in a grouped
// convolution, where every tensor must be packed, forward declines a tensor as "not packed"
// exactly when isTensorPacked is false for it, and never for overflow, whichever of x, w and y it
// is, at ranks 3, 4 and 5 and on the 2D shape of the other tests. It does not pin the replay's
// arithmetic against the SDK function (the overflow rows do, above). The other two tensors are
// packed in the layout class of the row so that the layout rule does not pre-empt the packing
// rule; the reordered rows are packed and are declined by the layout rule instead, so they must
// not be declined as not packed. The padded rows pad every position of both layouts by 1, 2, 4
// and 8 elements: a padded tensor keeps the stride order of its layout, so forward declines it
// for packing alone.
TEST(TestMiopenConvApplicabilityChecks, PackingVerdictAgreesWithIsTensorPacked)
{
    struct PackingRow
    {
        std::string name;
        std::vector<int64_t> strides;
        bool channelsLastClass;
        bool packed;
    };
    const auto gapped = [](std::vector<int64_t> strides) {
        for(auto& stride : strides)
        {
            stride *= 2;
        }
        return strides;
    };
    const auto rowsOf = [&gapped](const std::vector<int64_t>& dims) {
        const size_t rank = dims.size();

        std::vector<int64_t> reorderedOrder(rank);
        for(size_t i = 0; i < rank; ++i)
        {
            reorderedOrder[i] = static_cast<int64_t>(i);
        }

        std::vector<PackingRow> rows = {
            {"canonical channels-first", packedStrides(dims, false), false, true},
            {"canonical channels-last", packedStrides(dims, true), true, true},
            {"channels-first with gaps", gapped(packedStrides(dims, false)), false, false},
            {"channels-last with gaps", gapped(packedStrides(dims, true)), true, false},
            {"reordered", generateStrides(dims, reorderedOrder), false, true},
        };
        for(const bool channelsLastClass : {false, true})
        {
            for(size_t level = 0; level + 1 < rank; ++level)
            {
                for(const int64_t padding : {1, 2, 4, 8})
                {
                    rows.push_back({std::string(layoutLabel(channelsLastClass))
                                        + " padded at level " + std::to_string(level) + " by "
                                        + std::to_string(padding),
                                    paddedTensor(dims, channelsLastClass, level, padding).strides,
                                    channelsLastClass,
                                    false});
                }
            }
        }
        return rows;
    };

    std::vector<RankShapes> allShapes;
    for(const auto& shapes : rankShapes())
    {
        allShapes.push_back(groupedShapes(shapes));
    }
    allShapes.push_back(groupedShapes(convShapes()));

    for(const auto& shapes : allShapes)
    {
        const ConvCase uniform = uniformCase(shapes, false);

        for(const Role role : ROLES)
        {
            const auto& dims = tensorAt(uniform, role).dims;
            const char* name = roleName(ConvOp::FORWARD, role);

            for(const auto& row : rowsOf(dims))
            {
                SCOPED_TRACE("rank " + std::to_string(dims.size()) + ", " + roleTrace(role) + ", "
                             + row.name + ", strides " + vecToString(row.strides));
                const TensorSpec tensor = withStrides(dims, row.strides);
                ASSERT_EQ(isTensorPacked(tensor.dims, tensor.strides), row.packed);

                const std::string reason = declineReason(
                    ConvOp::FORWARD,
                    withTensorAt(uniformCase(shapes, row.channelsLastClass), role, tensor));

                EXPECT_EQ(reason.find(PACKED_REASON) != std::string::npos, !row.packed);
                if(!row.packed)
                {
                    EXPECT_THAT(reason,
                                ::testing::HasSubstr(std::string(operationName(ConvOp::FORWARD))
                                                     + ": tensor " + name + " is not packed"));
                }
                EXPECT_THAT(reason,
                            ::testing::Not(::testing::HasSubstr(strideOverflowReason(name))));
                EXPECT_THAT(reason, ::testing::Not(::testing::HasSubstr(overflowReason(name))));
            }
        }
    }
}

// Dims whose element count is INT64_MAX - 1 are not declined as overflowing: the product
// 3 * 3074457345618258602 is INT64_MAX - 1 and INT64_MAX / 3 is exactly 3074457345618258602, so
// the last factor lands on the division that bounds the element count. The canonical
// channels-first strides [INT64_MAX - 1, 3074457345618258602, 1] give terms and sums that all
// fit, and the tensor is packed, so every operation accepts it.
TEST(TestMiopenConvApplicabilityChecks, DimsWhoseElementCountIsJustBelowTheLimitAreAccepted)
{
    constexpr int64_t FACTOR = 3074457345618258602;
    static_assert(3 * FACTOR == std::numeric_limits<int64_t>::max() - 1);
    static_assert(std::numeric_limits<int64_t>::max() / 3 == FACTOR);

    const TensorSpec nearlyUnbounded = channelsFirst({1, 3, FACTOR});
    ASSERT_TRUE(isTensorPacked(nearlyUnbounded.dims, nearlyUnbounded.strides));
    expectAccepted(
        {"element count just below the limit", nearlyUnbounded, nearlyUnbounded, nearlyUnbounded});
}

// A negative dim has no element count: declined by name, not left to the stride arithmetic.
TEST(TestMiopenConvApplicabilityChecks, DeclinesNegativeDims)
{
    const TensorSpec negative = withUnitStrides({4, -2, 3, 3});
    const ConvCase packed = uniformCase(convShapes(), false);

    for(const Role role : ROLES)
    {
        expectDeclinedAtRole(withTensorAt(packed, role, negative), role, negativeDimReason);
    }
}
