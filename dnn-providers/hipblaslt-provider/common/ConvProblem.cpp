// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/ConvProblem.hpp"

#include <algorithm>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

namespace hipblaslt_plugin
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;

constexpr size_t MIN_RANK = 3;
constexpr size_t MAX_RANK = 5;

/// The node fields the three convolution attribute tables share.
struct ConvFields
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
    int64_t written = 0;
    const flatbuffers::Vector<int64_t>* prePadding = nullptr;
    const flatbuffers::Vector<int64_t>* postPadding = nullptr;
    const flatbuffers::Vector<int64_t>* stride = nullptr;
    const flatbuffers::Vector<int64_t>* dilation = nullptr;
    data_objects::ConvMode mode = data_objects::ConvMode::UNSET;
};

template <typename Attributes>
ConvFields commonFields(const Attributes& attributes)
{
    ConvFields fields;
    fields.prePadding = attributes.pre_padding();
    fields.postPadding = attributes.post_padding();
    fields.stride = attributes.stride();
    fields.dilation = attributes.dilation();
    fields.mode = attributes.conv_mode();
    return fields;
}

std::optional<ConvFields> convFields(const IGraph& graph, ConvDirection direction)
{
    if(graph.nodeCount() != 1 || graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }
    const auto& node = graph.getNodeWrapper(0);
    switch(direction)
    {
    case ConvDirection::FORWARD:
    {
        if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes)
        {
            return std::nullopt;
        }
        const auto& attributes = node.attributesAs<data_objects::ConvolutionFwdAttributes>();
        auto fields = commonFields(attributes);
        fields.x = attributes.x_tensor_uid();
        fields.w = attributes.w_tensor_uid();
        fields.y = attributes.y_tensor_uid();
        fields.written = fields.y;
        return fields;
    }
    case ConvDirection::BACKWARD_DATA:
    {
        if(node.attributesType() != data_objects::NodeAttributes::ConvolutionBwdAttributes)
        {
            return std::nullopt;
        }
        const auto& attributes = node.attributesAs<data_objects::ConvolutionBwdAttributes>();
        auto fields = commonFields(attributes);
        fields.x = attributes.dx_tensor_uid();
        fields.w = attributes.w_tensor_uid();
        fields.y = attributes.dy_tensor_uid();
        fields.written = fields.x;
        return fields;
    }
    case ConvDirection::BACKWARD_WEIGHTS:
    {
        if(node.attributesType() != data_objects::NodeAttributes::ConvolutionWrwAttributes)
        {
            return std::nullopt;
        }
        const auto& attributes = node.attributesAs<data_objects::ConvolutionWrwAttributes>();
        auto fields = commonFields(attributes);
        fields.x = attributes.x_tensor_uid();
        fields.w = attributes.dw_tensor_uid();
        fields.y = attributes.dy_tensor_uid();
        fields.written = fields.w;
        return fields;
    }
    default:
        break;
    }
    return std::nullopt;
}

/// The operand behind @p uid when it is real device memory of a supported element type and
/// rank, with positive extents and non-negative strides.
std::optional<ConvOperand>
    deviceOperand(const IGraph& graph, int64_t uid, data_objects::DataType& dataType)
{
    const auto& tensors = graph.getTensorMap();
    const auto found = tensors.find(uid);
    if(found == tensors.end() || found->second == nullptr)
    {
        return std::nullopt;
    }
    const auto& tensor = *found->second;
    if(tensor.virtual_() || tensor.is_runtime_pass_by_value()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor)
       || tensor.ragged_offset_tensor_uid().has_value() || tensor.ragged_offset_multiplier() != 1)
    {
        return std::nullopt;
    }
    dataType = tensor.data_type();
    if(dataType != data_objects::DataType::FLOAT && dataType != data_objects::DataType::HALF
       && dataType != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != strides->size()
       || dims->size() < MIN_RANK || dims->size() > MAX_RANK)
    {
        return std::nullopt;
    }
    ConvOperand operand{
        uid, {dims->begin(), dims->end()}, {strides->begin(), strides->end()}, std::nullopt, 0};
    for(size_t axis = 0; axis < operand.dims.size(); ++axis)
    {
        if(operand.dims[axis] < 1 || operand.strides[axis] < 0)
        {
            return std::nullopt;
        }
    }
    return operand;
}

/// True when no two elements of @p operand share an address.
bool hasDistinctElements(const ConvOperand& operand)
{
    std::vector<size_t> axes;
    for(size_t axis = 0; axis < operand.dims.size(); ++axis)
    {
        if(operand.dims[axis] > 1)
        {
            axes.push_back(axis);
        }
    }
    std::sort(axes.begin(), axes.end(), [&operand](size_t lhs, size_t rhs) {
        return operand.strides[lhs] < operand.strides[rhs];
    });
    int64_t span = 1;
    for(const auto axis : axes)
    {
        if(operand.strides[axis] < span
           || __builtin_mul_overflow(operand.strides[axis], operand.dims[axis], &span))
        {
            return false;
        }
    }
    return true;
}

std::optional<std::vector<int64_t>>
    convParameter(const flatbuffers::Vector<int64_t>* values, size_t size, int64_t minimum)
{
    if(values == nullptr || values->size() != size)
    {
        return std::nullopt;
    }
    std::vector<int64_t> result(values->begin(), values->end());
    for(const auto value : result)
    {
        if(value < minimum)
        {
            return std::nullopt;
        }
    }
    return result;
}

std::optional<int64_t> outputExtent(
    int64_t input, int64_t filter, int64_t prePad, int64_t postPad, int64_t stride, int64_t dil)
{
    int64_t reach = 0;
    int64_t padded = 0;
    if(__builtin_mul_overflow(dil, filter - 1, &reach)
       || __builtin_add_overflow(input, prePad, &padded)
       || __builtin_add_overflow(padded, postPad, &padded))
    {
        return std::nullopt;
    }
    const int64_t span = padded - reach - 1;
    if(span < 0)
    {
        return std::nullopt;
    }
    return span / stride + 1;
}

} // namespace

std::optional<ConvProblem> parseConvProblem(const IGraph& graph, ConvDirection direction)
{
    const auto fields = convFields(graph, direction);
    if(!fields.has_value())
    {
        return std::nullopt;
    }

    if(fields->mode != data_objects::ConvMode::CROSS_CORRELATION
       && fields->mode != data_objects::ConvMode::CONVOLUTION)
    {
        return std::nullopt;
    }
    if(fields->x == fields->w || fields->x == fields->y || fields->w == fields->y)
    {
        return std::nullopt;
    }

    auto xType = data_objects::DataType::UNSET;
    auto wType = data_objects::DataType::UNSET;
    auto yType = data_objects::DataType::UNSET;
    const auto x = deviceOperand(graph, fields->x, xType);
    const auto w = deviceOperand(graph, fields->w, wType);
    const auto y = deviceOperand(graph, fields->y, yType);
    if(!x.has_value() || !w.has_value() || !y.has_value() || xType != wType || xType != yType)
    {
        return std::nullopt;
    }

    // cuDNN's compute types for a convolution (cudnn-frontend docs/developer/graph-api.mdx,
    // "Compute Data Type"): FLOAT for every element type, HALF also for HALF; UNSET reads as
    // FLOAT. The GEMMs accumulate in f32, at least the precision either asks for.
    const auto computeType = graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT
       && !(computeType == data_objects::DataType::HALF && xType == data_objects::DataType::HALF))
    {
        return std::nullopt;
    }

    const size_t rank = x->dims.size();
    if(w->dims.size() != rank || y->dims.size() != rank)
    {
        return std::nullopt;
    }
    const ConvOperand& written
        = fields->written == fields->x ? *x : (fields->written == fields->w ? *w : *y);
    if(!hasDistinctElements(written))
    {
        return std::nullopt;
    }

    const size_t spatialRank = rank - 2;
    const auto prePadding = convParameter(fields->prePadding, spatialRank, 0);
    const auto postPadding = convParameter(fields->postPadding, spatialRank, 0);
    const auto stride = convParameter(fields->stride, spatialRank, 1);
    const auto dilation = convParameter(fields->dilation, spatialRank, 1);
    if(!prePadding.has_value() || !postPadding.has_value() || !stride.has_value()
       || !dilation.has_value())
    {
        return std::nullopt;
    }

    ConvProblem problem;
    problem.direction = direction;
    problem.dataType = xType;
    problem.mode = fields->mode;
    problem.n = x->dims[0];
    problem.c = x->dims[1];
    problem.k = w->dims[0];
    const int64_t cPerGroup = w->dims[1];
    if(problem.c % cPerGroup != 0)
    {
        return std::nullopt;
    }
    problem.groups = problem.c / cPerGroup;
    if(problem.k % problem.groups != 0 || y->dims[0] != problem.n || y->dims[1] != problem.k)
    {
        return std::nullopt;
    }

    for(size_t axis = 0; axis < spatialRank; ++axis)
    {
        const auto extent = outputExtent(x->dims[2 + axis],
                                         w->dims[2 + axis],
                                         (*prePadding)[axis],
                                         (*postPadding)[axis],
                                         (*stride)[axis],
                                         (*dilation)[axis]);
        if(!extent.has_value() || *extent != y->dims[2 + axis])
        {
            return std::nullopt;
        }
        problem.input.push_back(x->dims[2 + axis]);
        problem.output.push_back(y->dims[2 + axis]);
        problem.filter.push_back(w->dims[2 + axis]);
    }
    problem.prePadding = *prePadding;
    problem.postPadding = *postPadding;
    problem.stride = *stride;
    problem.dilation = *dilation;
    problem.x = *x;
    problem.w = *w;
    problem.y = *y;
    return problem;
}

} // namespace hipblaslt_plugin
