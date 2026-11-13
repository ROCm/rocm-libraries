// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvComputeType.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_naive
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using pack_features::checkedMultiply;

/// Tensor ranks the kernel addresses: N, C and one to three spatial axes.
constexpr size_t MIN_RANK = 3;
constexpr size_t MAX_RANK = 5;

/// A validated operand: rank 3 to 5, every extent positive, every stride non-negative.
struct Operand
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

/// The node fields the three convolution attribute tables share, read once.
struct ConvFields
{
    ConvNaiveOperands operands;
    int64_t writtenUid = 0;
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

/// The node's operands in forward roles, and which of them the kernel writes, or nullopt
/// when the graph is not one convolution node of @p direction.
std::optional<ConvFields> convFields(const MatchContext& context, ConvDirection direction)
{
    // One node per graph, and shapes fixed at plan build: the argument block is computed from
    // the graph's dims, which an override-shape graph would change at execute.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }

    const auto& node = context.graph.getNodeWrapper(0);
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
        fields.operands
            = {attributes.x_tensor_uid(), attributes.w_tensor_uid(), attributes.y_tensor_uid()};
        fields.writtenUid = attributes.y_tensor_uid();
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
        fields.operands
            = {attributes.dx_tensor_uid(), attributes.w_tensor_uid(), attributes.dy_tensor_uid()};
        fields.writtenUid = attributes.dx_tensor_uid();
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
        fields.operands
            = {attributes.x_tensor_uid(), attributes.dw_tensor_uid(), attributes.dy_tensor_uid()};
        fields.writtenUid = attributes.dw_tensor_uid();
        return fields;
    }
    default:
        break;
    }
    return std::nullopt;
}

int64_t elementBytes(data_objects::DataType dataType)
{
    return dataType == data_objects::DataType::FLOAT ? 4 : 2;
}

/// The operand behind @p uid when the kernel can read or write it: real device memory of a
/// supported element type and rank, with a positive extent on every axis.
std::optional<Operand> deviceOperand(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto found = tensors.find(uid);
    if(found == tensors.end() || found->second == nullptr)
    {
        return std::nullopt;
    }
    const auto& tensor = *found->second;

    // Device data the launch can address: not virtual, not a host scalar, not ragged.
    if(tensor.virtual_() || tensor.is_runtime_pass_by_value()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor)
       || tensor.ragged_offset_tensor_uid().has_value() || tensor.ragged_offset_multiplier() != 1)
    {
        return std::nullopt;
    }

    const auto dataType = tensor.data_type();
    if(dataType != data_objects::DataType::FLOAT && dataType != data_objects::DataType::HALF
       && dataType != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }

    // The kernel issues one scalar access per element; a pointer guaranteed less aligned
    // than an element is not one it was written for.
    if(tensor.alignment() < elementBytes(dataType))
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

    Operand operand;
    operand.dataType = dataType;
    operand.dims.assign(dims->begin(), dims->end());
    operand.strides.assign(strides->begin(), strides->end());
    for(size_t axis = 0; axis < operand.dims.size(); ++axis)
    {
        if(operand.dims[axis] < 1 || operand.strides[axis] < 0)
        {
            return std::nullopt;
        }
    }
    return operand;
}

/// True when no two elements of @p operand share an address. Required of the written
/// operand: one thread per element writes it, so an overlap is a race.
bool hasDistinctElements(const Operand& operand)
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
        if(operand.strides[axis] < span)
        {
            return false;
        }
        const auto next = checkedMultiply(operand.strides[axis], operand.dims[axis]);
        if(!next.has_value())
        {
            return false;
        }
        span = *next;
    }
    return true;
}

/// @p values as a vector of exactly @p size elements each at least @p minimum, or nullopt.
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

/// Spatial extent of the output for one axis, or nullopt when the dilated filter does not
/// fit the padded input.
std::optional<int64_t> outputExtent(
    int64_t input, int64_t filter, int64_t prePad, int64_t postPad, int64_t stride, int64_t dil)
{
    // dil * (filter - 1): the reach of the filter past its first tap.
    int64_t reach = 0;
    if(filter > 1)
    {
        const auto product = checkedMultiply(dil, filter - 1);
        if(!product.has_value())
        {
            return std::nullopt;
        }
        reach = *product;
    }

    int64_t padded = 0;
    if(__builtin_add_overflow(input, prePad, &padded)
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

/// Copies an operand's (n, c, spatial...) strides into the kernel's 5-slot form, padding the
/// missing leading spatial axes with stride 0.
void fillStrides(const Operand& operand,
                 int64_t spatialRank,
                 long long (&strides)[CONV_NAIVE_TENSOR_DIMS])
{
    strides[0] = operand.strides[0];
    strides[1] = operand.strides[1];
    const int64_t missing = CONV_NAIVE_SPATIAL_DIMS - spatialRank;
    for(int64_t axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        strides[2 + axis]
            = axis < missing ? 0 : operand.strides[static_cast<size_t>(2 + axis - missing)];
    }
}

/// Product of @p values, or nullopt on overflow.
std::optional<int64_t> product(const long long (&values)[CONV_NAIVE_SPATIAL_DIMS])
{
    std::optional<int64_t> result = 1;
    for(const auto value : values)
    {
        result = result.has_value() ? checkedMultiply(*result, value) : std::nullopt;
    }
    return result;
}

} // namespace

std::string_view engineName(ConvDirection direction)
{
    switch(direction)
    {
    case ConvDirection::FORWARD:
        return "hipkernel:CONV_NAIVE_FWD";
    case ConvDirection::BACKWARD_DATA:
        return "hipkernel:CONV_NAIVE_BWD";
    case ConvDirection::BACKWARD_WEIGHTS:
        return "hipkernel:CONV_NAIVE_WRW";
    default:
        break;
    }
    return "";
}

std::optional<ConvNaiveProblem> parseConvNaiveProblem(const MatchContext& context,
                                                      ConvDirection direction)
{
    const auto fields = convFields(context, direction);
    if(!fields.has_value())
    {
        return std::nullopt;
    }

    // Both modes are served: convolution reflects the filter, cross-correlation does not.
    if(fields->mode != data_objects::ConvMode::CROSS_CORRELATION
       && fields->mode != data_objects::ConvMode::CONVOLUTION)
    {
        return std::nullopt;
    }

    // Three distinct operands: an output aliasing an input is read while it is written.
    const auto& uids = fields->operands;
    if(uids.x == uids.w || uids.x == uids.y || uids.w == uids.y)
    {
        return std::nullopt;
    }

    const auto x = deviceOperand(context, uids.x);
    const auto w = deviceOperand(context, uids.w);
    const auto y = deviceOperand(context, uids.y);
    if(!x.has_value() || !w.has_value() || !y.has_value())
    {
        return std::nullopt;
    }

    // One element type for all three; mixed precision is another kernel.
    if(x->dataType != w->dataType || x->dataType != y->dataType)
    {
        return std::nullopt;
    }

    // cuDNN's compute types for the element type; the kernel accumulates in f32.
    if(!convServesComputeType(context.graph.getNodeWrapper(0).computeDataType(), x->dataType))
    {
        return std::nullopt;
    }

    const size_t rank = x->dims.size();
    if(w->dims.size() != rank || y->dims.size() != rank)
    {
        return std::nullopt;
    }
    const auto spatialRank = static_cast<int64_t>(rank - 2);

    const Operand& written
        = fields->writtenUid == uids.x ? *x : (fields->writtenUid == uids.w ? *w : *y);
    if(!hasDistinctElements(written))
    {
        return std::nullopt;
    }

    const auto prePadding = convParameter(fields->prePadding, rank - 2, 0);
    const auto postPadding = convParameter(fields->postPadding, rank - 2, 0);
    const auto stride = convParameter(fields->stride, rank - 2, 1);
    const auto dilation = convParameter(fields->dilation, rank - 2, 1);
    if(!prePadding.has_value() || !postPadding.has_value() || !stride.has_value()
       || !dilation.has_value())
    {
        return std::nullopt;
    }

    // Channels and groups. Groups are implied by the filter's channel count.
    const int64_t n = x->dims[0];
    const int64_t c = x->dims[1];
    const int64_t k = w->dims[0];
    const int64_t cPerGroup = w->dims[1];
    if(c % cPerGroup != 0)
    {
        return std::nullopt;
    }
    const int64_t groups = c / cPerGroup;
    if(k % groups != 0 || y->dims[0] != n || y->dims[1] != k)
    {
        return std::nullopt;
    }

    ConvNaiveProblem problem;
    problem.direction = direction;
    problem.dataType = x->dataType;
    problem.operands = uids;
    problem.spatialRank = spatialRank;

    auto& args = problem.args;
    args.n = n;
    args.c = c;
    args.k = k;
    args.groups = groups;
    args.flipFilter = fields->mode == data_objects::ConvMode::CONVOLUTION ? 1 : 0;

    const int64_t missing = CONV_NAIVE_SPATIAL_DIMS - spatialRank;
    for(int64_t axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        if(axis < missing)
        {
            args.inSpatial[axis] = 1;
            args.outSpatial[axis] = 1;
            args.filter[axis] = 1;
            args.prePadding[axis] = 0;
            args.stride[axis] = 1;
            args.dilation[axis] = 1;
            continue;
        }

        const auto tensorAxis = static_cast<size_t>(2 + axis - missing);
        const auto parameterAxis = static_cast<size_t>(axis - missing);
        const auto extent = outputExtent(x->dims[tensorAxis],
                                         w->dims[tensorAxis],
                                         (*prePadding)[parameterAxis],
                                         (*postPadding)[parameterAxis],
                                         (*stride)[parameterAxis],
                                         (*dilation)[parameterAxis]);
        // y must be exactly the shape this convolution produces.
        if(!extent.has_value() || *extent != y->dims[tensorAxis])
        {
            return std::nullopt;
        }

        args.inSpatial[axis] = x->dims[tensorAxis];
        args.outSpatial[axis] = y->dims[tensorAxis];
        args.filter[axis] = w->dims[tensorAxis];
        args.prePadding[axis] = (*prePadding)[parameterAxis];
        args.stride[axis] = (*stride)[parameterAxis];
        args.dilation[axis] = (*dilation)[parameterAxis];
    }

    fillStrides(*x, spatialRank, args.xStrides);
    fillStrides(*w, spatialRank, args.wStrides);
    fillStrides(*y, spatialRank, args.yStrides);

    const auto inElements = product(args.inSpatial);
    const auto outElements = product(args.outSpatial);
    const auto filterElements = product(args.filter);
    if(!inElements.has_value() || !outElements.has_value() || !filterElements.has_value())
    {
        return std::nullopt;
    }

    std::optional<int64_t> total;
    switch(direction)
    {
    case ConvDirection::FORWARD:
        total = pack_features::checkedProduct({n, k, *outElements});
        break;
    case ConvDirection::BACKWARD_DATA:
        total = pack_features::checkedProduct({n, c, *inElements});
        break;
    case ConvDirection::BACKWARD_WEIGHTS:
        total = pack_features::checkedProduct({k, cPerGroup, *filterElements});
        break;
    default:
        break;
    }
    const auto macs
        = pack_features::checkedProduct({n, k, *outElements, cPerGroup, *filterElements});
    if(!total.has_value() || !macs.has_value())
    {
        return std::nullopt;
    }
    args.total = *total;
    problem.multiplyAccumulates = *macs;
    return problem;
}

bool withinWorkloadBudget(const ConvNaiveProblem& problem,
                          const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    // Lanes are physical: count CUs, two per WGP where HIP reports WGPs (gfx10+).
    const int64_t reported = device_units::computeUnits(device);
    const int64_t computeUnits = reported > 0 ? reported : FALLBACK_COMPUTE_UNITS;
    const int64_t elementLanes = problem.direction == ConvDirection::BACKWARD_WEIGHTS
                                     ? NAIVE_WRW_LANES_PER_ELEMENT
                                     : int64_t{1};
    const auto outputLanes = pack_features::checkedMultiply(problem.args.total, elementLanes);
    const int64_t deviceLanes = computeUnits * LANES_PER_COMPUTE_UNIT;
    const int64_t occupiedLanes = std::max<int64_t>(
        1, outputLanes.has_value() ? std::min<int64_t>(deviceLanes, *outputLanes) : deviceLanes);
    const int64_t perLane = problem.multiplyAccumulates / occupiedLanes
                            + (problem.multiplyAccumulates % occupiedLanes != 0 ? 1 : 0);
    return perLane <= NAIVE_MACS_PER_LANE_BUDGET;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_naive

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
