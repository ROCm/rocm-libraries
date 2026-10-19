// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/attribute_table.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/operation_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/parameter_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/tensor_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/GraphPacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/LayerNormNode.hpp>
#include <hipdnn_frontend/node/LayernormBackwardNode.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/RMSNormBackwardNode.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>
#include <hipdnn_frontend/node/ResampleBwdNode.hpp>
#include <hipdnn_frontend/node/ResampleFwdNode.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

// The shim's own `graph` namespace (the v9 API) shadows hipDNN's.
namespace fe_graph = ::hipdnn_frontend::graph;

inline void checkFrontend(const hipdnn_frontend::Error& error, const std::string& context)
{
    if(error.is_good())
    {
        return;
    }
    throw ShimError(toCudnnStatus(error.get_code()), context + ": " + error.get_message());
}

inline hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor
    createHipdnnDescriptor(hipdnnBackendDescriptorType_t type, const std::string& context)
{
    hipdnnBackendDescriptor_t raw = nullptr;
    checkHipdnn(hipdnn_frontend::detail::hipdnnBackend()->backendCreateDescriptor(type, &raw),
                context);
    return {raw};
}

// Tensors an operation reads and writes, in slot order; unset slots are
// omitted.
struct OperationPorts
{
    std::vector<std::shared_ptr<TensorDescriptor>> inputs;
    std::vector<std::shared_ptr<TensorDescriptor>> outputs;
};

inline void addPort(std::vector<std::shared_ptr<TensorDescriptor>>& ports,
                    const std::shared_ptr<TensorDescriptor>& tensor)
{
    if(tensor)
    {
        ports.push_back(tensor);
    }
}

inline OperationPorts operationPorts(const ConvolutionForwardOperation& op)
{
    return {{op.x(), op.w()}, {op.y()}};
}

inline OperationPorts operationPorts(const ConvolutionBackwardDataOperation& op)
{
    return {{op.dy(), op.w()}, {op.dx()}};
}

inline OperationPorts operationPorts(const ConvolutionBackwardFilterOperation& op)
{
    return {{op.dy(), op.x()}, {op.dw()}};
}

inline OperationPorts operationPorts(const PointwiseOperation& op)
{
    OperationPorts ports;
    addPort(ports.inputs, op.in0());
    addPort(ports.inputs, op.in1());
    addPort(ports.inputs, op.in2());
    addPort(ports.outputs, op.out0());
    return ports;
}

inline OperationPorts operationPorts(const MatmulOperation& op)
{
    return {{op.a(), op.b()}, {op.c()}};
}

inline OperationPorts operationPorts(const ReductionOperation& op)
{
    return {{op.x()}, {op.y()}};
}

inline OperationPorts operationPorts(const ResampleForwardOperation& op)
{
    OperationPorts ports{{op.x()}, {op.y()}};
    addPort(ports.outputs, op.index());
    return ports;
}

inline OperationPorts operationPorts(const ResampleBackwardOperation& op)
{
    OperationPorts ports{{op.dy()}, {op.dx()}};
    addPort(ports.inputs, op.index());
    return ports;
}

inline OperationPorts operationPorts(const NormForwardOperation& op)
{
    OperationPorts ports;
    addPort(ports.inputs, op.x());
    addPort(ports.inputs, op.scale());
    addPort(ports.inputs, op.bias());
    addPort(ports.inputs, op.epsilon());
    addPort(ports.outputs, op.y());
    // Batchnorm inference reads the saved statistics; every other target
    // produces them.
    auto& statistics
        = op.target() == OperationTarget::BATCHNORM_INFERENCE ? ports.inputs : ports.outputs;
    addPort(statistics, op.mean());
    addPort(statistics, op.invVariance());
    addPort(ports.inputs, op.inputRunningMean());
    addPort(ports.inputs, op.inputRunningVariance());
    addPort(ports.inputs, op.expAvgFactor());
    addPort(ports.outputs, op.outputRunningMean());
    addPort(ports.outputs, op.outputRunningVariance());
    return ports;
}

inline OperationPorts operationPorts(const NormBackwardOperation& op)
{
    OperationPorts ports;
    addPort(ports.inputs, op.x());
    addPort(ports.inputs, op.dy());
    addPort(ports.inputs, op.scale());
    addPort(ports.inputs, op.mean());
    addPort(ports.inputs, op.invVariance());
    addPort(ports.inputs, op.epsilon());
    addPort(ports.outputs, op.dx());
    addPort(ports.outputs, op.dscale());
    addPort(ports.outputs, op.dbias());
    return ports;
}

// Calls `visitor` with the concrete class behind a finalized operation.
template <typename Visitor>
decltype(auto) visitOperation(Descriptor& descriptor, Visitor&& visitor)
{
    switch(operationCast(descriptor).target())
    {
    case OperationTarget::CONVOLUTION_FPROP:
        return visitor(descriptorCast<ConvolutionForwardOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR));
    case OperationTarget::CONVOLUTION_DGRAD:
        return visitor(descriptorCast<ConvolutionBackwardDataOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR));
    case OperationTarget::CONVOLUTION_WGRAD:
        return visitor(descriptorCast<ConvolutionBackwardFilterOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR));
    case OperationTarget::POINTWISE:
        return visitor(descriptorCast<PointwiseOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR));
    case OperationTarget::MATMUL:
        return visitor(
            descriptorCast<MatmulOperation>(descriptor, CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR));
    case OperationTarget::REDUCTION:
        return visitor(descriptorCast<ReductionOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR));
    case OperationTarget::RESAMPLE_FWD:
        return visitor(descriptorCast<ResampleForwardOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR));
    case OperationTarget::RESAMPLE_BWD:
        return visitor(descriptorCast<ResampleBackwardOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR));
    case OperationTarget::BATCHNORM:
    case OperationTarget::BATCHNORM_INFERENCE:
    case OperationTarget::LAYERNORM:
    case OperationTarget::RMSNORM:
        return visitor(descriptorCast<NormForwardOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR));
    case OperationTarget::BATCHNORM_BACKWARD:
    case OperationTarget::LAYERNORM_BACKWARD:
    case OperationTarget::RMSNORM_BACKWARD:
        return visitor(descriptorCast<NormBackwardOperation>(
            descriptor, CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR));
    case OperationTarget::NOT_SET:
    default:
        throw ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                        std::string(descriptorTypeName(descriptor.type()))
                            + " has no lowering target");
    }
}

// One hipdnn_frontend tensor per UID. The hipDNN packers keep the first tensor
// they see for a UID, so two shim tensors that share a UID must agree on
// everything that changes the lowered tensor.
class CanonicalTensors
{
public:
    void add(const std::shared_ptr<TensorDescriptor>& tensor)
    {
        if(!tensor)
        {
            return;
        }
        add(tensor->raggedOffset());
        const auto found = _byUid.find(tensor->uniqueId());
        if(found == _byUid.end())
        {
            _byUid.emplace(tensor->uniqueId(), Entry{tensor.get(), lower(*tensor)});
            return;
        }
        const Entry& entry = found->second;
        if(entry.source == tensor.get())
        {
            return;
        }
        requireSame(*entry.source, *tensor);
        // Alignment is a lower bound, so the weakest claim is the safe one.
        const std::int64_t alignment = tensor->forwardedAlignment().value_or(
            hipdnn_frontend::detail::DEFAULT_TENSOR_ALIGNMENT);
        if(alignment < entry.lowered->get_alignment())
        {
            entry.lowered->set_alignment(alignment);
        }
    }

    const std::shared_ptr<fe_graph::TensorAttributes>&
        at(const std::shared_ptr<TensorDescriptor>& tensor) const
    {
        return _byUid.at(tensor->uniqueId()).lowered;
    }

private:
    struct Entry
    {
        const TensorDescriptor* source;
        std::shared_ptr<fe_graph::TensorAttributes> lowered;
    };

    std::shared_ptr<fe_graph::TensorAttributes> lower(const TensorDescriptor& tensor) const
    {
        auto lowered = std::make_shared<fe_graph::TensorAttributes>();
        // set_value resets dims and strides to {1}, so it must come first for a
        // constant to keep the caller's shape.
        if(const std::optional<std::vector<std::byte>>& constant = tensor.constantValue())
        {
            setConstant(*lowered, tensor, *constant);
        }
        lowered->set_uid(tensor.uniqueId())
            .set_dim(tensor.dimensions())
            .set_stride(tensor.strides())
            .set_is_virtual(tensor.isVirtual());
        lowered->set_data_type(tensor.dataType());
        if(const std::optional<std::int64_t> alignment = tensor.forwardedAlignment())
        {
            lowered->set_alignment(*alignment);
        }
        if(tensor.isRuntimePassByValue())
        {
            lowered->set_is_pass_by_value(true);
        }
        if(const std::shared_ptr<TensorDescriptor>& ragged = tensor.raggedOffset())
        {
            lowered->set_ragged_offset(_byUid.at(ragged->uniqueId()).lowered);
        }
        return lowered;
    }

    template <typename T>
    static void setConstantAs(fe_graph::TensorAttributes& lowered,
                              const std::vector<std::byte>& bytes)
    {
        T value{};
        std::memcpy(static_cast<void*>(&value), bytes.data(), std::min(sizeof(T), bytes.size()));
        lowered.set_value(value);
    }

    static void setConstant(fe_graph::TensorAttributes& lowered,
                            const TensorDescriptor& tensor,
                            const std::vector<std::byte>& bytes)
    {
        switch(tensor.dataType())
        {
        case DataType_t::FLOAT:
            setConstantAs<float>(lowered, bytes);
            return;
        case DataType_t::DOUBLE:
            setConstantAs<double>(lowered, bytes);
            return;
        case DataType_t::HALF:
            setConstantAs<fe_graph::half>(lowered, bytes);
            return;
        case DataType_t::BFLOAT16:
            setConstantAs<fe_graph::bfloat16>(lowered, bytes);
            return;
        case DataType_t::UINT8:
            setConstantAs<std::uint8_t>(lowered, bytes);
            return;
        case DataType_t::INT32:
            setConstantAs<std::int32_t>(lowered, bytes);
            return;
        case DataType_t::INT64:
            setConstantAs<std::int64_t>(lowered, bytes);
            return;
        case DataType_t::BOOLEAN:
            lowered.set_value(!bytes.empty() && bytes.front() != std::byte{0});
            return;
        default:
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "CUDNN_ATTR_TENSOR_CONSTANT_VALUE of UID "
                                + std::to_string(tensor.uniqueId())
                                + ": hipDNN cannot carry a constant of this data type");
        }
    }

    static void requireSame(const TensorDescriptor& first, const TensorDescriptor& second)
    {
        const char* difference = nullptr;
        if(first.dataType() != second.dataType())
        {
            difference = "data type";
        }
        else if(first.dimensions() != second.dimensions())
        {
            difference = "dimensions";
        }
        else if(first.strides() != second.strides())
        {
            difference = "strides";
        }
        else if(first.isVirtual() != second.isVirtual())
        {
            difference = "virtualness";
        }
        else if(first.isRuntimePassByValue() != second.isRuntimePassByValue()
                || first.constantValue() != second.constantValue())
        {
            difference = "by-value scalar";
        }
        else if(raggedUid(first) != raggedUid(second))
        {
            difference = "ragged offset";
        }
        if(difference != nullptr)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_OPERATIONGRAPH_OPS: two tensors with UID "
                                + std::to_string(first.uniqueId()) + " differ in " + difference);
        }
    }

    static std::optional<std::int64_t> raggedUid(const TensorDescriptor& tensor)
    {
        return tensor.raggedOffset()
                   ? std::optional<std::int64_t>(tensor.raggedOffset()->uniqueId())
                   : std::nullopt;
    }

    std::unordered_map<std::int64_t, Entry> _byUid;
};

// Kahn's algorithm with the smallest original index first among ready
// operations, so an already topological OPS array keeps its order (providers
// match fusion patterns by node position). Disconnected components are fine.
inline std::vector<std::size_t> stableTopologicalOrder(const std::vector<OperationPorts>& ports)
{
    const std::size_t count = ports.size();
    std::unordered_map<std::int64_t, std::size_t> producers;
    for(std::size_t i = 0; i < count; ++i)
    {
        for(const std::shared_ptr<TensorDescriptor>& output : ports[i].outputs)
        {
            const auto [found, inserted] = producers.emplace(output->uniqueId(), i);
            if(!inserted && found->second != i)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                "CUDNN_ATTR_OPERATIONGRAPH_OPS: UID "
                                    + std::to_string(output->uniqueId())
                                    + " is written by operations " + std::to_string(found->second)
                                    + " and " + std::to_string(i));
            }
        }
    }

    std::vector<std::vector<std::size_t>> consumers(count);
    std::vector<std::size_t> pending(count, 0);
    for(std::size_t i = 0; i < count; ++i)
    {
        for(const std::shared_ptr<TensorDescriptor>& input : ports[i].inputs)
        {
            const auto producer = producers.find(input->uniqueId());
            if(producer != producers.end())
            {
                consumers[producer->second].push_back(i);
                ++pending[i];
            }
        }
    }

    std::set<std::size_t> ready;
    for(std::size_t i = 0; i < count; ++i)
    {
        if(pending[i] == 0)
        {
            ready.insert(i);
        }
    }
    std::vector<std::size_t> order;
    order.reserve(count);
    while(!ready.empty())
    {
        const std::size_t next = *ready.begin();
        ready.erase(ready.begin());
        order.push_back(next);
        for(const std::size_t consumer : consumers[next])
        {
            if(--pending[consumer] == 0)
            {
                ready.insert(consumer);
            }
        }
    }
    if(order.size() != count)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_OPERATIONGRAPH_OPS: the operations form a cycle");
    }
    return order;
}

struct LoweringContext
{
    const CanonicalTensors& tensors;
    std::unordered_map<std::int64_t, hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor>
        tensorDescs;
    std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> operations;

    const std::shared_ptr<fe_graph::TensorAttributes>&
        tensor(const std::shared_ptr<TensorDescriptor>& source) const
    {
        return tensors.at(source);
    }
};

// Drives a standalone hipdnn_frontend node through the same lifecycle the
// frontend graph uses, then emits its hipDNN operation descriptor.
template <typename Node, typename Attributes>
void lowerNode(Attributes attributes,
               DataType_t computeType,
               const OperationDescriptor& op,
               LoweringContext& context)
{
    attributes.set_compute_data_type(computeType);
    Node node(std::move(attributes), fe_graph::GraphAttributes{});
    const std::string label = std::string("lowering ") + descriptorTypeName(op.type());
    checkFrontend(node.pre_validate_node(), label);
    checkFrontend(node.infer_properties_node(), label);
    checkFrontend(node.post_validate_node(), label);
    checkFrontend(node.create_operation(context.tensorDescs, context.operations), label);
}

inline void lowerOperation(const ConvolutionForwardOperation& op, LoweringContext& context)
{
    const ConvolutionDescriptor& convolution = *op.convolution();
    fe_graph::ConvFpropAttributes attributes;
    attributes.set_x(context.tensor(op.x()))
        .set_w(context.tensor(op.w()))
        .set_y(context.tensor(op.y()))
        .set_pre_padding(convolution.prePaddings())
        .set_post_padding(convolution.postPaddings())
        .set_stride(convolution.filterStrides())
        .set_dilation(convolution.dilations())
        .set_convolution_mode(convolution.mode());
    lowerNode<fe_graph::ConvolutionFpropNode>(
        std::move(attributes), convolution.computeType(), op, context);
}

inline void lowerOperation(const ConvolutionBackwardDataOperation& op, LoweringContext& context)
{
    const ConvolutionDescriptor& convolution = *op.convolution();
    fe_graph::ConvDgradAttributes attributes;
    attributes.set_dy(context.tensor(op.dy()))
        .set_w(context.tensor(op.w()))
        .set_dx(context.tensor(op.dx()))
        .set_pre_padding(convolution.prePaddings())
        .set_post_padding(convolution.postPaddings())
        .set_stride(convolution.filterStrides())
        .set_dilation(convolution.dilations())
        .set_convolution_mode(convolution.mode());
    lowerNode<fe_graph::ConvolutionDgradNode>(
        std::move(attributes), convolution.computeType(), op, context);
}

inline void lowerOperation(const ConvolutionBackwardFilterOperation& op, LoweringContext& context)
{
    const ConvolutionDescriptor& convolution = *op.convolution();
    fe_graph::ConvWgradAttributes attributes;
    attributes.set_x(context.tensor(op.x()))
        .set_dy(context.tensor(op.dy()))
        .set_dw(context.tensor(op.dw()))
        .set_pre_padding(convolution.prePaddings())
        .set_post_padding(convolution.postPaddings())
        .set_stride(convolution.filterStrides())
        .set_dilation(convolution.dilations())
        .set_convolution_mode(convolution.mode());
    lowerNode<fe_graph::ConvolutionWgradNode>(
        std::move(attributes), convolution.computeType(), op, context);
}

// hipDNN carries pointwise parameters as float; a bound beyond float range
// saturates instead of becoming infinity.
inline float narrowParameter(double value) noexcept
{
    constexpr auto LIMIT = static_cast<double>(std::numeric_limits<float>::max());
    return static_cast<float>(std::clamp(value, -LIMIT, LIMIT));
}

inline void lowerOperation(const PointwiseOperation& op, LoweringContext& context)
{
    const PointwiseDescriptor& pointwise = *op.pointwise();
    fe_graph::PointwiseAttributes attributes;
    attributes.set_mode(pointwise.mode())
        .set_input_0(context.tensor(op.in0()))
        .set_output_0(context.tensor(op.out0()));
    if(op.in1())
    {
        attributes.set_input_1(context.tensor(op.in1()));
    }
    if(op.in2())
    {
        attributes.set_input_2(context.tensor(op.in2()));
    }
    // The descriptor reports only non-default parameters (plus the ReLU lower
    // clip when no slope is set); leaving the rest unset keeps them off the
    // hipDNN descriptor.
    if(const std::optional<double> value = pointwise.reluLowerClip())
    {
        attributes.set_relu_lower_clip(narrowParameter(*value));
    }
    if(const std::optional<double> value = pointwise.reluUpperClip())
    {
        attributes.set_relu_upper_clip(narrowParameter(*value));
    }
    if(const std::optional<double> value = pointwise.reluLowerClipSlope())
    {
        attributes.set_relu_lower_clip_slope(narrowParameter(*value));
    }
    if(const std::optional<double> value = pointwise.eluAlpha())
    {
        attributes.set_elu_alpha(narrowParameter(*value));
    }
    if(const std::optional<double> value = pointwise.softplusBeta())
    {
        attributes.set_softplus_beta(narrowParameter(*value));
    }
    if(const std::optional<double> value = pointwise.swishBeta())
    {
        attributes.set_swish_beta(narrowParameter(*value));
    }
    if(const std::optional<std::int64_t> axis = pointwise.axis())
    {
        attributes.set_axis(*axis);
    }
    lowerNode<fe_graph::PointwiseNode>(std::move(attributes), pointwise.computeType(), op, context);
}

inline void lowerOperation(const MatmulOperation& op, LoweringContext& context)
{
    fe_graph::MatmulAttributes attributes;
    attributes.set_a(context.tensor(op.a()))
        .set_b(context.tensor(op.b()))
        .set_c(context.tensor(op.c()));
    lowerNode<fe_graph::MatmulNode>(std::move(attributes), op.matmul()->computeType(), op, context);
}

inline void lowerOperation(const ReductionOperation& op, LoweringContext& context)
{
    const ReductionDescriptor& reduction = *op.reduction();
    fe_graph::ReductionAttributes attributes;
    attributes.set_mode(reduction.mode())
        .set_is_deterministic(reduction.isDeterministic())
        .set_x(context.tensor(op.x()))
        .set_y(context.tensor(op.y()));
    lowerNode<fe_graph::ReductionNode>(std::move(attributes), reduction.computeType(), op, context);
}

inline void lowerOperation(const ResampleForwardOperation& op, LoweringContext& context)
{
    const ResampleDescriptor& resample = *op.resample();
    fe_graph::ResampleFwdAttributes attributes;
    attributes.set_x(context.tensor(op.x()))
        .set_y(context.tensor(op.y()))
        .set_pre_padding(resample.prePaddings())
        .set_post_padding(resample.postPaddings())
        .set_stride(resample.strides())
        .set_window(resample.windowDims())
        .set_resample_mode(resample.mode())
        .set_padding_mode(resample.paddingMode());
    if(op.index())
    {
        attributes.set_index(context.tensor(op.index())).set_generate_index(true);
    }
    lowerNode<fe_graph::ResampleFwdNode>(
        std::move(attributes), resample.computeType(), op, context);
}

inline void lowerOperation(const ResampleBackwardOperation& op, LoweringContext& context)
{
    const ResampleDescriptor& resample = *op.resample();
    fe_graph::ResampleBwdAttributes attributes;
    attributes.set_dy(context.tensor(op.dy()))
        .set_dx(context.tensor(op.dx()))
        .set_pre_padding(resample.prePaddings())
        .set_post_padding(resample.postPaddings())
        .set_stride(resample.strides())
        .set_window(resample.windowDims())
        .set_resample_mode(resample.mode())
        .set_padding_mode(resample.paddingMode());
    if(op.index())
    {
        attributes.set_index(context.tensor(op.index()));
    }
    lowerNode<fe_graph::ResampleBwdNode>(
        std::move(attributes), resample.computeType(), op, context);
}

inline void lowerBatchnorm(const NormForwardOperation& op, LoweringContext& context)
{
    fe_graph::BatchnormAttributes attributes;
    attributes.set_x(context.tensor(op.x()))
        .set_scale(context.tensor(op.scale()))
        .set_bias(context.tensor(op.bias()))
        .set_epsilon(context.tensor(op.epsilon()))
        .set_y(context.tensor(op.y()));
    if(op.mean())
    {
        attributes.set_mean(context.tensor(op.mean()));
    }
    if(op.invVariance())
    {
        attributes.set_inv_variance(context.tensor(op.invVariance()));
    }
    // Finalize admits the running statistics all together or not at all; the
    // exponential-average factor is hipDNN's momentum.
    if(op.expAvgFactor())
    {
        attributes.set_prev_running_mean(context.tensor(op.inputRunningMean()))
            .set_prev_running_variance(context.tensor(op.inputRunningVariance()))
            .set_momentum(context.tensor(op.expAvgFactor()))
            .set_next_running_mean(context.tensor(op.outputRunningMean()))
            .set_next_running_variance(context.tensor(op.outputRunningVariance()));
    }
    lowerNode<fe_graph::BatchnormNode>(
        std::move(attributes), NormForwardOperation::computeType(), op, context);
}

inline void lowerBatchnormInference(const NormForwardOperation& op, LoweringContext& context)
{
    fe_graph::BatchnormInferenceAttributes attributes;
    attributes.set_x(context.tensor(op.x()))
        .set_mean(context.tensor(op.mean()))
        .set_inv_variance(context.tensor(op.invVariance()))
        .set_scale(context.tensor(op.scale()))
        .set_bias(context.tensor(op.bias()))
        .set_y(context.tensor(op.y()));
    lowerNode<fe_graph::BatchnormInferenceNode>(
        std::move(attributes), NormForwardOperation::computeType(), op, context);
}

inline void lowerLayernorm(const NormForwardOperation& op, LoweringContext& context)
{
    fe_graph::LayernormAttributes attributes;
    attributes.set_x(context.tensor(op.x()))
        .set_scale(context.tensor(op.scale()))
        .set_bias(context.tensor(op.bias()))
        .set_epsilon(context.tensor(op.epsilon()))
        .set_y(context.tensor(op.y()))
        .set_forward_phase(op.phase())
        .set_normalized_dim_count(op.normalizedDimCount());
    if(op.mean())
    {
        attributes.set_mean(context.tensor(op.mean()));
    }
    if(op.invVariance())
    {
        attributes.set_inv_variance(context.tensor(op.invVariance()));
    }
    lowerNode<fe_graph::LayerNormNode>(
        std::move(attributes), NormForwardOperation::computeType(), op, context);
}

inline void lowerRmsnorm(const NormForwardOperation& op, LoweringContext& context)
{
    fe_graph::RMSNormAttributes attributes;
    attributes.set_x(context.tensor(op.x()))
        .set_scale(context.tensor(op.scale()))
        .set_epsilon(context.tensor(op.epsilon()))
        .set_y(context.tensor(op.y()))
        .set_forward_phase(op.phase());
    if(op.bias())
    {
        attributes.set_bias(context.tensor(op.bias()));
    }
    if(op.invVariance())
    {
        attributes.set_inv_rms(context.tensor(op.invVariance()));
    }
    lowerNode<fe_graph::RMSNormNode>(
        std::move(attributes), NormForwardOperation::computeType(), op, context);
}

inline void lowerOperation(const NormForwardOperation& op, LoweringContext& context)
{
    switch(op.target())
    {
    case OperationTarget::BATCHNORM:
        lowerBatchnorm(op, context);
        return;
    case OperationTarget::BATCHNORM_INFERENCE:
        lowerBatchnormInference(op, context);
        return;
    case OperationTarget::LAYERNORM:
        lowerLayernorm(op, context);
        return;
    case OperationTarget::RMSNORM:
        lowerRmsnorm(op, context);
        return;
    default:
        throw ShimError(CUDNN_STATUS_INTERNAL_ERROR, "norm forward operation has no target");
    }
}

inline void lowerBatchnormBackward(const NormBackwardOperation& op, LoweringContext& context)
{
    fe_graph::BatchnormBackwardAttributes attributes;
    attributes.set_dy(context.tensor(op.dy()))
        .set_x(context.tensor(op.x()))
        .set_scale(context.tensor(op.scale()))
        .set_dx(context.tensor(op.dx()))
        .set_dscale(context.tensor(op.dscale()))
        .set_dbias(context.tensor(op.dbias()));
    if(op.mean())
    {
        attributes.set_mean(context.tensor(op.mean()));
    }
    if(op.invVariance())
    {
        attributes.set_inv_variance(context.tensor(op.invVariance()));
    }
    lowerNode<fe_graph::BatchnormBackwardNode>(
        std::move(attributes), NormBackwardOperation::computeType(), op, context);
}

inline void lowerLayernormBackward(const NormBackwardOperation& op, LoweringContext& context)
{
    fe_graph::LayernormBackwardAttributes attributes;
    attributes.set_dy(context.tensor(op.dy()))
        .set_x(context.tensor(op.x()))
        .set_scale(context.tensor(op.scale()))
        .set_dx(context.tensor(op.dx()))
        .set_dscale(context.tensor(op.dscale()))
        .set_dbias(context.tensor(op.dbias()))
        .set_normalized_dim_count(op.normalizedDimCount());
    if(op.mean())
    {
        attributes.set_mean(context.tensor(op.mean()));
    }
    if(op.invVariance())
    {
        attributes.set_inv_variance(context.tensor(op.invVariance()));
    }
    if(op.epsilon())
    {
        attributes.set_epsilon(context.tensor(op.epsilon()));
    }
    lowerNode<fe_graph::LayernormBackwardNode>(
        std::move(attributes), NormBackwardOperation::computeType(), op, context);
}

inline void lowerRmsnormBackward(const NormBackwardOperation& op, LoweringContext& context)
{
    fe_graph::RMSNormBackwardAttributes attributes;
    attributes.set_dy(context.tensor(op.dy()))
        .set_x(context.tensor(op.x()))
        .set_scale(context.tensor(op.scale()))
        .set_inv_rms(context.tensor(op.invVariance()))
        .set_dx(context.tensor(op.dx()))
        .set_dscale(context.tensor(op.dscale()))
        .set_compute_dbias(op.dbias() != nullptr);
    if(op.dbias())
    {
        attributes.set_dbias(context.tensor(op.dbias()));
    }
    lowerNode<fe_graph::RMSNormBackwardNode>(
        std::move(attributes), NormBackwardOperation::computeType(), op, context);
}

inline void lowerOperation(const NormBackwardOperation& op, LoweringContext& context)
{
    switch(op.target())
    {
    case OperationTarget::BATCHNORM_BACKWARD:
        lowerBatchnormBackward(op, context);
        return;
    case OperationTarget::LAYERNORM_BACKWARD:
        lowerLayernormBackward(op, context);
        return;
    case OperationTarget::RMSNORM_BACKWARD:
        lowerRmsnormBackward(op, context);
        return;
    default:
        throw ShimError(CUDNN_STATUS_INTERNAL_ERROR, "norm backward operation has no target");
    }
}

class OperationGraphDescriptor final : public Descriptor
{
public:
    OperationGraphDescriptor()
        : Descriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)
    {
    }

    // Valid once finalized. hipDNN has no getter for a graph's handle, so this
    // is the only record of it.
    cudnnHandle_t handle() const noexcept
    {
        return _handle;
    }

    hipdnnBackendDescriptor_t hipdnnGraph() const noexcept
    {
        return _graph ? _graph->get() : nullptr;
    }

    // Engine index space. The first call of any of these runs the hipDNN
    // heuristics (which need a HIP device); a failure leaves it to be retried.
    std::int64_t engineCount() const
    {
        return static_cast<std::int64_t>(engineSpace().byId.size());
    }

    // Engine IDs in heuristic rank order.
    const std::vector<std::int64_t>& rankedEngineIds() const
    {
        return engineSpace().rankedIds;
    }

    // Dense index (engines ordered by signed ascending ID) <-> engine ID.
    std::optional<std::int64_t> engineIdAt(std::int64_t index) const
    {
        const std::vector<Engine>& engines = engineSpace().byId;
        if(index < 0 || index >= static_cast<std::int64_t>(engines.size()))
        {
            return std::nullopt;
        }
        return engines[static_cast<std::size_t>(index)].id;
    }

    std::optional<std::int64_t> engineIndexOf(std::int64_t id) const
    {
        const std::vector<Engine>& engines = engineSpace().byId;
        const auto found = findEngine(engines, id);
        if(found == engines.end())
        {
            return std::nullopt;
        }
        return static_cast<std::int64_t>(found - engines.begin());
    }

    // The finalized hipDNN ENGINECFG (default knobs) and ENGINE heuristics
    // produced for `id`; null when `id` is not in the index space. Owned by
    // this graph.
    hipdnnBackendDescriptor_t defaultEngineConfig(std::int64_t id) const
    {
        const std::vector<Engine>& engines = engineSpace().byId;
        const auto found = findEngine(engines, id);
        return found == engines.end() ? nullptr : found->config.get();
    }

    hipdnnBackendDescriptor_t defaultEngine(std::int64_t id) const
    {
        const std::vector<Engine>& engines = engineSpace().byId;
        const auto found = findEngine(engines, id);
        return found == engines.end() ? nullptr : found->engine.get();
    }

protected:
    void finalizeImpl() override
    {
        const std::optional<cudnnHandle_t> handle
            = scalarAttribute<cudnnHandle_t>(CUDNN_ATTR_OPERATIONGRAPH_HANDLE);
        if(!handle || *handle == nullptr)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, "CUDNN_ATTR_OPERATIONGRAPH_HANDLE is required");
        }
        const std::vector<std::shared_ptr<Descriptor>> operations
            = descriptorArrayAttribute(CUDNN_ATTR_OPERATIONGRAPH_OPS);
        if(operations.empty())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_OPERATIONGRAPH_OPS is required (non-empty)");
        }
        for(const cudnnBackendAttributeName_t name :
            {CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED,
             CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED})
        {
            if(scalarAttribute<bool>(name).value_or(false))
            {
                throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                                attributeName(name)
                                    + " = true is not supported by the hipDNN cuDNN shim");
            }
        }

        std::vector<OperationPorts> ports;
        ports.reserve(operations.size());
        CanonicalTensors tensors;
        for(const std::shared_ptr<Descriptor>& operation : operations)
        {
            ports.push_back(
                visitOperation(*operation, [](const auto& op) { return operationPorts(op); }));
            for(const std::shared_ptr<TensorDescriptor>& tensor : ports.back().inputs)
            {
                tensors.add(tensor);
            }
            for(const std::shared_ptr<TensorDescriptor>& tensor : ports.back().outputs)
            {
                tensors.add(tensor);
            }
        }

        LoweringContext context{tensors, {}, {}};
        for(const std::size_t index : stableTopologicalOrder(ports))
        {
            visitOperation(*operations[index],
                           [&context](const auto& op) { lowerOperation(op, context); });
        }

        std::unique_ptr<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> graph;
        checkFrontend(hipdnn_frontend::detail::assembleGraphDescriptor(context.operations,
                                                                       *handle,
                                                                       std::nullopt,
                                                                       std::nullopt,
                                                                       std::nullopt,
                                                                       std::nullopt,
                                                                       false,
                                                                       "",
                                                                       graph),
                      "assembling the hipDNN operation graph");
        _handle = *handle;
        _graph = std::move(graph);
    }

    std::optional<StoredAttribute> lazyAttribute(cudnnBackendAttributeName_t name,
                                                 bool countOnly) const override
    {
        if(name != CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT)
        {
            return std::nullopt;
        }
        // The element count of this scalar is always 1; only the value needs the
        // engine sweep (and a device).
        const std::int64_t count = countOnly ? 0 : engineCount();
        StoredAttribute value;
        value.type = CUDNN_TYPE_INT64;
        value.count = 1;
        value.bytes.resize(sizeof(count));
        std::memcpy(value.bytes.data(), &count, sizeof(count));
        return value;
    }

private:
    struct Engine
    {
        std::int64_t id;
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor config;
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor engine;
    };

    struct EngineSpace
    {
        std::vector<std::int64_t> rankedIds;
        std::vector<Engine> byId;
    };

    static std::vector<Engine>::const_iterator findEngine(const std::vector<Engine>& engines,
                                                          std::int64_t id)
    {
        const auto found = std::lower_bound(
            engines.begin(), engines.end(), id, [](const Engine& engine, std::int64_t value) {
                return engine.id < value;
            });
        return found != engines.end() && found->id == id ? found : engines.end();
    }

    const EngineSpace& engineSpace() const
    {
        if(!isFinalized())
        {
            throw ShimError(CUDNN_STATUS_NOT_INITIALIZED,
                            "the engine index space needs a finalized "
                            "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR");
        }
        std::call_once(_engineOnce, [this] { _engines = enumerateEngines(hipdnnGraph()); });
        return _engines;
    }

    // Every applicable engine with its default config. hipDNN reports an engine
    // ID only after its config is finalized, which runs the engine's workspace
    // query; an engine whose query fails is dropped instead of failing the
    // whole graph.
    static EngineSpace enumerateEngines(hipdnnBackendDescriptor_t graph)
    {
        using hipdnn_frontend::detail::hipdnnBackend;
        using hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor;

        const ScopedHipdnnBackendDescriptor heuristic = createHipdnnDescriptor(
            HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, "creating the hipDNN engine heuristic");
        checkHipdnn(hipdnnBackend()->backendSetAttribute(heuristic.get(),
                                                         HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         static_cast<const void*>(&graph)),
                    "setting the hipDNN engine heuristic graph");
        const hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
        checkHipdnn(hipdnnBackend()->backendSetAttribute(heuristic.get(),
                                                         HIPDNN_ATTR_ENGINEHEUR_MODE,
                                                         HIPDNN_TYPE_HEUR_MODE,
                                                         1,
                                                         static_cast<const void*>(&mode)),
                    "setting the hipDNN engine heuristic mode");
        checkHipdnn(hipdnnBackend()->backendFinalize(heuristic.get()),
                    "finalizing the hipDNN engine heuristic");

        std::int64_t available = 0;
        checkHipdnn(hipdnnBackend()->backendGetAttribute(heuristic.get(),
                                                         HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         0,
                                                         &available,
                                                         nullptr),
                    "counting the hipDNN engine heuristic results");

        EngineSpace space;
        if(available <= 0)
        {
            return space;
        }
        std::vector<ScopedHipdnnBackendDescriptor> configs;
        std::vector<hipdnnBackendDescriptor_t> rawConfigs;
        configs.reserve(static_cast<std::size_t>(available));
        rawConfigs.reserve(static_cast<std::size_t>(available));
        for(std::int64_t i = 0; i < available; ++i)
        {
            configs.push_back(createHipdnnDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                                                     "creating a hipDNN engine config"));
            rawConfigs.push_back(configs.back().get());
        }
        std::int64_t returned = 0;
        checkHipdnn(hipdnnBackend()->backendGetAttribute(heuristic.get(),
                                                         HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         available,
                                                         &returned,
                                                         static_cast<void*>(rawConfigs.data())),
                    "reading the hipDNN engine heuristic results");

        const auto results
            = static_cast<std::size_t>(std::clamp<std::int64_t>(returned, 0, available));
        std::unordered_set<std::int64_t> seen;
        for(std::size_t i = 0; i < results; ++i)
        {
            const hipdnnStatus_t status = hipdnnBackend()->backendFinalize(rawConfigs[i]);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> message{};
                hipdnnBackend()->getLastErrorString(message.data(), message.size());
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] Dropping heuristic result "
                                   << i << ": its default engine config failed to finalize: "
                                   << message.data());
                continue;
            }
            hipdnnBackendDescriptor_t rawEngine = nullptr;
            checkHipdnn(hipdnnBackend()->backendGetAttribute(rawConfigs[i],
                                                             HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             nullptr,
                                                             static_cast<void*>(&rawEngine)),
                        "reading the engine of a hipDNN engine config");
            ScopedHipdnnBackendDescriptor engine(rawEngine);
            std::int64_t id = 0;
            checkHipdnn(hipdnnBackend()->backendGetAttribute(engine.get(),
                                                             HIPDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                                             HIPDNN_TYPE_INT64,
                                                             1,
                                                             nullptr,
                                                             static_cast<void*>(&id)),
                        "reading a hipDNN engine ID");
            if(!seen.insert(id).second)
            {
                continue;
            }
            space.rankedIds.push_back(id);
            space.byId.push_back(Engine{id, std::move(configs[i]), std::move(engine)});
        }
        std::sort(space.byId.begin(), space.byId.end(), [](const Engine& a, const Engine& b) {
            return a.id < b.id;
        });
        return space;
    }

    cudnnHandle_t _handle = nullptr;
    std::unique_ptr<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> _graph;
    // Declared after _graph so the cached configs and engines are released
    // before the graph they refer to.
    mutable std::once_flag _engineOnce;
    mutable EngineSpace _engines;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
