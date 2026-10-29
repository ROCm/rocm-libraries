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
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_backend.h>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/type_mapping.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_space.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
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

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// The shim's own `graph` namespace (the v9 API) shadows hipDNN's.
namespace fe_graph = ::hipdnn_frontend::graph;

// cuDNN's name for an operation kind, from upstream's DescriptorType_t names.
inline std::string operationKindName(cudnnBackendDescriptorType_t kind)
{
    std::ostringstream name;
    name << "CUDNN_BACKEND_" << toFe(kind).value_or(DescriptorType_t::NOT_SET);
    return name.str();
}

// Caller-chosen names that prefix graph errors, so each caller keeps its own
// wording.
struct LoweringLabels
{
    std::string handle;
    // OPS required, null records, UID conflicts, double writers, cycles.
    std::string operations;
    std::string dynamicShape;
    std::string overrideShape;
    // A constant tensor hipDNN cannot carry.
    std::string constantValue;
    // Names an operation kind in node-lifecycle errors; null means
    // operationKindName.
    std::string (*operationName)(cudnnBackendDescriptorType_t) = &operationKindName;
};

inline std::string operationLabel(const LoweringLabels& labels, cudnnBackendDescriptorType_t kind)
{
    return labels.operationName != nullptr ? labels.operationName(kind) : operationKindName(kind);
}

// Graph features the shim cannot lower.
struct GraphFlags
{
    bool dynamicShape = false;
    bool overrideShape = false;
};

// Tensors an operation reads and writes, in slot order; unset slots are
// omitted.
struct OperationPorts
{
    std::vector<TensorPtr> inputs;
    std::vector<TensorPtr> outputs;
};

inline void addPort(std::vector<TensorPtr>& ports, const TensorPtr& tensor)
{
    if(tensor)
    {
        ports.push_back(tensor);
    }
}

template <cudnnBackendDescriptorType_t K, OperationTarget T, typename S>
OperationPorts operationPorts(const ConvolutionOperationRecordOf<K, T, S>& op)
{
    const std::array<TensorPtr, 3>& tensors = op.tensors();
    return {{tensors[0], tensors[1]}, {tensors[2]}};
}

inline OperationPorts operationPorts(const PointwiseOperationRecord& op)
{
    OperationPorts ports;
    addPort(ports.inputs, op.in0());
    addPort(ports.inputs, op.in1());
    addPort(ports.inputs, op.in2());
    addPort(ports.outputs, op.out0());
    return ports;
}

inline OperationPorts operationPorts(const MatmulOperationRecord& op)
{
    return {{op.a(), op.b()}, {op.c()}};
}

inline OperationPorts operationPorts(const ReductionOperationRecord& op)
{
    return {{op.x()}, {op.y()}};
}

inline OperationPorts operationPorts(const ResampleForwardOperationRecord& op)
{
    OperationPorts ports{{op.x()}, {op.y()}};
    addPort(ports.outputs, op.index());
    return ports;
}

inline OperationPorts operationPorts(const ResampleBackwardOperationRecord& op)
{
    OperationPorts ports{{op.dy()}, {op.dx()}};
    addPort(ports.inputs, op.index());
    return ports;
}

inline OperationPorts operationPorts(const NormForwardOperationRecord& op)
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

inline OperationPorts operationPorts(const NormBackwardOperationRecord& op)
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

template <typename Visitor, typename First, typename... Rest>
decltype(auto) visitOperationAs(const OperationRecord& op,
                                Visitor& visitor,
                                OperationRecordList<First, Rest...> /*records*/)
{
    if constexpr(sizeof...(Rest) == 0)
    {
        return visitor(recordAs<First>(op));
    }
    else
    {
        if(op.kind() == First::KIND)
        {
            return visitor(recordAs<First>(op));
        }
        return visitOperationAs(op, visitor, OperationRecordList<Rest...>{});
    }
}

// Calls `visitor` with the concrete record behind an operation.
template <typename Visitor>
decltype(auto)
    visitOperation(const OperationRecord& op, const LoweringLabels& labels, Visitor&& visitor)
{
    if(op.target() == OperationTarget::NOT_SET)
    {
        throw ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                        operationLabel(labels, op.kind()) + " has no lowering target");
    }
    return visitOperationAs(op, visitor, OperationRecords{});
}

// One hipdnn_frontend tensor per UID. The hipDNN packers keep the first tensor
// they see for a UID, so two tensor records that share a UID must agree on
// everything that changes the lowered tensor. Records are told apart by
// address, so every added record must outlive this object.
class CanonicalTensors
{
public:
    explicit CanonicalTensors(const LoweringLabels& labels)
        : _labels(labels)
    {
    }

    void add(const TensorPtr& tensor)
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

    const std::shared_ptr<fe_graph::TensorAttributes>& at(const TensorPtr& tensor) const
    {
        return _byUid.at(tensor->uniqueId()).lowered;
    }

private:
    struct Entry
    {
        const TensorRecord* source;
        std::shared_ptr<fe_graph::TensorAttributes> lowered;
    };

    std::shared_ptr<fe_graph::TensorAttributes> lower(const TensorRecord& tensor) const
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
        if(const TensorPtr& ragged = tensor.raggedOffset())
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

    void setConstant(fe_graph::TensorAttributes& lowered,
                     const TensorRecord& tensor,
                     const std::vector<std::byte>& bytes) const
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
                            _labels.constantValue + " of UID " + std::to_string(tensor.uniqueId())
                                + ": hipDNN cannot carry a constant of this data type");
        }
    }

    void requireSame(const TensorRecord& first, const TensorRecord& second) const
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
                            _labels.operations + ": two tensors with UID "
                                + std::to_string(first.uniqueId()) + " differ in " + difference);
        }
    }

    static std::optional<std::int64_t> raggedUid(const TensorRecord& tensor)
    {
        return tensor.raggedOffset()
                   ? std::optional<std::int64_t>(tensor.raggedOffset()->uniqueId())
                   : std::nullopt;
    }

    const LoweringLabels& _labels;
    std::unordered_map<std::int64_t, Entry> _byUid;
};

// Kahn's algorithm with the smallest original index first among ready
// operations, so an already topological OPS array keeps its order (providers
// match fusion patterns by node position). Disconnected components are fine.
inline std::vector<std::size_t> stableTopologicalOrder(const std::vector<OperationPorts>& ports,
                                                       const LoweringLabels& labels)
{
    const std::size_t count = ports.size();
    std::unordered_map<std::int64_t, std::size_t> producers;
    for(std::size_t i = 0; i < count; ++i)
    {
        for(const TensorPtr& output : ports[i].outputs)
        {
            const auto [found, inserted] = producers.emplace(output->uniqueId(), i);
            if(!inserted && found->second != i)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                labels.operations + ": UID " + std::to_string(output->uniqueId())
                                    + " is written by operations " + std::to_string(found->second)
                                    + " and " + std::to_string(i));
            }
        }
    }

    std::vector<std::vector<std::size_t>> consumers(count);
    std::vector<std::size_t> pending(count, 0);
    for(std::size_t i = 0; i < count; ++i)
    {
        for(const TensorPtr& input : ports[i].inputs)
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
                        labels.operations + ": the operations form a cycle");
    }
    return order;
}

struct LoweringContext
{
    const CanonicalTensors& tensors;
    const LoweringLabels& labels;
    std::unordered_map<std::int64_t, hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor>
        tensorDescs;
    std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> operations;

    const std::shared_ptr<fe_graph::TensorAttributes>& tensor(const TensorPtr& source) const
    {
        return tensors.at(source);
    }
};

// Drives a standalone hipdnn_frontend node through the same lifecycle the
// frontend graph uses, then emits its hipDNN operation descriptor.
template <typename Node, typename Attributes>
void lowerNode(Attributes attributes,
               DataType_t computeType,
               const OperationRecord& op,
               LoweringContext& context)
{
    attributes.set_compute_data_type(computeType);
    Node node(std::move(attributes), fe_graph::GraphAttributes{});
    const std::string label = std::string("lowering ") + operationLabel(context.labels, op.kind());
    checkFrontend(node.pre_validate_node(), label);
    checkFrontend(node.infer_properties_node(), label);
    checkFrontend(node.post_validate_node(), label);
    checkFrontend(node.create_operation(context.tensorDescs, context.operations), label);
}

// The caller sets the tensors; the convolution parameters are the same for all
// three convolution nodes.
template <typename Node, typename Attributes, typename Record>
void lowerConvolution(Attributes attributes, const Record& op, LoweringContext& context)
{
    const ConvolutionRecord& convolution = *op.convolution();
    attributes.set_pre_padding(convolution.prePaddings())
        .set_post_padding(convolution.postPaddings())
        .set_stride(convolution.filterStrides())
        .set_dilation(convolution.dilations())
        .set_convolution_mode(convolution.mode());
    lowerNode<Node>(std::move(attributes), convolution.computeType(), op, context);
}

inline void lowerOperation(const ConvolutionForwardOperationRecord& op, LoweringContext& context)
{
    using Slots = ConvolutionForwardOperationRecord::Slots;
    const std::array<TensorPtr, 3>& tensors = op.tensors();
    fe_graph::ConvFpropAttributes attributes;
    attributes.set_x(context.tensor(tensors[Slots::X]))
        .set_w(context.tensor(tensors[Slots::W]))
        .set_y(context.tensor(tensors[Slots::Y]));
    lowerConvolution<fe_graph::ConvolutionFpropNode>(std::move(attributes), op, context);
}

inline void lowerOperation(const ConvolutionBackwardDataOperationRecord& op,
                           LoweringContext& context)
{
    using Slots = ConvolutionBackwardDataOperationRecord::Slots;
    const std::array<TensorPtr, 3>& tensors = op.tensors();
    fe_graph::ConvDgradAttributes attributes;
    attributes.set_dy(context.tensor(tensors[Slots::DY]))
        .set_w(context.tensor(tensors[Slots::W]))
        .set_dx(context.tensor(tensors[Slots::DX]));
    lowerConvolution<fe_graph::ConvolutionDgradNode>(std::move(attributes), op, context);
}

inline void lowerOperation(const ConvolutionBackwardFilterOperationRecord& op,
                           LoweringContext& context)
{
    using Slots = ConvolutionBackwardFilterOperationRecord::Slots;
    const std::array<TensorPtr, 3>& tensors = op.tensors();
    fe_graph::ConvWgradAttributes attributes;
    attributes.set_x(context.tensor(tensors[Slots::X]))
        .set_dy(context.tensor(tensors[Slots::DY]))
        .set_dw(context.tensor(tensors[Slots::DW]));
    lowerConvolution<fe_graph::ConvolutionWgradNode>(std::move(attributes), op, context);
}

// hipDNN carries pointwise parameters as float; a bound beyond float range
// saturates instead of becoming infinity.
inline float narrowParameter(double value) noexcept
{
    constexpr auto LIMIT = static_cast<double>(std::numeric_limits<float>::max());
    return static_cast<float>(std::clamp(value, -LIMIT, LIMIT));
}

inline void lowerOperation(const PointwiseOperationRecord& op, LoweringContext& context)
{
    const PointwiseRecord& pointwise = *op.pointwise();
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
    // The record holds only non-default parameters; leaving the rest unset
    // keeps them off the hipDNN descriptor.
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

inline void lowerOperation(const MatmulOperationRecord& op, LoweringContext& context)
{
    fe_graph::MatmulAttributes attributes;
    attributes.set_a(context.tensor(op.a()))
        .set_b(context.tensor(op.b()))
        .set_c(context.tensor(op.c()));
    lowerNode<fe_graph::MatmulNode>(std::move(attributes), op.matmul()->computeType(), op, context);
}

inline void lowerOperation(const ReductionOperationRecord& op, LoweringContext& context)
{
    const ReductionRecord& reduction = *op.reduction();
    fe_graph::ReductionAttributes attributes;
    attributes.set_mode(reduction.mode())
        .set_is_deterministic(reduction.isDeterministic())
        .set_x(context.tensor(op.x()))
        .set_y(context.tensor(op.y()));
    lowerNode<fe_graph::ReductionNode>(std::move(attributes), reduction.computeType(), op, context);
}

// The caller sets the tensors; the window parameters are the same for both
// resample nodes.
template <typename Node, typename Attributes, typename Record>
void lowerResample(Attributes attributes, const Record& op, LoweringContext& context)
{
    const ResampleRecord& resample = *op.resample();
    attributes.set_pre_padding(resample.prePaddings())
        .set_post_padding(resample.postPaddings())
        .set_stride(resample.strides())
        .set_window(resample.windowDims())
        .set_resample_mode(resample.mode())
        .set_padding_mode(resample.paddingMode());
    lowerNode<Node>(std::move(attributes), resample.computeType(), op, context);
}

inline void lowerOperation(const ResampleForwardOperationRecord& op, LoweringContext& context)
{
    fe_graph::ResampleFwdAttributes attributes;
    attributes.set_x(context.tensor(op.x())).set_y(context.tensor(op.y()));
    if(op.index())
    {
        attributes.set_index(context.tensor(op.index())).set_generate_index(true);
    }
    lowerResample<fe_graph::ResampleFwdNode>(std::move(attributes), op, context);
}

inline void lowerOperation(const ResampleBackwardOperationRecord& op, LoweringContext& context)
{
    fe_graph::ResampleBwdAttributes attributes;
    attributes.set_dy(context.tensor(op.dy())).set_dx(context.tensor(op.dx()));
    if(op.index())
    {
        attributes.set_index(context.tensor(op.index()));
    }
    lowerResample<fe_graph::ResampleBwdNode>(std::move(attributes), op, context);
}

inline void lowerBatchnorm(const NormForwardOperationRecord& op, LoweringContext& context)
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
    // The record holds the running statistics all together or not at all; the
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
        std::move(attributes), NormForwardOperationRecord::computeType(), op, context);
}

inline void lowerBatchnormInference(const NormForwardOperationRecord& op, LoweringContext& context)
{
    fe_graph::BatchnormInferenceAttributes attributes;
    attributes.set_x(context.tensor(op.x()))
        .set_mean(context.tensor(op.mean()))
        .set_inv_variance(context.tensor(op.invVariance()))
        .set_scale(context.tensor(op.scale()))
        .set_bias(context.tensor(op.bias()))
        .set_y(context.tensor(op.y()));
    lowerNode<fe_graph::BatchnormInferenceNode>(
        std::move(attributes), NormForwardOperationRecord::computeType(), op, context);
}

inline void lowerLayernorm(const NormForwardOperationRecord& op, LoweringContext& context)
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
        std::move(attributes), NormForwardOperationRecord::computeType(), op, context);
}

inline void lowerRmsnorm(const NormForwardOperationRecord& op, LoweringContext& context)
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
        std::move(attributes), NormForwardOperationRecord::computeType(), op, context);
}

inline void lowerOperation(const NormForwardOperationRecord& op, LoweringContext& context)
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

inline void lowerBatchnormBackward(const NormBackwardOperationRecord& op, LoweringContext& context)
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
        std::move(attributes), NormBackwardOperationRecord::computeType(), op, context);
}

inline void lowerLayernormBackward(const NormBackwardOperationRecord& op, LoweringContext& context)
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
        std::move(attributes), NormBackwardOperationRecord::computeType(), op, context);
}

inline void lowerRmsnormBackward(const NormBackwardOperationRecord& op, LoweringContext& context)
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
        std::move(attributes), NormBackwardOperationRecord::computeType(), op, context);
}

inline void lowerOperation(const NormBackwardOperationRecord& op, LoweringContext& context)
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

// The graph-level rules, checked before any hipDNN call.
inline void preflightGraph(const std::vector<std::shared_ptr<const OperationRecord>>& operations,
                           hipdnnHandle_t handle,
                           GraphFlags flags,
                           const LoweringLabels& labels)
{
    if(handle == nullptr)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, labels.handle + " is required");
    }
    if(operations.empty())
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, labels.operations + " is required (non-empty)");
    }
    for(const auto& [enabled, label] : {std::pair{flags.dynamicShape, &labels.dynamicShape},
                                        std::pair{flags.overrideShape, &labels.overrideShape}})
    {
        if(enabled)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            *label + " = true is not supported by the hipDNN cuDNN shim");
        }
    }
    for(std::size_t i = 0; i < operations.size(); ++i)
    {
        if(!operations[i])
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            labels.operations + ": operation " + std::to_string(i) + " is null");
        }
    }
}

// Lowers `operations` to a finalized hipDNN operation graph on `handle`. The
// per-tensor and per-operation hipDNN descriptors are released on return;
// hipDNN's refcounts keep what the graph references.
inline std::shared_ptr<const GraphCore>
    lowerGraph(const std::vector<std::shared_ptr<const OperationRecord>>& operations,
               hipdnnHandle_t handle,
               GraphFlags flags,
               const LoweringLabels& labels,
               std::string tag)
{
    preflightGraph(operations, handle, flags, labels);
    std::vector<OperationPorts> ports;
    ports.reserve(operations.size());
    CanonicalTensors tensors(labels);
    for(const std::shared_ptr<const OperationRecord>& operation : operations)
    {
        ports.push_back(
            visitOperation(*operation, labels, [](const auto& op) { return operationPorts(op); }));
        for(const TensorPtr& tensor : ports.back().inputs)
        {
            tensors.add(tensor);
        }
        for(const TensorPtr& tensor : ports.back().outputs)
        {
            tensors.add(tensor);
        }
    }

    LoweringContext context{tensors, labels, {}, {}};
    for(const std::size_t index : stableTopologicalOrder(ports, labels))
    {
        visitOperation(*operations[index], labels, [&context](const auto& op) {
            lowerOperation(op, context);
        });
    }

    std::unique_ptr<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> graph;
    checkFrontend(hipdnn_frontend::detail::assembleGraphDescriptor(context.operations,
                                                                   handle,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   false,
                                                                   "",
                                                                   graph),
                  "assembling the hipDNN operation graph");
    return std::make_shared<const GraphCore>(std::move(*graph), std::move(tag));
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
