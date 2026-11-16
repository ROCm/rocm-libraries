// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

/**
 * @file ConvEpilogueMatch.hpp
 * @brief The pointwise epilogue a forward convolution graph fuses after its conv node, as the
 *        conv engines of every provider read it.
 *
 * A fused graph is the convolution node followed by a chain of pointwise nodes, each taking the
 * previous one's result, the last one writing the graph's single non-virtual output. The chain
 * runs in fp32 on the convolution's result and is rounded once to the output type.
 *
 * The pattern -- the operations, the operand dtypes and their broadcast kind -- is what an
 * engine's code depends on. Operand uids and every value (clip bounds, slope, alpha, beta,
 * scalar operands) are runtime inputs, so one code object serves every graph of a pattern.
 */
namespace hipdnn_plugin_sdk::conv_epilogue
{

/// Operands and values an epilogue may read: the bounds of the kernels' argument blocks.
inline constexpr size_t MAX_OPERANDS = 4;
inline constexpr size_t MAX_SCALARS = 8;

/// One step of the chain, applied to the running fp32 value v.
enum class OpCode
{
    ADD_OPERAND, ///< v + operand (bias)
    ADD_SCALAR, ///< v + value
    MUL_OPERAND, ///< v * operand (scale)
    MUL_SCALAR, ///< v * value
    RELU, ///< relu / clipped relu / clamp / leaky relu: values lower, upper, slope
    GELU, ///< exact (erf)
    GELU_TANH, ///< tanh approximation
    SWISH, ///< v * sigmoid(beta v): value beta (silu at 1)
    SIGMOID,
    TANH,
    ELU, ///< value alpha
    ABS,
    SOFTPLUS, ///< log(1 + e^(beta v)) / beta, v past 20 / beta: value beta
};

/// A tensor operand the epilogue reads: a per-channel vector ([1, K, 1, ...]) or a one-element
/// device tensor, indexed by the channel or by 0.
struct Operand
{
    int64_t uid = 0;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    bool perChannel = true;
};

/// A value slot of the epilogue's value array: fixed at plan build (an attribute, a constant scalar
/// tensor, a runtime scalar with a default), or read at execute (a runtime pass-by-value scalar
/// tensor the caller supplies).
struct Scalar
{
    float value = 0.0F;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> runtime;
};

struct Op
{
    OpCode code = OpCode::ADD_OPERAND;
    /// The operand index (ADD_OPERAND, MUL_OPERAND) or the first scalar slot.
    int slot = 0;
};

/// The epilogue of a forward convolution graph. Empty for a single-node graph.
struct ConvEpilogue
{
    std::vector<Op> ops;
    std::vector<Operand> operands;
    std::vector<Scalar> scalars;
    /// The convolution node's own output (virtual in a fused graph).
    int64_t convOutputUid = 0;
    /// The graph's written output: the last pointwise node's.
    int64_t outputUid = 0;
    /// The written output's element type. Its precision decides the activations' form: an fp32
    /// output takes the few-ulp form, a 16-bit output the faster one (its error stays far below
    /// the output's rounding).
    hipdnn_flatbuffers_sdk::data_objects::DataType outputType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

    bool empty() const
    {
        return ops.empty();
    }
};

namespace detail
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using data_objects::PointwiseMode;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;

/// Pointwise nodes a fused graph may chain after its convolution.
inline constexpr size_t MAX_CHAIN = 8;

inline const data_objects::TensorAttributes* tensorOf(const IGraph& graph, int64_t uid)
{
    const auto& tensors = graph.getTensorMap();
    const auto found = tensors.find(uid);
    return found == tensors.end() ? nullptr : found->second;
}

inline bool isFloatElement(DataType dataType)
{
    return dataType == DataType::FLOAT || dataType == DataType::HALF
           || dataType == DataType::BFLOAT16;
}

inline int64_t elementBytes(DataType dataType)
{
    return dataType == DataType::FLOAT ? 4 : 2;
}

inline bool dimsEqual(const flatbuffers::Vector<int64_t>* dims,
                      const std::vector<int64_t>& expected)
{
    return dims != nullptr && dims->size() == expected.size()
           && std::equal(dims->begin(), dims->end(), expected.begin());
}

/// One stride per dim, none negative.
inline bool stridesWellFormed(const data_objects::TensorAttributes& tensor)
{
    const auto* strides = tensor.strides();
    const auto* dims = tensor.dims();
    if(strides == nullptr || dims == nullptr || strides->size() != dims->size())
    {
        return false;
    }
    return std::all_of(strides->begin(), strides->end(), [](int64_t s) { return s >= 0; });
}

inline bool ragged(const data_objects::TensorAttributes& tensor)
{
    return tensor.ragged_offset_tensor_uid().has_value() || tensor.ragged_offset_multiplier() != 1;
}

/// A virtual tensor between two steps of the chain: the convolution's shape, a floating element
/// type (the chain computes in fp32 whatever it declares), device data in no particular layout.
inline bool intermediateFits(const data_objects::TensorAttributes& tensor,
                             const std::vector<int64_t>& outputDims)
{
    return tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor)
           && !ragged(tensor) && isFloatElement(tensor.data_type())
           && dimsEqual(tensor.dims(), outputDims) && stridesWellFormed(tensor);
}

template <typename Optional>
inline bool nanValue(const Optional& value)
{
    return value.has_value() && std::isnan(*value);
}

/// The node's compute type is one the fp32 chain honours: UNSET or FLOAT, or a 16-bit type the
/// node's operand or the graph's output holds (cuDNN-style bias nodes computed in the bias type);
/// fp32 is at least the precision any of these asks for.
inline bool computeTypeServed(DataType compute, DataType operandType, DataType outputType)
{
    switch(compute)
    {
    case DataType::UNSET:
    case DataType::FLOAT:
        return true;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return compute == operandType || compute == outputType;
    default:
        return false;
    }
}

class Parser
{
public:
    Parser(const IGraph& graph, std::vector<int64_t> outputDims)
        : _graph(graph)
        , _dims(std::move(outputDims))
    {
    }

    /// Adds the non-chain input of a binary node: an operand tensor, or a value slot.
    /// Returns false when the tensor is not one the epilogue reads.
    bool addBinaryInput(int64_t uid, OpCode operandCode, OpCode scalarCode, DataType& inputType)
    {
        const auto* tensor = tensorOf(_graph, uid);
        if(tensor == nullptr || tensor->virtual_() || ragged(*tensor))
        {
            return false;
        }
        inputType = tensor->data_type();
        const auto* dims = tensor->dims();
        if(dims == nullptr)
        {
            return false;
        }

        if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor))
        {
            return addHostValue(*tensor, uid, scalarCode);
        }

        if(dims->size() != _dims.size())
        {
            return false;
        }
        bool scalar = true;
        for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
        {
            const int64_t extent = (*dims)[axis];
            if(extent == 1)
            {
                continue;
            }
            if(axis != 1 || extent != _dims[1])
            {
                return false;
            }
            scalar = false;
        }
        if(!stridesWellFormed(*tensor))
        {
            return false;
        }

        // Device data, read one element per channel (or element 0) with scalar loads.
        if(!isFloatElement(tensor->data_type())
           || tensor->alignment() < elementBytes(tensor->data_type()))
        {
            return false;
        }
        if(!scalar && (*tensor->strides())[1] != 1)
        {
            return false;
        }
        _epilogue.ops.push_back({operandCode, static_cast<int>(_epilogue.operands.size())});
        _epilogue.operands.push_back({uid, tensor->data_type(), !scalar});
        return _epilogue.operands.size() <= MAX_OPERANDS;
    }

    /// A host value (constant, runtime with a default, or supplied at execute): one element of a
    /// floating type, into a value slot.
    bool addHostValue(const data_objects::TensorAttributes& tensor, int64_t uid, OpCode code)
    {
        const auto* dims = tensor.dims();
        if(!std::all_of(dims->begin(), dims->end(), [](int64_t extent) { return extent == 1; }))
        {
            return false;
        }
        if(!isFloatElement(tensor.data_type()) && tensor.data_type() != DataType::DOUBLE)
        {
            return false;
        }
        Scalar value;
        try
        {
            auto operand = hipdnn_plugin_sdk::makeScalarOperand(
                _graph.getTensorMap(), uid, "conv epilogue scalar");
            if(operand.isRuntimeUserSupplied)
            {
                value.runtime = operand;
            }
            else
            {
                value.value = static_cast<float>(hipdnn_plugin_sdk::toDouble(operand.bakedDefault));
            }
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return false;
        }
        _epilogue.ops.push_back({code, static_cast<int>(_epilogue.scalars.size())});
        _epilogue.scalars.push_back(value);
        return _epilogue.scalars.size() <= MAX_SCALARS;
    }

    void addValues(OpCode code, std::initializer_list<float> values)
    {
        _epilogue.ops.push_back({code, static_cast<int>(_epilogue.scalars.size())});
        for(const float value : values)
        {
            Scalar slot;
            slot.value = value;
            _epilogue.scalars.push_back(slot);
        }
    }

    std::optional<ConvEpilogue> parse(int64_t convOutputUid)
    {
        _epilogue.convOutputUid = convOutputUid;
        _epilogue.outputUid = convOutputUid;
        const auto& graph = _graph;
        const uint32_t nodes = graph.nodeCount();
        if(nodes <= 1)
        {
            return _epilogue;
        }
        if(nodes - 1 > MAX_CHAIN)
        {
            return std::nullopt;
        }

        int64_t current = convOutputUid;
        std::vector<int64_t> chain{current};
        for(uint32_t index = 1; index < nodes; ++index)
        {
            // The value entering this node is an intermediate: virtual, the conv's shape.
            const auto* in = tensorOf(_graph, current);
            if(in == nullptr || !intermediateFits(*in, _dims))
            {
                return std::nullopt;
            }

            const auto& node = graph.getNodeWrapper(index);
            if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
            {
                return std::nullopt;
            }
            const auto& pw = node.attributesAs<data_objects::PointwiseAttributes>();
            if(pw.in_2_tensor_uid().has_value() || pw.axis_tensor_uid().has_value())
            {
                return std::nullopt;
            }
            const auto mode = pw.operation();
            const bool reluMode = mode == PointwiseMode::RELU_FWD;
            if(!reluMode
               && (pw.relu_lower_clip().has_value() || pw.relu_upper_clip().has_value()
                   || pw.relu_lower_clip_slope().has_value()))
            {
                return std::nullopt;
            }
            if((mode != PointwiseMode::SWISH_FWD && pw.swish_beta().has_value())
               || (mode != PointwiseMode::ELU_FWD && pw.elu_alpha().has_value())
               || (mode != PointwiseMode::SOFTPLUS_FWD && pw.softplus_beta().has_value()))
            {
                return std::nullopt;
            }
            if(nanValue(pw.relu_lower_clip()) || nanValue(pw.relu_upper_clip())
               || nanValue(pw.relu_lower_clip_slope()) || nanValue(pw.swish_beta())
               || nanValue(pw.elu_alpha()) || nanValue(pw.softplus_beta()))
            {
                return std::nullopt;
            }

            DataType operandType = DataType::UNSET;
            const bool binary = mode == PointwiseMode::ADD || mode == PointwiseMode::MUL;
            if(binary)
            {
                if(!pw.in_1_tensor_uid().has_value())
                {
                    return std::nullopt;
                }
                const int64_t in0 = pw.in_0_tensor_uid();
                const int64_t in1 = *pw.in_1_tensor_uid();
                // One input is the running value. Both cannot be: the other one is read as an
                // operand, and the running value is virtual, which addBinaryInput declines.
                if(in0 != current && in1 != current)
                {
                    return std::nullopt;
                }
                // The other input is never an earlier step's result: those are virtual, which
                // addBinaryInput declines.
                const int64_t other = in0 == current ? in1 : in0;
                const bool add = mode == PointwiseMode::ADD;
                if(!addBinaryInput(other,
                                   add ? OpCode::ADD_OPERAND : OpCode::MUL_OPERAND,
                                   add ? OpCode::ADD_SCALAR : OpCode::MUL_SCALAR,
                                   operandType))
                {
                    return std::nullopt;
                }
            }
            else
            {
                if(pw.in_0_tensor_uid() != current || pw.in_1_tensor_uid().has_value())
                {
                    return std::nullopt;
                }
                switch(mode)
                {
                case PointwiseMode::RELU_FWD:
                    addValues(
                        OpCode::RELU,
                        {pw.relu_lower_clip().value_or(0.0F),
                         pw.relu_upper_clip().value_or(std::numeric_limits<float>::infinity()),
                         pw.relu_lower_clip_slope().value_or(0.0F)});
                    break;
                case PointwiseMode::GELU_FWD:
                    _epilogue.ops.push_back({OpCode::GELU, 0});
                    break;
                case PointwiseMode::GELU_APPROX_TANH_FWD:
                    _epilogue.ops.push_back({OpCode::GELU_TANH, 0});
                    break;
                case PointwiseMode::SWISH_FWD:
                    addValues(OpCode::SWISH, {pw.swish_beta().value_or(1.0F)});
                    break;
                case PointwiseMode::SIGMOID_FWD:
                    _epilogue.ops.push_back({OpCode::SIGMOID, 0});
                    break;
                case PointwiseMode::TANH_FWD:
                    _epilogue.ops.push_back({OpCode::TANH, 0});
                    break;
                case PointwiseMode::ELU_FWD:
                    addValues(OpCode::ELU, {pw.elu_alpha().value_or(1.0F)});
                    break;
                case PointwiseMode::ABS:
                    _epilogue.ops.push_back({OpCode::ABS, 0});
                    break;
                case PointwiseMode::SOFTPLUS_FWD:
                {
                    const float beta = pw.softplus_beta().value_or(1.0F);
                    if(!(beta > 0.0F) || std::isinf(beta))
                    {
                        return std::nullopt;
                    }
                    addValues(OpCode::SOFTPLUS, {beta});
                    break;
                }
                default:
                    return std::nullopt;
                }
                if(_epilogue.scalars.size() > MAX_SCALARS)
                {
                    return std::nullopt;
                }
            }

            // Each step writes a tensor of its own. Its dims are checked where it is read: by the
            // next step as an intermediate, or by the caller as the graph's output (an operand
            // never has the output's dims).
            const int64_t out = pw.out_0_tensor_uid();
            if(std::find(chain.begin(), chain.end(), out) != chain.end())
            {
                return std::nullopt;
            }
            const auto* outTensor = tensorOf(_graph, out);
            if(outTensor == nullptr)
            {
                return std::nullopt;
            }
            if(!computeTypeServed(node.computeDataType(), operandType, outTensor->data_type()))
            {
                return std::nullopt;
            }
            chain.push_back(out);
            current = out;
        }

        // The last node writes the graph's output: real device memory, which the convolution's
        // own operand checks validate as its y. Every earlier one was checked as an intermediate.
        const auto* output = tensorOf(_graph, current);
        if(output == nullptr || output->virtual_())
        {
            return std::nullopt;
        }
        _epilogue.outputUid = current;
        _epilogue.outputType = output->data_type();
        return _epilogue;
    }

private:
    const IGraph& _graph;
    std::vector<int64_t> _dims;
    ConvEpilogue _epilogue;
};

} // namespace detail

/// The graph's pointwise chain after its convolution node, or nullopt when the graph has nodes
/// after the convolution that the epilogue cannot express. @p convOutputUid is the conv node's y
/// and @p outputDims its dims (n, k, spatial...); every tensor of the chain has them, and
/// per-channel operands broadcast along k. A single-node graph gives an empty epilogue with
/// outputUid the conv's y.
///
/// Gates every field: the node kinds and order, each pointwise mode and its attributes (relu
/// clips and slope, swish beta, elu alpha, softplus beta, axis, in_2), each node's compute type,
/// and for every tensor its dims (the broadcast kind), strides, data type, virtual flag,
/// pass-by-value and ragged fields and alignment. The written output's dims, layout and type are
/// the caller's to check, as the convolution's own y.
inline std::optional<ConvEpilogue>
    parseConvEpilogue(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      int64_t convOutputUid,
                      const std::vector<int64_t>& outputDims)
{
    if(outputDims.size() < 2)
    {
        return std::nullopt;
    }
    detail::Parser parser(graph, outputDims);
    return parser.parse(convOutputUid);
}

} // namespace hipdnn_plugin_sdk::conv_epilogue
