// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvEpilogue.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::conv_epilogue
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using data_objects::PointwiseMode;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// Pointwise nodes a fused graph may chain after its convolution.
constexpr size_t MAX_CHAIN = 8;

// The defines the generated code travels in (ConvEpilogue.h).
constexpr std::string_view EPI_DEFINE = "CONV_EPI";
constexpr std::string_view FIELDS_DEFINE = "CONV_EPI_CHAN_FIELDS";
constexpr std::string_view CHAN_DEFINE = "CONV_EPI_CHAN_CODE";
constexpr std::string_view CODE_DEFINE = "CONV_EPI_CODE";

const data_objects::TensorAttributes* tensorOf(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto found = tensors.find(uid);
    return found == tensors.end() ? nullptr : found->second;
}

bool isFloatElement(DataType dataType)
{
    return dataType == DataType::FLOAT || dataType == DataType::HALF
           || dataType == DataType::BFLOAT16;
}

int64_t elementBytes(DataType dataType)
{
    return dataType == DataType::FLOAT ? 4 : 2;
}

bool dimsEqual(const flatbuffers::Vector<int64_t>* dims, const std::vector<int64_t>& expected)
{
    return dims != nullptr && dims->size() == expected.size()
           && std::equal(dims->begin(), dims->end(), expected.begin());
}

/// Strides of a tensor's rank, none negative.
bool stridesWellFormed(const data_objects::TensorAttributes& tensor, size_t rank)
{
    const auto* strides = tensor.strides();
    if(strides == nullptr || strides->size() != rank)
    {
        return false;
    }
    return std::all_of(strides->begin(), strides->end(), [](int64_t s) { return s >= 0; });
}

bool ragged(const data_objects::TensorAttributes& tensor)
{
    return tensor.ragged_offset_tensor_uid().has_value() || tensor.ragged_offset_multiplier() != 1;
}

/// A virtual tensor between two steps of the chain: the convolution's shape, a floating element
/// type (the chain computes in fp32 whatever it declares), device data in no particular layout.
bool intermediateFits(const data_objects::TensorAttributes& tensor,
                      const std::vector<int64_t>& outputDims)
{
    return tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor)
           && !ragged(tensor) && isFloatElement(tensor.data_type())
           && dimsEqual(tensor.dims(), outputDims) && stridesWellFormed(tensor, outputDims.size());
}

template <typename Optional>
bool nanValue(const Optional& value)
{
    return value.has_value() && std::isnan(*value);
}

/// The node's compute type is one the fp32 chain honours: UNSET or FLOAT, or a 16-bit type the
/// node's operand or the graph's output holds (cuDNN-style bias nodes computed in the bias type);
/// fp32 is at least the precision any of these asks for.
bool computeTypeServed(DataType compute, DataType operandType, DataType outputType)
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
    Parser(const MatchContext& context, std::vector<int64_t> outputDims)
        : _context(context)
        , _dims(std::move(outputDims))
    {
    }

    /// Adds the non-chain input of a binary node: an operand tensor, or a value slot.
    /// Returns false when the tensor is not one the epilogue reads.
    bool addBinaryInput(int64_t uid, OpCode operandCode, OpCode scalarCode, DataType& inputType)
    {
        const auto* tensor = tensorOf(_context, uid);
        if(tensor == nullptr || tensor->virtual_() || ragged(*tensor))
        {
            return false;
        }
        inputType = tensor->data_type();
        const auto* dims = tensor->dims();
        if(dims == nullptr || dims->size() == 0)
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
        if(!stridesWellFormed(*tensor, _dims.size()))
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
        return _epilogue.operands.size() <= CONV_EPI_OPERANDS;
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
                _context.graph.getTensorMap(), uid, "conv epilogue scalar");
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
        return _epilogue.scalars.size() <= CONV_EPI_SCALARS;
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
        const auto& graph = _context.graph;
        const auto nodes = static_cast<uint32_t>(graph.nodeCount());
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
            const auto* in = tensorOf(_context, current);
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
                // Exactly one input is the running value.
                if((in0 == current) == (in1 == current))
                {
                    return std::nullopt;
                }
                const int64_t other = in0 == current ? in1 : in0;
                if(std::find(chain.begin(), chain.end(), other) != chain.end())
                {
                    return std::nullopt;
                }
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
                if(_epilogue.scalars.size() > CONV_EPI_SCALARS)
                {
                    return std::nullopt;
                }
            }

            const int64_t out = pw.out_0_tensor_uid();
            if(std::find(chain.begin(), chain.end(), out) != chain.end()
               || std::any_of(_epilogue.operands.begin(),
                              _epilogue.operands.end(),
                              [out](const Operand& operand) { return operand.uid == out; }))
            {
                return std::nullopt;
            }
            const auto* outTensor = tensorOf(_context, out);
            if(outTensor == nullptr || !dimsEqual(outTensor->dims(), _dims))
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
        const auto* output = tensorOf(_context, current);
        if(output == nullptr || output->virtual_())
        {
            return std::nullopt;
        }
        _epilogue.outputUid = current;
        return _epilogue;
    }

private:
    const MatchContext& _context;
    std::vector<int64_t> _dims;
    ConvEpilogue _epilogue;
};

std::string_view loaderFor(DataType dataType)
{
    switch(dataType)
    {
    case DataType::FLOAT:
        return "convEpiLoadF32";
    case DataType::HALF:
        return "convEpiLoadF16";
    case DataType::BFLOAT16:
        return "convEpiLoadBf16";
    default:
        break;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "conv epilogue operand of an unsupported type");
}

std::string slot(int index)
{
    return "e.scalar[" + std::to_string(index) + "]";
}

std::string reg(int index)
{
    return "c.c" + std::to_string(index);
}

} // namespace

std::optional<ConvEpilogue> parseConvEpilogue(const MatchContext& context,
                                              int64_t convOutputUid,
                                              const std::vector<int64_t>& outputDims)
{
    if(outputDims.size() < 2)
    {
        return std::nullopt;
    }
    Parser parser(context, outputDims);
    return parser.parse(convOutputUid);
}

std::string patternOf(const ConvEpilogue& epilogue)
{
    // The generated code is a function of this text alone.
    std::string pattern;
    for(const auto& op : epilogue.ops)
    {
        if(!pattern.empty())
        {
            pattern += '|';
        }
        switch(op.code)
        {
        case OpCode::ADD_OPERAND:
        case OpCode::MUL_OPERAND:
        {
            const auto& operand = epilogue.operands[static_cast<size_t>(op.slot)];
            pattern += op.code == OpCode::ADD_OPERAND ? "add" : "mul";
            pattern += operand.perChannel ? ".k." : ".1.";
            pattern += data_objects::EnumNameDataType(operand.dataType);
            break;
        }
        case OpCode::ADD_SCALAR:
            pattern += "add.value";
            break;
        case OpCode::MUL_SCALAR:
            pattern += "mul.value";
            break;
        case OpCode::RELU:
            pattern += "relu";
            break;
        case OpCode::GELU:
            pattern += "gelu";
            break;
        case OpCode::GELU_TANH:
            pattern += "gelu_tanh";
            break;
        case OpCode::SWISH:
            pattern += "swish";
            break;
        case OpCode::SIGMOID:
            pattern += "sigmoid";
            break;
        case OpCode::TANH:
            pattern += "tanh";
            break;
        case OpCode::ELU:
            pattern += "elu";
            break;
        case OpCode::ABS:
            pattern += "abs";
            break;
        case OpCode::SOFTPLUS:
            pattern += "softplus";
            break;
        default:
            break;
        }
    }
    return pattern;
}

void addEpilogueDefines(compilation::KernelCompileOptions& options, const ConvEpilogue& epilogue)
{
    if(epilogue.empty())
    {
        return;
    }
    std::string fields;
    std::string chan;
    for(size_t i = 0; i < epilogue.operands.size(); ++i)
    {
        const auto& operand = epilogue.operands[i];
        const int index = static_cast<int>(i);
        fields += "float c" + std::to_string(index) + ";";
        // A column past the edge reads channel 0, whose value its discarded result never
        // uses: the load is unconditional, so the store code stays one basic block (a load
        // under a branch there costs the tile kernels their register allocation).
        chan += reg(index) + "=" + std::string(loaderFor(operand.dataType)) + "(e.operand["
                + std::to_string(index) + "]," + (operand.perChannel ? "ok?ch:0" : "0") + ");";
    }
    std::string code;
    for(const auto& op : epilogue.ops)
    {
        switch(op.code)
        {
        case OpCode::ADD_OPERAND:
            code += "v+=" + reg(op.slot) + ";";
            break;
        case OpCode::MUL_OPERAND:
            code += "v*=" + reg(op.slot) + ";";
            break;
        case OpCode::ADD_SCALAR:
            code += "v+=" + slot(op.slot) + ";";
            break;
        case OpCode::MUL_SCALAR:
            code += "v*=" + slot(op.slot) + ";";
            break;
        case OpCode::RELU:
            code += "v=convEpiRelu(v," + slot(op.slot) + "," + slot(op.slot + 1) + ","
                    + slot(op.slot + 2) + ");";
            break;
        case OpCode::GELU:
            code += "v=convEpiGelu(v);";
            break;
        case OpCode::GELU_TANH:
            code += "v=convEpiGeluTanh(v);";
            break;
        case OpCode::SWISH:
            code += "v=convEpiSwish(v," + slot(op.slot) + ");";
            break;
        case OpCode::SIGMOID:
            code += "v=convEpiSigmoid(v);";
            break;
        case OpCode::TANH:
            code += "v=convEpiTanh(v);";
            break;
        case OpCode::ELU:
            code += "v=convEpiElu(v," + slot(op.slot) + ");";
            break;
        case OpCode::ABS:
            code += "v=fabsf(v);";
            break;
        case OpCode::SOFTPLUS:
            code += "v=convEpiSoftplus(v," + slot(op.slot) + ");";
            break;
        default:
            break;
        }
    }
    options.add(std::string(EPI_DEFINE), std::string("1"));
    options.add(std::string(FIELDS_DEFINE), fields);
    options.add(std::string(CHAN_DEFINE), chan);
    options.add(std::string(CODE_DEFINE), code);
}

void removeEpilogueDefines(compilation::KernelCompileOptions& options)
{
    for(const auto name : {EPI_DEFINE, FIELDS_DEFINE, CHAN_DEFINE, CODE_DEFINE})
    {
        options.remove(std::string(name));
    }
}

ConvEpiArgs epilogueArgs(const ConvEpilogue& epilogue,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         long long chanStride,
                         int chans)
{
    ConvEpiArgs args{};
    for(size_t i = 0; i < epilogue.operands.size() && i < CONV_EPI_OPERANDS; ++i)
    {
        args.operand[i] = hipdnn_plugin_sdk::findDeviceBuffer(
                              epilogue.operands[i].uid, deviceBuffers, numDeviceBuffers)
                              .ptr;
    }
    for(size_t i = 0; i < epilogue.scalars.size() && i < CONV_EPI_SCALARS; ++i)
    {
        const auto& scalar = epilogue.scalars[i];
        args.scalar[i] = scalar.runtime.has_value()
                             ? static_cast<float>(hipdnn_plugin_sdk::toDouble(
                                   hipdnn_plugin_sdk::resolveScalarOperand(
                                       *scalar.runtime, deviceBuffers, numDeviceBuffers)))
                             : scalar.value;
    }
    args.chanStride = chanStride < 1 ? 1 : chanStride;
    args.chans = chans < 1 ? 1 : chans;
    return args;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_epilogue

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
