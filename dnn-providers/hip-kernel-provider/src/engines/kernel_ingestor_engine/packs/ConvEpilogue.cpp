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
using hipdnn_plugin_sdk::ingestor::MatchContext;

// The defines the generated code travels in (ConvEpilogue.h).
constexpr std::string_view EPI_DEFINE = "CONV_EPI";
constexpr std::string_view FIELDS_DEFINE = "CONV_EPI_CHAN_FIELDS";
constexpr std::string_view CHAN_DEFINE = "CONV_EPI_CHAN_CODE";
constexpr std::string_view CODE_DEFINE = "CONV_EPI_CODE";

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

/// A relu step's form, which its attributes fix at plan build: "relu" (lower clip only),
/// "relu.clamp" (an upper clip), "relu.leaky" (a lower-clip slope) or "relu.leaky.clamp".
std::string reluForm(const ConvEpilogue& epilogue, const Op& op)
{
    const auto& scalars = epilogue.scalars;
    const bool upper = !std::isinf(scalars[static_cast<size_t>(op.slot) + 1].value)
                       || scalars[static_cast<size_t>(op.slot) + 1].value < 0.0F;
    const bool slope = scalars[static_cast<size_t>(op.slot) + 2].value != 0.0F;
    std::string form = "relu";
    if(slope)
    {
        form += ".leaky";
    }
    if(upper)
    {
        form += ".clamp";
    }
    return form;
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
    return hipdnn_plugin_sdk::conv_epilogue::parseConvEpilogue(
        context.graph, convOutputUid, outputDims);
}

std::string patternOf(const ConvEpilogue& epilogue)
{
    // The generated code is a function of this text alone. The activations built on exp2 come in
    // two forms, the few-ulp one for an fp32 output (".fp32") and a faster one for 16-bit outputs.
    const std::string form = epilogue.outputType == data_objects::DataType::FLOAT
                                 ? std::string(".fp32")
                                 : std::string();
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
            pattern += reluForm(epilogue, op);
            break;
        case OpCode::GELU:
            pattern += "gelu" + form;
            break;
        case OpCode::GELU_TANH:
            pattern += "gelu_tanh" + form;
            break;
        case OpCode::SWISH:
            pattern += "swish" + form;
            break;
        case OpCode::SIGMOID:
            pattern += "sigmoid" + form;
            break;
        case OpCode::TANH:
            pattern += "tanh" + form;
            break;
        case OpCode::ELU:
            pattern += "elu" + form;
            break;
        case OpCode::ABS:
            pattern += "abs";
            break;
        case OpCode::SOFTPLUS:
            pattern += "softplus" + form;
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
    // The activations' form: few-ulp for an fp32 output, the faster form for a 16-bit output.
    const std::string precision
        = epilogue.outputType == data_objects::DataType::FLOAT ? "<1>" : "<0>";
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
        {
            // The form is the activation (relu, clipped relu or clamp, leaky relu); the clip
            // and slope values stay runtime arguments.
            const auto form = reluForm(epilogue, op);
            if(form == "relu")
            {
                code += "v=convEpiReluLower(v," + slot(op.slot) + ");";
            }
            else if(form == "relu.clamp")
            {
                code += "v=convEpiReluClamp(v," + slot(op.slot) + "," + slot(op.slot + 1) + ");";
            }
            else if(form == "relu.leaky")
            {
                code += "v=convEpiReluLeaky(v," + slot(op.slot) + "," + slot(op.slot + 2) + ");";
            }
            else
            {
                code += "v=convEpiRelu(v," + slot(op.slot) + "," + slot(op.slot + 1) + ","
                        + slot(op.slot + 2) + ");";
            }
            break;
        }
        case OpCode::GELU:
            code += "v=convEpiGelu" + precision + "(v);";
            break;
        case OpCode::GELU_TANH:
            code += "v=convEpiGeluTanh" + precision + "(v);";
            break;
        case OpCode::SWISH:
            code += "v=convEpiSwish" + precision + "(v," + slot(op.slot) + ");";
            break;
        case OpCode::SIGMOID:
            code += "v=convEpiSigmoid" + precision + "(v);";
            break;
        case OpCode::TANH:
            code += "v=convEpiTanh" + precision + "(v);";
            break;
        case OpCode::ELU:
            code += "v=convEpiElu" + precision + "(v," + slot(op.slot) + ");";
            break;
        case OpCode::ABS:
            code += "v=fabsf(v);";
            break;
        case OpCode::SOFTPLUS:
            code += "v=convEpiSoftplus" + precision + "(v," + slot(op.slot) + ");";
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
