// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <set>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenUnaryActivationChecks.hpp"

namespace miopen_plugin::unary_activation_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

namespace
{

// Prefix for messages emitted before the pointwise mode is known, i.e. before the graph has
// been confirmed to hold a single PointwiseAttributes node.
constexpr auto GENERIC_OP_NAME = "Unary activation";

// Validates the ReLU-family parameters (lower clip, upper clip, lower clip slope) of a node
// whose mode is already known to be RELU_FWD. Throws when the combination cannot be mapped
// onto a MIOpen activation.
//
// NOTE: Keep the branch order below identical to the RELU_FWD/RELU_BWD arm of
// MiopenUtils::mapPointwiseModeToMiopenActivation. If this check accepts a parameter
// combination that the mapping cannot represent, MIOpen may silently execute the op as a
// standard ReLU, leading to incorrect results.
void checkReluParamsSupported(const PointwiseAttributes& attrs)
{
    const auto lowerClip = attrs.relu_lower_clip();
    const auto upperClip = attrs.relu_upper_clip();
    const auto lowerClipSlope = attrs.relu_lower_clip_slope();

    if(lowerClip && upperClip)
    {
        // Clamp. MIOpen's CLAMP floors at lower_clip and has no slope parameter, and this
        // branch wins over the leaky one below, so a non-zero slope would be silently dropped
        // and the op would compute a flat floor instead of a ramp. A zero slope is a no-op
        // that CLAMP already represents exactly, so it stays allowed.
        if(lowerClipSlope && *lowerClipSlope != 0.f)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Relu plan builder: clamp with a non-zero relu_lower_clip_slope is not "
                "supported");
        }
        return; // Clamp
    }
    if(upperClip)
    {
        // Clipped ReLU is likewise flat below the knee, so the same reasoning applies.
        if(lowerClipSlope && *lowerClipSlope != 0.f)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Relu plan builder: clipped relu with a non-zero relu_lower_clip_slope is not "
                "supported");
        }
        return; // Clipped ReLU
    }
    if(lowerClipSlope)
    {
        // Leaky ReLU. MIOpen's LEAKYRELU is parameterized by slope alone, with the knee fixed
        // at 0; it cannot represent a non-zero lower_clip (a shifted knee). Accepting such a
        // combination here would silently drop the lower_clip and produce incorrect results, so
        // decline it instead.
        if(lowerClip && *lowerClip != 0.f)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Relu plan builder: leaky relu (relu_lower_clip_slope) with a non-zero "
                "lower_clip is not supported");
        }
        return; // Leaky ReLU
    }
    if(lowerClip && *lowerClip != 0.f)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Relu plan builder: standard relu with a non-zero lower_clip is not supported");
    }

    // Standard ReLU (including lower_clip == 0.0, which is a no-op lower clip).
}

// Generic IO-tensor validation shared by every unary pointwise activation:
// non-virtual, FLOAT/HALF only, matching dtypes, rank in [1, 4], matching element counts.
// `opName` is used only to make exception/log messages activation-specific
// (e.g. "Relu plan builder: ...").
void checkTensorsSupported(
    const PointwiseAttributes& attrs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::string& opName)
{
    const auto& inputTensor
        = miopen_utils::findTensorAttributes(tensorMap, attrs.in_0_tensor_uid());
    const auto& outputTensor
        = miopen_utils::findTensorAttributes(tensorMap, attrs.out_0_tensor_uid());

    if(inputTensor.virtual_() || outputTensor.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            opName + " plan builder: input and output tensors must be non-virtual");
    }

    const auto inputDtype = inputTensor.data_type();
    const auto outputDtype = outputTensor.data_type();

    if((inputDtype != DataType::FLOAT && inputDtype != DataType::HALF)
       || (outputDtype != DataType::FLOAT && outputDtype != DataType::HALF))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            opName + " plan builder: only FLOAT and HALF IO dtypes are supported");
    }

    if(inputDtype != outputDtype)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            opName + " plan builder: input and output tensors must have the same data type");
    }

    const auto* inputDims = inputTensor.dims();
    const auto* outputDims = outputTensor.dims();
    const auto* inputStrides = inputTensor.strides();
    const auto* outputStrides = outputTensor.strides();

    if(inputDims == nullptr || outputDims == nullptr || inputStrides == nullptr
       || outputStrides == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            opName + " plan builder: tensor dims or strides are null");
    }

    if(inputDims->size() != inputStrides->size() || outputDims->size() != outputStrides->size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            opName + " plan builder: tensor dims and strides size mismatch");
    }

    const auto rank = inputDims->size();

    if(rank < 1 || rank > 4)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            opName + " plan builder: tensor rank must be between 1 and 4, got "
                + std::to_string(rank));
    }

    int64_t inputElementCount = 1;
    for(const auto dim : *inputDims)
    {
        inputElementCount *= static_cast<int64_t>(dim);
    }

    int64_t outputElementCount = 1;
    for(const auto dim : *outputDims)
    {
        outputElementCount *= static_cast<int64_t>(dim);
    }

    if(inputElementCount != outputElementCount)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            opName + " plan builder: input and output tensors must have the same element count");
    }
}

} // namespace

bool isSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            GENERIC_OP_NAME << " plan builder is applicable only for single-node graphs. Graph has "
                            << opGraph.nodeCount() << " nodes");
        return false;
    }

    if(!opGraph.hasOnlySupportedAttributes(
           std::set<NodeAttributes>{NodeAttributes::PointwiseAttributes}))
    {
        HIPDNN_PLUGIN_LOG_INFO(GENERIC_OP_NAME << " plan builder is not applicable for this graph");
        return false;
    }

    if(opGraph.getNode(0).compute_data_type() != DataType::FLOAT)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            GENERIC_OP_NAME << " plan builder only supports nodes with an fp32 compute_data_type");
        return false;
    }

    const auto& attrs = opGraph.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

    try
    {
        // NOTE: Only add a mode here once MiopenUtils::mapPointwiseModeToMiopenActivation can
        // faithfully represent it AND MIOpen's forward activation is the right kernel for it.
        // Two traps to avoid:
        //  - The mapping covers more modes than are safe to accept. SOFTPLUS_* throws unless
        //    beta == 1, and GELU/SWISH have no mapping at all, so accepting them here would make
        //    isApplicable() return true and buildPlan() throw afterwards.
        //  - The mapping also covers the *_BWD modes, but MiopenUnaryActivationPlan always calls
        //    miopenActivationForward, so accepting a *_BWD mode would silently compute the
        //    forward activation instead of its gradient.
        // The display name is bound here so that the tensor checks below can name the activation.
        const char* opName = nullptr;

        switch(attrs.operation())
        {
        case PointwiseMode::RELU_FWD:
            opName = "Relu";
            checkReluParamsSupported(attrs);
            break;
        case PointwiseMode::SIGMOID_FWD:
            opName = "Sigmoid";
            break;
        case PointwiseMode::TANH_FWD:
            opName = "Tanh";
            break;
        default:
            HIPDNN_PLUGIN_LOG_INFO(
                GENERIC_OP_NAME
                << " plan builder: unsupported pointwise mode. Supported modes: RELU_FWD, "
                   "SIGMOID_FWD, TANH_FWD");
            return false;
        }

        checkTensorsSupported(attrs, opGraph.getTensorMap(), opName);
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }

    return true;
}

} // namespace miopen_plugin::unary_activation_applicability
