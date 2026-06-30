// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <set>
#include <unordered_map>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenSigmoidApplicabilityChecks.hpp"

namespace miopen_plugin
{

namespace sigmoid_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

void checkSigmoidModeSupported(const PointwiseAttributes& attrs)
{
    if(attrs.operation() != PointwiseMode::SIGMOID_FWD)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Sigmoid plan builder: unsupported pointwise mode. "
            "Supported mode: SIGMOID_FWD");
    }
}

void checkSigmoidTensorsSupported(
    const PointwiseAttributes& attrs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto& inputTensor
        = miopen_utils::findTensorAttributes(tensorMap, attrs.in_0_tensor_uid());
    const auto& outputTensor
        = miopen_utils::findTensorAttributes(tensorMap, attrs.out_0_tensor_uid());

    if(inputTensor.virtual_() || outputTensor.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Sigmoid plan builder: input and output tensors must be non-virtual");
    }

    const auto inputDtype = inputTensor.data_type();
    const auto outputDtype = outputTensor.data_type();

    if((inputDtype != DataType::FLOAT && inputDtype != DataType::HALF)
       || (outputDtype != DataType::FLOAT && outputDtype != DataType::HALF))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Sigmoid plan builder: only FLOAT and HALF IO dtypes are supported");
    }

    if(inputDtype != outputDtype)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Sigmoid plan builder: input and output tensors must have the same data type");
    }

    const auto* inputDims = inputTensor.dims();
    const auto* outputDims = outputTensor.dims();

    if(inputDims == nullptr || outputDims == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Sigmoid plan builder: tensor dims are null");
    }

    const auto rank = inputDims->size();

    if(rank < 1 || rank > 4)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("Sigmoid plan builder: tensor rank must be between 1 and 4, got ")
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
            "Sigmoid plan builder: input and output tensors must have the same element count");
    }
}

bool isSigmoidSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Sigmoid plan builder is applicable only for single-node graphs. Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }

    if(!opGraph.hasOnlySupportedAttributes(
           std::set<NodeAttributes>{NodeAttributes::PointwiseAttributes}))
    {
        HIPDNN_PLUGIN_LOG_INFO("Sigmoid plan builder is not applicable for this graph");
        return false;
    }

    if(opGraph.getNode(0).compute_data_type() != DataType::FLOAT)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Sigmoid plan builder only supports nodes with an fp32 compute_data_type");
        return false;
    }

    const auto& attrs = opGraph.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

    try
    {
        checkSigmoidModeSupported(attrs);
        checkSigmoidTensorsSupported(attrs, opGraph.getTensorMap());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }

    return true;
}

} // namespace sigmoid_applicability

} // namespace miopen_plugin
