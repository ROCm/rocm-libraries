// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <set>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenPointwiseApplicabilityChecks.hpp"

namespace miopen_plugin
{

namespace pointwise_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

void checkPointwiseModeSupported(const PointwiseAttributes& attrs)
{
    if(attrs.operation() != PointwiseMode::RELU_FWD)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Pointwise plan builder: unsupported pointwise mode. "
            "Supported modes for standalone pointwise: RELU_FWD");
    }
}

void checkPointwiseTensorsSupported(
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
            "Pointwise plan builder: input and output tensors must be non-virtual for a "
            "standalone pointwise operation");
    }
}

bool isSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Pointwise plan builder is applicable only for single-node graphs. Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }

    if(!opGraph.hasOnlySupportedAttributes(
           std::set<NodeAttributes>{NodeAttributes::PointwiseAttributes}))
    {
        HIPDNN_PLUGIN_LOG_INFO("Pointwise plan builder is not applicable for this graph");
        return false;
    }

    if(opGraph.getNode(0).compute_data_type() != DataType::FLOAT)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "Pointwise plan builder only supports nodes with an fp32 compute_data_type");
        return false;
    }

    const auto& attrs = opGraph.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

    try
    {
        checkPointwiseModeSupported(attrs);
        checkPointwiseTensorsSupported(attrs, opGraph.getTensorMap());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }

    return true;
}

} // namespace pointwise_applicability

} // namespace miopen_plugin
