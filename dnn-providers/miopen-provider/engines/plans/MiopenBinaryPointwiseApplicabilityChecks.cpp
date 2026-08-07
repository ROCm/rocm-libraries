// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <set>
#include <string>
#include <unordered_map>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenBinaryPointwiseApplicabilityChecks.hpp"

namespace miopen_plugin
{

namespace binary_pointwise_applicability
{
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

void checkModeSupported(const PointwiseAttributes& attrs)
{
    const auto mode = attrs.operation();
    if(mode != PointwiseMode::ADD && mode != PointwiseMode::SUB && mode != PointwiseMode::MUL)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan builder: unsupported pointwise mode. "
            "Supported mode: ADD, SUB, MUL");
    }
}

void checkTensorsSupported(
    const PointwiseAttributes& attrs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Extract both inputs (in_0 AND in_1) and the single output (out_0)
    const auto& inputTensor0
        = miopen_utils::findTensorAttributes(tensorMap, attrs.in_0_tensor_uid());

    // Since in_1 is an optional field in FlatBuffers, we ensure it's valid before unpacking
    if(!attrs.in_1_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan builder: binary operations require a valid in_1_tensor_uid");
    }

    const auto& inputTensor1
        = miopen_utils::findTensorAttributes(tensorMap, *attrs.in_1_tensor_uid());
    const auto& outputTensor
        = miopen_utils::findTensorAttributes(tensorMap, attrs.out_0_tensor_uid());

    // Virtual tensor checks
    if(inputTensor0.virtual_() || inputTensor1.virtual_() || outputTensor.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan builder: input and output tensors must be non-virtual");
    }

    // Datatype validations
    const auto input0Dtype = inputTensor0.data_type();
    const auto input1Dtype = inputTensor1.data_type();
    const auto outputDtype = outputTensor.data_type();

    if((input0Dtype != DataType::FLOAT && input0Dtype != DataType::HALF)
       || (input1Dtype != DataType::FLOAT && input1Dtype != DataType::HALF)
       || (outputDtype != DataType::FLOAT && outputDtype != DataType::HALF))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan builder: only FLOAT and HALF IO dtypes are supported");
    }

    if(input0Dtype != outputDtype || input1Dtype != outputDtype)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan: all input and output tensors must share the same data type");
    }

    // Dimension and Rank checks
    const auto* input0Dims = inputTensor0.dims();
    const auto* input1Dims = inputTensor1.dims();
    const auto* outputDims = outputTensor.dims();

    if(input0Dims == nullptr || input1Dims == nullptr || outputDims == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Binary pointwise plan builder: tensor dims are null");
    }

    const auto rank = input0Dims->size();

    if(rank != input1Dims->size() || rank != outputDims->size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan builder: input and output tensors must have matching "
            "dimensions/ranks");
    }

    if(rank < 1 || rank > 4)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("Binary pointwise plan builder: tensor rank must be between 1 and 4, got ")
                + std::to_string(rank));
    }

    // Compute element count to verify consistency
    int64_t input0ElementCount = 1;
    for(const auto dim : *input0Dims)
    {
        input0ElementCount *= static_cast<int64_t>(dim);
    }

    int64_t input1ElementCount = 1;
    for(const auto dim : *input1Dims)
    {
        input1ElementCount *= static_cast<int64_t>(dim);
    }

    int64_t outputElementCount = 1;
    for(const auto dim : *outputDims)
    {
        outputElementCount *= static_cast<int64_t>(dim);
    }

    if(input0ElementCount != outputElementCount || input1ElementCount != outputElementCount)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan builder: all tensors must have the exact same total element "
            "count");
    }

    const auto* input0Strides = inputTensor0.strides();
    const auto* input1Strides = inputTensor1.strides();
    const auto* outputStrides = outputTensor.strides();

    if(input0Strides == nullptr || input1Strides == nullptr || outputStrides == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan builder: tensor strides are null");
    }

    if(input0Strides->size() != rank || input1Strides->size() != rank
       || outputStrides->size() != rank)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise plan builder: stride dimensionality does not match tensor rank");
    }
}

bool isBinaryPointwiseSupported(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Binary pointwise plan builder is applicable only for single-node graphs. Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }

    if(!opGraph.hasOnlySupportedAttributes(
           std::set<NodeAttributes>{NodeAttributes::PointwiseAttributes}))
    {
        HIPDNN_PLUGIN_LOG_INFO("Binary pointwise plan builder is not applicable for this graph");
        return false;
    }

    if(opGraph.getNode(0).compute_data_type() != DataType::FLOAT)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Binary pointwise builder only supports nodes with an fp32 compute_data_type");
        return false;
    }

    const auto& attrs = opGraph.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

    try
    {
        checkModeSupported(attrs);
        checkTensorsSupported(attrs, opGraph.getTensorMap());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }

    return true;
}

}

}
