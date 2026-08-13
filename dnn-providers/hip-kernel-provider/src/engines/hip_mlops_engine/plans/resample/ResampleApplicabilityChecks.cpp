// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResampleApplicabilityChecks.hpp"

#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResamplePlanUtils.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <unordered_set>

namespace hip_kernel_provider::resample
{
using namespace hip_kernel_provider::core::utils;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

void ResampleValidator::checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds)
{
    std::vector<TensorDescriptor> tensors;
    tensors.reserve(tensorIds.size());

    for(const auto& id : tensorIds)
    {
        auto attr = _tensorMap.at(id);
        if(attr->value_type() == data_objects::TensorValue::NONE)
        {
            tensors.emplace_back(attr);
        }
    }

    validateConsistentDimensions(tensors);
    validatePackedTensors(tensors);
    validateConsistentLayouts(tensors);
}

void ResampleValidator::checkTensorDataTypesSupported(
    const data_objects::ResampleFwdAttributes& resampleAttr)
{
    const std::unordered_set<data_objects::DataType> allowedIoTypes{
        data_objects::DataType::FLOAT,
        data_objects::DataType::HALF,
        data_objects::DataType::BFLOAT16};

    const auto& xTensor = findTensorAttributes(_tensorMap, resampleAttr.x_tensor_uid());
    const auto& yTensor = findTensorAttributes(_tensorMap, resampleAttr.y_tensor_uid());

    validateDataTypeIsSupported(xTensor.data_type(),
                                allowedIoTypes,
                                "ResampleFwd supports FLOAT, HALF, and BFLOAT16 x tensors.");
    if(yTensor.data_type() != xTensor.data_type())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleFwd requires x and y tensors to have the same data type.");
    }

    const bool hasIndex = resampleAttr.index_tensor_uid().has_value();
    if(hasIndex)
    {
        const auto& indexTensor
            = findTensorAttributes(_tensorMap, resampleAttr.index_tensor_uid().value());
        if(indexTensor.data_type() != data_objects::DataType::INT32)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "ResampleFwd index tensor must have INT32 data type.");
        }
    }
}

void ResampleValidator::checkTensorShapesSupported(
    const data_objects::ResampleFwdAttributes& resampleAttr)
{
    const auto& xTensor = findTensorAttributes(_tensorMap, resampleAttr.x_tensor_uid());
    const auto& yTensor = findTensorAttributes(_tensorMap, resampleAttr.y_tensor_uid());

    const auto xDims = tensorDims(xTensor);
    const auto yDims = tensorDims(yTensor);
    validateResampleOutputShape(xDims,
                                yDims,
                                toStdVector(resampleAttr.pre_padding()),
                                toStdVector(resampleAttr.post_padding()),
                                toStdVector(resampleAttr.stride()),
                                toStdVector(resampleAttr.window()),
                                "ResampleFwd");

    if(resampleAttr.index_tensor_uid().has_value())
    {
        const auto& indexTensor
            = findTensorAttributes(_tensorMap, resampleAttr.index_tensor_uid().value());
        validateResampleIndexShape(indexTensor, yDims, "ResampleFwd");
    }
}

void ResampleValidator::checkTensorConfigSupported(
    const data_objects::ResampleFwdAttributes& resampleAttr)
{
    if(resampleAttr.resample_mode() == data_objects::ResampleMode::NOT_SET)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "ResampleFwd mode must be set.");
    }
    if(resampleAttr.padding_mode() == data_objects::PaddingMode::PADDING_NOT_SET)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "ResampleFwd padding mode must be set.");
    }
    const bool generateIndex
        = resampleAttr.generate_index().has_value() && resampleAttr.generate_index().value();
    if(generateIndex && !resampleAttr.index_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "ResampleFwd generate_index requires an index tensor.");
    }
    if(resampleAttr.index_tensor_uid().has_value()
       && resampleAttr.resample_mode() != data_objects::ResampleMode::MAXPOOL)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleFwd index tensor is supported only for maxpool mode.");
    }

    std::vector<int64_t> tensorIds{resampleAttr.x_tensor_uid(), resampleAttr.y_tensor_uid()};
    if(resampleAttr.index_tensor_uid().has_value())
    {
        tensorIds.push_back(resampleAttr.index_tensor_uid().value());
    }

    checkTensorLayoutsAndDimsSupported(tensorIds);
    checkTensorDataTypesSupported(resampleAttr);
    checkTensorShapesSupported(resampleAttr);
}

} // namespace hip_kernel_provider::resample
