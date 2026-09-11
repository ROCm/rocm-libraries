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

void ResampleValidator::checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                                      const std::optional<int64_t>& indexTensorId)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for Resample.");
    }

    const std::unordered_set<data_objects::DataType> allowedIoTypes{
        data_objects::DataType::FLOAT,
        data_objects::DataType::HALF,
        data_objects::DataType::BFLOAT16};

    const auto& refTensor = findTensorAttributes(_tensorMap, ioTensorIds[0]);
    validateDataTypeIsSupported(refTensor.data_type(),
                                allowedIoTypes,
                                "Resample supports FLOAT, HALF, and BFLOAT16 IO tensors.");

    for(size_t i = 1; i < ioTensorIds.size(); ++i)
    {
        const auto& tensorAttr = findTensorAttributes(_tensorMap, ioTensorIds[i]);
        validateDataTypeIsSupported(tensorAttr.data_type(),
                                    allowedIoTypes,
                                    "Resample supports FLOAT, HALF, and BFLOAT16 IO tensors.");

        if(tensorAttr.data_type() != refTensor.data_type())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Resample requires all IO tensors to have the same data type.");
        }
    }

    if(indexTensorId.has_value())
    {
        const auto& indexTensor = findTensorAttributes(_tensorMap, indexTensorId.value());
        if(indexTensor.data_type() != data_objects::DataType::INT32)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Resample index tensor must have INT32 data type.");
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
                                toStdVector(resampleAttr.window()));

    if(resampleAttr.index_tensor_uid().has_value())
    {
        const auto& indexTensor
            = findTensorAttributes(_tensorMap, resampleAttr.index_tensor_uid().value());
        validateResampleIndexShape(indexTensor, yDims, "ResampleFwd");
    }
}

void ResampleValidator::checkBwdTensorShapesSupported(
    const data_objects::ResampleBwdAttributes& resampleBwdAttr)
{
    const auto& dyTensor = findTensorAttributes(_tensorMap, resampleBwdAttr.dy_tensor_uid());
    const auto& dxTensor = findTensorAttributes(_tensorMap, resampleBwdAttr.dx_tensor_uid());

    const auto dyDims = tensorDims(dyTensor);
    const auto dxDims = tensorDims(dxTensor);
    validateResampleBwdOutputShape(dyDims,
                                   dxDims,
                                   toStdVector(resampleBwdAttr.pre_padding()),
                                   toStdVector(resampleBwdAttr.post_padding()),
                                   toStdVector(resampleBwdAttr.stride()),
                                   toStdVector(resampleBwdAttr.window()));

    if(resampleBwdAttr.index_tensor_uid().has_value())
    {
        const auto& indexTensor
            = findTensorAttributes(_tensorMap, resampleBwdAttr.index_tensor_uid().value());
        validateResampleIndexShape(indexTensor, dyDims, "ResampleBwd");
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
    checkTensorDataTypesSupported({tensorIds[0], tensorIds[1]}, resampleAttr.index_tensor_uid());
    checkTensorShapesSupported(resampleAttr);
}

void ResampleValidator::checkBwdTensorConfigSupported(
    const data_objects::ResampleBwdAttributes& resampleBwdAttr)
{
    const auto resampleMode = resampleBwdAttr.resample_mode();
    if(resampleMode != data_objects::ResampleMode::MAXPOOL
       && resampleMode != data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING
       && resampleMode != data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "ResampleBwd resample mode is unsupported.");
    }

    if(resampleBwdAttr.padding_mode() == data_objects::PaddingMode::PADDING_NOT_SET)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "ResampleBwd padding mode must be set.");
    }

    if(resampleBwdAttr.index_tensor_uid().has_value()
       && resampleBwdAttr.resample_mode() != data_objects::ResampleMode::MAXPOOL)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleBwd index tensor is supported only for maxpool mode.");
    }

    if(resampleBwdAttr.resample_mode() == data_objects::ResampleMode::MAXPOOL
       && !resampleBwdAttr.index_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleBwd with maxpool mode requires an index tensor.");
    }

    std::vector<int64_t> tensorIds{resampleBwdAttr.dy_tensor_uid(),
                                   resampleBwdAttr.dx_tensor_uid()};
    if(resampleBwdAttr.index_tensor_uid().has_value())
    {
        tensorIds.push_back(resampleBwdAttr.index_tensor_uid().value());
    }

    checkTensorLayoutsAndDimsSupported(tensorIds);
    checkTensorDataTypesSupported({tensorIds[0], tensorIds[1]}, resampleBwdAttr.index_tensor_uid());
    checkBwdTensorShapesSupported(resampleBwdAttr);
}

} // namespace hip_kernel_provider::resample
