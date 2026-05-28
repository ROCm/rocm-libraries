// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResampleApplicabilityChecks.hpp"

#include "core/Utils.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <unordered_set>

namespace hip_kernel_provider::resample
{
using namespace hip_kernel_provider::core::utils;

namespace
{
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

std::vector<int64_t> tensorDims(const data_objects::TensorAttributes& tensor)
{
    return {tensor.dims()->begin(), tensor.dims()->end()};
}

void validateSpatialVector(const std::vector<int64_t>& values,
                           size_t spatialDims,
                           const std::string& name,
                           bool allowZero)
{
    if(values.size() != spatialDims)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleFwd " + name + " rank must match the number of spatial dimensions.");
    }

    for(const auto value : values)
    {
        if(allowZero ? value < 0 : value <= 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "ResampleFwd " + name + " values must be "
                    + std::string(allowZero ? "non-negative." : "positive."));
        }
    }
}

} // namespace

std::vector<int64_t> ResampleValidator::toVector(const flatbuffers::Vector<int64_t>* values)
{
    if(values == nullptr)
    {
        return {};
    }
    return {values->begin(), values->end()};
}

void ResampleValidator::checkTensorLayoutsAndDimsSupported()
{
    std::vector<TensorDescriptor> tensors;
    tensors.reserve(_tensorMap.size());

    for(const auto& [id, attr] : _tensorMap)
    {
        (void)id;
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

    const auto& xTensor
        = findTensorAttributes(_tensorMap, resampleAttr.x_tensor_uid());
    const auto& yTensor
        = findTensorAttributes(_tensorMap, resampleAttr.y_tensor_uid());

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
    const auto& xTensor
        = findTensorAttributes(_tensorMap, resampleAttr.x_tensor_uid());
    const auto& yTensor
        = findTensorAttributes(_tensorMap, resampleAttr.y_tensor_uid());

    const auto xDims = tensorDims(xTensor);
    const auto yDims = tensorDims(yTensor);
    if(xDims.size() != yDims.size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "ResampleFwd x and y tensors must have the same rank.");
    }

    const auto spatialDims = xDims.size() - 2;
    const auto prePadding = toVector(resampleAttr.pre_padding());
    const auto postPadding = toVector(resampleAttr.post_padding());
    const auto stride = toVector(resampleAttr.stride());
    const auto window = toVector(resampleAttr.window());

    validateSpatialVector(prePadding, spatialDims, "pre_padding", true);
    validateSpatialVector(postPadding, spatialDims, "post_padding", true);
    validateSpatialVector(stride, spatialDims, "stride", false);
    validateSpatialVector(window, spatialDims, "window", false);

    if(xDims[0] != yDims[0] || xDims[1] != yDims[1])
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleFwd preserves the batch and channel dimensions.");
    }

    for(size_t i = 0; i < spatialDims; ++i)
    {
        const auto expected
            = (xDims[i + 2] + prePadding[i] + postPadding[i] - window[i]) / stride[i] + 1;
        if(expected <= 0 || yDims[i + 2] != expected)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "ResampleFwd y spatial dimensions must match the resample parameters.");
        }
    }

    if(resampleAttr.index_tensor_uid().has_value())
    {
        const auto& indexTensor
            = findTensorAttributes(_tensorMap, resampleAttr.index_tensor_uid().value());
        const auto indexDims = tensorDims(indexTensor);
        if(indexDims != yDims)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "ResampleFwd index tensor must have the same shape as y.");
        }
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
    if(resampleAttr.generate_index().has_value() && resampleAttr.generate_index().value()
       && !resampleAttr.index_tensor_uid().has_value())
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

    checkTensorLayoutsAndDimsSupported();
    checkTensorDataTypesSupported(resampleAttr);
    checkTensorShapesSupported(resampleAttr);
}

} // namespace hip_kernel_provider::resample
