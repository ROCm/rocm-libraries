// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenPointwiseTensorChecks.hpp"

#include <optional>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace miopen_plugin::pointwise_applicability
{

void validatePointwiseIoTensors(
    std::initializer_list<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> tensors,
    const std::string& opName)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

    std::optional<DataType> commonDtype;

    for(const auto* tensor : tensors)
    {
        if(tensor->virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, opName + ": tensors must be non-virtual");
        }

        const auto dtype = tensor->data_type();
        if(dtype != DataType::FLOAT && dtype != DataType::HALF)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                opName + ": only FLOAT and HALF IO dtypes are supported");
        }

        if(!commonDtype.has_value())
        {
            commonDtype = dtype;
        }
        else if(*commonDtype != dtype)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                opName + ": all tensors must have the same data type");
        }

        const auto* dims = tensor->dims();
        const auto* strides = tensor->strides();

        if(dims == nullptr || strides == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, opName + ": tensor dims or strides are null");
        }

        if(dims->size() != strides->size())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, opName + ": tensor dims and strides size mismatch");
        }
    }
}

} // namespace miopen_plugin::pointwise_applicability
