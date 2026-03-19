// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <limits>

namespace miopen_plugin
{

MiopenTensor::MiopenTensor(const hipdnn_data_sdk::data_objects::TensorAttributes& tensor)
    : _uid(tensor.uid())
{
    THROW_ON_MIOPEN_FAILURE(miopenCreateTensorDescriptor(&_descriptor));

    PLUGIN_THROW_IF_NULL(tensor.dims(),
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor dims pointer is null for tensor UID: " + std::to_string(_uid));
    PLUGIN_THROW_IF_NULL(tensor.strides(),
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor strides pointer is null for tensor UID: " + std::to_string(_uid));

    // Validate number of dimensions fits in int (miopenSetTensorDescriptorV2 nbDims parameter is int)
    if(tensor.dims()->size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Number of tensor dimensions (" + std::to_string(tensor.dims()->size())
                + ") exceeds int range for tensor UID: " + std::to_string(_uid));
    }

    // Convert int64_t dims/strides to size_t for miopenSetTensorDescriptorV2
    std::vector<size_t> dims;
    dims.reserve(tensor.dims()->size());
    for(auto d : *tensor.dims())
    {
        if(d < 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Negative tensor dimension value " + std::to_string(d)
                    + " for tensor UID: " + std::to_string(_uid));
        }
        dims.push_back(static_cast<size_t>(d));
    }

    std::vector<size_t> strides;
    strides.reserve(tensor.strides()->size());
    for(auto s : *tensor.strides())
    {
        if(s < 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Negative tensor stride value " + std::to_string(s)
                    + " for tensor UID: " + std::to_string(_uid));
        }
        strides.push_back(static_cast<size_t>(s));
    }
    THROW_ON_MIOPEN_FAILURE(miopenSetTensorDescriptorV2(
        _descriptor,
        miopen_utils::tensorDataTypeToMiopenDataType(tensor.data_type()),
        static_cast<int>(dims.size()),
        dims.data(),
        strides.data()));
}

MiopenTensor::MiopenTensor(MiopenTensor&& other) noexcept
    : _uid(other._uid)
    , _descriptor(other._descriptor)
{
    other._descriptor = nullptr;
}

MiopenTensor& MiopenTensor::operator=(MiopenTensor&& other) noexcept
{
    if(this != &other)
    {
        if(_descriptor != nullptr)
        {
            LOG_ON_MIOPEN_FAILURE(miopenDestroyTensorDescriptor(_descriptor));
        }

        _uid = other._uid;
        _descriptor = other._descriptor;

        other._descriptor = nullptr;
    }
    return *this;
}

MiopenTensor::~MiopenTensor()
{
    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyTensorDescriptor(_descriptor));
    }
}

int64_t MiopenTensor::uid() const
{
    return _uid;
}

miopenTensorDescriptor_t MiopenTensor::tensorDescriptor() const
{
    return _descriptor;
}

}
