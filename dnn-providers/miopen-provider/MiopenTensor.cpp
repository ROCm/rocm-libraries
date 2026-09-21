// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <limits>

namespace miopen_plugin
{

namespace
{
template <typename Container>
miopenTensorDescriptor_t
    createTensorDescriptor(int64_t uid,
                           hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                           const Container& inputDims,
                           const Container& inputStrides)
{
    // Validate dims and strides size match
    if(inputDims.size() != inputStrides.size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Dims and strides size mismatch for tensor UID: " + std::to_string(uid)
                + ": dims.size()=" + std::to_string(inputDims.size())
                + ", strides.size()=" + std::to_string(inputStrides.size()));
    }

    // Validate number of dimensions fits in int (miopenSetTensorDescriptorV2 nbDims parameter is int)
    if(inputDims.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Number of tensor dimensions (" + std::to_string(inputDims.size())
                + ") exceeds int range for tensor UID: " + std::to_string(uid));
    }

    // Convert int64_t dims/strides to size_t for miopenSetTensorDescriptorV2
    std::vector<size_t> dims;
    dims.reserve(inputDims.size());
    for(auto d : inputDims)
    {
        if(d < 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Negative tensor dimension value " + std::to_string(d)
                    + " for tensor UID: " + std::to_string(uid));
        }
        dims.push_back(static_cast<size_t>(d));
    }

    std::vector<size_t> strides;
    strides.reserve(inputStrides.size());
    for(auto s : inputStrides)
    {
        if(s < 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Negative tensor stride value " + std::to_string(s)
                    + " for tensor UID: " + std::to_string(uid));
        }
        strides.push_back(static_cast<size_t>(s));
    }

    // Create and configure the descriptor
    miopenTensorDescriptor_t descriptor;
    THROW_ON_MIOPEN_FAILURE(miopenCreateTensorDescriptor(&descriptor));

    try
    {
        THROW_ON_MIOPEN_FAILURE(
            miopenSetTensorDescriptorV2(descriptor,
                                        miopen_utils::tensorDataTypeToMiopenDataType(dataType),
                                        static_cast<int>(dims.size()),
                                        dims.data(),
                                        strides.data()));
        return descriptor;
    }
    catch(...)
    {
        miopenDestroyTensorDescriptor(descriptor);
        throw;
    }
}
} // namespace

MiopenTensor::MiopenTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
    : _uid(tensor.uid())
{
    PLUGIN_THROW_IF_NULL(tensor.dims(),
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor dims pointer is null for tensor UID: " + std::to_string(_uid));
    PLUGIN_THROW_IF_NULL(tensor.strides(),
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor strides pointer is null for tensor UID: " + std::to_string(_uid));

    _descriptor
        = createTensorDescriptor(_uid, tensor.data_type(), *tensor.dims(), *tensor.strides());
}

MiopenTensor::MiopenTensor(int64_t uid,
                           hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                           const std::vector<int64_t>& inputDims,
                           const std::vector<int64_t>& inputStrides)
    : _uid(uid)
    , _descriptor(createTensorDescriptor(uid, dataType, inputDims, inputStrides))
{
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
