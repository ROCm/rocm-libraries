// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

MiopenTensor::MiopenTensor(const hipdnn_sdk::data_objects::TensorAttributes& tensor)
    : _uid(tensor.uid())
{
    THROW_ON_MIOPEN_FAILURE(miopenCreateTensorDescriptor(&_descriptor));

    std::vector<int> dims(tensor.dims()->begin(), tensor.dims()->end());
    std::vector<int> strides(tensor.strides()->begin(), tensor.strides()->end());
    THROW_ON_MIOPEN_FAILURE(
        miopenSetTensorDescriptor(_descriptor,
                                  miopen_utils::tensorDataTypeToMiopenDataType(tensor.data_type()),
                                  static_cast<int>(dims.size()),
                                  reinterpret_cast<int*>(dims.data()),
                                  reinterpret_cast<int*>(strides.data())));
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
