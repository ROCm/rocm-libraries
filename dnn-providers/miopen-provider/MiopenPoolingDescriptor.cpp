// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenPoolingDescriptor.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

MiopenPoolingDescriptor::MiopenPoolingDescriptor(miopenPoolingMode_t mode,
                                                 const std::vector<int>& window,
                                                 const std::vector<int>& padding,
                                                 const std::vector<int>& stride,
                                                 miopenIndexType_t indexType)
{
    if(window.size() != padding.size() || window.size() != stride.size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenPoolingDescriptor: window, padding and stride must have the same length");
    }

    // MIOpen pooling handles 4D and 5D tensors only, i.e. 2 or 3 spatial dims.
    if(window.size() != 2 && window.size() != 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenPoolingDescriptor: only 2 or 3 spatial dimensions are supported, got "
                + std::to_string(window.size()));
    }

    THROW_ON_MIOPEN_FAILURE(miopenCreatePoolingDescriptor(&_descriptor));

    try
    {
        THROW_ON_MIOPEN_FAILURE(miopenSetNdPoolingDescriptor(_descriptor,
                                                             mode,
                                                             static_cast<int>(window.size()),
                                                             window.data(),
                                                             padding.data(),
                                                             stride.data()));

        // Setting the N-d descriptor resets the index type and the workspace index mode, so both
        // must be applied afterwards. Image mode records a flattened spatial offset into the
        // unpadded input, hipDNN's index convention. MIOpen's default for two spatial dimensions
        // is mask mode, an offset inside the window.
        THROW_ON_MIOPEN_FAILURE(miopenSetPoolingIndexType(_descriptor, indexType));
        THROW_ON_MIOPEN_FAILURE(
            miopenSetPoolingWorkSpaceIndexMode(_descriptor, miopenPoolingWorkspaceIndexImage));
    }
    catch(...)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyPoolingDescriptor(_descriptor));
        _descriptor = nullptr;
        throw;
    }
}

MiopenPoolingDescriptor::MiopenPoolingDescriptor(MiopenPoolingDescriptor&& other) noexcept
    : _descriptor(other._descriptor)
{
    other._descriptor = nullptr;
}

MiopenPoolingDescriptor&
    MiopenPoolingDescriptor::operator=(MiopenPoolingDescriptor&& other) noexcept
{
    if(this == &other)
    {
        return *this;
    }

    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyPoolingDescriptor(_descriptor));
    }

    _descriptor = other._descriptor;
    other._descriptor = nullptr;

    return *this;
}

MiopenPoolingDescriptor::~MiopenPoolingDescriptor()
{
    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyPoolingDescriptor(_descriptor));
    }
}

miopenPoolingDescriptor_t MiopenPoolingDescriptor::poolingDescriptor() const
{
    return _descriptor;
}

} // namespace miopen_plugin
