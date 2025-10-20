// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenActivationDescriptor.hpp"
#include "MiopenUtils.hpp"

#include <hipdnn_sdk/plugin/PluginException.hpp>

namespace miopen_legacy_plugin
{

MiopenActivationDescriptor::MiopenActivationDescriptor(
    const hipdnn_sdk::data_objects::PointwiseAttributes& pointwiseAttrs)
{
    const auto params = miopen_utils::mapPointwiseModeToMiopenActivation(pointwiseAttrs);
    if(!params.has_value())
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported pointwise mode for activation descriptor: "
                + std::to_string(static_cast<int>(pointwiseAttrs.operation())));
    }

    THROW_ON_MIOPEN_FAILURE(miopenCreateActivationDescriptor(&_descriptor));
    THROW_ON_MIOPEN_FAILURE(miopenSetActivationDescriptor(
        _descriptor, params->mode, params->alpha, params->beta, params->gamma));
}

MiopenActivationDescriptor::MiopenActivationDescriptor(MiopenActivationDescriptor&& other) noexcept
    : _descriptor(other._descriptor)
{
    other._descriptor = nullptr;
}

MiopenActivationDescriptor&
    MiopenActivationDescriptor::operator=(MiopenActivationDescriptor&& other) noexcept
{
    if(this == &other)
    {
        return *this;
    }

    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyActivationDescriptor(_descriptor));
    }

    _descriptor = other._descriptor;
    other._descriptor = nullptr;
    return *this;
}

MiopenActivationDescriptor::~MiopenActivationDescriptor()
{
    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyActivationDescriptor(_descriptor));
    }
}

miopenActivationDescriptor_t MiopenActivationDescriptor::activationDescriptor() const
{
    return _descriptor;
}

} // namespace miopen_legacy_plugin
