// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenActivationDescriptor.hpp"
#include "MiopenUtils.hpp"

#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <optional>

namespace miopen_legacy_plugin
{

namespace
{

struct ActivationParams
{
    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
};

std::optional<ActivationParams>
    mapPointwiseModeToMiopenActivation(const hipdnn_sdk::data_objects::PointwiseAttributes& attrs)
{
    using PM = hipdnn_sdk::data_objects::PointwiseMode;

    switch(attrs.operation())
    {
    case PM::RELU_FWD:
    case PM::RELU_BWD:
    {
        if(attrs.relu_lower_clip() && attrs.relu_upper_clip())
        {
            // CLAMP
            return ActivationParams{miopenActivationCLAMP,
                                    static_cast<double>(*attrs.relu_lower_clip()),
                                    static_cast<double>(*attrs.relu_upper_clip()),
                                    0.0};
        }
        if(attrs.relu_upper_clip())
        {
            // Clipped ReLU
            return ActivationParams{miopenActivationCLIPPEDRELU,
                                    static_cast<double>(*attrs.relu_upper_clip()),
                                    0.0,
                                    0.0};
        }
        if(attrs.relu_lower_clip_slope())
        {
            // Leaky ReLU
            return ActivationParams{miopenActivationLEAKYRELU,
                                    static_cast<double>(*attrs.relu_lower_clip_slope()),
                                    0.0,
                                    0.0};
        }
        // Standard ReLU
        return ActivationParams{miopenActivationRELU, 0.0, 0.0, 0.0};
    }
    case PM::SIGMOID_FWD:
    case PM::SIGMOID_BWD:
        return ActivationParams{miopenActivationLOGISTIC, 0.0, 0.0, 0.0};
    case PM::TANH_FWD:
    case PM::TANH_BWD:
        return ActivationParams{miopenActivationTANH, 1.0, 1.0, 0.0};
    case PM::ELU_FWD:
    case PM::ELU_BWD:
    {
        double alpha = attrs.elu_alpha() ? static_cast<double>(*attrs.elu_alpha()) : 1.0;
        return ActivationParams{miopenActivationELU, alpha, 0.0, 0.0};
    }
    case PM::SOFTPLUS_FWD:
    case PM::SOFTPLUS_BWD:
        return ActivationParams{miopenActivationSOFTRELU, 0.0, 0.0, 0.0};
    case PM::ABS:
        return ActivationParams{miopenActivationABS, 0.0, 0.0, 0.0};
    case PM::IDENTITY:
        return ActivationParams{miopenActivationPASTHRU, 0.0, 0.0, 0.0};
    default:
        return std::nullopt;
    }
}

} // namespace

MiopenActivationDescriptor::MiopenActivationDescriptor(
    const hipdnn_sdk::data_objects::PointwiseAttributes& pointwiseAttrs)
{
    const auto params = mapPointwiseModeToMiopenActivation(pointwiseAttrs);
    if(!params.has_value())
    {
        // TODO: make a common pointwise mode to string function
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
