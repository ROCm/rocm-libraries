// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Creates a finalized HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR from a KnobSetting.
/// The descriptor is ready to be passed to EngineConfigDescriptor via
/// HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES.
inline Error createKnobSettingDescriptor(const hipdnn_frontend::KnobSetting& setting,
                                         ScopedHipdnnBackendDescriptor& outDesc)
{
    ScopedHipdnnBackendDescriptor desc(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    if(!desc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create knob setting descriptor for " + setting.knobId()};
    }

    // Set knob ID
    HIPDNN_CHECK_ERROR(setDescriptorAttrString(
        desc.get(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, setting.knobId(), "knob ID"));

    // Set knob value, dispatching on the variant type
    HIPDNN_CHECK_ERROR(std::visit(
        [&](auto&& val) -> Error {
            using T = std::decay_t<decltype(val)>;
            if constexpr(std::is_same_v<T, int64_t>)
            {
                return setDescriptorAttrScalar(desc.get(),
                                               HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                               HIPDNN_TYPE_INT64,
                                               val,
                                               "knob value (int64)");
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                return setDescriptorAttrScalar(desc.get(),
                                               HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                               HIPDNN_TYPE_DOUBLE,
                                               val,
                                               "knob value (double)");
            }
            else if constexpr(std::is_same_v<T, std::string>)
            {
                return setDescriptorAttrString(
                    desc.get(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, val, "knob value (string)");
            }
            else
            {
                return {ErrorCode::INVALID_VALUE, "Unsupported knob value type"};
            }
        },
        setting.value()));

    HIPDNN_CHECK_ERROR(finalizeDescriptor(desc.get(), "knob setting descriptor"));

    outDesc = std::move(desc);
    return {};
}

/// Applies knob settings to an engine config descriptor using the descriptor-based
/// C API path. Creates a KnobSettingDescriptor per setting and passes them to
/// the engine config via HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES.
inline Error
    applyKnobSettingsViaDescriptors(hipdnnBackendDescriptor_t engineConfigDesc,
                                    const std::vector<hipdnn_frontend::KnobSetting>& settings)
{
    if(settings.empty())
    {
        return {};
    }

    std::vector<ScopedHipdnnBackendDescriptor> knobDescs;
    knobDescs.reserve(settings.size());
    std::vector<hipdnnBackendDescriptor_t> knobDescPtrs;
    knobDescPtrs.reserve(settings.size());

    for(const auto& setting : settings)
    {
        ScopedHipdnnBackendDescriptor knobDesc;
        HIPDNN_CHECK_ERROR(createKnobSettingDescriptor(setting, knobDesc));
        knobDescs.push_back(std::move(knobDesc));
        knobDescPtrs.push_back(knobDescs.back().get());
    }

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(engineConfigDesc,
                                             HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             static_cast<int64_t>(knobDescPtrs.size()),
                                             static_cast<const void*>(knobDescPtrs.data())),
        "Failed to set knob settings on engine config via descriptors.");
    return {};
}

} // namespace hipdnn_frontend::detail
