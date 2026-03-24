// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <string>
#include <variant>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Unpacks a finalized HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR into a frontend KnobSetting.
///
/// This is the inverse of createKnobSettingDescriptor() in KnobPacker.hpp.
/// Reads the knob ID and polymorphic value from the backend descriptor via
/// getAttribute and reconstructs a frontend KnobSetting.
///
/// @param knobDesc A finalized knob choice backend descriptor
/// @param outSetting The KnobSetting to populate
/// @return Error on failure, empty Error on success
[[nodiscard]] inline Error unpackKnobSettingDescriptor(hipdnnBackendDescriptor_t knobDesc,
                                                       KnobSetting& outSetting)
{
    // Read knob ID string
    std::string knobId;
    HIPDNN_CHECK_ERROR(getDescriptorAttrString(
        knobDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE_EXT, knobId, "knob setting knob ID"));

    if(knobId.empty())
    {
        return {ErrorCode::INVALID_VALUE, "Knob setting descriptor has empty knob ID"};
    }

    // Determine the value type by querying the default_value_type attribute.
    // The KnobSettingDescriptor stores a polymorphic value (int64, double, or string).
    // We need to determine its type first, then read the appropriate value.
    //
    // We attempt int64 first; if that fails, try double; then string.
    // This approach works because getAttribute will succeed for the type that was set.
    KnobValueVariant value;

    // Try reading as int64
    {
        int64_t intVal = 0;
        auto status = hipdnnBackend()->backendGetAttribute(knobDesc,
                                                            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                                            HIPDNN_TYPE_INT64,
                                                            1,
                                                            nullptr,
                                                            &intVal);
        if(status == HIPDNN_STATUS_SUCCESS)
        {
            value = intVal;
            outSetting = KnobSetting(std::move(knobId), std::move(value));
            return {};
        }
    }

    // Try reading as double
    {
        double doubleVal = 0.0;
        auto status = hipdnnBackend()->backendGetAttribute(knobDesc,
                                                            HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT,
                                                            HIPDNN_TYPE_DOUBLE,
                                                            1,
                                                            nullptr,
                                                            &doubleVal);
        if(status == HIPDNN_STATUS_SUCCESS)
        {
            value = doubleVal;
            outSetting = KnobSetting(std::move(knobId), std::move(value));
            return {};
        }
    }

    // Try reading as string (CHAR)
    {
        std::string strVal;
        auto err = getDescriptorAttrString(
            knobDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE_EXT, strVal, "knob setting value (string)");
        if(err.is_good())
        {
            value = std::move(strVal);
            outSetting = KnobSetting(std::move(knobId), std::move(value));
            return {};
        }
    }

    return {ErrorCode::HIPDNN_BACKEND_ERROR,
            "Failed to read knob value from knob setting descriptor for knob '" + knobId + "'"};
}

/// Unpacks an array of knob choice descriptors from a parent descriptor.
///
/// @param parentDesc The parent descriptor (e.g., engine config) containing knob choices
/// @param attrName The attribute name for the knob choices array
/// @param outSettings Output vector of KnobSettings
/// @return Error on failure, empty Error on success
[[nodiscard]] inline Error
    unpackKnobSettingsFromDescriptor(hipdnnBackendDescriptor_t parentDesc,
                                     hipdnnBackendAttributeName_t attrName,
                                     std::vector<KnobSetting>& outSettings)
{
    auto [knobDescs, err] = getDescriptorAttrDescArray(parentDesc, attrName, "knob choices");
    if(err.is_bad())
    {
        return err;
    }

    outSettings.clear();
    outSettings.reserve(knobDescs.size());

    for(size_t i = 0; i < knobDescs.size(); ++i)
    {
        KnobSetting setting("", KnobValueVariant{int64_t{0}});
        auto unpackErr = unpackKnobSettingDescriptor(knobDescs[i].get(), setting);
        if(unpackErr.is_bad())
        {
            return {unpackErr.code,
                    "Failed to unpack knob setting at index " + std::to_string(i) + ": "
                        + unpackErr.get_message()};
        }
        outSettings.push_back(std::move(setting));
    }

    return {};
}

} // namespace hipdnn_frontend::detail
