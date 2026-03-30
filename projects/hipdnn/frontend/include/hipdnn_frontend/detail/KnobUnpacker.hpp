// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Unpacks a finalized HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR into a frontend Knob.
///
/// This lifts knob metadata from a backend KnobDescriptor by reading each
/// attribute via the C-API getAttribute pattern. It reconstructs the frontend
/// Knob with its ID, description, default value, deprecation flag, and
/// constraints directly from descriptor attributes.
///
/// @param knobDesc A finalized knob info backend descriptor
/// @return Pair of error and unpacked knob. On failure, the optional is empty.
[[nodiscard]] inline std::pair<Error, std::optional<Knob>>
    unpackKnobDescriptor(hipdnnBackendDescriptor_t knobDesc)
{
    // Read knob ID
    std::string knobId;
    auto err
        = getDescriptorAttrString(knobDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, knobId, "knob info ID");
    if(err.is_bad())
    {
        return {err, std::nullopt};
    }

    if(knobId.empty())
    {
        return {{ErrorCode::INVALID_VALUE, "Knob info descriptor has empty knob ID"}, std::nullopt};
    }

    // Read description
    std::string description;
    err = getDescriptorAttrString(
        knobDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, description, "knob info description");
    if(err.is_bad())
    {
        return {err, std::nullopt};
    }

    // Read deprecated flag
    bool deprecated = false;
    err = getDescriptorAttrScalar(knobDesc,
                                  HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT,
                                  HIPDNN_TYPE_BOOLEAN,
                                  deprecated,
                                  "knob info deprecated flag");
    if(err.is_bad())
    {
        return {err, std::nullopt};
    }

    // Read the default value type first
    int64_t defaultValueTypeRaw = 0;
    err = getDescriptorAttrScalar(knobDesc,
                                  HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE_EXT,
                                  HIPDNN_TYPE_INT64,
                                  defaultValueTypeRaw,
                                  "knob info default value type");
    if(err.is_bad())
    {
        return {err, std::nullopt};
    }

    const auto defaultValueType = static_cast<hipdnnBackendAttributeType_t>(defaultValueTypeRaw);
    KnobValueVariant defaultValue;
    std::shared_ptr<IConstraint> constraint = std::make_shared<EmptyConstraint>();

    // Read the default value and matching constraint fields based on type.
    switch(defaultValueType)
    {
    case HIPDNN_TYPE_INT64:
    {
        int64_t intVal = 0;
        err = getDescriptorAttrScalar(knobDesc,
                                      HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT,
                                      HIPDNN_TYPE_INT64,
                                      intVal,
                                      "knob info default value (int64)");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }
        defaultValue = intVal;

        std::optional<int64_t> minVal;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                              HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT,
                                              HIPDNN_TYPE_INT64,
                                              minVal,
                                              "knob info min value (int64)");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }

        std::optional<int64_t> maxVal;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                              HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT,
                                              HIPDNN_TYPE_INT64,
                                              maxVal,
                                              "knob info max value (int64)");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }

        std::optional<int64_t> stride;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                              HIPDNN_ATTR_KNOB_INFO_STRIDE_EXT,
                                              HIPDNN_TYPE_INT64,
                                              stride,
                                              "knob info stride");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }

        std::vector<int64_t> validValuesVec;
        err = getDescriptorAttrVec(knobDesc,
                                   HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT_EXT,
                                   validValuesVec,
                                   "knob info valid values (int64)");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }

        if(minVal.has_value() || maxVal.has_value() || stride.has_value()
           || !validValuesVec.empty())
        {
            std::unordered_set<int64_t> validValues(validValuesVec.begin(), validValuesVec.end());
            constraint = std::make_shared<IntConstraint>(
                minVal.value_or(0), maxVal.value_or(0), stride.value_or(1), std::move(validValues));
        }
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        double doubleVal = 0.0;
        err = getDescriptorAttrScalar(knobDesc,
                                      HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT,
                                      HIPDNN_TYPE_DOUBLE,
                                      doubleVal,
                                      "knob info default value (double)");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }
        defaultValue = doubleVal;

        std::optional<double> minVal;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                              HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT,
                                              HIPDNN_TYPE_DOUBLE,
                                              minVal,
                                              "knob info min value (double)");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }

        std::optional<double> maxVal;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                              HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT,
                                              HIPDNN_TYPE_DOUBLE,
                                              maxVal,
                                              "knob info max value (double)");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }

        if(minVal.has_value() || maxVal.has_value())
        {
            constraint
                = std::make_shared<FloatConstraint>(minVal.value_or(0.0), maxVal.value_or(0.0));
        }
        break;
    }
    case HIPDNN_TYPE_CHAR:
    {
        std::string strVal;
        err = getDescriptorAttrString(knobDesc,
                                      HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT,
                                      strVal,
                                      "knob info default value (string)");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }
        defaultValue = std::move(strVal);

        std::optional<int32_t> stringMaxLength;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                              HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH_EXT,
                                              HIPDNN_TYPE_INT32,
                                              stringMaxLength,
                                              "knob info string max length");
        if(err.is_bad())
        {
            return {err, std::nullopt};
        }

        // Read valid values string - uses the raw C-API since the format
        // is a null-separated buffer
        std::vector<std::string> validValuesList;
        {
            int64_t count = 0;
            auto countStatus = hipdnnBackend()->backendGetAttribute(
                knobDesc,
                HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING_EXT,
                HIPDNN_TYPE_CHAR,
                0,
                &count,
                nullptr);

            if(countStatus != HIPDNN_STATUS_SUCCESS && countStatus != HIPDNN_STATUS_NOT_SUPPORTED)
            {
                return {{ErrorCode::HIPDNN_BACKEND_ERROR,
                         "Knob '" + knobId + "': failed to query valid string values count"},
                        std::nullopt};
            }

            if(countStatus == HIPDNN_STATUS_SUCCESS && count > 0)
            {
                std::vector<char> buffer(static_cast<size_t>(count));
                int64_t actualCount = 0;
                auto getStatus = hipdnnBackend()->backendGetAttribute(
                    knobDesc,
                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING_EXT,
                    HIPDNN_TYPE_CHAR,
                    count,
                    &actualCount,
                    buffer.data());

                if(getStatus != HIPDNN_STATUS_SUCCESS)
                {
                    return {{ErrorCode::HIPDNN_BACKEND_ERROR,
                             "Knob '" + knobId + "': failed to read valid string values"},
                            std::nullopt};
                }

                if(actualCount > 0 && static_cast<size_t>(actualCount) <= buffer.size())
                {
                    // Parse null-separated string buffer
                    const char* data = buffer.data();
                    const char* end = data + actualCount;
                    while(data < end)
                    {
                        std::string val(data);
                        if(!val.empty())
                        {
                            validValuesList.push_back(std::move(val));
                        }
                        data += std::strlen(data) + 1;
                    }
                }
            }
        }

        if(stringMaxLength.has_value() || !validValuesList.empty())
        {
            std::unordered_set<std::string> validValues(validValuesList.begin(),
                                                        validValuesList.end());
            constraint = std::make_shared<StringConstraint>(stringMaxLength.value_or(0),
                                                            std::move(validValues));
        }
        break;
    }
    default:
        return {{ErrorCode::INVALID_VALUE,
                 "Knob '" + knobId
                     + "' has unknown default value type: " + std::to_string(defaultValueTypeRaw)},
                std::nullopt};
    }

    auto [knobErr, knob] = Knob::tryCreate(
        knobId, description, std::move(defaultValue), deprecated, std::move(constraint));
    if(knobErr.is_bad())
    {
        return {knobErr, std::nullopt};
    }

    return {{}, std::move(knob)};
}

/// Unpacks knob info descriptors from a backend engine descriptor via the
/// C-API descriptor path (HIPDNN_ATTR_ENGINE_KNOB_INFO as descriptors).
///
/// This is the C-API-based alternative to the flatbuffer path used by
/// getKnobsForEngine() in Knob.hpp.
///
/// @param engineDesc A finalized engine backend descriptor
/// @param outKnobs Output vector of Knob objects
/// @return Error on failure, empty Error on success
[[nodiscard]] inline Error unpackKnobsFromDescriptors(hipdnnBackendDescriptor_t engineDesc,
                                                      std::vector<Knob>& outKnobs)
{
    auto [knobDescs, err] = getDescriptorAttrDescArray(
        engineDesc, HIPDNN_ATTR_ENGINE_KNOB_INFO, "knob info descriptors");
    if(err.is_bad())
    {
        outKnobs.clear();
        return err;
    }

    outKnobs.clear();
    outKnobs.reserve(knobDescs.size());

    for(size_t i = 0; i < knobDescs.size(); ++i)
    {
        auto [knobErr, knob] = unpackKnobDescriptor(knobDescs[i].get());
        if(knobErr.is_bad())
        {
            return {knobErr.code,
                    "Failed to unpack knob info at index " + std::to_string(i) + ": "
                        + knobErr.get_message()};
        }
        if(!knob.has_value())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Failed to unpack knob info at index " + std::to_string(i)
                        + ": missing knob without an error"};
        }
        outKnobs.push_back(std::move(knob.value()));
    }

    return {};
}

} // namespace hipdnn_frontend::detail
