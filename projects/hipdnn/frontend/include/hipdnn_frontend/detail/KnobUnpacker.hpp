// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Unpacks int64 default value and constraint fields from a knob descriptor.
[[nodiscard]] inline std::pair<Error, std::pair<KnobValueVariant, std::shared_ptr<IConstraint>>>
    unpackIntKnobFields(hipdnnBackendDescriptor_t knobDesc)
{
    int64_t intVal = 0;
    HIPDNN_FE_TRY(getDescriptorAttrScalar(knobDesc,
                                          HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE,
                                          HIPDNN_TYPE_INT64,
                                          intVal,
                                          "knob info default value (int64)"));

    std::optional<int64_t> minVal;
    HIPDNN_FE_TRY(getDescriptorAttrOptionalScalar(knobDesc,
                                                  HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE,
                                                  HIPDNN_TYPE_INT64,
                                                  minVal,
                                                  "knob info min value (int64)"));

    std::optional<int64_t> maxVal;
    HIPDNN_FE_TRY(getDescriptorAttrOptionalScalar(knobDesc,
                                                  HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE,
                                                  HIPDNN_TYPE_INT64,
                                                  maxVal,
                                                  "knob info max value (int64)"));

    std::optional<int64_t> stride;
    HIPDNN_FE_TRY(getDescriptorAttrOptionalScalar(
        knobDesc, HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64, stride, "knob info stride"));

    std::vector<int64_t> validValuesVec;
    HIPDNN_FE_TRY(getDescriptorAttrVec(knobDesc,
                                       HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT,
                                       validValuesVec,
                                       "knob info valid values (int64)"));

    std::shared_ptr<IConstraint> constraint = std::make_shared<EmptyConstraint>();
    if(minVal.has_value() || maxVal.has_value() || stride.has_value() || !validValuesVec.empty())
    {
        std::unordered_set<int64_t> validValues(validValuesVec.begin(), validValuesVec.end());
        constraint = std::make_shared<IntConstraint>(
            minVal.value_or(0), maxVal.value_or(0), stride.value_or(1), std::move(validValues));
    }

    return {{ErrorCode::OK, ""}, {KnobValueVariant{intVal}, std::move(constraint)}};
}

/// Unpacks double default value and constraint fields from a knob descriptor.
[[nodiscard]] inline std::pair<Error, std::pair<KnobValueVariant, std::shared_ptr<IConstraint>>>
    unpackFloatKnobFields(hipdnnBackendDescriptor_t knobDesc)
{
    double doubleVal = 0.0;
    HIPDNN_FE_TRY(getDescriptorAttrScalar(knobDesc,
                                          HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE,
                                          HIPDNN_TYPE_DOUBLE,
                                          doubleVal,
                                          "knob info default value (double)"));

    std::optional<double> minVal;
    HIPDNN_FE_TRY(getDescriptorAttrOptionalScalar(knobDesc,
                                                  HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE,
                                                  HIPDNN_TYPE_DOUBLE,
                                                  minVal,
                                                  "knob info min value (double)"));

    std::optional<double> maxVal;
    HIPDNN_FE_TRY(getDescriptorAttrOptionalScalar(knobDesc,
                                                  HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE,
                                                  HIPDNN_TYPE_DOUBLE,
                                                  maxVal,
                                                  "knob info max value (double)"));

    std::shared_ptr<IConstraint> constraint = std::make_shared<EmptyConstraint>();
    if(minVal.has_value() || maxVal.has_value())
    {
        constraint = std::make_shared<FloatConstraint>(minVal.value_or(0.0), maxVal.value_or(0.0));
    }

    return {{ErrorCode::OK, ""}, {KnobValueVariant{doubleVal}, std::move(constraint)}};
}

/// Unpacks string default value and constraint fields from a knob descriptor.
[[nodiscard]] inline std::pair<Error, std::pair<KnobValueVariant, std::shared_ptr<IConstraint>>>
    unpackStringKnobFields(hipdnnBackendDescriptor_t knobDesc, const std::string& knobId)
{
    std::string strVal;
    HIPDNN_FE_TRY(getDescriptorAttrString(
        knobDesc, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, strVal, "knob info default value (string)"));

    std::optional<int32_t> stringMaxLength;
    HIPDNN_FE_TRY(getDescriptorAttrOptionalScalar(knobDesc,
                                                  HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH,
                                                  HIPDNN_TYPE_INT32,
                                                  stringMaxLength,
                                                  "knob info string max length"));

    // Read valid values string - uses the raw C-API since the format
    // is a null-separated buffer
    std::vector<std::string> validValuesList;
    {
        int64_t count = 0;
        auto countStatus
            = hipdnnBackend()->backendGetAttribute(knobDesc,
                                                   HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                   HIPDNN_TYPE_CHAR,
                                                   0,
                                                   &count,
                                                   nullptr);

        if(countStatus != HIPDNN_STATUS_SUCCESS && countStatus != HIPDNN_STATUS_NOT_SUPPORTED)
        {
            return {{ErrorCode::HIPDNN_BACKEND_ERROR,
                     "Knob '" + knobId + "': failed to query valid string values count"},
                    {}};
        }

        if(countStatus == HIPDNN_STATUS_SUCCESS && count > 0)
        {
            std::vector<char> buffer(static_cast<size_t>(count));
            int64_t actualCount = 0;
            auto getStatus
                = hipdnnBackend()->backendGetAttribute(knobDesc,
                                                       HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                                       HIPDNN_TYPE_CHAR,
                                                       count,
                                                       &actualCount,
                                                       buffer.data());

            if(getStatus != HIPDNN_STATUS_SUCCESS)
            {
                return {{ErrorCode::HIPDNN_BACKEND_ERROR,
                         "Knob '" + knobId + "': failed to read valid string values"},
                        {}};
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

    std::shared_ptr<IConstraint> constraint = std::make_shared<EmptyConstraint>();
    if(stringMaxLength.has_value() || !validValuesList.empty())
    {
        std::unordered_set<std::string> validValues(validValuesList.begin(), validValuesList.end());
        constraint = std::make_shared<StringConstraint>(stringMaxLength.value_or(0),
                                                        std::move(validValues));
    }

    return {{ErrorCode::OK, ""}, {KnobValueVariant{std::move(strVal)}, std::move(constraint)}};
}

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
    // Read knob ID — cannot use HIPDNN_FE_TRY because we need knobId
    // for the empty-check immediately after.
    std::string knobId;
    auto err
        = getDescriptorAttrString(knobDesc, HIPDNN_ATTR_KNOB_INFO_TYPE, knobId, "knob info ID");
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
    HIPDNN_FE_TRY(getDescriptorAttrString(
        knobDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, description, "knob info description"));

    // Read deprecated flag
    bool deprecated = false;
    HIPDNN_FE_TRY(getDescriptorAttrScalar(knobDesc,
                                          HIPDNN_ATTR_KNOB_INFO_DEPRECATED,
                                          HIPDNN_TYPE_BOOLEAN,
                                          deprecated,
                                          "knob info deprecated flag"));

    // Read the default value type first
    int64_t defaultValueTypeRaw = 0;
    HIPDNN_FE_TRY(getDescriptorAttrScalar(knobDesc,
                                          HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                                          HIPDNN_TYPE_INT64,
                                          defaultValueTypeRaw,
                                          "knob info default value type"));

    const auto defaultValueType = static_cast<hipdnnBackendAttributeType_t>(defaultValueTypeRaw);
    KnobValueVariant defaultValue;
    std::shared_ptr<IConstraint> constraint;

    // Read the default value and matching constraint fields based on type.
    switch(defaultValueType)
    {
    case HIPDNN_TYPE_INT64:
    {
        auto [fieldErr, fields] = unpackIntKnobFields(knobDesc);
        if(fieldErr.is_bad())
        {
            return {fieldErr, std::nullopt};
        }
        defaultValue = std::move(fields.first);
        constraint = std::move(fields.second);
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        auto [fieldErr, fields] = unpackFloatKnobFields(knobDesc);
        if(fieldErr.is_bad())
        {
            return {fieldErr, std::nullopt};
        }
        defaultValue = std::move(fields.first);
        constraint = std::move(fields.second);
        break;
    }
    case HIPDNN_TYPE_CHAR:
    {
        auto [fieldErr, fields] = unpackStringKnobFields(knobDesc, knobId);
        if(fieldErr.is_bad())
        {
            return {fieldErr, std::nullopt};
        }
        defaultValue = std::move(fields.first);
        constraint = std::move(fields.second);
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

/// Unpacks knob descriptors from a backend engine descriptor into Knob objects.
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

    std::vector<Knob> unpackedKnobs;
    unpackedKnobs.reserve(knobDescs.size());
    std::unordered_set<std::string> usedKnobIds;
    size_t skippedCount = 0;

    for(const auto& knobDesc : knobDescs)
    {
        auto [knobErr, knob] = unpackKnobDescriptor(knobDesc.get());
        if(knobErr.is_bad())
        {
            HIPDNN_FE_LOG_WARN("Skipping knob: unpack failed: " << knobErr.get_message());
            ++skippedCount;
            continue;
        }
        if(!knob.has_value())
        {
            HIPDNN_FE_LOG_WARN("Skipping knob: unpack returned empty result");
            ++skippedCount;
            continue;
        }

        if(!usedKnobIds.insert(knob->knobId()).second)
        {
            HIPDNN_FE_LOG_WARN("Skipping knob with duplicate ID: " << knob->knobId());
            ++skippedCount;
            continue;
        }

        unpackedKnobs.push_back(std::move(knob.value()));
    }

    outKnobs = std::move(unpackedKnobs);

    if(skippedCount > 0)
    {
        std::string summary = "Loaded " + std::to_string(outKnobs.size()) + " knobs, skipped "
                              + std::to_string(skippedCount) + " invalid/duplicate knobs";
        HIPDNN_FE_LOG_WARN(summary);
        return {ErrorCode::OK, std::move(summary)};
    }

    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::detail
