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
/// constraints by round-tripping through flatbuffer serialization.
///
/// @param knobDesc A finalized knob info backend descriptor
/// @return Pair of Error and Knob; on error, the Knob should be ignored.
///         Uses Knob::tryFromFlatbuffer internally.
[[nodiscard]] inline std::pair<Error, Knob>
    unpackKnobDescriptor(hipdnnBackendDescriptor_t knobDesc)
{
    namespace fb = hipdnn_data_sdk::data_objects;

    // Read knob ID
    std::string knobId;
    auto err = getDescriptorAttrString(
        knobDesc, HIPDNN_ATTR_KNOB_INFO_TYPE_EXT, knobId, "knob info ID");
    if(err.is_bad())
    {
        return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
    }

    if(knobId.empty())
    {
        return {{ErrorCode::INVALID_VALUE, "Knob info descriptor has empty knob ID"},
                Knob::tryFromFlatbuffer({nullptr, 0}).second};
    }

    // Read description
    std::string description;
    err = getDescriptorAttrString(
        knobDesc, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT, description, "knob info description");
    if(err.is_bad())
    {
        return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
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
        return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
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
        return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
    }

    const auto defaultValueType
        = static_cast<hipdnnBackendAttributeType_t>(defaultValueTypeRaw);

    // Build a KnobT that we will serialize to flatbuffer for round-tripping
    // through Knob::tryFromFlatbuffer (since Knob's constructor is private).
    fb::KnobT knobT;
    knobT.knob_id = knobId;
    knobT.description = description;
    knobT.deprecated = deprecated;

    // Read default value based on type and set it on the KnobT
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
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }
        fb::IntValueT intValue;
        intValue.value = intVal;
        knobT.default_value.Set(intValue);
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
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }
        fb::FloatValueT floatValue;
        floatValue.value = doubleVal;
        knobT.default_value.Set(floatValue);
        break;
    }
    case HIPDNN_TYPE_CHAR:
    {
        std::string strVal;
        err = getDescriptorAttrString(
            knobDesc,
            HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT,
            strVal,
            "knob info default value (string)");
        if(err.is_bad())
        {
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }
        fb::StringValueT stringValue;
        stringValue.value = std::move(strVal);
        knobT.default_value.Set(std::move(stringValue));
        break;
    }
    default:
        return {{ErrorCode::INVALID_VALUE,
                 "Knob '" + knobId + "' has unknown default value type: "
                     + std::to_string(defaultValueTypeRaw)},
                Knob::tryFromFlatbuffer({nullptr, 0}).second};
    }

    // Read constraint fields based on default value type and set them on KnobT
    switch(defaultValueType)
    {
    case HIPDNN_TYPE_INT64:
    {
        std::optional<int64_t> minVal;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                               HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT,
                                               HIPDNN_TYPE_INT64,
                                               minVal,
                                               "knob info min value (int64)");
        if(err.is_bad())
        {
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }

        std::optional<int64_t> maxVal;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                               HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT,
                                               HIPDNN_TYPE_INT64,
                                               maxVal,
                                               "knob info max value (int64)");
        if(err.is_bad())
        {
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }

        std::optional<int64_t> stride;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                               HIPDNN_ATTR_KNOB_INFO_STRIDE_EXT,
                                               HIPDNN_TYPE_INT64,
                                               stride,
                                               "knob info stride");
        if(err.is_bad())
        {
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }

        std::vector<int64_t> validValuesVec;
        err = getDescriptorAttrVec(knobDesc,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT_EXT,
                                    validValuesVec,
                                    "knob info valid values (int64)");
        if(err.is_bad())
        {
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }

        if(minVal.has_value() || maxVal.has_value() || stride.has_value()
           || !validValuesVec.empty())
        {
            fb::IntConstraintT intConstraint;
            intConstraint.min_value = minVal.value_or(0);
            intConstraint.max_value = maxVal.value_or(0);
            intConstraint.step = stride.value_or(1);
            intConstraint.valid_values = std::move(validValuesVec);
            knobT.constraint.Set(std::move(intConstraint));
        }
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        std::optional<double> minVal;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                               HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT,
                                               HIPDNN_TYPE_DOUBLE,
                                               minVal,
                                               "knob info min value (double)");
        if(err.is_bad())
        {
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }

        std::optional<double> maxVal;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                               HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT,
                                               HIPDNN_TYPE_DOUBLE,
                                               maxVal,
                                               "knob info max value (double)");
        if(err.is_bad())
        {
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
        }

        if(minVal.has_value() || maxVal.has_value())
        {
            fb::FloatConstraintT floatConstraint;
            floatConstraint.min_value = minVal.value_or(0.0);
            floatConstraint.max_value = maxVal.value_or(0.0);
            knobT.constraint.Set(floatConstraint);
        }
        break;
    }
    case HIPDNN_TYPE_CHAR:
    {
        std::optional<int32_t> stringMaxLength;
        err = getDescriptorAttrOptionalScalar(knobDesc,
                                               HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH_EXT,
                                               HIPDNN_TYPE_INT32,
                                               stringMaxLength,
                                               "knob info string max length");
        if(err.is_bad())
        {
            return {err, Knob::tryFromFlatbuffer({nullptr, 0}).second};
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

                if(getStatus == HIPDNN_STATUS_SUCCESS && actualCount > 0)
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
            fb::StringConstraintT stringConstraint;
            stringConstraint.max_length = stringMaxLength.value_or(0);
            stringConstraint.valid_values = std::move(validValuesList);
            knobT.constraint.Set(std::move(stringConstraint));
        }
        break;
    }
    default:
        break;
    }

    // Serialize to flatbuffer and use the existing factory method.
    // This round-trip ensures all validation in tryFromFlatbuffer is applied.
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::Knob::Pack(builder, &knobT));

    hipdnnBackendFlatbufferData_t fbData;
    fbData.ptr = builder.GetBufferPointer();
    fbData.size = builder.GetSize();

    return Knob::tryFromFlatbuffer(fbData);
}

/// Unpacks knob info descriptors from a backend engine descriptor via the
/// C-API descriptor path (HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES as descriptors).
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
    auto [knobDescs, err]
        = getDescriptorAttrDescArray(engineDesc,
                                     HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                     "knob info descriptors");
    if(err.is_bad())
    {
        // The attribute may not support descriptor-based retrieval; this is expected
        // when the backend only supports flatbuffer-based knob info.
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
        outKnobs.push_back(std::move(knob));
    }

    return {};
}

} // namespace hipdnn_frontend::detail
