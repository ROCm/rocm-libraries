// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <HipdnnBackendFlatbufferData.h>

#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <hipdnn_frontend/Utilities.hpp>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/KnobWrapper.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace hipdnn_frontend
{

// Knob information class - describes available knobs for an engine
class Knob
{
public:
    // Factory function to create from flatbuffer
    // Returns {Error, Knob}. On error, the Knob is default-constructed (should be ignored).
    static std::pair<Error, Knob> tryFromFlatbuffer(hipdnnBackendFlatbufferData_t fbData);

    // Accessors
    const std::string& knobId() const
    {
        return _knobId;
    }

    const std::string& description() const
    {
        return _description;
    }

    bool isDeprecated() const
    {
        return _deprecated;
    }

    KnobValueType valueType() const
    {
        return getKnobValueTypeFromVariant(_defaultValue);
    }

    const KnobValueVariant& defaultValue() const
    {
        return _defaultValue;
    }

    // Get constraint
    const IConstraint* constraint() const
    {
        return _constraint.get();
    }

    // Validate a knob setting against this knob's constraints
    Error validate(const KnobSetting& setting) const
    {
        // Validate against constraint if present
        if(_constraint)
        {
            return _constraint->validateKnobSetting(setting);
        }

        return {ErrorCode::OK, ""};
    }

    // String representation for logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "Knob{knobIdStr=\"" << _knobId << "\", description=\"" << _description
            << "\", defaultValue=";

        variantToStream(oss, _defaultValue);

        oss << ", deprecated=" << (_deprecated ? "true" : "false");

        if(_constraint)
        {
            oss << ", constraint=" << _constraint->toString();
        }

        oss << "}";
        return oss.str();
    }

private:
    // Private default constructor - allows factory function to create empty Knob then populate
    Knob() = default;

    // Private constructor - use flatbuffer factory function to create instances
    Knob(std::string knobIdStr,
         std::string description,
         KnobValueVariant defaultValue,
         bool deprecated)
        : _knobId(std::move(knobIdStr))
        , _description(std::move(description))
        , _defaultValue(std::move(defaultValue))
        , _deprecated(deprecated)
    {
    }

    static void variantToStream(std::ostringstream& oss, const KnobValueVariant& variant)
    {
        std::visit(
            [&oss](auto&& value) {
                if constexpr(std::is_same_v<std::decay_t<decltype(value)>, std::string>)
                {
                    oss << "\"" << value << "\"";
                }
                else
                {
                    oss << value;
                }
            },
            variant);
    }

    std::string _knobId;
    std::string _description;
    KnobValueVariant _defaultValue;
    bool _deprecated;

    // Constraint (polymorphic)
    std::shared_ptr<IConstraint> _constraint;
};

// Factory method implementation
inline std::pair<Error, Knob> Knob::tryFromFlatbuffer(hipdnnBackendFlatbufferData_t fbData)
{
    if(fbData.ptr == nullptr || fbData.size == 0)
    {
        return {{ErrorCode::INVALID_VALUE, "Flatbuffer data is nullptr or has zero size"}, {}};
    }

    hipdnn_data_sdk::flatbuffer_utilities::KnobWrapper knobWrapper(fbData.ptr, fbData.size);

    if(!knobWrapper.isValid())
    {
        return {{ErrorCode::INVALID_VALUE, "Knob flatbuffer failed verification"}, {}};
    }

    auto fbKnob = &knobWrapper.getKnob();

    // Unpack to native KnobT - all conversions done automatically by FlatBuffers
    std::unique_ptr<hipdnn_data_sdk::data_objects::KnobT> knobT(fbKnob->UnPack());

    // Get knob_id for error messages
    std::string knobId = knobT->knob_id;

    // Extract default value from the union
    KnobValueVariant defaultValue;
    switch(knobT->default_value.type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        defaultValue = knobT->default_value.AsIntValue()->value;
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        defaultValue = knobT->default_value.AsFloatValue()->value;
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
        defaultValue = knobT->default_value.AsStringValue()->value; // Already std::string
        break;
    case hipdnn_data_sdk::data_objects::KnobValue::NONE:
        // NONE default_value is invalid - knobs must have a default value
        return {{ErrorCode::INVALID_VALUE,
                 "Knob '" + knobId + "' has NONE default_value - knobs must have a default value"},
                {}};
    default:
        return {{ErrorCode::INVALID_VALUE, "Knob '" + knobId + "' has unknown default_value type"},
                {}};
    }

    // Create the knob - strings are already std::string in KnobT
    Knob knob(std::move(knobT->knob_id),
              std::move(knobT->description),
              std::move(defaultValue),
              knobT->deprecated);

    // Handle constraints using the native union types
    switch(knobT->constraint.type)
    {
    case hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint:
    {
        auto* c = knobT->constraint.AsIntConstraint();
        std::unordered_set<int64_t> validValues(c->valid_values.begin(), c->valid_values.end());
        knob._constraint = std::make_unique<IntConstraint>(
            c->min_value, c->max_value, c->step, std::move(validValues));
        break;
    }
    case hipdnn_data_sdk::data_objects::KnobConstraint::FloatConstraint:
    {
        auto* c = knobT->constraint.AsFloatConstraint();
        knob._constraint = std::make_unique<FloatConstraint>(c->min_value, c->max_value);
        break;
    }
    case hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint:
    {
        auto* c = knobT->constraint.AsStringConstraint();
        std::unordered_set<std::string> validValues(c->valid_values.begin(), c->valid_values.end());
        knob._constraint
            = std::make_unique<StringConstraint>(c->max_length, std::move(validValues));
        break;
    }
    case hipdnn_data_sdk::data_objects::KnobConstraint::NONE:
        knob._constraint = std::make_unique<EmptyConstraint>();
        break;
    default:
        return {{ErrorCode::INVALID_VALUE, "Knob '" + knobId + "' has unknown constraint type"},
                {}};
    }

    // Validate that the default_value satisfies the constraint
    if(knob._constraint)
    {
        KnobSetting defaultSetting(knob._knobId, knob._defaultValue);
        auto validationError = knob._constraint->validateKnobSetting(defaultSetting);
        if(validationError.code != ErrorCode::OK)
        {
            return {{ErrorCode::INVALID_VALUE,
                     "Knob '" + knob._knobId + "' has default_value that violates its constraint: "
                         + validationError.err_msg},
                    {}};
        }
    }

    return {{ErrorCode::OK, ""}, knob};
}

namespace detail
{
inline Error getKnobsForEngine(std::vector<Knob>& knobs, hipdnnBackendDescriptor_t engineDesc)
{
    int64_t knobCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(engineDesc,
                                             HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                             HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                             0,
                                             &knobCount,
                                             nullptr),
        "Failed to get knob count from engine descriptor.");

    if(knobCount == 0)
    {
        knobs.clear();
        return {ErrorCode::OK, ""};
    }

    std::vector<hipdnnBackendFlatbufferData_t> flatbufferDataArray(static_cast<size_t>(knobCount));

    int64_t actualCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(engineDesc,
                                             HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                             HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                             knobCount,
                                             &actualCount,
                                             flatbufferDataArray.data()),
        "Failed to get knob flatbuffer data from engine descriptor.");

    if(actualCount != knobCount)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Mismatch between expected and actual knob count."};
    }

    knobs.clear();
    knobs.reserve(static_cast<size_t>(actualCount));

    std::unordered_set<std::string> usedKnobIds;
    size_t skippedCount = 0;

    for(size_t i = 0; i < static_cast<size_t>(actualCount); ++i)
    {
        auto [error, knob] = Knob::tryFromFlatbuffer(flatbufferDataArray[i]);

        if(error.code != ErrorCode::OK)
        {
            // Log error and skip this knob - don't fail the entire operation
            HIPDNN_FE_LOG_ERROR("Skipping invalid knob at index " << i << ": " << error.err_msg);
            ++skippedCount;
            continue;
        }

        // Check for duplicate knob IDs
        if(!usedKnobIds.insert(knob.knobId()).second)
        {
            HIPDNN_FE_LOG_ERROR("Skipping knob with duplicate ID: " << knob.knobId());
            ++skippedCount;
            continue;
        }

        knobs.emplace_back(std::move(knob));
    }

    // If any knobs were skipped, include that information in the return message
    if(skippedCount > 0)
    {
        std::ostringstream oss;
        oss << "Loaded " << knobs.size() << " knobs, skipped " << skippedCount
            << " invalid/duplicate knobs";
        HIPDNN_FE_LOG_WARN(oss.str());
        return {ErrorCode::OK, oss.str()};
    }

    return {ErrorCode::OK, ""};
}

} // namespace detail

} // namespace hipdnn_frontend
