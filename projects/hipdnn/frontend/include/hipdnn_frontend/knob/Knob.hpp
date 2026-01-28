// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <HipdnnBackendFlatbufferData.h>

#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <spdlog/fmt/fmt.h>
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
    static Knob fromFlatbuffer(const hipdnn_data_sdk::data_objects::Knob* fbKnob)
    {
        if(fbKnob == nullptr)
        {
            throw std::invalid_argument("Null flatbuffer Knob pointer");
        }

        // Extract default value based on type
        KnobValueVariant defaultValue;
        switch(fbKnob->default_value_type())
        {
        case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        {
            auto intVal = fbKnob->default_value_as_IntValue();
            defaultValue = intVal != nullptr ? intVal->value() : 0;
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        {
            auto floatVal = fbKnob->default_value_as_FloatValue();
            defaultValue = floatVal != nullptr ? floatVal->value() : 0.0;
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
        {
            auto stringVal = fbKnob->default_value_as_StringValue();
            defaultValue = stringVal != nullptr && stringVal->value() != nullptr
                               ? stringVal->value()->str()
                               : std::string("");
            break;
        }
        default:
            throw std::invalid_argument("Unknown knob value type");
        }

        // Create the knob
        Knob knob(fbKnob->knob_id() != nullptr ? fbKnob->knob_id()->str() : "",
                  fbKnob->description() != nullptr ? fbKnob->description()->str() : "",
                  defaultValue,
                  fbKnob->deprecated());

        // Set constraint if present
        switch(fbKnob->constraint_type())
        {
        case hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint:
        {
            auto fbConstraint = fbKnob->constraint_as_IntConstraint();
            if(fbConstraint != nullptr)
            {
                std::unordered_set<int64_t> validValues
                    = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdUnorderedSet(
                        fbConstraint->valid_values());

                knob._constraint = std::make_unique<IntConstraint>(fbConstraint->min_value(),
                                                                   fbConstraint->max_value(),
                                                                   fbConstraint->step(),
                                                                   std::move(validValues));
            }
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobConstraint::FloatConstraint:
        {
            auto fbConstraint = fbKnob->constraint_as_FloatConstraint();
            if(fbConstraint != nullptr)
            {
                knob._constraint = std::make_unique<FloatConstraint>(fbConstraint->min_value(),
                                                                     fbConstraint->max_value());
            }
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint:
        {
            auto fbConstraint = fbKnob->constraint_as_StringConstraint();
            if(fbConstraint != nullptr)
            {
                std::unordered_set<std::string> validValues
                    = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdUnorderedSet(
                        fbConstraint->valid_values());
                knob._constraint = std::make_unique<StringConstraint>(fbConstraint->max_length(),
                                                                      std::move(validValues));
            }
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobConstraint::NONE:
            // No constraint
            break;
        default:
            throw std::invalid_argument("Unknown knob constraint");
            break;
        }

        return knob;
    }

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

    for(size_t i = 0; i < static_cast<size_t>(actualCount); ++i)
    {
        const auto& fbData = flatbufferDataArray[i];
        if(fbData.ptr == nullptr || fbData.size == 0)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Invalid flatbuffer data for knob at index " + std::to_string(i)};
        }

        hipdnn_data_sdk::flatbuffer_utilities::KnobWrapper knobWrapper(fbData.ptr, fbData.size);

        if(!knobWrapper.isValid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Knob flatbuffer failed verification at index " + std::to_string(i)};
        }
        try
        {
            knobs.emplace_back(Knob::fromFlatbuffer(&knobWrapper.getKnob()));
            if(!usedKnobIds.insert(knobs.back().knobId()).second)
            {
                return {ErrorCode::INVALID_VALUE,
                        "Engine description had knob with duplicate ID: " + knobs.back().knobId()};
            }
        }
        catch(const std::exception& e)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    std::string("Failed to create Knob from flatbuffer: ") + e.what()};
        }
    }

    return {ErrorCode::OK, ""};
}

} // namespace detail

} // namespace hipdnn_frontend

template <>
struct fmt::formatter<hipdnn_frontend::Knob> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(const hipdnn_frontend::Knob& knob, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(knob.toString().c_str(), ctx);
    }
};
