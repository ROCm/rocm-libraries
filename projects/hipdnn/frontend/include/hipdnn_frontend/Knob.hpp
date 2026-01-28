// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/flatbuffer_utilities/KnobWrapper.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>

#include <HipdnnBackendFlatbufferData.h>

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

using KnobValueVariant = std::variant<int64_t, double, std::string>;

// Type alias for knob IDs
typedef std::string KnobType_t; // NOLINT(readability-identifier-naming)

// KnobSetting class - represents a knob value setting
class KnobSetting
{
public:
    // Constructors
    KnobSetting(std::string knobId, KnobValueVariant value)
        : _knobId(std::move(knobId))
        , _value(std::move(value))
    {
    }

    template <typename T>
    KnobSetting(std::string knobId, const T& value)
        : _knobId(std::move(knobId))
        , _value(value)
    {
    }

    // Accessors
    const std::string& knobId() const
    {
        return _knobId;
    }

    const KnobValueVariant& value() const
    {
        return _value;
    }

    // Mutator
    template <typename T>
    void setValue(const T& value)
    {
        _value = value;
    }

    // Serialization
    flatbuffers::Offset<hipdnn_data_sdk::data_objects::KnobSetting>
        packKnobSetting(flatbuffers::FlatBufferBuilder& builder) const
    {
        // Create the appropriate KnobValue based on the variant type
        flatbuffers::Offset<void> valueOffset = 0;
        hipdnn_data_sdk::data_objects::KnobValue valueType
            = hipdnn_data_sdk::data_objects::KnobValue::NONE;

        std::visit(
            [&builder, &valueOffset, &valueType](auto&& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr(std::is_same_v<T, int64_t>)
                {
                    valueOffset
                        = hipdnn_data_sdk::data_objects::CreateIntValue(builder, value).Union();
                    valueType = hipdnn_data_sdk::data_objects::KnobValue::IntValue;
                }
                else if constexpr(std::is_same_v<T, double>)
                {
                    valueOffset
                        = hipdnn_data_sdk::data_objects::CreateFloatValue(builder, value).Union();
                    valueType = hipdnn_data_sdk::data_objects::KnobValue::FloatValue;
                }
                else if constexpr(std::is_same_v<T, std::string>)
                {
                    valueOffset = hipdnn_data_sdk::data_objects::CreateStringValueDirect(
                                      builder, value.c_str())
                                      .Union();
                    valueType = hipdnn_data_sdk::data_objects::KnobValue::StringValue;
                }
            },
            _value);

        return hipdnn_data_sdk::data_objects::CreateKnobSettingDirect(
            builder, _knobId.c_str(), valueType, valueOffset);
    }

    // String representation
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "KnobSetting{knobIdStr=" << _knobId << ", value=";

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
            _value);

        oss << "}";
        return oss.str();
    }

private:
    std::string _knobId;
    KnobValueVariant _value;
};

// Abstract constraint interface
class IConstraint
{
public:
    virtual ~IConstraint() = default;

    // Validate a knob setting against this constraint
    virtual Error validateKnobSetting(const KnobSetting& setting) const = 0;

    // String representation for logging
    virtual std::string toString() const = 0;
};

// Integer constraint implementation
class IntConstraint : public IConstraint
{
public:
    IntConstraint(int64_t minValue,
                  int64_t maxValue,
                  int64_t step = 1,
                  std::unordered_set<int64_t> validValues = {})
        : _minValue(minValue)
        , _maxValue(maxValue)
        , _step(step)
        , _validValues(std::move(validValues))
    {
    }

    Error validateKnobSetting(const KnobSetting& setting) const override
    {
        auto value = std::get_if<int64_t>(&setting.value());
        if(value == nullptr)
        {
            return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain an integer value"};
        }

        int64_t val = *value;

        // If explicit valid values are specified, check against them
        if(!_validValues.empty())
        {
            if(_validValues.count(val) == 0)
            {
                std::ostringstream oss;
                oss << "Value " << val << " is not in the list of valid values: ";
                std::vector<int64_t> sortedValues(_validValues.begin(), _validValues.end());
                std::sort(sortedValues.begin(), sortedValues.end());
                return {ErrorCode::INVALID_VALUE, oss.str()};
            }
            return {ErrorCode::OK, ""};
        }

        // Otherwise check min/max/step
        if(val < _minValue || val > _maxValue)
        {
            std::ostringstream oss;
            oss << "Value " << val << " is out of range [" << _minValue << ", " << _maxValue << "]";
            return {ErrorCode::INVALID_VALUE, oss.str()};
        }

        if(_step > 1 && ((val - _minValue) % _step) != 0)
        {
            std::ostringstream oss;
            oss << "Value " << val << " does not satisfy step constraint (step=" << _step
                << ", min=" << _minValue << ")";
            return {ErrorCode::INVALID_VALUE, oss.str()};
        }

        return {ErrorCode::OK, ""};
    }

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "IntConstraint{min=" << _minValue << ", max=" << _maxValue << ", step=" << _step;
        if(!_validValues.empty())
        {
            std::vector<int64_t> sortedValues(_validValues.begin(), _validValues.end());
            std::sort(sortedValues.begin(), sortedValues.end());
            oss << ", validValues=";
            hipdnn_data_sdk::utilities::vecToStream(oss, sortedValues);
        }
        oss << "}";
        return oss.str();
    }

    int64_t getMinValue() const
    {
        return _minValue;
    }
    int64_t getMaxValue() const
    {
        return _maxValue;
    }
    int64_t getStep() const
    {
        return _step;
    }
    const std::unordered_set<int64_t>& getValidValues() const
    {
        return _validValues;
    }

private:
    int64_t _minValue;
    int64_t _maxValue;
    int64_t _step;
    std::unordered_set<int64_t> _validValues;
};

// Float constraint implementation
class FloatConstraint : public IConstraint
{
public:
    FloatConstraint(double minValue, double maxValue)
        : _minValue(minValue)
        , _maxValue(maxValue)
    {
    }

    Error validateKnobSetting(const KnobSetting& setting) const override
    {
        auto value = std::get_if<double>(&setting.value());
        if(value == nullptr)
        {
            return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain a float value"};
        }

        double val = *value;

        if(val < _minValue || val > _maxValue)
        {
            std::ostringstream oss;
            oss << "Value " << val << " is out of range [" << _minValue << ", " << _maxValue << "]";
            return {ErrorCode::INVALID_VALUE, oss.str()};
        }

        return {ErrorCode::OK, ""};
    }

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "FloatConstraint{min=" << _minValue << ", max=" << _maxValue << "}";
        return oss.str();
    }

    double getMinValue() const
    {
        return _minValue;
    }
    double getMaxValue() const
    {
        return _maxValue;
    }

private:
    double _minValue;
    double _maxValue;
};

// String constraint implementation
class StringConstraint : public IConstraint
{
public:
    StringConstraint(int32_t maxLength, std::unordered_set<std::string> validValues = {})
        : _maxLength(maxLength)
        , _validValues(std::move(validValues))
    {
    }

    Error validateKnobSetting(const KnobSetting& setting) const override
    {
        auto value = std::get_if<std::string>(&setting.value());
        if(value == nullptr)
        {
            return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain a string value"};
        }

        const std::string& val = *value;

        // If explicit valid values are specified, check against them
        if(!_validValues.empty())
        {
            if(_validValues.count(val) == 0)
            {
                std::ostringstream oss;
                oss << "Value \"" << val << "\" is not in the list of valid values: ";
                std::vector<std::string> sortedValues(_validValues.begin(), _validValues.end());
                std::sort(sortedValues.begin(), sortedValues.end());
                hipdnn_data_sdk::utilities::stringVecToStream(oss, sortedValues);
                return {ErrorCode::INVALID_VALUE, oss.str()};
            }
            return {ErrorCode::OK, ""};
        }

        // Otherwise check max length
        if(static_cast<int32_t>(val.length()) > _maxLength)
        {
            std::ostringstream oss;
            oss << "String length " << val.length() << " exceeds maximum length " << _maxLength;
            return {ErrorCode::INVALID_VALUE, oss.str()};
        }

        return {ErrorCode::OK, ""};
    }

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "StringConstraint{maxLength=" << _maxLength;
        if(!_validValues.empty())
        {
            std::vector<std::string> sortedValues(_validValues.begin(), _validValues.end());
            std::sort(sortedValues.begin(), sortedValues.end());
            oss << ", validValues=";

            hipdnn_data_sdk::utilities::stringVecToStream(oss, sortedValues);
        }
        oss << "}";
        return oss.str();
    }

    int32_t getMaxLength() const
    {
        return _maxLength;
    }
    const std::unordered_set<std::string>& getValidValues() const
    {
        return _validValues;
    }

private:
    int32_t _maxLength;
    std::unordered_set<std::string> _validValues;
};

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

    const std::string& getDescription() const
    {
        return _description;
    }

    bool isDeprecated() const
    {
        return _deprecated;
    }

    KnobValueType getValueType() const
    {
        return getKnobValueTypeFromVariant(_defaultValue);
    }

    const KnobValueVariant& getDefaultValue() const
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
            knobs.emplace_back(Knob::fromFlatbuffer(knobWrapper.getKnob()));
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

template <>
struct fmt::formatter<hipdnn_frontend::KnobSetting> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(const hipdnn_frontend::KnobSetting& setting, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(setting.toString().c_str(), ctx);
    }
};
