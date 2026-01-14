// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>

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

#define KNOB_TYPES int64_t, double, std::string

// Type alias for knob IDs
typedef int64_t KnobType_t; // NOLINT(readability-identifier-naming)

// Abstract knob interface
class IKnob
{
public:
    virtual ~IKnob() = default;

    virtual const std::variant<KNOB_TYPES>& getChoice() const = 0;
};

// Abstract constraint interface
class IConstraint
{
public:
    virtual ~IConstraint() = default;

    // Validate a knob setting against this constraint
    virtual Error validateKnobChoice(const IKnob& knob) const = 0;

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

    Error validateKnobChoice(const IKnob& knob) const override
    {
        auto value = std::get_if<int64_t>(&knob.getChoice());
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
                oss << "Value " << val << " is not in the list of valid values: [";
                bool first = true;
                for(const auto& validVal : _validValues)
                {
                    if(!first)
                    {
                        oss << ", ";
                    }
                    oss << validVal;
                    first = false;
                }
                oss << "]";
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

    Error validateKnobChoice(const IKnob& knob) const override
    {
        auto value = std::get_if<double>(&knob.getChoice());
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

    Error validateKnobChoice(const IKnob& knob) const override
    {
        auto value = std::get_if<std::string>(&knob.getChoice());
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
                oss << "Value \"" << val << "\" is not in the list of valid values: [";
                bool first = true;
                for(const auto& validVal : _validValues)
                {
                    if(!first)
                    {
                        oss << ", ";
                    }
                    oss << "\"" << validVal << "\"";
                    first = false;
                }
                oss << "]";
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
class Knob : public IKnob
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
        std::variant<KNOB_TYPES> defaultValue;
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
        Knob knob(fbKnob->knob_id_str() != nullptr ? fbKnob->knob_id_str()->str() : "",
                  fbKnob->description() != nullptr ? fbKnob->description()->str() : "",
                  defaultValue,
                  fbKnob->deprecated());

        // Set constraint if present
        switch(fbKnob->constraints_type())
        {
        case hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint:
        {
            auto fbConstraint = fbKnob->constraints_as_IntConstraint();
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
            auto fbConstraint = fbKnob->constraints_as_FloatConstraint();
            if(fbConstraint != nullptr)
            {
                knob._constraint = std::make_unique<FloatConstraint>(fbConstraint->min_value(),
                                                                     fbConstraint->max_value());
            }
            break;
        }
        case hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint:
        {
            auto fbConstraint = fbKnob->constraints_as_StringConstraint();
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
        default:
            // No constraint
            break;
        }

        return knob;
    }

    // Accessors
    int64_t getKnobId() const
    {
        return _knobId;
    }

    const std::string& getKnobIdStr() const
    {
        return _knobIdStr;
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

    const std::variant<KNOB_TYPES>& getDefaultValue() const
    {
        return _defaultValue;
    }

    template <typename T>
    void setChoice(const T& value)
    {
        _choice = value;
    }

    const std::variant<KNOB_TYPES>& getChoice() const override
    {
        return _choice;
    }

    // Get constraint
    const IConstraint* getConstraint() const
    {
        return _constraint.get();
    }

    // Validate a knob setting against this knob's constraints
    Error validate() const
    {
        // Validate against constraint if present
        if(_constraint)
        {
            return _constraint->validateKnobChoice(*this);
        }

        return {ErrorCode::OK, ""};
    }

    // Flatbuffer pack method
    // TODO: Implement once flatbuffer schemas are available
    // flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob> pack(
    //     flatbuffers::FlatBufferBuilder& builder) const;

    // Helper to hash the string ID to the int ID
    static int64_t makeKnobId(const std::string& strID)
    {
        return static_cast<int64_t>(std::hash<std::string>{}(strID));
    }

    // String representation for logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "Knob{knobId=" << _knobId << ", knobIdStr=\"" << _knobIdStr << "\", description=\""
            << _description << "\", defaultValue=";

        variantToStream(oss, _defaultValue);

        oss << ", choice=";

        variantToStream(oss, _choice);

        oss << ", deprecated=" << (_deprecated ? "true" : "false");

        if(_constraint)
        {
            oss << ", constraint=" << _constraint->toString();
        }

        oss << "}";
        return oss.str();
    }

private:
    // Private constructor - use factory function to create instances
    Knob(std::string knobIdStr,
         std::string description,
         std::variant<KNOB_TYPES> defaultValue,
         bool deprecated)
        : _knobIdStr(std::move(knobIdStr))
        , _knobId(makeKnobId(_knobIdStr))
        , _description(std::move(description))
        , _defaultValue(std::move(defaultValue))
        , _deprecated(deprecated)
    {
    }

    static void variantToStream(std::ostringstream& oss, const std::variant<KNOB_TYPES>& variant)
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

    std::string _knobIdStr;
    int64_t _knobId;
    std::string _description;
    std::variant<KNOB_TYPES> _defaultValue;
    std::variant<KNOB_TYPES> _choice;
    bool _deprecated;

    // Constraint (polymorphic)
    std::unique_ptr<IConstraint> _constraint;

    // Allow factory function access to private members
    // friend Knob fromFlatbuffer(const hipdnn_data_sdk::data_objects::Knob* fbKnob);
};

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
