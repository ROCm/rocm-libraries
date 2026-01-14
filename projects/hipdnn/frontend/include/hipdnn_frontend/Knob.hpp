// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>

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

// Type alias for knob IDs
typedef int64_t KnobType_t; // NOLINT(readability-identifier-naming)

// Forward declarations
class KnobSetting;

// Abstract constraint interface
class IConstraint
{
public:
    virtual ~IConstraint() = default;

    // Validate a knob setting against this constraint
    virtual Error validateKnobChoice(const Knob& knob) const = 0;

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

    Error validateKnobChoice(const Knob& knob) const override;

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

    Error validateKnobChoice(const Knob& knob) const override;

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

    Error validateKnobChoice(const Knob& knob) const override;

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "StringConstraint{maxLength=" << _maxLength;
        if(!_validValues.empty())
        {
            std::vector<std::string> sortedValues(_validValues.begin(), _validValues.end());
            std::sort(sortedValues.begin(), sortedValues.end());
            oss << ", validValues=[";
            for(size_t i = 0; i < sortedValues.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << "\"" << sortedValues[i] << "\"";
            }
            oss << "]";
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
    // TODO: Implement once flatbuffer schemas are available
    // static Knob fromFlatbuffer(const hipdnn_data_sdk::data_objects::Knob* fbKnob);

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

    template <typename T>
    std::optional<T> getDefaultValue() const
    {
        if(auto* val = std::get_if<T>(&_defaultValue))
        {
            return *val;
        }
        return std::nullopt;
    }

    template <typename T>
    void setChoice(T value)
    {
        _choice = value;
    }

    template <typename T>
    std::optional<T> getChoice() const
    {
        if(auto* val = std::get_if<T>(&_choice))
        {
            return *val;
        }
        return std::nullopt;
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
         std::variant<int64_t, double, std::string> defaultValue,
         bool deprecated)
        : _knobIdStr(std::move(knobIdStr))
        , _knobId(makeKnobId(_knobIdStr))
        , _description(std::move(description))
        , _defaultValue(std::move(defaultValue))
        , _deprecated(deprecated)
    {
    }

    void variantToStream(std::ostringstream& oss,
                         const std::variant<int64_t, double, std::string>& variant) const
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
    std::variant<int64_t, double, std::string> _defaultValue;
    std::variant<int64_t, double, std::string> _choice;
    bool _deprecated;

    // Constraint (polymorphic)
    std::unique_ptr<IConstraint> _constraint;

    // Allow factory function access to private members
    // TODO: Uncomment when implementing factory
    // friend Knob fromFlatbuffer(const hipdnn_data_sdk::data_objects::Knob* fbKnob);

    // Allow test helper to create Knob instances for testing
    friend class KnobTestHelper;
};

// Constraint validation implementations (defined after KnobSetting is complete)
inline Error IntConstraint::validateKnobChoice(const Knob& knob) const
{
    auto value = knob.getChoice<int64_t>();
    if(!value.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain an integer value"};
    }

    int64_t val = value.value();

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

inline Error FloatConstraint::validateKnobChoice(const Knob& knob) const
{
    auto value = knob.getChoice<double>();
    if(!value.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain a float value"};
    }

    double val = value.value();

    if(val < _minValue || val > _maxValue)
    {
        std::ostringstream oss;
        oss << "Value " << val << " is out of range [" << _minValue << ", " << _maxValue << "]";
        return {ErrorCode::INVALID_VALUE, oss.str()};
    }

    return {ErrorCode::OK, ""};
}

inline Error StringConstraint::validateKnobChoice(const Knob& knob) const
{
    auto value = knob.getChoice<std::string>();
    if(!value.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain a string value"};
    }

    const std::string& val = value.value();

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
