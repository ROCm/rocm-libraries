// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <spdlog/fmt/fmt.h>
#include <sstream>
#include <string>
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
    virtual Error validateKnobSetting(const KnobSetting& knobSetting) const = 0;

    // String representation for logging
    virtual std::string toString() const = 0;
};

// Integer constraint implementation
class IntConstraint : public IConstraint
{
public:
    IntConstraint(int64_t minValue,
                  int64_t maxValue,
                  int64_t stride = 1,
                  std::vector<int64_t> validValues = {})
        : _minValue(minValue)
        , _maxValue(maxValue)
        , _stride(stride)
        , _validValues(std::move(validValues))
    {
    }

    Error validateKnobSetting(const KnobSetting& knobSetting) const override;

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "IntConstraint{min=" << _minValue << ", max=" << _maxValue << ", stride=" << _stride;
        if(!_validValues.empty())
        {
            oss << ", validValues=[";
            for(size_t i = 0; i < _validValues.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << _validValues[i];
            }
            oss << "]";
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
    int64_t getStride() const
    {
        return _stride;
    }
    const std::vector<int64_t>& getValidValues() const
    {
        return _validValues;
    }

private:
    int64_t _minValue;
    int64_t _maxValue;
    int64_t _stride;
    std::vector<int64_t> _validValues;
};

// Float constraint implementation
class FloatConstraint : public IConstraint
{
public:
    FloatConstraint(double minValue, double maxValue, std::vector<double> validValues = {})
        : _minValue(minValue)
        , _maxValue(maxValue)
        , _validValues(std::move(validValues))
    {
    }

    Error validateKnobSetting(const KnobSetting& knobSetting) const override;

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "FloatConstraint{min=" << _minValue << ", max=" << _maxValue;
        if(!_validValues.empty())
        {
            oss << ", validValues=[";
            for(size_t i = 0; i < _validValues.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << _validValues[i];
            }
            oss << "]";
        }
        oss << "}";
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
    const std::vector<double>& getValidValues() const
    {
        return _validValues;
    }

private:
    double _minValue;
    double _maxValue;
    std::vector<double> _validValues;
};

// String constraint implementation
class StringConstraint : public IConstraint
{
public:
    StringConstraint(int32_t maxLength, std::vector<std::string> validValues = {})
        : _maxLength(maxLength)
        , _validValues(std::move(validValues))
    {
    }

    Error validateKnobSetting(const KnobSetting& knobSetting) const override;

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "StringConstraint{maxLength=" << _maxLength;
        if(!_validValues.empty())
        {
            oss << ", validValues=[";
            for(size_t i = 0; i < _validValues.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << "\"" << _validValues[i] << "\"";
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
    const std::vector<std::string>& getValidValues() const
    {
        return _validValues;
    }

private:
    int32_t _maxLength;
    std::vector<std::string> _validValues;
};

// Knob setting class - represents a configured knob value
class KnobSetting
{
public:
    // Factory function to create from flatbuffer
    // TODO: Implement once flatbuffer schemas are available
    // static KnobSetting fromFlatbuffer(const hipdnn_data_sdk::data_objects::KnobSetting* fbKnobSetting);

    // Constructors for different value types
    KnobSetting(int64_t knobId, int64_t value)
        : _knobId(knobId)
        , _value(value)
    {
    }

    KnobSetting(int64_t knobId, double value)
        : _knobId(knobId)
        , _value(value)
    {
    }

    KnobSetting(int64_t knobId, const std::string& value)
        : _knobId(knobId)
        , _value(value)
    {
    }

    // String representation for logging
    std::string toString() const
    {
        return std::visit(
            [this](auto&& value) -> std::string {
                std::ostringstream oss;
                oss << "KnobSetting{knobId=" << _knobId << ", value=";
                if constexpr(std::is_same_v<std::decay_t<decltype(value)>, std::string>)
                {
                    oss << "\"" << value << "\"";
                }
                else
                {
                    oss << value;
                }
                oss << "}";
                return oss.str();
            },
            _value);
    }

    // Accessors
    int64_t getKnobId() const
    {
        return _knobId;
    }

    KnobValueType getValueType() const
    {
        return getKnobValueTypeFromVariant(_value);
    }

    // Get value (templated)
    template <typename T>
    std::optional<T> getValue() const
    {
        if(auto* val = std::get_if<T>(&_value))
        {
            return *val;
        }
        return std::nullopt;
    }

    // Flatbuffer pack method
    // TODO: Implement once flatbuffer schemas are available
    // flatbuffers::Offset<hipdnn_data_sdk::data_objects::KnobSetting> pack(
    //     flatbuffers::FlatBufferBuilder& builder) const;

private:
    int64_t _knobId;
    std::variant<int64_t, double, std::string> _value;
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

    // Get default value (templated)
    template <typename T>
    std::optional<T> getDefaultValue() const
    {
        if(auto* val = std::get_if<T>(&_defaultValue))
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

    // Convert to KnobSetting with default value
    KnobSetting toDefaultKnobSetting() const
    {
        return std::visit(
            [this](auto&& value) -> KnobSetting { return KnobSetting(_knobId, value); },
            _defaultValue);
    }

    // Validate a knob setting against this knob's constraints
    Error validateKnobSetting(const KnobSetting& knobSetting) const
    {
        // Check that the knob IDs match
        if(knobSetting.getKnobId() != _knobId)
        {
            return {ErrorCode::INVALID_VALUE, "KnobSetting knob ID does not match Knob knob ID"};
        }

        // Check that the value types match
        if(knobSetting.getValueType() != getValueType())
        {
            return {ErrorCode::INVALID_VALUE,
                    "KnobSetting value type does not match Knob value type for knob '" + _knobIdStr
                        + "'"};
        }

        // Validate against constraint if present
        if(_constraint)
        {
            return _constraint->validateKnobSetting(knobSetting);
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
            _defaultValue);

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

    std::string _knobIdStr;
    int64_t _knobId;
    std::string _description;
    std::variant<int64_t, double, std::string> _defaultValue;
    bool _deprecated;

    // Constraint (polymorphic)
    std::unique_ptr<IConstraint> _constraint;

    // Allow factory function access to private members
    // TODO: Uncomment when implementing factory
    // friend Knob fromFlatbuffer(const hipdnn_data_sdk::data_objects::Knob* fbKnob);
};

// Constraint validation implementations (defined after KnobSetting is complete)
inline Error IntConstraint::validateKnobSetting(const KnobSetting& knobSetting) const
{
    auto value = knobSetting.getValue<int64_t>();
    if(!value.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain an integer value"};
    }

    int64_t val = value.value();

    // If explicit valid values are specified, check against them
    if(!_validValues.empty())
    {
        bool found = false;
        for(auto validVal : _validValues)
        {
            if(val == validVal)
            {
                found = true;
                break;
            }
        }
        if(!found)
        {
            std::ostringstream oss;
            oss << "Value " << val << " is not in the list of valid values: [";
            for(size_t i = 0; i < _validValues.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << _validValues[i];
            }
            oss << "]";
            return {ErrorCode::INVALID_VALUE, oss.str()};
        }
        return {ErrorCode::OK, ""};
    }

    // Otherwise check min/max/stride
    if(val < _minValue || val > _maxValue)
    {
        std::ostringstream oss;
        oss << "Value " << val << " is out of range [" << _minValue << ", " << _maxValue << "]";
        return {ErrorCode::INVALID_VALUE, oss.str()};
    }

    if(_stride > 1 && ((val - _minValue) % _stride) != 0)
    {
        std::ostringstream oss;
        oss << "Value " << val << " does not satisfy stride constraint (stride=" << _stride
            << ", min=" << _minValue << ")";
        return {ErrorCode::INVALID_VALUE, oss.str()};
    }

    return {ErrorCode::OK, ""};
}

inline Error FloatConstraint::validateKnobSetting(const KnobSetting& knobSetting) const
{
    auto value = knobSetting.getValue<double>();
    if(!value.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain a float value"};
    }

    double val = value.value();

    // If explicit valid values are specified, check against them
    if(!_validValues.empty())
    {
        bool found = false;
        for(auto validVal : _validValues)
        {
            if(val == validVal)
            {
                found = true;
                break;
            }
        }
        if(!found)
        {
            std::ostringstream oss;
            oss << "Value " << val << " is not in the list of valid values: [";
            for(size_t i = 0; i < _validValues.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << _validValues[i];
            }
            oss << "]";
            return {ErrorCode::INVALID_VALUE, oss.str()};
        }
        return {ErrorCode::OK, ""};
    }

    // Otherwise check min/max
    if(val < _minValue || val > _maxValue)
    {
        std::ostringstream oss;
        oss << "Value " << val << " is out of range [" << _minValue << ", " << _maxValue << "]";
        return {ErrorCode::INVALID_VALUE, oss.str()};
    }

    return {ErrorCode::OK, ""};
}

inline Error StringConstraint::validateKnobSetting(const KnobSetting& knobSetting) const
{
    auto value = knobSetting.getValue<std::string>();
    if(!value.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "KnobSetting does not contain a string value"};
    }

    const std::string& val = value.value();

    // If explicit valid values are specified, check against them
    if(!_validValues.empty())
    {
        bool found = false;
        for(const auto& validVal : _validValues)
        {
            if(val == validVal)
            {
                found = true;
                break;
            }
        }
        if(!found)
        {
            std::ostringstream oss;
            oss << "Value \"" << val << "\" is not in the list of valid values: [";
            for(size_t i = 0; i < _validValues.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << "\"" << _validValues[i] << "\"";
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

// Validate knob settings against knob constraints
inline Error validateKnobSettings(const std::unordered_map<int64_t, Knob>& knobs,
                                  const std::unordered_map<int64_t, KnobSetting>& settings)
{
    for(const auto& setting : settings)
    {
        auto found = knobs.find(setting.first);
        if(found == knobs.end())
        {
            return {ErrorCode::INVALID_VALUE,
                    fmt::format("KnobSetting {} isn't a knob supported by this engine",
                                setting.second)};
        }

        Error err = found->second.validateKnobSetting(setting.second);
        if(err.code != ErrorCode::OK)
        {
            return err;
        }
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

template <>
struct fmt::formatter<hipdnn_frontend::KnobSetting> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(const hipdnn_frontend::KnobSetting& setting, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(setting.toString().c_str(), ctx);
    }
};
