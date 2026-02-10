// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>

#include <hipdnn_frontend/knob/KnobSetting.hpp>

namespace hipdnn_frontend
{

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

} // namespace hidpnn_frontend
