// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file KnobConstraint.hpp
 * @brief Constraint classes for validating knob values
 *
 * This file defines constraint interfaces and implementations used to validate
 * knob settings. Constraints specify valid ranges or allowed values for knobs.
 */

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>

#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <unordered_set>

namespace hipdnn_frontend
{

/**
 * @class IConstraint
 * @brief Abstract interface for knob value constraints
 *
 * IConstraint defines the interface for validating knob settings. Different
 * constraint types (integer, float, string) implement this interface.
 *
 * @see IntConstraint, FloatConstraint, StringConstraint
 */
class IConstraint
{
public:
    /// @brief Virtual destructor
    virtual ~IConstraint() = default;

    /**
     * @brief Validate a knob setting against this constraint
     * @param setting The KnobSetting to validate
     * @return Error indicating success or describing the validation failure
     */
    virtual Error validateKnobSetting(const KnobSetting& setting) const = 0;

    /**
     * @brief Get a string representation of this constraint
     * @return Human-readable string for debugging/logging
     */
    virtual std::string toString() const = 0;
};

/**
 * @class IntConstraint
 * @brief Constraint for integer-valued knobs
 *
 * Validates that integer knob values are within a specified range and/or
 * match allowed values. Supports min/max bounds, step increments, and
 * explicit allowed value lists.
 */
class IntConstraint : public IConstraint
{
public:
    /**
     * @brief Construct an IntConstraint
     * @param minValue Minimum allowed value
     * @param maxValue Maximum allowed value
     * @param step Step increment (value must be minValue + n*step)
     * @param validValues Explicit set of allowed values (if non-empty, overrides range)
     */
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

        const int64_t val = *value;

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

    /// @brief Get the minimum allowed value
    int64_t getMinValue() const
    {
        return _minValue;
    }

    /// @brief Get the maximum allowed value
    int64_t getMaxValue() const
    {
        return _maxValue;
    }

    /// @brief Get the step increment
    int64_t getStep() const
    {
        return _step;
    }

    /// @brief Get the set of explicitly allowed values
    const std::unordered_set<int64_t>& getValidValues() const
    {
        return _validValues;
    }

private:
    int64_t _minValue; ///< Minimum allowed value
    int64_t _maxValue; ///< Maximum allowed value
    int64_t _step; ///< Step increment
    std::unordered_set<int64_t> _validValues; ///< Explicit allowed values
};

/**
 * @class FloatConstraint
 * @brief Constraint for floating-point valued knobs
 *
 * Validates that float knob values are within a specified range.
 */
class FloatConstraint : public IConstraint
{
public:
    /**
     * @brief Construct a FloatConstraint
     * @param minValue Minimum allowed value
     * @param maxValue Maximum allowed value
     */
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

        const double val = *value;

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

    /// @brief Get the minimum allowed value
    double getMinValue() const
    {
        return _minValue;
    }

    /// @brief Get the maximum allowed value
    double getMaxValue() const
    {
        return _maxValue;
    }

private:
    double _minValue; ///< Minimum allowed value
    double _maxValue; ///< Maximum allowed value
};

/**
 * @class StringConstraint
 * @brief Constraint for string-valued knobs
 *
 * Validates that string knob values have an acceptable length and/or
 * match allowed values.
 */
class StringConstraint : public IConstraint
{
public:
    /**
     * @brief Construct a StringConstraint
     * @param maxLength Maximum allowed string length
     * @param validValues Explicit set of allowed strings (if non-empty, overrides length check)
     */
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

    /// @brief Get the maximum allowed string length
    int32_t getMaxLength() const
    {
        return _maxLength;
    }

    /// @brief Get the set of explicitly allowed strings
    const std::unordered_set<std::string>& getValidValues() const
    {
        return _validValues;
    }

private:
    int32_t _maxLength; ///< Maximum string length
    std::unordered_set<std::string> _validValues; ///< Explicit allowed values
};

// Empty constraint implementation - represents an unconstrained knob
class EmptyConstraint : public IConstraint
{
public:
    EmptyConstraint() = default;

    Error validateKnobSetting(const KnobSetting& /*setting*/) const override
    {
        // Always valid - no constraints apply
        return {ErrorCode::OK, ""};
    }

    std::string toString() const override
    {
        return "EmptyConstraint{}";
    }
};
} // namespace hipdnn_frontend
