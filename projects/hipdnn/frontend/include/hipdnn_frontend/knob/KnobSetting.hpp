// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file KnobSetting.hpp
 * @brief Knob value settings for engine configuration
 *
 * This file defines the KnobSetting class which represents a specific value
 * assignment for an engine configuration knob.
 */

#pragma once

#include <string>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>

namespace hipdnn_frontend
{

/// Variant type for knob values (integer, float, or string)
using KnobValueVariant = std::variant<int64_t, double, std::string>;

/// Type alias for knob identifiers
typedef std::string KnobType_t; // NOLINT(readability-identifier-naming)

/**
 * @class KnobSetting
 * @brief Represents a specific value assignment for an engine configuration knob
 *
 * KnobSetting pairs a knob identifier with a value. This is used when creating
 * execution plans with custom engine configurations.
 *
 * @code{.cpp}
 * // Create knob settings
 * std::vector<KnobSetting> settings = {
 *     KnobSetting("HIPDNN_KNOB_TYPE_TILE_SIZE", int64_t{256}),
 *     KnobSetting("HIPDNN_KNOB_TYPE_SPLIT_K", int64_t{2})
 * };
 *
 * // Use in execution plan creation
 * graph.create_execution_plan_ext(engineId, settings);
 * @endcode
 *
 * @see Knob, Graph::create_execution_plan_ext()
 */
class KnobSetting
{
public:
    /**
     * @brief Construct a KnobSetting with ID and variant value
     * @param knobId The knob identifier string
     * @param value The value as a variant (int64_t, double, or string)
     */
    KnobSetting(std::string knobId, KnobValueVariant value)
        : _knobId(std::move(knobId))
        , _value(std::move(value))
    {
    }

    /**
     * @brief Construct a KnobSetting with ID and typed value
     * @tparam T Value type (int64_t, double, or std::string)
     * @param knobId The knob identifier string
     * @param value The value
     */
    template <typename T>
    KnobSetting(std::string knobId, const T& value)
        : _knobId(std::move(knobId))
        , _value(value)
    {
    }

    /**
     * @brief Get the knob identifier
     * @return The knob ID string
     */
    const std::string& knobId() const
    {
        return _knobId;
    }

    /**
     * @brief Get the knob value
     * @return The value as a variant
     */
    const KnobValueVariant& value() const
    {
        return _value;
    }

    /**
     * @brief Set the knob value
     * @tparam T Value type (int64_t, double, or std::string)
     * @param value The new value
     */
    template <typename T>
    void setValue(const T& value)
    {
        _value = value;
    }

    /**
     * @brief Serialize this KnobSetting to a FlatBuffer
     * @param builder The FlatBufferBuilder to use
     * @return FlatBuffer offset for the serialized KnobSetting
     */
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

    /**
     * @brief Get a string representation of this KnobSetting
     * @return Human-readable string describing the setting
     */
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
    std::string _knobId; ///< The knob identifier
    KnobValueVariant _value; ///< The knob value
};

} // namespace hipdnn_frontend
