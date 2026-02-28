// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>

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

} // namespace hipdnn_frontend

template <>
struct fmt::formatter<hipdnn_frontend::KnobSetting> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(const hipdnn_frontend::KnobSetting& setting, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(setting.toString().c_str(), ctx);
    }
};
