// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>

namespace hipdnn_tests::knob_expectation
{

using ExpectedKnobValue = std::variant<int64_t, double, std::string>;

struct CanonicalKnobSetting
{
    std::string knobId;
    std::string encodedValue;
};

inline std::string encodeKnobValue(const ExpectedKnobValue& value)
{
    return std::visit(
        [](const auto& typedValue) -> std::string {
            using T = std::decay_t<decltype(typedValue)>;
            std::ostringstream oss;
            if constexpr(std::is_same_v<T, int64_t>)
            {
                oss << "int64:" << typedValue;
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                oss << "float64:" << std::setprecision(std::numeric_limits<double>::max_digits10)
                    << typedValue;
            }
            else
            {
                oss << "string:" << std::quoted(typedValue);
            }
            return oss.str();
        },
        value);
}

inline CanonicalKnobSetting makeExpectedKnobSetting(std::string knobId,
                                                    const ExpectedKnobValue& value)
{
    return {std::move(knobId), encodeKnobValue(value)};
}

inline std::string serializeExpectedKnobSettings(int64_t engineId,
                                                 std::vector<CanonicalKnobSetting> settings)
{
    std::sort(settings.begin(),
              settings.end(),
              [](const CanonicalKnobSetting& lhs, const CanonicalKnobSetting& rhs) {
                  return lhs.knobId < rhs.knobId;
              });

    std::ostringstream oss;
    oss << "engine=" << engineId;
    for(const auto& setting : settings)
    {
        oss << "|" << setting.knobId << "=" << setting.encodedValue;
    }
    return oss.str();
}

inline std::string serializeActualKnobSettings(
    const hipdnn_data_sdk::flatbuffer_utilities::EngineConfigWrapper& engineConfig)
{
    std::vector<CanonicalKnobSetting> settings;
    settings.reserve(engineConfig.knobSettingCount());

    for(const auto& knobSetting : engineConfig.knobSettingWrappers())
    {
        switch(knobSetting->valueType())
        {
        case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
            settings.push_back(makeExpectedKnobSetting(
                knobSetting->knobId(),
                knobSetting->valueAs<hipdnn_data_sdk::data_objects::IntValue>().value()));
            break;
        case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
            settings.push_back(makeExpectedKnobSetting(
                knobSetting->knobId(),
                knobSetting->valueAs<hipdnn_data_sdk::data_objects::FloatValue>().value()));
            break;
        case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
        {
            auto* value
                = knobSetting->valueAs<hipdnn_data_sdk::data_objects::StringValue>().value();
            settings.push_back(makeExpectedKnobSetting(knobSetting->knobId(),
                                                       value != nullptr ? value->str() : ""));
            break;
        }
        default:
            settings.push_back(
                {knobSetting->knobId(),
                 "unknown:" + std::to_string(static_cast<int>(knobSetting->valueType()))});
            break;
        }
    }

    return serializeExpectedKnobSettings(engineConfig.engineId(), std::move(settings));
}

} // namespace hipdnn_tests::knob_expectation
