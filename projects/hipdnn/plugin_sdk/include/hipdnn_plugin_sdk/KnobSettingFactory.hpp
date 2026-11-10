// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>

namespace hipdnn_plugin_sdk
{

class KnobSettingFactory
{
public:
    static flatbuffers::DetachedBuffer createIntKnobSetting(const std::string& knobId,
                                                            int64_t value)
    {
        flatbuffers::FlatBufferBuilder builder;
        const auto knobIdOffset = builder.CreateString(knobId);
        const auto intValue = hipdnn_flatbuffers_sdk::data_objects::CreateIntValue(builder, value);
        const auto knobSetting = hipdnn_flatbuffers_sdk::data_objects::CreateKnobSetting(
            builder,
            knobIdOffset,
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue,
            intValue.Union());
        builder.Finish(knobSetting);
        return builder.Release();
    }

    static flatbuffers::DetachedBuffer createFloatKnobSetting(const std::string& knobId,
                                                              double value)
    {
        flatbuffers::FlatBufferBuilder builder;
        const auto knobIdOffset = builder.CreateString(knobId);
        const auto floatValue
            = hipdnn_flatbuffers_sdk::data_objects::CreateFloatValue(builder, value);
        const auto knobSetting = hipdnn_flatbuffers_sdk::data_objects::CreateKnobSetting(
            builder,
            knobIdOffset,
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue,
            floatValue.Union());
        builder.Finish(knobSetting);
        return builder.Release();
    }

    static flatbuffers::DetachedBuffer createStringKnobSetting(const std::string& knobId,
                                                               const std::string& value)
    {
        flatbuffers::FlatBufferBuilder builder;
        const auto knobIdOffset = builder.CreateString(knobId);
        const auto strValueOffset = builder.CreateString(value);
        const auto stringValue
            = hipdnn_flatbuffers_sdk::data_objects::CreateStringValue(builder, strValueOffset);
        const auto knobSetting = hipdnn_flatbuffers_sdk::data_objects::CreateKnobSetting(
            builder,
            knobIdOffset,
            hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue,
            stringValue.Union());
        builder.Finish(knobSetting);
        return builder.Release();
    }
};

} // namespace hipdnn_plugin_sdk
