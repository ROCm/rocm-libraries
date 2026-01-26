// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <vector>

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_plugin_sdk
{

class KnobFactory
{
public:
    static flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob>
        createIntKnob(flatbuffers::FlatBufferBuilder& builder,
                      const std::string& name,
                      const std::string& description,
                      int64_t defaultValue,
                      int64_t min,
                      int64_t max,
                      int64_t step,
                      const std::vector<int64_t>& options)
    {
        auto knobIdStr = builder.CreateString(name);
        auto descStr = builder.CreateString(description);
        auto defaultValOffset
            = hipdnn_data_sdk::data_objects::CreateIntValue(builder, defaultValue);
        auto optionsVector = builder.CreateVector(options);
        auto constraintOffset = hipdnn_data_sdk::data_objects::CreateIntConstraint(
            builder, min, max, step, optionsVector);

        return hipdnn_data_sdk::data_objects::CreateKnob(
            builder,
            knobIdStr,
            descStr,
            hipdnn_data_sdk::data_objects::KnobValue::IntValue,
            defaultValOffset.Union(),
            hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
            constraintOffset.Union(),
            false);
    }

    static flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob>
        createFloatKnob(flatbuffers::FlatBufferBuilder& builder,
                        const std::string& name,
                        const std::string& description,
                        float defaultValue,
                        float min,
                        float max)
    {
        auto knobIdStr = builder.CreateString(name);
        auto descStr = builder.CreateString(description);
        auto defaultValOffset = hipdnn_data_sdk::data_objects::CreateFloatValue(
            builder, static_cast<double>(defaultValue));
        auto constraintOffset = hipdnn_data_sdk::data_objects::CreateFloatConstraint(
            builder, static_cast<double>(min), static_cast<double>(max));

        return hipdnn_data_sdk::data_objects::CreateKnob(
            builder,
            knobIdStr,
            descStr,
            hipdnn_data_sdk::data_objects::KnobValue::FloatValue,
            defaultValOffset.Union(),
            hipdnn_data_sdk::data_objects::KnobConstraint::FloatConstraint,
            constraintOffset.Union(),
            false);
    }

    static flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob>
        createStringKnob(flatbuffers::FlatBufferBuilder& builder,
                         const std::string& name,
                         const std::string& description,
                         const std::string& defaultValue,
                         const std::vector<std::string>& options)
    {
        auto knobIdStr = builder.CreateString(name);
        auto descStr = builder.CreateString(description);
        auto defaultValStr = builder.CreateString(defaultValue);
        auto defaultValOffset
            = hipdnn_data_sdk::data_objects::CreateStringValue(builder, defaultValStr);

        std::vector<flatbuffers::Offset<flatbuffers::String>> optionsOffsets;
        optionsOffsets.reserve(options.size());
        for(const auto& opt : options)
        {
            optionsOffsets.push_back(builder.CreateString(opt));
        }
        auto optionsVector = builder.CreateVector(optionsOffsets);
        auto constraintOffset
            = hipdnn_data_sdk::data_objects::CreateStringConstraint(builder, 0, optionsVector);

        return hipdnn_data_sdk::data_objects::CreateKnob(
            builder,
            knobIdStr,
            descStr,
            hipdnn_data_sdk::data_objects::KnobValue::StringValue,
            defaultValOffset.Union(),
            hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint,
            constraintOffset.Union(),
            false);
    }
};

}
