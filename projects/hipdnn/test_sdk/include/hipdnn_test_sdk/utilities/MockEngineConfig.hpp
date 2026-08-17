// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>

namespace hipdnn_test_sdk::utilities
{

class MockEngineConfig : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig
{
public:
    MOCK_METHOD(const hipdnn_flatbuffers_sdk::data_objects::EngineConfig&,
                getEngineConfig,
                (),
                (const, override));
    MOCK_METHOD(bool, isValid, (), (const, override));
    MOCK_METHOD(int64_t, engineId, (), (const, override));
    MOCK_METHOD(uint32_t, knobSettingCount, (), (const, override));
    MOCK_METHOD((const std::vector<
                    std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnobSetting>>&),
                knobSettingWrappers,
                (),
                (const, override));
    MOCK_METHOD(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IKnobSetting&,
                getKnobSettingByName,
                (const std::string& knobName),
                (const, override));
    MOCK_METHOD(bool, hasKnobSetting, (const std::string& knobName), (const, override));
};

}
