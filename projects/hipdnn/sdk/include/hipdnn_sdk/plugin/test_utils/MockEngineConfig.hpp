// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/EngineConfigWrapper.hpp>

namespace hipdnn_plugin
{

class MockEngineConfig : public IEngineConfig
{
public:
    MOCK_METHOD(const hipdnn_sdk::data_objects::EngineConfig&,
                getEngineConfig,
                (),
                (const, override));
    MOCK_METHOD(bool, isValid, (), (const, override));
    MOCK_METHOD(int64_t, engineId, (), (const, override));
};

}
