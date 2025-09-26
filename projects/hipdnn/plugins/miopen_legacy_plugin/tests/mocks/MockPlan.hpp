// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/PlanInterface.hpp"
#include <gmock/gmock.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

namespace miopen_legacy_plugin
{

class MockPlan : public IPlan
{
public:
    MOCK_METHOD(void,
                execute,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace),
                (const, override));
};

}
