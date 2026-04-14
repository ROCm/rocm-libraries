// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"

namespace miopen_plugin
{

class MockPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    MOCK_METHOD(size_t, getWorkspaceSize, (const HipdnnMiopenHandle& handle), (const, override));
    MOCK_METHOD(void,
                execute,
                (const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace),
                (const, override));
};

}
