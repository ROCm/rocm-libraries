// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnHipKernelHandle.hpp"

namespace hip_kernel_plugin
{

class MockPlan : public hipdnn_plugin_sdk::IPlan<HipdnnHipKernelHandle>
{
public:
    MOCK_METHOD(size_t, getWorkspaceSize, (const HipdnnHipKernelHandle& handle), (const, override));
    MOCK_METHOD(void,
                execute,
                (const HipdnnHipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace),
                (const, override));
};

}
