// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/EnginePlugin.hpp"
#include "plugin/SharedLibrary.hpp"

#include <cstdint>
#include <gmock/gmock.h>
#include <string_view>
#include <vector>

namespace hipdnn_backend
{
namespace plugin
{

class MockEnginePlugin : public EnginePlugin
{
public:
    // Mock all public methods from EnginePlugin
    MOCK_METHOD(hipdnnEnginePluginHandle_t, createHandle, (), (const));
    MOCK_METHOD(std::vector<int64_t>, getAllEngineIds, (), (const));
    MOCK_METHOD(void, destroyHandle, (hipdnnEnginePluginHandle_t handle), (const));
    MOCK_METHOD(void, setStream, (hipdnnEnginePluginHandle_t handle, hipStream_t stream), (const));
    MOCK_METHOD(std::vector<int64_t>,
                getApplicableEngineIds,
                (hipdnnEnginePluginHandle_t handle, const hipdnnPluginConstData_t* opGraph),
                (const));
    MOCK_METHOD(void,
                getEngineDetails,
                (hipdnnEnginePluginHandle_t handle,
                 int64_t engineId,
                 const hipdnnPluginConstData_t* opGraph,
                 hipdnnPluginConstData_t* engineDetails),
                (const));
    MOCK_METHOD(void,
                destroyEngineDetails,
                (hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* engineDetails),
                (const));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (hipdnnEnginePluginHandle_t handle,
                 const hipdnnPluginConstData_t* engineConfig,
                 const hipdnnPluginConstData_t* opGraph),
                (const));
    MOCK_METHOD(hipdnnEnginePluginExecutionContext_t,
                createExecutionContext,
                (hipdnnEnginePluginHandle_t handle,
                 const hipdnnPluginConstData_t* engineConfig,
                 const hipdnnPluginConstData_t* opGraph),
                (const));
    MOCK_METHOD(void,
                destroyExecutionContext,
                (hipdnnEnginePluginHandle_t handle,
                 hipdnnEnginePluginExecutionContext_t executionContext),
                (const));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (hipdnnEnginePluginHandle_t handle,
                 hipdnnEnginePluginExecutionContext_t executionContext),
                (const));
    MOCK_METHOD(void,
                executeOpGraph,
                (hipdnnEnginePluginHandle_t handle,
                 hipdnnEnginePluginExecutionContext_t executionContext,
                 void* workspace,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers),
                (const));

    // Mock inherited methods from PluginBase
    MOCK_METHOD(std::string_view, name, (), (const));
    MOCK_METHOD(std::string_view, version, (), (const));
    MOCK_METHOD(hipdnnPluginType_t, type, (), (const));
    MOCK_METHOD(hipdnnPluginStatus_t, setLoggingCallback, (hipdnnCallback_t callback), (const));
};

} // namespace plugin
} // namespace hipdnn_backend
