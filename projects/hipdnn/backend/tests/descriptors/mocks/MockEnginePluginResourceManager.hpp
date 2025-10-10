// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/EnginePluginResourceManager.hpp"
#include <gmock/gmock.h>

namespace hipdnn_backend
{
namespace plugin
{

class MockEnginePluginResourceManager : public EnginePluginResourceManager
{
public:
    MOCK_METHOD(void, setStream, (hipStream_t stream), (const, override));
    MOCK_METHOD(void,
                executeOpGraph,
                (hipdnnBackendDescriptor_t executionPlan, hipdnnBackendDescriptor_t variantPack),
                (const, override));
    MOCK_METHOD(std::vector<int64_t>,
                getApplicableEngineIds,
                (const hipdnn_backend::GraphDescriptor* graphDesc),
                (const, override));
    MOCK_METHOD(void,
                getEngineDetails,
                (int64_t engineId,
                 const hipdnn_backend::GraphDescriptor* graphDesc,
                 hipdnnPluginConstData_t* engineDetails),
                (const, override));
    MOCK_METHOD(void,
                destroyEngineDetails,
                (int64_t engineId, hipdnnPluginConstData_t* engineDetails),
                (const, override));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (int64_t engineId,
                 const hipdnnPluginConstData_t* engineConfig,
                 const hipdnn_backend::GraphDescriptor* graphDesc),
                (const, override));
    MOCK_METHOD(hipdnnEnginePluginExecutionContext_t,
                createExecutionContext,
                (int64_t engineId,
                 const hipdnnPluginConstData_t* engineConfig,
                 const hipdnn_backend::GraphDescriptor* graphDesc),
                (const, override));
    MOCK_METHOD(void,
                destroyExecutionContext,
                (int64_t engineId, hipdnnEnginePluginExecutionContext_t executionContext),
                (const, override));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (int64_t engineId, hipdnnEnginePluginExecutionContext_t executionContext),
                (const, override));
};

}
}
