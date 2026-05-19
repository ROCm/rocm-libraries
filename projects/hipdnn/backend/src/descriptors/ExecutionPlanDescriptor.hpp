// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <cstdint>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <vector>

namespace hipdnn_backend
{

class EngineConfigDescriptor;

namespace plugin
{
class EngineExecutionContextWrapper;
class EnginePluginResourceManager;
}

class ExecutionPlanDescriptor : public HipdnnBackendDescriptorImpl<ExecutionPlanDescriptor>
{
private:
    std::shared_ptr<const EngineConfigDescriptor> _engineConfig;
    std::shared_ptr<const plugin::EngineExecutionContextWrapper> _executionContext;
    std::shared_ptr<plugin::EnginePluginResourceManager> _pluginResourceManager;
    std::vector<int64_t> _tensorUids;
    int64_t _workspaceSize = INVALID_WORKSPACE_SIZE;
    int64_t _engineId = INVALID_ENGINE_ID;

    void getWorkspaceSize(hipdnnBackendAttributeType_t attributeType,
                          int64_t requestedElementCount,
                          int64_t* elementCount,
                          void* arrayOfElements) const;

    // Deprecated: No longer needed since we can get the handle from the engine config.
    // Kept for backward compatibility.
    static void setHandle(hipdnnBackendAttributeType_t attributeType,
                          int64_t elementCount,
                          const void* arrayOfElements);

    void setEngineConfig(hipdnnBackendAttributeType_t attributeType,
                         int64_t elementCount,
                         const void* arrayOfElements);

    void getEngineConfig(hipdnnBackendAttributeType_t attributeType,
                         int64_t requestedElementCount,
                         int64_t* elementCount,
                         void* arrayOfElements) const;

    void getTensorUids(hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedElementCount,
                       int64_t* elementCount,
                       void* arrayOfElements) const;

public:
    void finalize() override;

    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;

    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    // Throws an exception if the descriptor is not finalized.
    virtual std::shared_ptr<const EngineConfigDescriptor> getEngineConfig() const;
    virtual int64_t getEngineId() const;
    virtual const std::vector<int64_t>& getTensorUids() const;
    virtual hipdnnEnginePluginExecutionContext_t getExecutionContext() const;
    virtual void serializeBackendPlan(size_t requestedByteSize,
                                      size_t* planByteSize,
                                      uint8_t* serializedPlan) const;
    virtual void deserializeBackendPlan(
        const std::shared_ptr<plugin::EnginePluginResourceManager>& pluginResourceManager,
        const uint8_t* serializedPlan,
        size_t planByteSize);

    static hipdnnBackendDescriptorType_t getStaticType();

    static constexpr int64_t INVALID_WORKSPACE_SIZE = -1;
    static constexpr int64_t INVALID_ENGINE_ID = -1;

    std::string toString() const override;
};

} // namespace hipdnn_backend
