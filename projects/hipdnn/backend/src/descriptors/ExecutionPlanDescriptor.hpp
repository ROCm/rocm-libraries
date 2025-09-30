// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

namespace hipdnn_backend
{

class EngineConfigDescriptor;

namespace plugin
{
class EngineExecutionContextWrapper;
}

class ExecutionPlanDescriptor : public HipdnnBackendDescriptorImpl<ExecutionPlanDescriptor>
{
private:
    std::shared_ptr<const EngineConfigDescriptor> _engineConfig;
    std::shared_ptr<const plugin::EngineExecutionContextWrapper> _executionContext;

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
    virtual hipdnnEnginePluginExecutionContext_t getExecutionContext() const;

    static hipdnnBackendDescriptorType_t getStaticType();
};

} // namespace hipdnn_backend
