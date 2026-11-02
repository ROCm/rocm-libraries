// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <mutex>
#include <optional>
#include <vector>

struct hipdnnHandle;

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
    // Required device-pointer byte alignment per tensor, parallel to _tensorUids
    // (_tensorAlignments[i] applies to _tensorUids[i]).
    std::vector<int64_t> _tensorAlignments;
    int64_t _workspaceSize = INVALID_WORKSPACE_SIZE;
    int64_t _engineId = INVALID_ENGINE_ID;
    bool _isOverrideShapeEnabled = false;

    // Cached serialized execution-plan blob. Populated lazily on the first
    // serializeBackendPlan() call and reused by all subsequent size/fill calls.
    // A finalized plan is immutable, so this never needs invalidation.
    mutable std::vector<uint8_t> _serializedPlanCache;
    // Guards _serializedPlanCache. serializeBackendPlan() is const and can run on several threads.
    mutable std::mutex _serializedPlanCacheMutex;

    // Process-unique plan id. Logs use it to tell plans apart, because a plan address
    // can be reused after the plan is destroyed.
    const uint64_t _planId = nextPlanId();

    // Id of the graph that the plan was finalized from. A deserialized plan has no id.
    std::optional<std::array<uint8_t, 16>> _graphId;

    // Number of successful hipdnnBackendExecute calls on this plan. The backend
    // logs this count so users can see how often each compiled plan runs.
    std::atomic<uint64_t> _executionCount{0};

    // Device of the handle the plan was finalized or deserialized on. Empty when it could not
    // be resolved, which disables the execute-time device check.
    std::optional<int> _device;

    static uint64_t nextPlanId();

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
    virtual const std::vector<int64_t>& getTensorAlignments() const;
    virtual bool isOverrideShapeEnabled() const;
    virtual hipdnnEnginePluginExecutionContext_t getExecutionContext() const;
    virtual void serializeBackendPlan(size_t requestedByteSize,
                                      size_t* planByteSize,
                                      uint8_t* serializedPlan) const;
    virtual void deserializeBackendPlan(
        const std::shared_ptr<plugin::EnginePluginResourceManager>& pluginResourceManager,
        const uint8_t* serializedPlan,
        size_t planByteSize);

    // Records one successful submission of this plan and returns the new count (1-based).
    // Thread-safe: concurrent submissions of the same plan get distinct counts.
    uint64_t recordExecution();

    // Binds the plan to the device of handle's stream; a failed query leaves the plan unbound.
    void recordDevice(const hipdnnHandle& handle);

    // No-op for an unbound plan. Otherwise throws BAD_PARAM if handle's stream is on another
    // device, or INTERNAL_ERROR if that device cannot be resolved.
    void validateExecutionDevice(const hipdnnHandle& handle) const;

    static hipdnnBackendDescriptorType_t getStaticType();

    static constexpr int64_t INVALID_WORKSPACE_SIZE = -1;
    static constexpr int64_t INVALID_ENGINE_ID = -1;

    std::string toString() const override;
};

} // namespace hipdnn_backend
