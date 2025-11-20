// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>

#include "PluginCore.hpp"

namespace hipdnn_backend
{
namespace plugin
{

class EnginePlugin : public PluginBase
{
protected:
    // The constructor is protected to prevent direct instantiation of the class.
    EnginePlugin(SharedLibrary&& lib);

    // We need this to allow mocking this class
    EnginePlugin();

public:
    // Functions that don't require a handle (called first)
    virtual std::vector<int64_t> getAllEngineIds() const;

    // Handle lifecycle functions
    virtual hipdnnEnginePluginHandle_t createHandle() const;
    virtual void setStream(hipdnnEnginePluginHandle_t handle, hipStream_t stream) const;

    // Engine discovery and configuration functions
    virtual std::vector<int64_t>
        getApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                               const hipdnnPluginConstData_t* opGraph) const;
    virtual void getEngineDetails(hipdnnEnginePluginHandle_t handle,
                                  int64_t engineId,
                                  const hipdnnPluginConstData_t* opGraph,
                                  hipdnnPluginConstData_t* engineDetails) const;
    virtual size_t getWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                    const hipdnnPluginConstData_t* engineConfig,
                                    const hipdnnPluginConstData_t* opGraph) const;
    virtual size_t getWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                    hipdnnEnginePluginExecutionContext_t executionContext) const;

    // Execution functions
    [[nodiscard]] virtual hipdnnEnginePluginExecutionContext_t
        createExecutionContext(hipdnnEnginePluginHandle_t handle,
                               const hipdnnPluginConstData_t* engineConfig,
                               const hipdnnPluginConstData_t* opGraph) const;
    virtual void executeOpGraph(hipdnnEnginePluginHandle_t handle,
                                hipdnnEnginePluginExecutionContext_t executionContext,
                                void* workspace,
                                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                uint32_t numDeviceBuffers) const;

    // Cleanup functions (called in reverse order)
    virtual void
        destroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                hipdnnEnginePluginExecutionContext_t executionContext) const;
    virtual void destroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                      hipdnnPluginConstData_t* engineDetails) const;
    virtual void destroyHandle(hipdnnEnginePluginHandle_t handle) const;

    static hipdnnPluginType_t getPluginType()
    {
        return HIPDNN_PLUGIN_TYPE_ENGINE;
    }

private:
    void resolveSymbols();

#ifndef NDEBUG
    bool _initialized = false;
#endif

    mutable std::vector<int64_t> _allEngineIds;

    hipdnnPluginStatus_t (*_funcGetAllEngineIds)(int64_t*, uint32_t, uint32_t*);
    hipdnnPluginStatus_t (*_funcCreateHandle)(hipdnnEnginePluginHandle_t*);
    hipdnnPluginStatus_t (*_funcDestroyHandle)(hipdnnEnginePluginHandle_t);
    hipdnnPluginStatus_t (*_funcSetStream)(hipdnnEnginePluginHandle_t, hipStream_t);
    hipdnnPluginStatus_t (*_funcGetApplicableEngineIds)(
        hipdnnEnginePluginHandle_t, const hipdnnPluginConstData_t*, int64_t*, uint32_t, uint32_t*);
    hipdnnPluginStatus_t (*_funcGetEngineDetails)(hipdnnEnginePluginHandle_t,
                                                  int64_t,
                                                  const hipdnnPluginConstData_t*,
                                                  hipdnnPluginConstData_t*);
    hipdnnPluginStatus_t (*_funcDestroyEngineDetails)(hipdnnEnginePluginHandle_t,
                                                      hipdnnPluginConstData_t*);
    hipdnnPluginStatus_t (*_funcGetWorkspaceSize)(hipdnnEnginePluginHandle_t,
                                                  const hipdnnPluginConstData_t*,
                                                  const hipdnnPluginConstData_t*,
                                                  size_t*);
    hipdnnPluginStatus_t (*_funcCreateExecutionContext)(hipdnnEnginePluginHandle_t,
                                                        const hipdnnPluginConstData_t*,
                                                        const hipdnnPluginConstData_t*,
                                                        hipdnnEnginePluginExecutionContext_t*);
    hipdnnPluginStatus_t (*_funcDestroyExecutionContext)(hipdnnEnginePluginHandle_t,
                                                         hipdnnEnginePluginExecutionContext_t);
    hipdnnPluginStatus_t (*_funcGetWorkspaceSizeFromExecutionContext)(
        hipdnnEnginePluginHandle_t, hipdnnEnginePluginExecutionContext_t, size_t*);
    hipdnnPluginStatus_t (*_funcExecuteOpGraph)(hipdnnEnginePluginHandle_t,
                                                hipdnnEnginePluginExecutionContext_t,
                                                void*,
                                                const hipdnnPluginDeviceBuffer_t*,
                                                uint32_t);

    friend class PluginManagerBase<EnginePlugin>;
};

} // namespace plugin
} // hipdnn_backend
