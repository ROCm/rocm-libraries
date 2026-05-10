// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>

#include "PluginCore.hpp"

namespace hipdnn_backend::plugin
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
    virtual bool supportsExecutionContextSerialization() const;
    virtual void serializeExecutionContext(hipdnnEnginePluginHandle_t handle,
                                           hipdnnEnginePluginExecutionContext_t executionContext,
                                           hipdnnPluginConstData_t* serializedContext) const;
    virtual void
        destroySerializedExecutionContext(hipdnnEnginePluginHandle_t handle,
                                          hipdnnPluginConstData_t* serializedContext) const;
    [[nodiscard]] virtual hipdnnEnginePluginExecutionContext_t createExecutionContextFromSerialized(
        hipdnnEnginePluginHandle_t handle, const hipdnnPluginConstData_t* serializedContext) const;
    virtual void executeOpGraph(hipdnnEnginePluginHandle_t handle,
                                hipdnnEnginePluginExecutionContext_t executionContext,
                                void* workspace,
                                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                uint32_t numDeviceBuffers) const;

    /**
     * @brief Reports whether this plugin exports the optional override-execute
     *        symbol (`hipdnnEnginePluginExecuteOpGraphWithOverrides`).
     *
     * Plugins that report an older API version are filtered out for graphs
     * that opt in to overridable tensor shapes. Per RFC 0008 §4.6, callers
     * must still check this predicate before override dispatch so a plugin
     * that reports a recent-enough API version but omits the optional symbol
     * yields a clean `HIPDNN_STATUS_NOT_SUPPORTED` rather than a
     * missing-symbol crash.
     */
    virtual bool hasOverrideExecute() const;

    /**
     * @brief Override-aware execute dispatch (RFC 0008 §4.5).
     *
     * Wraps the optional `hipdnnEnginePluginExecuteOpGraphWithOverrides`
     * symbol. Callers must guard with `hasOverrideExecute()`; invoking this
     * method on a plugin that did not export the symbol throws
     * `HIPDNN_STATUS_NOT_SUPPORTED`.
     *
     * Pointer-lifetime contract: all override pointers (and the inner
     * per-tensor shape/stride buffers reachable through them) must remain
     * valid for the duration of the call. The plugin is forbidden from
     * retaining any of them. See RFC 0008 §4.6.
     */
    virtual void executeOpGraphWithOverrides(hipdnnEnginePluginHandle_t handle,
                                             hipdnnEnginePluginExecutionContext_t executionContext,
                                             void* workspace,
                                             const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                             uint32_t numDeviceBuffers,
                                             uint32_t numOverrides,
                                             const int64_t* overrideUniqueIds,
                                             const uint32_t* overrideLengths,
                                             const int64_t* const* overrideShapes,
                                             const int64_t* const* overrideStrides) const;

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
    hipdnnPluginStatus_t (*_funcSerializeExecutionContext)(hipdnnEnginePluginHandle_t,
                                                           hipdnnEnginePluginExecutionContext_t,
                                                           hipdnnPluginConstData_t*);
    hipdnnPluginStatus_t (*_funcDestroySerializedExecutionContext)(hipdnnEnginePluginHandle_t,
                                                                   hipdnnPluginConstData_t*);
    hipdnnPluginStatus_t (*_funcCreateExecutionContextFromSerialized)(
        hipdnnEnginePluginHandle_t,
        const hipdnnPluginConstData_t*,
        hipdnnEnginePluginExecutionContext_t*);
    hipdnnPluginStatus_t (*_funcGetWorkspaceSizeFromExecutionContext)(
        hipdnnEnginePluginHandle_t, hipdnnEnginePluginExecutionContext_t, size_t*);
    hipdnnPluginStatus_t (*_funcExecuteOpGraph)(hipdnnEnginePluginHandle_t,
                                                hipdnnEnginePluginExecutionContext_t,
                                                void*,
                                                const hipdnnPluginDeviceBuffer_t*,
                                                uint32_t);

    // Optional override-execute symbol (RFC 0008 §4.5). Resolved via
    // tryAssignSymbol; nullptr when the plugin does not export it.
    hipdnnPluginStatus_t (*_funcExecuteOpGraphWithOverrides)(hipdnnEnginePluginHandle_t,
                                                             hipdnnEnginePluginExecutionContext_t,
                                                             void*,
                                                             const hipdnnPluginDeviceBuffer_t*,
                                                             uint32_t,
                                                             uint32_t,
                                                             const int64_t*,
                                                             const uint32_t*,
                                                             const int64_t* const*,
                                                             const int64_t* const*)
        = nullptr; ///< Default nullptr is load-bearing: tryAssignSymbol leaves this untouched when the symbol is absent.

    friend class PluginManagerBase<EnginePlugin>;
};

} // namespace hipdnn_backend::plugin
