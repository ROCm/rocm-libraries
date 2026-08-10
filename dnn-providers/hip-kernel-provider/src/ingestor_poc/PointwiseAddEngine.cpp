// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ingestor_poc/PointwiseAddEngine.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <mutex>

#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>

#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "ingestor_poc/NativeMatchers.hpp"
#include "ingestor_poc/PointwiseAddDispatchHandler.hpp"
#include "ingestor_poc/PointwiseAddPack.hpp"

namespace hip_kernel_provider::ingestor_poc
{

namespace
{

int currentDeviceId()
{
    int deviceId = 0;
    if(hipGetDevice(&deviceId) != hipSuccess)
    {
        // Catalogs are cached per device, so failing to resolve one is not fatal:
        // device 0 keys the cache consistently and matching is unaffected.
        return 0;
    }
    return deviceId;
}

/**
 * @brief Registers this pack's native implementations, then builds its state manager.
 *
 * Order matters: assembling the state manager resolves the heuristic's score symbol, so
 * the implementations must already be registered.
 *
 * Registration deliberately does not run at static-init time. An engine that is never
 * constructed should not have mutated process-wide registries, and a duplicate-symbol
 * throw from a global constructor would escape during dlopen() and terminate the
 * process; from here it surfaces as a failed plugin creation the host can report.
 */
std::shared_ptr<hipdnn_plugin_sdk::ingestor::KernelIngestorStateManager<Handle>>
    registerThenBuildStateManager(const PointwiseAddDispatchHandler& dispatchHandler)
{
    static std::once_flag s_registered;
    std::call_once(s_registered, [&dispatchHandler]() {
        registerPointwiseAddMatchers();
        registerPointwiseAddDispatch(dispatchHandler);
    });

    return makePointwiseAddStateManager();
}

/**
 * @brief A generic engine plus the provider-owned machinery its kernels reach through
 *        the native registry.
 *
 * The registry holds the dispatch handler by pointer, so something must own that handler
 * for as long as any plan built from it can execute. That is this type: the compiler and
 * handler are declared before the engine, so they are constructed first and destroyed
 * last, and the engine's plans never outlive the engine.
 */
class PointwiseAddEngine : public hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>
{
public:
    explicit PointwiseAddEngine(const device::IDevicePropertyProvider& devicePropertyProvider)
        : _dispatchHandler(_kernelCompiler, devicePropertyProvider.getDeviceProperties())
        , _engine(registerThenBuildStateManager(_dispatchHandler),
                  devicePropertyProvider.getDeviceProperties(),
                  currentDeviceId())
    {
    }

    int64_t id() const override
    {
        return _engine.id();
    }

    bool isApplicable(
        Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override
    {
        return _engine.isApplicable(handle, opGraph);
    }

    void getDetails(Handle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override
    {
        _engine.getDetails(handle, opGraph, detailsOut);
    }

    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override
    {
        return _engine.getMaxWorkspaceSize(handle, opGraph, engineConfig);
    }

    void initializeExecutionContext(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Context& executionContext) const override
    {
        _engine.initializeExecutionContext(handle, opGraph, engineConfig, executionContext);
    }

private:
    HipMlopsKernelCompiler _kernelCompiler;
    PointwiseAddDispatchHandler _dispatchHandler;
    hipdnn_plugin_sdk::ingestor::GenericEngine<Handle, Settings, Context> _engine;
};

} // namespace

std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>>
    makePointwiseAddEngine(const device::IDevicePropertyProvider& devicePropertyProvider)
{
    return std::make_unique<PointwiseAddEngine>(devicePropertyProvider);
}

} // namespace hip_kernel_provider::ingestor_poc

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
