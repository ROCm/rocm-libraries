// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <mutex>

#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>

#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/HandleDeviceResolver.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddDispatchHandler.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddMatchers.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddPack.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace
{

/**
 * @brief The dispatch handler this pack's UDD resolves to.
 *
 * Process-lifetime, because the registry that holds it is. The registry stores a
 * non-owning pointer, and a provider's container is created and destroyed along with
 * its handles, so a handler owned by an engine would be freed while the registration
 * still pointed at it -- and the next container's engine would resolve that dangling
 * pointer rather than its own handler.
 *
 * The compiler it holds is a static for the same reason, and matches the module cache
 * behind it, which is already process-wide.
 */
const PointwiseAddDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const PointwiseAddDispatchHandler s_dispatchHandler(s_kernelCompiler);
    return s_dispatchHandler;
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
    registerThenBuildStateManager()
{
    static std::once_flag s_registered;
    std::call_once(s_registered, []() {
        registerPointwiseAddMatchers();
        registerPointwiseAddDispatch(dispatchHandler());
    });

    return makePointwiseAddStateManager();
}

/**
 * @brief A generic engine over this pack's descriptor set.
 *
 * Owns only what is scoped to one engine: the device resolver, and the state manager's
 * catalog cache. The dispatch handler and its compiler are deliberately not members --
 * see dispatchHandler() for why their lifetime has to exceed this engine's.
 */
class PointwiseAddEngine : public hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>
{
public:
    PointwiseAddEngine()
        : _engine(buildPointwiseAddDescriptorSet().engine,
                  registerThenBuildStateManager(),
                  _deviceResolver)
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
    HandleDeviceResolver _deviceResolver;
    hipdnn_plugin_sdk::ingestor::GenericEngine<Handle, Settings, Context> _engine;
};

} // namespace

std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> makePointwiseAddEngine()
{
    return std::make_unique<PointwiseAddEngine>();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
