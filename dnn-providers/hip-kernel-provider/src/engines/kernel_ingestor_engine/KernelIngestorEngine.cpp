// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <mutex>
#include <utility>

#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>

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
 * non-owning pointer, and a provider's container is created and destroyed along with its
 * handles (see SharedContainerManager), so a handler owned by an engine or a container
 * would be freed while the registration still pointed at it -- and the next container's
 * engine would resolve that dangling pointer rather than its own handler. A function-
 * local static outlives every container the process ever builds, which is the only
 * lifetime that satisfies DispatchRegistry's non-owning-pointer contract without the
 * registry taking ownership itself; NativeRegistry's own doc explains why it does not.
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
 * @brief The device resolver every descriptor-backed engine in this provider shares.
 *
 * Container/process-lifetime, not per-engine: it is a device-property cache with no
 * engine-specific state (see HandleDeviceResolver's own doc for why a device's
 * properties are safe to share across every handle and every engine that asks about
 * that device), so giving each engine its own instance would only duplicate the cache
 * for no isolation benefit. A function-local static matches how Container and
 * EnginePluginImpl already share state that outlives any one engine or container
 * instance -- dispatchHandler() above is the same pattern for the same reason.
 */
const HandleDeviceResolver& deviceResolver()
{
    static const HandleDeviceResolver s_deviceResolver;
    return s_deviceResolver;
}

} // namespace

/// The body registerNativeIngestorSymbols() runs at most once per process. Broken out
/// so a test can drive it directly and observe rollback on a forced conflict, which the
/// once_flag wrapper below would otherwise make unreachable a second time.
void registerNativeIngestorSymbolsOnce()
{
    registerPointwiseAddMatchers();
    try
    {
        registerPointwiseAddDispatch(dispatchHandler());
    }
    catch(...)
    {
        unregisterPointwiseAddMatchers();
        throw;
    }
}

void registerNativeIngestorSymbols()
{
    // call_once leaves the flag unset on a throw, so a retry re-runs the body -- which
    // must roll back what it installed, or the retry fails on ITS OWN partial state
    // instead of the original conflict.
    static std::once_flag s_registered;
    std::call_once(s_registered, registerNativeIngestorSymbolsOnce);
}

std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> makePointwiseAddEngine()
{
    auto set = buildPointwiseAddDescriptorSet();
    // Moves the UED out of `set` in its own statement, fully sequenced before the move
    // of the (now engine-less) remainder below -- not two moves racing inside one call's
    // argument list. makePointwiseAddStateManager() never reads set.engine, so its
    // moved-from state here is inert.
    auto engine = std::move(set.engine);
    return std::make_unique<hipdnn_plugin_sdk::ingestor::GenericEngine<Handle, Settings, Context>>(
        std::move(engine), makePointwiseAddStateManager(std::move(set)), deviceResolver());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
