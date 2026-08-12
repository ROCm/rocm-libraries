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
 * Process-lifetime, because DispatchRegistry stores a non-owning pointer to it while a
 * provider's Container -- and any engine or handler it owns -- is created and destroyed
 * per handle (see SharedContainerManager). A function-local static is the only lifetime
 * that satisfies that non-owning-pointer contract; see NativeRegistry's doc for why the
 * registry itself does not take ownership.
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
 * engine-specific state (see HandleDeviceResolver's doc), so a per-engine instance
 * would only duplicate the cache with no isolation benefit.
 */
const HandleDeviceResolver& deviceResolver()
{
    static const HandleDeviceResolver s_deviceResolver;
    return s_deviceResolver;
}

} // namespace

/// The body registerNativeIngestorSymbols() runs at most once per process; broken out
/// so a test can drive it directly and observe rollback on a forced conflict.
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
    // call_once leaves the flag unset on a throw, so a retry re-runs the body, which
    // must roll back what it installed or the retry fails on its own partial state.
    static std::once_flag s_registered;
    std::call_once(s_registered, registerNativeIngestorSymbolsOnce);
}

std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> makePointwiseAddEngine()
{
    auto set = buildPointwiseAddDescriptorSet();
    // Moved out of `set` in its own statement, fully sequenced before the move of the
    // (now engine-less) remainder below. makePointwiseAddStateManager() never reads
    // set.engine, so its moved-from state here is inert.
    auto engine = std::move(set.engine);
    return std::make_unique<hipdnn_plugin_sdk::ingestor::GenericEngine<Handle, Settings, Context>>(
        std::move(engine), makePointwiseAddStateManager(std::move(set)), deviceResolver());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
