// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <utility>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/CodeObjectTarget.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/RestoredIngestorPlan.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace
{

using serialization::IngestorPlanRefusal;

} // namespace

std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorPlan(const uint8_t* data,
                        size_t size,
                        const Handle& handle,
                        const IngestorPlanRestoreEnvironment& environment)
{
    if(!environment.loadedIngestorEngineName || environment.deviceResolver == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "an ingestor plan restore needs an engine name lookup and a device resolver");
    }

    // Header, body digest, verifier and field structure.
    serialization::IngestorPlanPayload payload = serialization::decodeIngestorPlan(data, size);
    const std::string kernel
        = "saved kernel " + hipdnn_plugin_sdk::ingestor::toString(payload.kernelDescriptorId);
    const std::string dispatch = serialization::ingestorPlanMessageText(payload.dispatchSymbol);

    const int64_t engineId = hipdnn_data_sdk::utilities::engineNameToId(payload.engineName);
    const std::string engineIdText = hipdnn_data_sdk::utilities::formatEngineIdHex(engineId);
    const std::string savedEngineName = serialization::ingestorPlanMessageText(payload.engineName);
    const auto loadedName = environment.loadedIngestorEngineName(engineId);
    if(!loadedName.has_value())
    {
        serialization::refuseIngestorPlan(
            IngestorPlanRefusal::INCOMPATIBLE,
            "no ingestor engine named '" + savedEngineName + "' (id " + engineIdText
                + ") is loaded for " + kernel
                + "; install and load the descriptor set that defines the engine");
    }
    if(*loadedName != payload.engineName)
    {
        serialization::refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                                          "the loaded ingestor engine with id " + engineIdText
                                              + " is named '" + *loadedName + "', not '"
                                              + savedEngineName + "' as " + kernel + " names");
    }

    const auto deviceOrdinal = environment.deviceResolver->deviceId(handle);
    if(deviceOrdinal < 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "cannot resolve the handle's device to restore " + kernel);
    }

    const std::string deviceArch
        = environment.deviceResolver->deviceProperties(deviceOrdinal).gcnArchName;
    if(!isCodeObjectTargetCompatible(payload.sourceKind, payload.target, deviceArch))
    {
        serialization::refuseIngestorPlan(
            IngestorPlanRefusal::INCOMPATIBLE,
            kernel + " holds a " + hipdnn_plugin_sdk::ingestor::toString(payload.sourceKind)
                + " code object built for target '" + payload.target + "', which device "
                + std::to_string(deviceOrdinal) + " ('" + deviceArch + "') cannot run");
    }

    // Registration is idempotent. It makes the dispatch names resolvable whichever entry
    // point ran first.
    registerNativeIngestorSymbols();
    const auto* handler
        = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::tryResolve(payload.dispatchSymbol);
    if(handler == nullptr)
    {
        serialization::refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                                          "no dispatch handler is registered under '" + dispatch
                                              + "', which " + kernel + " names");
    }

    hipdnn_plugin_sdk::ingestor::SavedLaunchInputs inputs;
    inputs.dispatchSymbol = payload.dispatchSymbol;
    inputs.values = std::move(payload.launchValues);

    hipdnn_plugin_sdk::ingestor::SavedKernelCode code;
    code.kernelId = payload.kernelDescriptorId;
    code.sourceKind = payload.sourceKind;
    code.symbol = std::move(payload.symbol);
    code.target = std::move(payload.target);
    code.sha256 = std::move(payload.sha256);
    code.recordedSignature = std::move(payload.recordedSignature);
    code.codeObject = std::move(payload.codeObject);

    // The recorded signature, the launch values and the module load.
    auto prepared = handler->restoreLaunch(inputs, std::move(code), deviceOrdinal);
    if(prepared == nullptr)
    {
        serialization::refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                                          "the dispatch handler registered under '" + dispatch
                                              + "' does not support restoring " + kernel);
    }

    return std::make_unique<RestoredIngestorPlan<Handle>>(
        *handler,
        std::move(prepared),
        static_cast<size_t>(payload.workspaceBytes),
        kernel + " (dispatch '" + dispatch + "')");
}

std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorPlan(const uint8_t* data, size_t size, const Handle& handle)
{
    IngestorPlanRestoreEnvironment environment;
    environment.loadedIngestorEngineName
        = [&handle](int64_t engineId) { return loadedIngestorEngineName(handle, engineId); };
    environment.deviceResolver = &deviceResolver();
    return restoreIngestorPlan(data, size, handle, environment);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
