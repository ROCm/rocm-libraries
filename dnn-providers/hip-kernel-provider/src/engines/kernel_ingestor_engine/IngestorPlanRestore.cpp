// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/CodeObjectTarget.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/RestoredIngestorPlan.hpp"
#include "engines/kernel_ingestor_engine/RestoredSlotProgramPlan.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanPayload.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include <algorithm>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{
using hipdnn_plugin_sdk::ingestor::DispatchRegistry;
using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::PreparedDispatch;
using hipdnn_plugin_sdk::ingestor::SavedKernelCode;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using serialization::IngestorPlanRefusal;

// A UID list as `{5, 9}`, in the order it is given.
std::string formatUids(const std::vector<int64_t>& uids)
{
    std::string text = "{";

    for(size_t index = 0; index < uids.size(); ++index)
    {
        text += (index == 0 ? "" : ", ") + std::to_string(uids[index]);
    }

    return text + "}";
}

// The UIDs @p prepared reads from device_buffers as host pointers. A prepared dispatch that no
// ingestor pack built binds none that this provider can see.
void addBoundUids(const PreparedDispatch& prepared, std::set<int64_t>& bound)
{
    const auto* ingestor = dynamic_cast<const IngestorPreparedDispatch*>(&prepared);

    if(ingestor == nullptr)
    {
        return;
    }

    const auto uids = ingestor->userSuppliedPassByValueUids();
    bound.insert(uids.begin(), uids.end());
}

// RFC 0009 load check 10. The UIDs the plan lists as user-supplied must be the ones the restored
// handlers bind, and no others: a handler that cannot bind a listed UID would run on a value it
// never reads, and one that binds an unlisted UID would read a pointer the caller never supplies.
void requireListedUidsAreBound(const std::vector<int64_t>& listed,
                               const std::set<int64_t>& bound,
                               const std::string& plan)
{
    const std::vector<int64_t> boundSorted(bound.begin(), bound.end());

    if(listed != boundSorted)
    {
        serialization::refuseIngestorPlan(
            IngestorPlanRefusal::INCOMPATIBLE,
            plan + " lists the runtime pass-by-value UIDs " + formatUids(listed)
                + ", but the dispatch handlers that restore it bind " + formatUids(boundSorted));
    }
}

// The handler registered under @p dispatchSymbol.
const IKernelDispatchHandler<Handle>& requireHandler(const std::string& dispatchSymbol,
                                                     const std::string& subject)
{
    // Registration is idempotent. It makes the dispatch names resolvable whichever entry
    // point ran first.
    registerNativeIngestorSymbols();
    const auto* handler = DispatchRegistry<Handle>::tryResolve(dispatchSymbol);

    if(handler == nullptr)
    {
        serialization::refuseIngestorPlan(
            IngestorPlanRefusal::INCOMPATIBLE,
            "no dispatch handler is registered under '"
                + serialization::ingestorPlanMessageText(dispatchSymbol) + "', which " + subject
                + " names");
    }

    return *handler;
}

std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> restoreSingleKernelPlan(
    serialization::IngestorPlanPayload payload, int deviceOrdinal, const std::string& deviceArch)
{
    const std::string kernel
        = "saved kernel " + hipdnn_plugin_sdk::ingestor::toString(payload.kernelDescriptorId);
    const std::string dispatch = serialization::ingestorPlanMessageText(payload.dispatchSymbol);

    if(!isCodeObjectTargetCompatible(payload.sourceKind, payload.target, deviceArch))
    {
        serialization::refuseIngestorPlan(
            IngestorPlanRefusal::INCOMPATIBLE,
            kernel + " holds a " + hipdnn_plugin_sdk::ingestor::toString(payload.sourceKind)
                + " plan saved for '" + payload.target + "', which cannot run on device "
                + std::to_string(deviceOrdinal) + " ('" + deviceArch + "')");
    }

    const auto& handler = requireHandler(payload.dispatchSymbol, kernel);
    SavedLaunchInputs inputs;
    inputs.dispatchSymbol = payload.dispatchSymbol;
    inputs.values = std::move(payload.launchValues);
    SavedKernelCode code;
    code.kernelId = payload.kernelDescriptorId;
    code.sourceKind = payload.sourceKind;
    code.symbol = std::move(payload.symbol);
    code.target = std::move(payload.target);
    code.sha256 = std::move(payload.sha256);
    code.recordedSignature = std::move(payload.recordedSignature);
    code.codeObject = std::move(payload.codeObject);

    // The recorded signature, the launch values and the module load.
    auto prepared = handler.restoreLaunch(inputs, std::move(code), deviceOrdinal);

    if(prepared == nullptr)
    {
        serialization::refuseIngestorPlan(IngestorPlanRefusal::INCOMPATIBLE,
                                          "the dispatch handler registered under '" + dispatch
                                              + "' does not support restoring " + kernel);
    }

    std::set<int64_t> bound;
    addBoundUids(*prepared, bound);
    requireListedUidsAreBound(payload.runtimePassByValueUids, bound, kernel);

    return std::make_unique<RestoredIngestorPlan<Handle>>(
        handler,
        std::move(prepared),
        static_cast<size_t>(payload.workspaceBytes),
        kernel + " (dispatch '" + dispatch + "')");
}

// The decoder refuses these. A payload built in memory reaches here without it, so each is
// checked again: a stage with no kernel, an index past the code objects, a shared region larger
// than the workspace that contains it.
void requireWellFormed(const serialization::IngestorMultiKernelPayload& payload)
{
    if(payload.stages.empty())
    {
        serialization::refuseIngestorPlan(IngestorPlanRefusal::DAMAGED, "the plan has no stages");
    }

    if(payload.sharedWorkspaceBytes > payload.workspaceBytes)
    {
        serialization::refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                                          "the shared workspace region ("
                                              + std::to_string(payload.sharedWorkspaceBytes)
                                              + " bytes) is larger than the plan's workspace ("
                                              + std::to_string(payload.workspaceBytes) + " bytes)");
    }

    for(const auto& stage : payload.stages)
    {
        if(stage.kernels.empty())
        {
            serialization::refuseIngestorPlan(IngestorPlanRefusal::DAMAGED,
                                              "stage '" + stage.slot + "' launches no kernel");
        }

        for(const auto& kernel : stage.kernels)
        {
            if(kernel.codeObjectIndex >= payload.codeObjects.size())
            {
                serialization::refuseIngestorPlan(
                    IngestorPlanRefusal::DAMAGED,
                    "launch '" + kernel.launchName + "' of stage '" + stage.slot
                        + "' names code object " + std::to_string(kernel.codeObjectIndex)
                        + ", but the plan holds " + std::to_string(payload.codeObjects.size())
                        + " code objects");
            }
        }
    }
}

std::string describeStage(const serialization::IngestorStagePayload& stage)
{
    return "stage " + (stage.slot.empty() ? std::string("(unslotted)") : "'" + stage.slot + "'")
           + " (saved kernel " + hipdnn_plugin_sdk::ingestor::toString(stage.kernelDescriptorId)
           + ")";
}

std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreMultiKernelPlan(serialization::IngestorMultiKernelPayload payload,
                           int deviceOrdinal,
                           const std::string& deviceArch)
{
    requireWellFormed(payload);

    // Every code object the device cannot run is refused before any handler loads one.
    for(size_t index = 0; index < payload.codeObjects.size(); ++index)
    {
        const auto& object = payload.codeObjects[index];

        if(!isCodeObjectTargetCompatible(object.sourceKind, object.target, deviceArch))
        {
            serialization::refuseIngestorPlan(
                IngestorPlanRefusal::INCOMPATIBLE,
                "code object " + std::to_string(index) + " holds a "
                    + hipdnn_plugin_sdk::ingestor::toString(object.sourceKind) + " plan saved for '"
                    + object.target + "', which cannot run on device "
                    + std::to_string(deviceOrdinal) + " ('" + deviceArch + "')");
        }
    }

    // How many kernels still need each code object. The last one takes the bytes, and every
    // earlier one gets a copy.
    std::vector<size_t> usesLeft(payload.codeObjects.size(), 0);

    for(const auto& stage : payload.stages)
    {
        for(const auto& kernel : stage.kernels)
        {
            ++usesLeft[kernel.codeObjectIndex];
        }
    }

    std::vector<RestoredSlotProgramPlan<Handle>::Stage> stages;
    stages.reserve(payload.stages.size());
    std::set<int64_t> bound;

    for(auto& stage : payload.stages)
    {
        const std::string subject = describeStage(stage);
        const auto& handler = requireHandler(stage.dispatchSymbol, subject);
        SavedLaunchInputs inputs;
        inputs.dispatchSymbol = stage.dispatchSymbol;
        inputs.values = std::move(stage.launchValues);
        std::vector<SavedKernelCode> codes;
        codes.reserve(stage.kernels.size());

        for(auto& kernel : stage.kernels)
        {
            auto& object = payload.codeObjects[kernel.codeObjectIndex];
            SavedKernelCode code;
            code.kernelId = stage.kernelDescriptorId;
            code.sourceKind = object.sourceKind;
            code.symbol = std::move(kernel.symbol);
            code.target = object.target;
            code.sha256 = object.sha256;
            code.recordedSignature = std::move(kernel.recordedSignature);
            code.launchName = std::move(kernel.launchName);

            if(--usesLeft[kernel.codeObjectIndex] == 0)
            {
                code.codeObject = std::move(object.codeObject);
            }
            else
            {
                code.codeObject = object.codeObject;
            }

            codes.push_back(std::move(code));
        }

        // A handler that does not opt in restores one kernel through restoreLaunch. One that
        // does restores every stage, one kernel included, through restoreLaunches.
        std::unique_ptr<PreparedDispatch> prepared;

        if(handler.supportsSavingAllLaunches())
        {
            prepared = handler.restoreLaunches(inputs, std::move(codes), deviceOrdinal);
        }
        else if(codes.size() == 1)
        {
            prepared = handler.restoreLaunch(inputs, std::move(codes.front()), deviceOrdinal);
        }
        else
        {
            serialization::refuseIngestorPlan(
                IngestorPlanRefusal::INCOMPATIBLE,
                subject + " launches " + std::to_string(codes.size())
                    + " kernels, but the dispatch handler registered under '"
                    + serialization::ingestorPlanMessageText(stage.dispatchSymbol)
                    + "' does not restore several launches");
        }

        if(prepared == nullptr)
        {
            serialization::refuseIngestorPlan(
                IngestorPlanRefusal::INCOMPATIBLE,
                "the dispatch handler registered under '"
                    + serialization::ingestorPlanMessageText(stage.dispatchSymbol)
                    + "' does not support restoring " + subject);
        }

        addBoundUids(*prepared, bound);
        stages.push_back({std::move(stage.slot), &handler, std::move(prepared)});
    }

    requireListedUidsAreBound(payload.runtimePassByValueUids, bound, "the saved plan");

    return std::make_unique<RestoredSlotProgramPlan<Handle>>(
        std::move(stages),
        static_cast<size_t>(payload.sharedWorkspaceBytes),
        static_cast<size_t>(payload.workspaceBytes));
}

} // namespace

std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorSavedPlan(serialization::IngestorSavedPlan&& contents,
                             const Handle& handle,
                             const IngestorPlanRestoreEnvironment& environment)
{
    if(environment.deviceResolver == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "an ingestor plan restore needs a device resolver");
    }

    const auto deviceOrdinal = environment.deviceResolver->deviceId(handle);

    if(deviceOrdinal < 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "cannot resolve the handle's device to restore a plan");
    }

    const std::string deviceArch
        = environment.deviceResolver->deviceProperties(deviceOrdinal).gcnArchName;

    if(auto* single = std::get_if<serialization::IngestorPlanPayload>(&contents))
    {
        return restoreSingleKernelPlan(std::move(*single), deviceOrdinal, deviceArch);
    }

    return restoreMultiKernelPlan(
        std::move(std::get<serialization::IngestorMultiKernelPayload>(contents)),
        deviceOrdinal,
        deviceArch);
}

std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorSavedPlan(const serialization::IngestorSavedPlan& contents,
                             const Handle& handle,
                             const IngestorPlanRestoreEnvironment& environment)
{
    return restoreIngestorSavedPlan(
        serialization::IngestorSavedPlan(contents), handle, environment);
}

std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorPlan(const uint8_t* data,
                        size_t size,
                        const Handle& handle,
                        const IngestorPlanRestoreEnvironment& environment)
{
    // Header, body digest, verifier and field structure.
    return restoreIngestorSavedPlan(
        serialization::decodeIngestorSavedPlan(data, size), handle, environment);
}

std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>>
    restoreIngestorPlan(const uint8_t* data, size_t size, const Handle& handle)
{
    IngestorPlanRestoreEnvironment environment;
    environment.deviceResolver = &deviceResolver();

    return restoreIngestorPlan(data, size, handle, environment);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
