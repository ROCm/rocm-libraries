// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SlotProgramBenchmark.hpp>

#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/RestoredIngestorPlan.hpp"
#include "engines/kernel_ingestor_engine/RestoredSlotProgramPlan.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"
#include "engines/kernel_ingestor_engine/serialization/SerializableSourceKind.hpp"
#include "version.h"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using serialization::IngestorPlanRefusal;

void refuseNonIngestorPlanSave()
{
    serialization::refuseIngestorPlanSave(IngestorPlanRefusal::INCOMPATIBLE,
                                          "this plan was not built by an ingestor engine; its "
                                          "engine does not support saving execution plans");
}

namespace
{

using hipdnn_plugin_sdk::ingestor::DescriptorId;
using hipdnn_plugin_sdk::ingestor::GenericPlan;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MetadataValues;

[[noreturn]] void refuseUnchosenPlan()
{
    serialization::refuseIngestorPlanSave(
        IngestorPlanRefusal::INCOMPATIBLE,
        "the plan has not chosen a kernel yet; execute it once, or build it with "
        "benchmarking off, then save");
}

// One kernel a stage launches, with its code object read.
struct CapturedLaunch
{
    std::string launchName;
    KernelSourceKind sourceKind{};
    std::vector<KernelArgument> signature;
    IngestorCodeObject code;
};

// Everything one built kernel contributes to a saved plan.
struct CapturedStage
{
    std::string slot;
    DescriptorId kernelId{};
    std::string dispatchSymbol;
    MetadataValues launchValues;
    std::vector<int64_t> passByValueUids;
    std::vector<CapturedLaunch> launches;
};

// Refuses a source kind that saving does not support.
void requireSerializableSourceKind(KernelSourceKind kind, const std::string& subject)
{
    if(!serialization::isSerializableSourceKind(kind))
    {
        serialization::refuseIngestorPlanSave(IngestorPlanRefusal::INCOMPATIBLE,
                                              subject + " has source kind '"
                                                  + hipdnn_plugin_sdk::ingestor::toString(kind)
                                                  + "', which saving does not yet support");
    }
}

CapturedStage captureStage(const GenericPlan<Handle>& plan, const std::string& slot)
{
    const auto& kernel = plan.kernel();
    const std::string subject
        = hipdnn_plugin_sdk::ingestor::describeDescriptor("kernel", kernel.name, kernel.kernelId)
          + (slot.empty() ? std::string() : " of slot '" + slot + "'");
    const auto& handler = plan.handler();
    const auto* prepared = dynamic_cast<const IngestorPreparedDispatch*>(&plan.prepared());

    // The launches the dispatch runs. A dispatch that reports none (the legacy single path) runs
    // the one `source` of its kernel, through kernelCode().
    auto launches
        = prepared != nullptr ? prepared->launchedKernels() : std::vector<IngestorLaunchedKernel>{};
    if(launches.empty())
    {
        // The dispatch reports no launches, so it runs the one `source`, which a kernel with
        // several launches leaves default. Read it through `sources` first, and refuse the
        // kernel rather than save the default.
        const auto* source = hipdnn_plugin_sdk::ingestor::singleLaunchSource(kernel);
        if(source == nullptr)
        {
            serialization::refuseIngestorPlanSave(
                IngestorPlanRefusal::INCOMPATIBLE,
                subject
                    + " has several launches ('sources'), and its dispatch handler does not "
                      "report them; saving supports only a handler that does");
        }
        requireSerializableSourceKind(source->kind, subject);
        if(prepared != nullptr)
        {
            launches.push_back({{}, &prepared->kernelCode(), source->kind, source->signature});
        }
    }
    else
    {
        for(const auto& launch : launches)
        {
            requireSerializableSourceKind(launch.sourceKind, subject);
        }
        if(launches.size() > 1)
        {
            if(!handler.supportsSavingAllLaunches())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "the dispatch handler of " + subject + " launches "
                        + std::to_string(launches.size())
                        + " kernels but does not report supportsSavingAllLaunches()");
            }
            for(const auto& launch : launches)
            {
                if(launch.launchName.empty())
                {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                        "the dispatch handler of " + subject
                            + " launches several kernels and names one of them with an empty "
                              "launch name");
                }
            }
        }
    }

    auto inputs = handler.saveLaunchInputs(plan.prepared());
    if(!inputs.has_value())
    {
        serialization::refuseIngestorPlanSave(IngestorPlanRefusal::INCOMPATIBLE,
                                              "this dispatch handler does not support saving "
                                                  + subject);
    }

    if(hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::tryResolve(inputs->dispatchSymbol)
       != &handler)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "dispatch alias '" + inputs->dispatchSymbol
                + "' is not registered to this handler, which saved " + subject);
    }

    if(prepared == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "prepared dispatch exposes no kernel code for " + subject);
    }

    CapturedStage stage;
    stage.slot = slot;
    stage.kernelId = kernel.kernelId;
    stage.dispatchSymbol = std::move(inputs->dispatchSymbol);
    stage.launchValues = std::move(inputs->values);
    stage.passByValueUids = prepared->userSuppliedPassByValueUids();
    stage.launches.reserve(launches.size());
    for(auto& launch : launches)
    {
        stage.launches.push_back({std::move(launch.launchName),
                                  launch.sourceKind,
                                  std::move(launch.signature),
                                  launch.code->readCodeObject()});
    }
    return stage;
}

// The kind 1 payload of one stage that launches one kernel.
serialization::IngestorPlanPayload singleKernelPayload(CapturedStage stage, uint64_t workspaceBytes)
{
    auto& launch = stage.launches.front();
    serialization::IngestorPlanPayload payload;
    payload.kernelDescriptorId = stage.kernelId;
    payload.workspaceBytes = workspaceBytes;
    payload.dispatchSymbol = std::move(stage.dispatchSymbol);
    payload.launchValues = std::move(stage.launchValues);
    payload.sourceKind = launch.sourceKind;
    payload.symbol = std::move(launch.code.symbol);
    payload.target = std::move(launch.code.target);
    payload.sha256 = std::move(launch.code.sha256);
    payload.recordedSignature = std::move(launch.signature);
    payload.codeObject = std::move(launch.code.bytes);
    payload.runtimePassByValueUids = std::move(stage.passByValueUids);
    payload.providerVersion = HIP_KERNEL_PROVIDER_VERSION_STRING;
    return payload;
}

// The kind 2 payload of @p stages, in order. A code object that several kernels use is stored
// once, keyed by its SHA-256.
serialization::IngestorMultiKernelPayload multiKernelPayload(std::vector<CapturedStage> stages,
                                                             uint64_t sharedWorkspaceBytes,
                                                             uint64_t workspaceBytes)
{
    serialization::IngestorMultiKernelPayload payload;
    payload.sharedWorkspaceBytes = sharedWorkspaceBytes;
    payload.workspaceBytes = workspaceBytes;
    payload.providerVersion = HIP_KERNEL_PROVIDER_VERSION_STRING;

    std::unordered_map<std::string, uint32_t> indexBySha256;
    std::set<int64_t> uids;
    payload.stages.reserve(stages.size());
    for(auto& stage : stages)
    {
        serialization::IngestorStagePayload saved;
        saved.slot = std::move(stage.slot);
        saved.kernelDescriptorId = stage.kernelId;
        saved.dispatchSymbol = std::move(stage.dispatchSymbol);
        saved.launchValues = std::move(stage.launchValues);
        uids.insert(stage.passByValueUids.begin(), stage.passByValueUids.end());

        saved.kernels.reserve(stage.launches.size());
        for(auto& launch : stage.launches)
        {
            const auto known = indexBySha256.find(launch.code.sha256);
            uint32_t index = 0;
            if(known != indexBySha256.end())
            {
                index = known->second;
            }
            else
            {
                index = static_cast<uint32_t>(payload.codeObjects.size());
                indexBySha256.emplace(launch.code.sha256, index);
                payload.codeObjects.push_back({launch.sourceKind,
                                               std::move(launch.code.target),
                                               std::move(launch.code.sha256),
                                               std::move(launch.code.bytes)});
            }
            saved.kernels.push_back({std::move(launch.launchName),
                                     std::move(launch.code.symbol),
                                     std::move(launch.signature),
                                     index});
        }
        payload.stages.push_back(std::move(saved));
    }
    payload.runtimePassByValueUids.assign(uids.begin(), uids.end());
    return payload;
}

serialization::IngestorSavedPlan
    captureSlotProgram(const hipdnn_plugin_sdk::ingestor::SlotProgramPlan<Handle>& program,
                       const Handle& handle)
{
    std::vector<CapturedStage> stages;
    stages.reserve(program.stages().size());
    for(const auto& stage : program.stages())
    {
        stages.push_back(captureStage(*stage.plan, stage.slot));
    }
    return multiKernelPayload(
        std::move(stages), program.alignedSharedBytes(), program.getWorkspaceSize(handle));
}

} // namespace

serialization::IngestorSavedPlan captureIngestorPlan(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                                                     const Handle& handle)
{
    if(dynamic_cast<const RestoredIngestorPlan<Handle>*>(&plan) != nullptr
       || dynamic_cast<const RestoredSlotProgramPlan<Handle>*>(&plan) != nullptr)
    {
        serialization::refuseIngestorPlanSave(
            IngestorPlanRefusal::INCOMPATIBLE,
            "this plan was loaded from a saved payload; hipDNN saves a loaded plan from its "
            "original bytes and does not call the provider");
    }

    if(const auto* benchmark
       = dynamic_cast<const hipdnn_plugin_sdk::ingestor::SlotProgramBenchmark<Handle>*>(&plan))
    {
        // The program it served is the plan; the others it timed are not.
        const auto* served = benchmark->servedProgram();
        if(served == nullptr)
        {
            refuseUnchosenPlan();
        }
        return captureSlotProgram(*served, handle);
    }
    if(const auto* program
       = dynamic_cast<const hipdnn_plugin_sdk::ingestor::SlotProgramPlan<Handle>*>(&plan))
    {
        return captureSlotProgram(*program, handle);
    }

    const auto* saveable = plan.saveablePlan();
    if(saveable == nullptr)
    {
        if(dynamic_cast<const hipdnn_plugin_sdk::ingestor::BenchmarkPlan<Handle>*>(&plan)
           != nullptr)
        {
            refuseUnchosenPlan();
        }
        refuseNonIngestorPlanSave();
    }

    auto stage = captureStage(*saveable, {});
    const uint64_t workspaceBytes = saveable->getWorkspaceSize(handle);
    // A plan served by one launch is a kind 1 payload, whatever its handler can save.
    if(stage.launches.size() == 1)
    {
        return singleKernelPayload(std::move(stage), workspaceBytes);
    }
    std::vector<CapturedStage> stages;
    stages.push_back(std::move(stage));
    return multiKernelPayload(std::move(stages), 0, workspaceBytes);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
