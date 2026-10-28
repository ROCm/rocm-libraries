// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <utility>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/RestoredIngestorPlan.hpp"
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

serialization::IngestorPlanPayload captureIngestorPlan(const hipdnn_plugin_sdk::IPlan<Handle>& plan,
                                                       const Handle& handle)
{
    if(dynamic_cast<const RestoredIngestorPlan<Handle>*>(&plan) != nullptr)
    {
        serialization::refuseIngestorPlanSave(
            IngestorPlanRefusal::INCOMPATIBLE,
            "this plan was loaded from a saved payload; re-saving is not supported; keep the "
            "original bytes");
    }

    const auto* saveable = plan.saveablePlan();
    if(saveable == nullptr)
    {
        if(dynamic_cast<const hipdnn_plugin_sdk::ingestor::BenchmarkPlan<Handle>*>(&plan)
           != nullptr)
        {
            serialization::refuseIngestorPlanSave(
                IngestorPlanRefusal::INCOMPATIBLE,
                "the plan has not chosen a kernel yet; execute it once, or build it with "
                "benchmarking off, then save");
        }
        refuseNonIngestorPlanSave();
    }

    const auto& kernel = saveable->kernel();
    const std::string kernelName
        = hipdnn_plugin_sdk::ingestor::describeDescriptor("kernel", kernel.name, kernel.kernelId);
    if(!serialization::isSerializableSourceKind(kernel.source.kind))
    {
        serialization::refuseIngestorPlanSave(
            IngestorPlanRefusal::INCOMPATIBLE,
            kernelName + " has source kind '"
                + hipdnn_plugin_sdk::ingestor::toString(kernel.source.kind)
                + "', which saving does not yet support");
    }

    const auto& handler = saveable->handler();
    auto inputs = handler.saveLaunchInputs(saveable->prepared());
    if(!inputs.has_value())
    {
        serialization::refuseIngestorPlanSave(IngestorPlanRefusal::INCOMPATIBLE,
                                              "this dispatch handler does not support saving "
                                                  + kernelName);
    }

    if(hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::tryResolve(inputs->dispatchSymbol)
       != &handler)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "dispatch alias '" + inputs->dispatchSymbol
                + "' is not registered to this handler, which saved " + kernelName);
    }

    const auto* prepared = dynamic_cast<const IngestorPreparedDispatch*>(&saveable->prepared());
    if(prepared == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "prepared dispatch exposes no kernel code for " + kernelName);
    }
    auto code = prepared->kernelCode().readCodeObject();

    serialization::IngestorPlanPayload payload;
    payload.kernelDescriptorId = kernel.kernelId;
    payload.workspaceBytes = saveable->getWorkspaceSize(handle);
    payload.dispatchSymbol = std::move(inputs->dispatchSymbol);
    payload.launchValues = std::move(inputs->values);
    payload.sourceKind = kernel.source.kind;
    payload.symbol = std::move(code.symbol);
    payload.target = std::move(code.target);
    payload.sha256 = std::move(code.sha256);
    payload.recordedSignature = kernel.source.signature;
    payload.codeObject = std::move(code.bytes);
    payload.providerVersion = HIP_KERNEL_PROVIDER_VERSION_STRING;
    return payload;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
