// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipMlopsEngine.hpp"
#include "core/Utils.hpp"
#include "device/ScopedDevice.hpp"
#include "hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <optional>

namespace hip_kernel_provider
{

HipMlopsEngine::HipMlopsEngine(int64_t id)
    : _id(id)
{
}

int64_t HipMlopsEngine::id() const
{
    return _id;
}

static void initializeHipKernelSettings(
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] Settings& executionSettings)
{
}

bool HipMlopsEngine::isApplicable(
    Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[HipMlopsEngine::isApplicable] ";

    const auto& tensorMap = opGraph.getTensorMap();
    HIP_KERNEL_RETURN_FALSE_IF(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(tensorMap),
                               "Ragged tensors are not supported");
    HIP_KERNEL_RETURN_FALSE_IF(
        !hip_kernel_provider::core::utils::hasOnly16ByteAlignedTensors(tensorMap),
        "All tensors must be aligned to a multiple of 16 bytes");

    // This is wrong if we ever have more than 1 plan builder thats applicable.
    // If this is the case, we should split plan builders accross multiple engines.
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            return true;
        }
    }
    return false;
}

void HipMlopsEngine::getDetails(Handle& handle,
                                const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                                hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobsVector;

    // Collect custom knobs from plan builders
    for(const auto& planBuilder : _planBuilders)
    {
        auto customKnobs = planBuilder->getCustomKnobs(handle, opGraph);

        if(customKnobs.empty())
        {
            continue;
        }

        for(const auto& knobT : customKnobs)
        {
            auto knobOffset = hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &knobT);
            knobsVector.push_back(knobOffset);
        }

        // Only one plan builder should be applicable for a given graph and return custom knobs.
        // Stop after finding the first one to avoid duplicates.
        break;
    }

    auto knobs = builder.CreateVector(knobsVector);

    const std::vector<int32_t> behaviorNotesVector{
        static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)};
    auto behaviorNotes = builder.CreateVector(behaviorNotesVector);

    auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
        builder, _id, knobs, behaviorNotes);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t HipMlopsEngine::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    Settings baseExecutionSettings;
    initializeHipKernelSettings(engineConfig, baseExecutionSettings);
    size_t workspaceSize = 0;
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            Settings executionSettings = baseExecutionSettings;
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            workspaceSize
                = std::max(workspaceSize,
                           planBuilder->getMaxWorkspaceSize(handle, opGraph, executionSettings));
        }
    }
    return workspaceSize;
}

void HipMlopsEngine::initializeExecutionContext(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    Context& executionContext) const
{
    Settings executionSettings;
    initializeHipKernelSettings(engineConfig, executionSettings);

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            break;
        }
    }

    executionContext.setExecutionSettings(executionSettings);

    // The plan builders read device properties from the current device and compile and
    // load their programs against it. Make that the device of the handle's stream for the
    // build, so the plan is built for the device the handle names rather than whichever
    // device the caller left current. A default stream token already names the current
    // device. Best effort: if the stream's device cannot be resolved or made current, the
    // build proceeds on the current device as before, and the plan's kernels still resolve
    // their device from the stream at every launch.
    std::optional<device::ScopedDevice> binding;
    auto* const stream = handle.getStream();
    int streamDevice = -1;
    if(!hipdnn_plugin_sdk::isDefaultStream(stream)
       && hipdnn_plugin_sdk::getDeviceFromStream(stream, &streamDevice) == hipSuccess
       && streamDevice >= 0)
    {
        binding.emplace(streamDevice);
        if(!binding->bound())
        {
            HIPDNN_PLUGIN_LOG_WARN("HipMlopsEngine: cannot make device "
                                   << streamDevice
                                   << " current to build a plan; building on the current device");
        }
    }

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->buildPlan(handle, opGraph, engineConfig, executionContext);
            break;
        }
    }
}

void HipMlopsEngine::addPlanBuilder(
    std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>> planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

}
