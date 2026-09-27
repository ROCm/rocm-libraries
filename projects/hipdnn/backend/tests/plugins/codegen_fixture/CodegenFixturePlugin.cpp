// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CodegenFixturePlugin.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace codegen_fixture
{

CodegenFixtureEngineManager& CodegenFixtureHandle::getEngineManager() const
{
    return container->getEngineManager();
}

size_t CodegenFixturePlan::getWorkspaceSize(const CodegenFixtureHandle& /*handle*/) const
{
    return 0;
}

void CodegenFixturePlan::execute(const CodegenFixtureHandle& /*handle*/,
                                 const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                                 uint32_t /*numDeviceBuffers*/,
                                 void* /*workspace*/) const
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "The codegen fixture plugin ships no kernels and cannot execute");
}

int64_t CodegenFixtureEngine::id() const
{
    return K_FIXTURE_ENGINE_ID;
}

bool CodegenFixtureEngine::isApplicable(
    CodegenFixtureHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    return false;
}

void CodegenFixtureEngine::getDetails(
    CodegenFixtureHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const
{
    // Engine ID only: no knobs, no behavior notes, and — matching the container's
    // missing getEngineName — no name.
    flatbuffers::FlatBufferBuilder builder;

    auto engineDetails
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, K_FIXTURE_ENGINE_ID);
    builder.Finish(engineDetails);

    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t CodegenFixtureEngine::getMaxWorkspaceSize(
    const CodegenFixtureHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    return 0;
}

void CodegenFixtureEngine::initializeExecutionContext(
    const CodegenFixtureHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    CodegenFixtureContext& executionContext) const
{
    executionContext.setPlan(std::make_unique<CodegenFixturePlan>());
}

CodegenFixtureContainer::CodegenFixtureContainer()
{
    _engineManager.addEngine(std::make_unique<CodegenFixtureEngine>());
}

uint32_t CodegenFixtureContainer::copyEngineIds(int64_t* engineIds,
                                                uint32_t maxEngines,
                                                uint32_t& numEngines)
{
    constexpr uint32_t TOTAL_ENGINES = 1;

    // A zero capacity is the backend asking how many IDs exist before it
    // allocates room for them; the entry point guarantees engineIds is non-null
    // for every other capacity.
    if(maxEngines == 0)
    {
        numEngines = TOTAL_ENGINES;
        return TOTAL_ENGINES;
    }

    engineIds[0] = K_FIXTURE_ENGINE_ID;
    numEngines = TOTAL_ENGINES;

    return TOTAL_ENGINES;
}

CodegenFixtureEngineManager& CodegenFixtureContainer::getEngineManager()
{
    return _engineManager;
}

} // namespace codegen_fixture
