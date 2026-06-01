// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CkDslConvImplicitGemmEngine.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <vector>

namespace ck_dsl_provider {

CkDslConvImplicitGemmEngine::CkDslConvImplicitGemmEngine(std::int64_t id,
                                                         CompileServiceBridge& bridge,
                                                         JitCache& cache)
    : _id(id), _planBuilder(std::make_unique<ConvImplicitGemmPlanBuilder>(bridge, cache)) {}

std::int64_t CkDslConvImplicitGemmEngine::id() const {
    return _id;
}

bool CkDslConvImplicitGemmEngine::isApplicable(
    ::CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    return _planBuilder->isApplicable(handle, opGraph);
}

void CkDslConvImplicitGemmEngine::getDetails(
    ::CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const {
    // EngineDetails FlatBuffer with this engine's id + an empty knob
    // vector (no custom knobs in M1; ConvImplicitGemmPlanBuilder::
    // getCustomKnobs returns {}). The handle's detached-buffer map
    // owns the FlatBuffer until the SDK explicitly releases it.
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> emptyKnobs;
    auto knobs = builder.CreateVector(emptyKnobs);
    auto engineDetails =
        hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, _id, knobs);
    builder.Finish(engineDetails);

    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

std::size_t CkDslConvImplicitGemmEngine::getMaxWorkspaceSize(
    const ::CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const {
    if (!_planBuilder->isApplicable(handle, opGraph)) {
        return 0;
    }
    CkDslSettings settings;
    return _planBuilder->getMaxWorkspaceSize(handle, opGraph, settings);
}

void CkDslConvImplicitGemmEngine::initializeExecutionContext(
    const ::CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    CkDslContext& executionContext) const {
    if (!_planBuilder->isApplicable(handle, opGraph)) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "CkDslConvImplicitGemmEngine::initializeExecutionContext called on a graph "
            "the plan builder reports as not applicable; the SDK should have skipped this "
            "engine after isApplicable returned false.");
    }
    CkDslSettings settings;
    _planBuilder->initializeExecutionSettings(handle, opGraph, engineConfig, settings);
    executionContext.setExecutionSettings(settings);
    _planBuilder->buildPlan(handle, opGraph, engineConfig, executionContext);
}

}  // namespace ck_dsl_provider
