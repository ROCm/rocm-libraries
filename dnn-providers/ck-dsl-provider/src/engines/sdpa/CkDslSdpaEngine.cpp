// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CkDslSdpaEngine.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <vector>

namespace ck_dsl_provider {

CkDslSdpaEngine::CkDslSdpaEngine(std::int64_t id, CompileServiceBridge& bridge, JitCache& cache)
    : _id(id),
      _fwdPlanBuilder(std::make_unique<SdpaFwdPlanBuilder>(bridge, cache)),
      _bwdPlanBuilder(std::make_unique<SdpaBwdPlanBuilder>(bridge, cache)) {}

std::int64_t CkDslSdpaEngine::id() const {
    return _id;
}

bool CkDslSdpaEngine::isApplicable(
    ::CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    // The two op kinds are disjoint, so at most one builder accepts the
    // graph; the engine is applicable if either does.
    return _fwdPlanBuilder->isApplicable(handle, opGraph) ||
           _bwdPlanBuilder->isApplicable(handle, opGraph);
}

void CkDslSdpaEngine::getDetails(
    ::CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const {
    // EngineDetails FlatBuffer with this engine's id + an empty knob
    // vector (no custom knobs in M1; SdpaFwdPlanBuilder::getCustomKnobs
    // returns {}). The handle's detached-buffer map owns the FlatBuffer
    // until the SDK explicitly releases it.
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

std::size_t CkDslSdpaEngine::getMaxWorkspaceSize(
    const ::CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const {
    CkDslSettings settings;
    if (_fwdPlanBuilder->isApplicable(handle, opGraph)) {
        return _fwdPlanBuilder->getMaxWorkspaceSize(handle, opGraph, settings);
    }
    if (_bwdPlanBuilder->isApplicable(handle, opGraph)) {
        return _bwdPlanBuilder->getMaxWorkspaceSize(handle, opGraph, settings);
    }
    return 0;
}

void CkDslSdpaEngine::initializeExecutionContext(
    const ::CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    CkDslContext& executionContext) const {
    // Route to whichever builder accepts the graph. The two op kinds are
    // disjoint, so the forward check is tried first and the backward
    // second; if neither applies the SDK called us on a graph it should
    // have skipped after isApplicable returned false.
    CkDslSettings settings;
    if (_fwdPlanBuilder->isApplicable(handle, opGraph)) {
        _fwdPlanBuilder->initializeExecutionSettings(handle, opGraph, engineConfig, settings);
        executionContext.setExecutionSettings(settings);
        _fwdPlanBuilder->buildPlan(handle, opGraph, engineConfig, executionContext);
        return;
    }
    if (_bwdPlanBuilder->isApplicable(handle, opGraph)) {
        _bwdPlanBuilder->initializeExecutionSettings(handle, opGraph, engineConfig, settings);
        executionContext.setExecutionSettings(settings);
        _bwdPlanBuilder->buildPlan(handle, opGraph, engineConfig, executionContext);
        return;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "CkDslSdpaEngine::initializeExecutionContext called on a graph "
        "neither the forward nor backward plan builder reports as applicable; the SDK "
        "should have skipped this engine after isApplicable returned false.");
}

}  // namespace ck_dsl_provider
