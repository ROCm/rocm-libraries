// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CkDslConvImplicitGemmEngine.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <vector>

namespace ck_dsl_provider {

CkDslConvImplicitGemmEngine::CkDslConvImplicitGemmEngine(int64_t id) : _id(id) {}

int64_t CkDslConvImplicitGemmEngine::id() const {
    return _id;
}

bool CkDslConvImplicitGemmEngine::isApplicable(
    ::CkDslHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const {
    // I-1 skeleton: no graphs match. The adapter + plan builder that
    // would justify a true return land in milestones I-6/I-7.
    return false;
}

void CkDslConvImplicitGemmEngine::getDetails(
    ::CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const {
    // The SDK contract requires us to publish engine details as a
    // FlatBuffer whose lifetime is managed by the handle. For I-1 we
    // emit an EngineDetails table containing this engine's id and an
    // empty knob vector; once knobs are real (I-7+) we will populate
    // them from the plan builder.
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

size_t CkDslConvImplicitGemmEngine::getMaxWorkspaceSize(
    const ::CkDslHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const {
    return 0;
}

void CkDslConvImplicitGemmEngine::initializeExecutionContext(
    const ::CkDslHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    CkDslContext& /*executionContext*/) const {
    // Unreachable in the I-1 skeleton: isApplicable() always returns
    // false, so the SDK must not call this. If it does, fail loudly
    // rather than silently leaving the context uninitialised.
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "CkDslConvImplicitGemmEngine::initializeExecutionContext was called but the engine "
        "reported no applicable plans (I-1 skeleton). Real planning lands in milestone I-7.");
}

}  // namespace ck_dsl_provider
