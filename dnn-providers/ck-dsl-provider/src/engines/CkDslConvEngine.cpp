// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslConvEngine.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>

#include <algorithm>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace ck_dsl_plugin {

CkDslConvEngine::CkDslConvEngine(int64_t id) : id_(id) {}

int64_t CkDslConvEngine::id() const {
    return id_;
}

bool CkDslConvEngine::isApplicable(
    CkDslHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const {
    for (const auto& b : plan_builders_)
        if (b->isApplicable(handle, opGraph)) return true;
    return false;
}

void CkDslConvEngine::getDetails(
    CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    hipdnnPluginConstData_t& detailsOut) const {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobs_vec;
    for (const auto& pb : plan_builders_) {
        auto custom = pb->getCustomKnobs(handle, opGraph);
        for (const auto& k : custom)
            knobs_vec.push_back(hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &k));
        if (!custom.empty()) break;
    }
    auto knobs = builder.CreateVector(knobs_vec);
    auto details = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, id_, knobs);
    builder.Finish(details);
    auto detached = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detached->data();
    detailsOut.size = detached->size();
    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detached));
}

size_t CkDslConvEngine::getMaxWorkspaceSize(
    const CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const {
    size_t max_ws = 0;
    for (const auto& pb : plan_builders_) {
        if (pb->isApplicable(handle, opGraph)) {
            CkDslSettings s;
            pb->initializeExecutionSettings(handle, opGraph, engineConfig, s);
            max_ws = std::max(max_ws, pb->getMaxWorkspaceSize(handle, opGraph, s));
        }
    }
    return max_ws;
}

void CkDslConvEngine::initializeExecutionContext(
    const CkDslHandle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    CkDslContext& ctx) const {
    CkDslSettings settings;
    for (const auto& pb : plan_builders_)
        if (pb->isApplicable(handle, opGraph)) {
            pb->initializeExecutionSettings(handle, opGraph, engineConfig, settings);
            break;
        }
    ctx.setExecutionSettings(settings);
    for (const auto& pb : plan_builders_)
        if (pb->isApplicable(handle, opGraph)) {
            pb->buildPlan(handle, opGraph, engineConfig, ctx);
            break;
        }
}

void CkDslConvEngine::addPlanBuilder(
    std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<CkDslHandle, CkDslSettings, CkDslContext>>
        builder) {
    plan_builders_.push_back(std::move(builder));
}

}  // namespace ck_dsl_plugin
