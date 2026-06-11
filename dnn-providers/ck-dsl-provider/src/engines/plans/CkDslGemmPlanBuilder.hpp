// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"

namespace ck_dsl_plugin {

class CkDslGemmPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<CkDslHandle, CkDslSettings, CkDslContext> {
   public:
    bool isApplicable(
        const CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
    size_t getMaxWorkspaceSize(const CkDslHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const CkDslSettings& settings) const override;
    void initializeExecutionSettings(
        const CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        CkDslSettings& settings) const override;
    void buildPlan(const CkDslHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   CkDslContext& ctx) const override;
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

}  // namespace ck_dsl_plugin
