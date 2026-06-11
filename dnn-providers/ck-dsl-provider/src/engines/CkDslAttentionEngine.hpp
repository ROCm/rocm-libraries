// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>
#include <memory>
#include <vector>

#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"

namespace ck_dsl_plugin {

// SDPA/FMHA engine: claims SdpaAttributes graphs and routes them to ck_dsl
// attention kernels. Same shape as CkDslGemmEngine; the difference is which
// plan builders are attached (forward today; backward is a follow-on).
class CkDslAttentionEngine
    : public hipdnn_plugin_sdk::IEngine<CkDslHandle, CkDslSettings, CkDslContext> {
   public:
    explicit CkDslAttentionEngine(int64_t id);

    int64_t id() const override;
    bool isApplicable(
        CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
    void getDetails(CkDslHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;
    size_t getMaxWorkspaceSize(const CkDslHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;
    void initializeExecutionContext(
        const CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        CkDslContext& ctx) const override;

    void addPlanBuilder(
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<CkDslHandle, CkDslSettings, CkDslContext>>
            builder);

   private:
    int64_t id_;
    std::vector<
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<CkDslHandle, CkDslSettings, CkDslContext>>>
        plan_builders_;
};

}  // namespace ck_dsl_plugin
