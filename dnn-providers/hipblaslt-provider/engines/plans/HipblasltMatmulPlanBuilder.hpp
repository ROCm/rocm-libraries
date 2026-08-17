// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PlanBuilderInterface.hpp"
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipblaslt_plugin
{

class HipblasltMatmulPlanBuilder : public IPlanBuilder
{
public:
    HipblasltMatmulPlanBuilder() = default;
    ~HipblasltMatmulPlanBuilder() override = default;

    // Disallow copy and assignment
    HipblasltMatmulPlanBuilder(const HipblasltMatmulPlanBuilder&) = delete;
    HipblasltMatmulPlanBuilder& operator=(const HipblasltMatmulPlanBuilder&) = delete;

    bool isApplicable(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
    size_t getWorkspaceSize(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;
};

} // namespace hipblaslt_plugin
