// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PlanBuilderInterface.hpp"

namespace hipblaslt_plugin
{

class HipblasltMxMatmulPlanBuilder : public IPlanBuilder
{
public:
    HipblasltMxMatmulPlanBuilder() = default;
    ~HipblasltMxMatmulPlanBuilder() override = default;

    HipblasltMxMatmulPlanBuilder(const HipblasltMxMatmulPlanBuilder&) = delete;
    HipblasltMxMatmulPlanBuilder& operator=(const HipblasltMxMatmulPlanBuilder&) = delete;

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
