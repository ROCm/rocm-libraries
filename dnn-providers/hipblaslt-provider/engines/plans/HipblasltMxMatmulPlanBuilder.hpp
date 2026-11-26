// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipblasltMxScaleLayout.hpp"
#include "PlanBuilderInterface.hpp"

namespace hipblaslt_plugin
{

// The checks of isApplicable() that need only the graph: node pattern, data types,
// shapes, operand, scale and output layouts, alignment and block size. The #10811
// decline (Workarounds.hpp) also needs only the graph but runs before this, in
// isApplicable(). Throws HipdnnPluginException(BAD_PARAM) with the reason for the
// first rule the graph breaks.
void checkMxGraphConstraints(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

// The layout rule of a scale format, also needing no device, for a graph that passed
// checkMxGraphConstraints(). The swizzled formats run on hipBLASLt's Tensile kernels,
// which exist only for K-contiguous A and B (hipBLAS TN, #13715); other layouts are
// declined with HipdnnPluginException(BAD_PARAM).
void checkMxLayoutForScaleFormat(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                                 MxScaleFormat scaleFormat);

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
