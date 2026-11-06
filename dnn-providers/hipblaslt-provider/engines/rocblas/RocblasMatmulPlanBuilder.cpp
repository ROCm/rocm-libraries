// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/rocblas/RocblasMatmulPlanBuilder.hpp"

#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/MatmulProblem.hpp"
#include "engines/rocblas/RocblasGemmPlan.hpp"

namespace hipblaslt_plugin
{

bool RocblasMatmulPlanBuilder::isApplicable(
    const HipdnnEnginePluginHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return gemmForMatmul(opGraph).has_value();
}

size_t RocblasMatmulPlanBuilder::getWorkspaceSize(
    const HipdnnEnginePluginHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    return 0;
}

void RocblasMatmulPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    auto call = gemmForMatmul(opGraph);
    if(!call.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "rocBLAS matmul does not apply to this graph");
    }
    HIPDNN_PLUGIN_LOG_INFO(
        "Building rocBLAS matmul plan for node: " << opGraph.getNodeWrapper(0).name());
    executionContext.setPlan(std::make_unique<RocblasGemmPlan>(std::vector<GemmCall>{*call}));
}

} // namespace hipblaslt_plugin
