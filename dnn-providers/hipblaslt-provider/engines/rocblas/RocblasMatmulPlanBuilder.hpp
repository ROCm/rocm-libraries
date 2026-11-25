// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "common/GemmCall.hpp"
#include "engines/plans/PlanBuilderInterface.hpp"

namespace hipblaslt_plugin
{

/**
 * @brief A single-node matmul through rocBLAS.
 *
 * Applicability is gemmForMatmul (common/MatmulProblem.hpp): a matmul the strided-batched
 * GEMM computes, with no fused bias or activation, which rocBLAS has no epilogue for.
 */
class RocblasMatmulPlanBuilder : public IPlanBuilder
{
public:
    bool isApplicable(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    /// rocBLAS allocates its own scratch from the handle.
    size_t getWorkspaceSize(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;
};

} // namespace hipblaslt_plugin
