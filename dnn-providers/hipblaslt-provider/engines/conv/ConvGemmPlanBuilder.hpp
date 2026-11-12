// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "common/ConvRepackFormulation.hpp"
#include "engines/plans/PlanBuilderInterface.hpp"
#include "engines/plans/PlanInterface.hpp"

namespace hipblaslt_plugin
{

/**
 * @brief Explicit-GEMM convolution, forward, backward data and backward weights, over one BLAS
 *        backend.
 *
 * Backend-neutral: the convolution resolves to GEMM calls, directly on its tensors or around
 * repack kernels (common/ConvRepackFormulation.hpp), and the backend only runs the GEMMs,
 * through the plan factory it is constructed with. Built in
 * the shape of a kernel-ingestor native pack -- graph match (parseConvProblem), applicability
 * (the catalog row and the backend's own plan), a catalog of variants (the rows) and a dispatch
 * (the plan) -- so it can move onto universal descriptors without restructuring.
 */
class ConvGemmPlanBuilder : public IPlanBuilder
{
public:
    /// Builds the backend's plan for @p calls, or throws when the backend cannot run them.
    using PlanFactory = std::function<std::unique_ptr<IPlan>(const HipdnnEnginePluginHandle&,
                                                             std::vector<GemmCall>)>;

    /// @p maxRepackWorkspaceBytes bounds the repack workspace of a plan (convGemmCalls); tests
    /// lower it to exercise image slicing on small shapes. @p tf32 lets every fp32 GEMM run as
    /// TF32 where the backend can (GemmCall::tf32), and declines convolutions with no fp32 GEMM:
    /// the builder of a TF32 engine (ConvTf32.hpp). @p reducedPrecisionReduction builds only
    /// reducedPrecisionGemmCalls plans, and only while the switch is set: the builder of a
    /// reduced-precision-reduction engine (ConvReducedPrecision.hpp).
    explicit ConvGemmPlanBuilder(PlanFactory factory,
                                 size_t maxRepackWorkspaceBytes = MAX_REPACK_WORKSPACE_BYTES,
                                 bool tf32 = false,
                                 bool reducedPrecisionReduction = false);

    bool isApplicable(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getWorkspaceSize(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;

private:
    /// The plan calls for @p opGraph in whichever direction it is, or nullopt.
    std::optional<ConvGemmPlanCalls>
        callsFor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 ColumnLayout layout = ColumnLayout::FOLLOW_Y) const;

    /// The backend's plan for @p calls' GEMMs, wrapped in their repack steps when there are any.
    std::unique_ptr<IPlan> planFor(const HipdnnEnginePluginHandle& handle,
                                   ConvGemmPlanCalls calls) const;

    PlanFactory _factory;
    size_t _maxRepackWorkspaceBytes;
    bool _tf32;
    bool _reducedPrecisionReduction;
};

} // namespace hipblaslt_plugin
