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
    /// When a forward or backward data GEMM straight into a 16-bit y or dx with a long reduction
    /// runs as fp32 partials summed once (convGemmCalls' narrowPartials) instead of one GEMM:
    /// where the backend may split the reduction itself and round the parts to the output type.
    enum class NarrowPartials
    {
        NEVER,
        /// hipBLASLt on gfx11: one output pixel over 524288 bf16 terms reads 1.60 against
        /// 1.26 on gfx1100; gfx12 and CDNA are exact.
        ON_GFX11,
        ALWAYS,
    };

    /// Builds the backend's plan for @p calls, or throws when the backend cannot run them.
    using PlanFactory = std::function<std::unique_ptr<IPlan>(const HipdnnEnginePluginHandle&,
                                                             std::vector<GemmCall>)>;

    /// @p maxRepackWorkspaceBytes bounds the repack workspace of a plan (convGemmCalls); tests
    /// lower it to exercise image slicing on small shapes. @p tf32 lets every fp32 GEMM run as
    /// TF32 where the backend can (GemmCall::tf32), and declines convolutions with no fp32 GEMM:
    /// the builder of a TF32 engine (HIPBLASLT_CONV_TF32, ROCBLAS_CONV_TF32, which report
    /// TENSOR_CORE). @p reducedPrecisionReduction builds only reducedPrecisionGemmCalls plans:
    /// the builder of a reduced-precision-reduction engine (HIPBLASLT_CONV_RPR, ROCBLAS_CONV_RPR,
    /// which report REDUCED_PRECISION_REDUCTION). @p splitReduction makes
    /// the split reduction (convGemmCalls) the default plan rather than only a benchmarking
    /// candidate, and @p layout does the same for a layout candidate; tests use them to check
    /// those plans. @p narrowPartials: see NarrowPartials. @p maxBackwardWeightsRepackBytes, when
    /// set, bounds a further backward weights benchmarking candidate's repack workspace.
    /// @p nativeEpilogue: the backend applies a fused graph's epilogue in its GEMM where the
    /// pattern and the GEMM allow (hipBLASLt; fuseEpilogue), instead of in a repack pass.
    explicit ConvGemmPlanBuilder(PlanFactory factory,
                                 size_t maxRepackWorkspaceBytes = MAX_REPACK_WORKSPACE_BYTES,
                                 bool tf32 = false,
                                 bool reducedPrecisionReduction = false,
                                 bool splitReduction = false,
                                 ColumnLayout layout = ColumnLayout::FOLLOW_Y,
                                 NarrowPartials narrowPartials = NarrowPartials::NEVER,
                                 std::optional<size_t> maxBackwardWeightsRepackBytes = std::nullopt,
                                 bool nativeEpilogue = false);

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
    /// The plan calls for @p opGraph in whichever direction it is, or nullopt. @p
    /// wideBackwardWeights: a backward weights plan under maxBackwardWeightsRepackBytes, nullopt
    /// for another direction or without that cap.
    std::optional<ConvGemmPlanCalls>
        callsFor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                 ColumnLayout layout = ColumnLayout::FOLLOW_Y,
                 bool splitReduction = false,
                 bool wideBackwardWeights = false) const;

    /// The backend's plan for @p calls' GEMMs, wrapped in their repack steps when there are any.
    std::unique_ptr<IPlan> planFor(const HipdnnEnginePluginHandle& handle,
                                   ConvGemmPlanCalls calls) const;

    PlanFactory _factory;
    size_t _maxRepackWorkspaceBytes;
    bool _tf32;
    bool _reducedPrecisionReduction;
    bool _splitReduction;
    ColumnLayout _layout;
    NarrowPartials _narrowPartials;
    std::optional<size_t> _maxBackwardWeightsRepackBytes;
    bool _nativeEpilogue;
};

} // namespace hipblaslt_plugin
