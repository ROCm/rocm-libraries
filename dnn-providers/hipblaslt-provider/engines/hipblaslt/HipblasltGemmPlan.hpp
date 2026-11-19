// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <vector>

#include "HipblasltMatmulDesc.hpp"
#include "HipblasltMatrixLayout.hpp"
#include "common/GemmCall.hpp"
#include "engines/plans/PlanInterface.hpp"

namespace hipblaslt_plugin
{

/**
 * @brief An ordered list of GEMM calls run through hipBLASLt, each with the algorithm its
 *        heuristic picked when the plan was built.
 *
 * Construction queries hipBLASLt for every call and throws when it finds no algorithm, so
 * building the plan is also its applicability check.
 *
 * A deterministic plan offers hipBLASLt no workspace (HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
 * 0, as MIOpen's deterministic GEMMs do), so no algorithm can split a reduction across
 * workgroups into a workspace; the convolution's own repack workspace is separate.
 */
class HipblasltGemmPlan : public IPlan
{
public:
    HipblasltGemmPlan(const HipdnnEnginePluginHandle& handle,
                      const std::vector<GemmCall>& calls,
                      bool deterministic = false);

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    struct PreparedCall
    {
        GemmCall call;
        HipblasltMatmulDesc desc;
        HipblasltMatrixLayout a;
        HipblasltMatrixLayout b;
        HipblasltMatrixLayout d;
        hipblasLtMatmulHeuristicResult_t heuristic{};
        /// The addend's layout (GemmCall::c); without one, c is d.
        std::optional<HipblasltMatrixLayout> c;
    };

    std::vector<PreparedCall> _calls;
    size_t _workspaceSize = 0;
};

} // namespace hipblaslt_plugin
