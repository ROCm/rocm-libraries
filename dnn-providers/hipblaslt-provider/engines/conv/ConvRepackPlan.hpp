// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <vector>

#include "common/ConvGemmFormulation.hpp"
#include "engines/plans/PlanInterface.hpp"

namespace hipblaslt_plugin
{

/**
 * @brief A convolution that repacks around its GEMMs: for each stage, its repack steps before,
 *        the backend's plan for its GEMM calls, then its repack steps after, all on the
 *        handle's stream.
 *
 * The workspace holds the repack tensors first, then the most any stage's backend plan asks
 * for (stages run one after another, so they share it). The repack tensors reach the backend
 * as extra device buffers under their own uids, so the backend runs GEMMs over workspace and
 * graph tensors alike, unchanged.
 */
class ConvRepackPlan : public IPlan
{
public:
    /// @p gemmPlans holds one plan per stage of @p steps, null for a stage without GEMM calls.
    ConvRepackPlan(ConvGemmPlanCalls steps, std::vector<std::unique_ptr<IPlan>> gemmPlans);

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvGemmPlanCalls _steps;
    std::vector<std::unique_ptr<IPlan>> _gemmPlans;
    /// Bytes from the start of the workspace to the backend's part of it.
    size_t _gemmWorkspaceOffset = 0;
};

} // namespace hipblaslt_plugin
