// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "common/ConvGemmFormulation.hpp"
#include "engines/plans/PlanInterface.hpp"

namespace hipblaslt_plugin
{

/**
 * @brief A convolution that repacks around its GEMMs: the plan's repack steps before, the
 *        backend's GEMM plan, then the repack steps after, all on the handle's stream.
 *
 * The workspace holds the repack tensors first, then whatever the backend's plan asks for. The
 * repack tensors reach the backend as extra device buffers under their own uids, so the
 * backend runs GEMMs over workspace and graph tensors alike, unchanged.
 */
class ConvRepackPlan : public IPlan
{
public:
    ConvRepackPlan(ConvGemmPlanCalls steps, std::unique_ptr<IPlan> gemmPlan);

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvGemmPlanCalls _steps;
    std::unique_ptr<IPlan> _gemmPlan;
    /// Bytes from the start of the workspace to the backend's part of it.
    size_t _gemmWorkspaceOffset = 0;
};

} // namespace hipblaslt_plugin
