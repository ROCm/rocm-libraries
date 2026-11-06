// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/conv/ConvRepackPlan.hpp"

#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/conv/RepackProgram.hpp"

namespace hipblaslt_plugin
{

namespace
{

/// The backend's part of the workspace starts on this boundary.
constexpr size_t GEMM_WORKSPACE_ALIGNMENT = 256;

} // namespace

ConvRepackPlan::ConvRepackPlan(ConvGemmPlanCalls steps, std::unique_ptr<IPlan> gemmPlan)
    : _steps(std::move(steps))
    , _gemmPlan(std::move(gemmPlan))
    , _gemmWorkspaceOffset((_steps.workspaceBytes + GEMM_WORKSPACE_ALIGNMENT - 1)
                           / GEMM_WORKSPACE_ALIGNMENT * GEMM_WORKSPACE_ALIGNMENT)
{
    PLUGIN_THROW_IF_FALSE(_gemmPlan != nullptr,
                          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                          "a repack plan needs a GEMM plan");
}

size_t ConvRepackPlan::getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const
{
    return _gemmWorkspaceOffset + _gemmPlan->getWorkspaceSize(handle);
}

void ConvRepackPlan::execute(const HipdnnEnginePluginHandle& handle,
                             const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                             uint32_t numDeviceBuffers,
                             void* workspace) const
{
    PLUGIN_THROW_IF_FALSE(workspace != nullptr || _steps.workspaceBytes == 0,
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "the explicit-GEMM convolution needs its workspace");

    // The workspace tensors first, so their uids resolve to them whatever else is passed.
    auto* base = static_cast<char*>(workspace);
    std::vector<hipdnnPluginDeviceBuffer_t> buffers;
    buffers.reserve(_steps.workspace.size() + numDeviceBuffers);
    for(const auto& tensor : _steps.workspace)
    {
        buffers.push_back(hipdnnPluginDeviceBuffer_t{tensor.uid, base + tensor.offset});
    }
    buffers.insert(buffers.end(), deviceBuffers, deviceBuffers + numDeviceBuffers);
    const auto count = static_cast<uint32_t>(buffers.size());

    const auto& program = handle.repackProgram();
    const auto run = [&](const std::vector<RepackStep>& steps) {
        for(const auto& step : steps)
        {
            const auto src
                = hipdnn_plugin_sdk::findDeviceBuffer(step.srcUid, buffers.data(), count);
            const auto dst
                = hipdnn_plugin_sdk::findDeviceBuffer(step.dstUid, buffers.data(), count);
            program.launch(step, src.ptr, dst.ptr, handle.getStream());
        }
    };

    run(_steps.before);
    _gemmPlan->execute(
        handle, buffers.data(), count, base == nullptr ? nullptr : base + _gemmWorkspaceOffset);
    run(_steps.after);
}

} // namespace hipblaslt_plugin
