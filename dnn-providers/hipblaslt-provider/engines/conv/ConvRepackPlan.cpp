// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/conv/ConvRepackPlan.hpp"

#include <algorithm>
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

ConvRepackPlan::ConvRepackPlan(ConvGemmPlanCalls steps,
                               std::vector<std::unique_ptr<IPlan>> gemmPlans)
    : _steps(std::move(steps))
    , _gemmPlans(std::move(gemmPlans))
    , _gemmWorkspaceOffset((_steps.workspaceBytes + GEMM_WORKSPACE_ALIGNMENT - 1)
                           / GEMM_WORKSPACE_ALIGNMENT * GEMM_WORKSPACE_ALIGNMENT)
{
    PLUGIN_THROW_IF_FALSE(_gemmPlans.size() == _steps.stages.size(),
                          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                          "a repack plan needs one GEMM plan slot per stage");
    for(size_t i = 0; i < _gemmPlans.size(); ++i)
    {
        PLUGIN_THROW_IF_FALSE((_gemmPlans[i] != nullptr) == !_steps.stages[i].calls.empty(),
                              HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                              "a stage with GEMM calls needs a GEMM plan, and only such a stage");
    }
}

size_t ConvRepackPlan::getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const
{
    size_t gemmBytes = 0;
    for(const auto& plan : _gemmPlans)
    {
        if(plan != nullptr)
        {
            gemmBytes = std::max(gemmBytes, plan->getWorkspaceSize(handle));
        }
    }
    return _gemmWorkspaceOffset + gemmBytes;
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
            program.launch(step,
                           static_cast<const char*>(src.ptr) + step.srcByteOffset,
                           static_cast<char*>(dst.ptr) + step.dstByteOffset,
                           handle.getStream());
        }
    };

    void* gemmWorkspace = base == nullptr ? nullptr : base + _gemmWorkspaceOffset;
    for(size_t i = 0; i < _steps.stages.size(); ++i)
    {
        run(_steps.stages[i].before);
        if(_gemmPlans[i] != nullptr)
        {
            _gemmPlans[i]->execute(handle, buffers.data(), count, gemmWorkspace);
        }
        run(_steps.stages[i].after);
    }
}

} // namespace hipblaslt_plugin
