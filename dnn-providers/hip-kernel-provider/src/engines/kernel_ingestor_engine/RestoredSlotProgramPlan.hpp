// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The plan every kind 2 payload restores as: one prepared dispatch and its handler for each
/// stage, run in the order the payload records them, over the workspace the payload records. An
/// unslotted payload is one stage with no slot name.
///
/// Execute mirrors SlotProgramPlan::execute. Stages run in order on the one stream the caller
/// executes on, and every stage receives the unshifted base workspace, because a stage's handler
/// adds its own offsets to it at launch. SlotProgramPlan itself cannot be rebuilt from a payload:
/// its stages are GenericPlans, which need a dispatcher and a MatchContext that a loaded plan
/// does not have.
///
/// The provider never captures a restored plan: hipDNN saves a loaded plan from the bytes it was
/// loaded from. saveablePlan() returns nullptr, and a direct call of the save hook recognises
/// this type and refuses it.
///
/// Immutable after construction, so several threads can execute it at once.
template <typename THandle>
class RestoredSlotProgramPlan final : public hipdnn_plugin_sdk::IPlan<THandle>
{
public:
    struct Stage
    {
        /// The slot name; empty for the one stage of an unslotted plan.
        std::string slot;
        /// Must outlive the plan. Dispatch handlers live for the process.
        const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<THandle>* handler = nullptr;
        std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch> prepared;
    };

    /// @param stages In execution order; never empty, and every stage has a handler and a
    ///        prepared dispatch.
    /// @param sharedWorkspaceBytes The aligned shared region at the start of the workspace.
    /// @param workspaceBytes The total workspace the plan reports and requires.
    /// @throws HipdnnPluginException with INTERNAL_ERROR when @p stages breaks those rules.
    RestoredSlotProgramPlan(std::vector<Stage> stages,
                            size_t sharedWorkspaceBytes,
                            size_t workspaceBytes)
        : _stages(std::move(stages))
        , _sharedWorkspaceBytes(sharedWorkspaceBytes)
        , _workspaceBytes(workspaceBytes)
    {
        if(_stages.empty())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           "a restored slot program has no stages");
        }
        for(const auto& stage : _stages)
        {
            if(stage.handler == nullptr || stage.prepared == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "stage '" + stage.slot + "' of a restored slot program has no "
                        + (stage.handler == nullptr ? "dispatch handler" : "prepared launch"));
            }
        }
    }

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    size_t getWorkspaceSize(const THandle& /*handle*/) const override
    {
        return _workspaceBytes;
    }

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void execute(const THandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        if(_workspaceBytes > 0 && workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "restored slot program requires " + std::to_string(_workspaceBytes)
                    + " workspace bytes (" + std::to_string(_sharedWorkspaceBytes)
                    + " shared) but none was provided");
        }

        for(const auto& stage : _stages)
        {
            stage.handler->launch(
                handle, *stage.prepared, deviceBuffers, numDeviceBuffers, workspace);
        }
    }

private:
    std::vector<Stage> _stages;
    size_t _sharedWorkspaceBytes;
    size_t _workspaceBytes;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
