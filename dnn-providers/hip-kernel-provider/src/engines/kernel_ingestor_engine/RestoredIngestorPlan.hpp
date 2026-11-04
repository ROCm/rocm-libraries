// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// A plan restored from a saved payload: one prepared dispatch and the handler that
/// launches it. It reports the workspace size the payload stores.
///
/// A restored plan cannot be saved again. saveablePlan() returns nullptr, and a save
/// recognises this type and refuses it.
///
/// Immutable after construction, so several threads can execute it at once.
template <typename THandle>
class RestoredIngestorPlan final : public hipdnn_plugin_sdk::IPlan<THandle>
{
public:
    /// @param handler Must outlive the plan. Dispatch handlers live for the process.
    /// @param descriptionText Names the plan in failure messages.
    RestoredIngestorPlan(
        const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<THandle>& handler,
        std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch> prepared,
        size_t workspaceBytes,
        std::string descriptionText)
        : _handler(&handler)
        , _prepared(std::move(prepared))
        , _workspaceBytes(workspaceBytes)
        , _description(std::move(descriptionText))
    {
        if(_prepared == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, _description + " has no prepared launch");
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
                _description + " requires " + std::to_string(_workspaceBytes)
                    + " workspace bytes but none was provided");
        }

        _handler->launch(handle, *_prepared, deviceBuffers, numDeviceBuffers, workspace);
    }

    /// Names the kernel and the dispatch of this plan.
    const std::string& description() const
    {
        return _description;
    }

private:
    const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<THandle>* _handler;
    std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch> _prepared;
    size_t _workspaceBytes;
    std::string _description;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
