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
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief An executable plan for one descriptor-selected kernel.
 *
 * Holds the kernel the heuristic chose, the handler that launches it, and the launch
 * detail that handler resolved from the graph at build time: execute() resolves and
 * re-matches nothing.
 *
 * Immutable after construction, as IPlan requires: the same plan may execute
 * concurrently from several threads with different device buffers.
 */
template <typename THandle>
class GenericPlan : public IPlan<THandle>
{
public:
    /**
     * @param dispatcher The selected kernel and its resolved dispatch handler.
     * @param context    Bound graph and device state. Read during construction to
     *                   prepare the launch; not retained, because a plan outlives the
     *                   graph reference this carries.
     * @param bound      What matching resolved about the graph, from the catalog that
     *                   selected this kernel.
     */
    GenericPlan(KernelDispatcher<THandle> dispatcher,
                const MatchContext& context,
                const BoundTokens& bound)
        : _dispatcher(std::move(dispatcher))
        , _workspaceBytes(_dispatcher.handler->workspaceBytes(context, bound, _dispatcher.kernel))
        , _prepared(_dispatcher.handler->prepare(context, bound, _dispatcher.kernel))
    {
        if(_prepared == nullptr)
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "dispatch handler for kernel '"
                                            + toString(_dispatcher.kernel.kernelId)
                                            + "' prepared no launch");
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
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                        "kernel '" + toString(_dispatcher.kernel.kernelId)
                                            + "' requires " + std::to_string(_workspaceBytes)
                                            + " workspace bytes but none was provided");
        }

        _dispatcher.handler->launch(handle, *_prepared, deviceBuffers, numDeviceBuffers, workspace);
    }

    /// The kernel this plan launches. Exposed for the resolved-plan diagnostics
    /// (RFC 0017 §10) and for tests asserting which kernel selection chose.
    const KernelDefinition& kernel() const
    {
        return _dispatcher.kernel;
    }

private:
    KernelDispatcher<THandle> _dispatcher;
    size_t _workspaceBytes;
    std::unique_ptr<PreparedDispatch> _prepared;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
