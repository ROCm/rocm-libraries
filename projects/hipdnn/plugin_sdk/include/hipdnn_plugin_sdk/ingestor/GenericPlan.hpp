// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// An executable plan for one descriptor-selected kernel; execute() re-matches
/// nothing and may run concurrently from multiple threads with different buffers.
template <typename THandle>
class GenericPlan : public IPlan<THandle>
{
public:
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

    /// The kernel's own scratch requirement, as getWorkspaceSize() answers it.
    size_t scratchBytes() const
    {
        return _workspaceBytes;
    }

    const KernelDefinition& kernel() const
    {
        return _dispatcher.kernel;
    }

private:
    KernelDispatcher<THandle> _dispatcher;
    size_t _workspaceBytes;
    std::unique_ptr<PreparedDispatch> _prepared;
};

/// One slot's built kernel inside a SlotProgramPlan. Shared, because a program benchmark runs
/// many programs over the same built kernels.
template <typename THandle>
struct SlotStage
{
    std::string slot;
    std::shared_ptr<GenericPlan<THandle>> plan;
};

/// The executable plan of a slotted engine (RFC 0021 §5, §6): one built kernel per slot, run
/// in slot order on the one stream the caller executes on. The workspace is the aligned shared
/// region plus the largest scratch any slot needs, since slots run one at a time and reuse the
/// scratch. Every kernel receives the unshifted base: a native handler reads its `ws.*` offset
/// tokens in prepare() and adds them to the base at launch.
template <typename THandle>
class SlotProgramPlan : public IPlan<THandle>
{
public:
    /// @param alignedSharedBytes The shared region, already rounded by alignSharedWorkspace().
    SlotProgramPlan(std::vector<SlotStage<THandle>> stages, size_t alignedSharedBytes)
        : _stages(std::move(stages))
        , _alignedSharedBytes(alignedSharedBytes)
    {
        for(const auto& stage : _stages)
        {
            _maxScratchBytes = std::max(_maxScratchBytes, stage.plan->scratchBytes());
        }
    }

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    size_t getWorkspaceSize(const THandle& /*handle*/) const override
    {
        return _alignedSharedBytes + _maxScratchBytes;
    }

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void execute(const THandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        const auto total = _alignedSharedBytes + _maxScratchBytes;
        if(total > 0 && workspace == nullptr)
        {
            throw HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "slot program requires " + std::to_string(total) + " workspace bytes ("
                    + std::to_string(_alignedSharedBytes) + " shared + "
                    + std::to_string(_maxScratchBytes) + " scratch) but none was provided");
        }

        for(const auto& stage : _stages)
        {
            stage.plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
        }
    }

    /// The built kernels, one per slot, in the order they execute.
    const std::vector<SlotStage<THandle>>& stages() const
    {
        return _stages;
    }

private:
    std::vector<SlotStage<THandle>> _stages;
    size_t _alignedSharedBytes;
    size_t _maxScratchBytes = 0;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
