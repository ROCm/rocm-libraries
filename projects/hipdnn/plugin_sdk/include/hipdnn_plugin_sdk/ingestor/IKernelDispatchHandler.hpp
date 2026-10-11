// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Launch state derived from the graph, resolved once at plan build. Must not
/// reference the MatchContext or BoundTokens it was built from.
class PreparedDispatch
{
public:
    virtual ~PreparedDispatch() = default;
};

/// Native escape hatch for a UDD: sizes, prepares, and launches a kernel.
template <typename THandle>
class IKernelDispatchHandler
{
public:
    virtual ~IKernelDispatchHandler() = default;

    /// Global scratch this kernel requires, in bytes. Lives on this interface
    /// because the query arrives before a kernel is chosen.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual size_t workspaceBytes(const MatchContext& context,
                                  const BoundTokens& bound,
                                  const KernelDefinition& kernel) const
        = 0;

    /// Resolves everything @p kernel's launch needs while @p context is still valid.
    /// The returned object is owned by the plan and MUST NOT reference @p context or
    /// @p bound.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                                      const BoundTokens& bound,
                                                      const KernelDefinition& kernel) const
        = 0;

    /// @note May run concurrently across threads; must not mutate @p prepared or the
    ///       handler.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual void launch(const THandle& handle,
                        const PreparedDispatch& prepared,
                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers,
                        void* workspace) const
        = 0;

    /// Returns the launch inputs of @p prepared and the versioned name they are saved
    /// under. The default returns nullopt: the handler does not support saving.
    // NOLINTBEGIN(portability-template-virtual-member-function)
    virtual std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& /*prepared*/) const
    {
        return std::nullopt;
    }
    // NOLINTEND(portability-template-virtual-member-function)

    /// Rebuilds a prepared dispatch from saved launch inputs and kernel code, with the
    /// code loaded on @p deviceOrdinal. The default returns nullptr: the handler does
    /// not support restoring.
    // The code is taken by value, so an override can move the bytes into the plan.
    // NOLINTBEGIN(portability-template-virtual-member-function, performance-unnecessary-value-param)
    virtual std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& /*inputs*/,
                                                            SavedKernelCode /*code*/,
                                                            int /*deviceOrdinal*/) const
    {
        return nullptr;
    }
    // NOLINTEND(portability-template-virtual-member-function, performance-unnecessary-value-param)
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
