// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>

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
    /// The launch names this handler launches: the keys of the `sources` map of every kernel
    /// of its pack (RFC 0017 §15.1). The empty set, which is the default, is a
    /// single-launch handler that reads `kernel_source`, so no existing handler changes.
    ///
    /// The state manager drops any kernel whose launch-name set differs from this one when
    /// it builds its catalog, so a kernel that reaches `prepare` carries exactly these
    /// launches.
    ///
    /// This declaration is provisional: it stands in for the `launches[]` dispatch-descriptor
    /// program, which does not exist yet. Until it does, the handler also owns launch order,
    /// workspace and the hand-off between stages.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual std::set<std::string> launchNames() const
    {
        return {};
    }

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

    /// Returns true when this handler saves launch inputs and restores launches. The
    /// default returns false. Return true only from a handler that overrides both
    /// saveLaunchInputs and restoreLaunch.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual bool supportsSaving() const
    {
        return false;
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
