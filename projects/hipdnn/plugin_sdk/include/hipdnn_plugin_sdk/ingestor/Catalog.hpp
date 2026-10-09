// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <vector>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// An engine's kernels that fit one graph on one device, plus sort state. An engine
/// is applicable exactly when its catalog is non-empty.
struct Catalog
{
    /// Points into the owning KernelIngestorStateManager's kernel set, which is built
    /// once at construction and never mutated, and the manager can be neither copied nor
    /// moved, so every pointer stays valid for the manager's lifetime.
    std::vector<const KernelDefinition*> entries;
    bool isSorted = false;
    /// True when `entries` came from a benchmarked record rather than the heuristic,
    /// distinct from `isSorted`: this asks whether the order can still be replaced by a
    /// later measurement, since a measured order arriving after a memoized heuristic
    /// sort must still win.
    bool orderedFromRecord = false;
    BoundTokens bound; ///< What graph-scoped matchers resolved, merged across packs.
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
