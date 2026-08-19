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
    std::vector<KernelDefinition> entries;
    bool isSorted = false;
    /// True when `entries` came from a benchmarked record rather than the heuristic.
    ///
    /// Distinct from `isSorted` because the two answer different questions: `isSorted`
    /// asks whether ordering has happened at all, this asks whether the ordering can
    /// still be improved. A heuristically sorted catalog is memoized and would otherwise
    /// keep being served after a benchmark sweep has produced a measured order for the
    /// same graph and device -- the sweep writes its record *after* the sort that cached
    /// it, so without this the measured order would never replace the guess.
    bool orderedFromRecord = false;
    BoundTokens bound; ///< What graph-scoped matchers resolved, merged across packs.
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
