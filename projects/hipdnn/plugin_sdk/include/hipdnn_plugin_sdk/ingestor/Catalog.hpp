// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
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
    /// True when `entries` came from a benchmarked record rather than the heuristic,
    /// distinct from `isSorted`: this asks whether the order can still be replaced by a
    /// later measurement, since a measured order arriving after a memoized heuristic
    /// sort must still win.
    bool orderedFromRecord = false;
    BoundTokens bound; ///< What graph-scoped matchers resolved, merged across packs.
};

/// A Catalog that points at its state manager's kernel definitions instead of copying
/// them. The definitions are immutable and owned by the engine's shared state, which
/// `owner` keeps alive, so a view stays valid for as long as anyone holds it. A view is
/// never mutated once published; Catalog copies are for callers that need to own one.
struct CatalogView
{
    std::vector<const KernelDefinition*> entries;
    bool isSorted = false;
    bool orderedFromRecord = false; ///< As in Catalog.
    BoundTokens bound; ///< As in Catalog.
    std::shared_ptr<const void> owner;

    /// The same catalog, with every kernel copied out.
    Catalog toCatalog() const
    {
        Catalog catalog;
        catalog.entries.reserve(entries.size());
        for(const auto* entry : entries)
        {
            catalog.entries.push_back(*entry);
        }
        catalog.isSorted = isSorted;
        catalog.orderedFromRecord = orderedFromRecord;
        catalog.bound = bound;
        return catalog;
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
