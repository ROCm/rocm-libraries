// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>

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
    /// The benchmarked record that ordered `entries`, or null when the heuristic did (or
    /// nothing has yet); while null, a later measured order still replaces the heuristic one.
    /// Carried here so plan build and configuration prediction read one snapshot even after
    /// the winner cache evicts the record. Immutable, so safe to share across threads.
    std::shared_ptr<const WinnerRecord> measuredRecord;
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
    std::shared_ptr<const WinnerRecord> measuredRecord; ///< As in Catalog.
    BoundTokens bound; ///< As in Catalog.
    std::shared_ptr<const void> owner;

    /// The same catalog as a plain Catalog, for a caller that needs to own and mutate the
    /// entry list. Both hold pointers into the state manager's immutable kernel set, so
    /// this copies the list, never the definitions -- but the result drops `owner`, and so
    /// is only valid while the state manager that produced this view is alive.
    Catalog toCatalog() const
    {
        Catalog catalog;
        catalog.entries = entries;
        catalog.isSorted = isSorted;
        catalog.measuredRecord = measuredRecord;
        catalog.bound = bound;
        return catalog;
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
