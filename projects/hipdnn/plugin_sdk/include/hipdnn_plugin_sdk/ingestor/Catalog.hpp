// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <vector>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief An engine's kernels that fit one graph on one device, plus their sort state.
 *
 * Built once per (graph, device) by running the matchers, then cached. An engine is
 * applicable exactly when its catalog is non-empty.
 *
 * `isSorted` is what makes ranking lazy: applicability only needs to know whether any
 * kernel survived, so the heuristic is not loaded or run until something actually asks
 * for an order — a knob query or a plan build. Once sorted, `entries` is in
 * heuristic-ranked order and stays that way for the life of the cache entry.
 */
struct Catalog
{
    std::vector<KernelDefinition> entries;
    bool isSorted = false;
    /// What this engine's graph-scoped matchers resolved about the graph, merged across
    /// the packs that survived. Cached with the entries because RFC 0017 §8.1 keeps the
    /// bound token state alongside the catalog, so a plan build reads these rather than
    /// re-running a matcher to recover them.
    ///
    /// "The packs that survived" is enforced by KernelIngestorStateManager::buildCatalog(),
    /// which accumulates each pack's bindings in a pack-scoped view and merges it into
    /// this map only once that pack's own graph-scoped matchers all pass; a pruned
    /// pack's view is discarded unmerged. A shared matcher's bindings still reach every
    /// surviving pack that lists it, because the memo each pack merges from is keyed by
    /// matcher id, not by pack.
    ///
    /// Merged rather than kept per pack: a token name means the same thing to every pack
    /// in an engine. Two packs binding one name to DIFFERENT values is an authoring
    /// error, rejected at merge time since only a runtime match reveals it.
    BoundTokens bound;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
