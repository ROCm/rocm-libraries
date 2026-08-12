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
 * `isSorted` makes ranking lazy: applicability only needs to know whether any kernel
 * survived, so the heuristic does not run until something asks for an order. Once
 * sorted, `entries` stays in heuristic-ranked order for the life of the cache entry.
 */
struct Catalog
{
    std::vector<KernelDefinition> entries;
    bool isSorted = false;
    /// What this engine's graph-scoped matchers resolved about the graph, merged across
    /// the packs that survived (RFC 0017 §8.1 keeps bound state alongside the catalog).
    ///
    /// Merged only once a pack's own graph-scoped matchers all pass; a pruned pack's
    /// bindings are discarded unmerged (KernelIngestorStateManager::buildCatalog()).
    /// Merged rather than kept per pack because a token name means the same thing to
    /// every pack in an engine — two packs binding one name to different values is an
    /// authoring error, rejected at merge time.
    BoundTokens bound;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
