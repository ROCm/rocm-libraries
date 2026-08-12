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
 * `isSorted` defers ranking until something asks for an order; once sorted, `entries`
 * stays in heuristic-ranked order for the life of the cache entry.
 */
struct Catalog
{
    std::vector<KernelDefinition> entries;
    bool isSorted = false;
    /// What this engine's graph-scoped matchers resolved, merged across surviving packs
    /// (RFC 0017 §8.1); a pruned pack's bindings are discarded unmerged. One token name
    /// binding to different values across packs is an authoring error, rejected at merge.
    BoundTokens bound;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
