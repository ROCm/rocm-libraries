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
    std::vector<KernelDefinition> entries;
    bool isSorted = false;
    /// The benchmarked record that ordered `entries`, or null when the heuristic did (or
    /// nothing has yet); while null, a later measured order still replaces the heuristic one.
    /// Carried here so plan build and configuration prediction read one snapshot even after
    /// the winner cache evicts the record. Immutable, so safe to share across threads.
    std::shared_ptr<const WinnerRecord> measuredRecord;
    BoundTokens bound; ///< What graph-scoped matchers resolved, merged across packs.
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
