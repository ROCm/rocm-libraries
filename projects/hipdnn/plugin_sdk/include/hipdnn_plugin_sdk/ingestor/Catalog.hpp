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
    /// nothing has yet). Distinct from `isSorted`: this asks whether the order can still be
    /// replaced by a later measurement, since a measured order arriving after a memoized
    /// heuristic sort must still win.
    ///
    /// The record travels with the order it produced -- its measured times and the
    /// `(kernel, pack, dispatch)` ids they belong to -- so plan build and configuration
    /// prediction read one snapshot. Looking the record up again would split them once the
    /// bounded winner cache evicts it while this catalog stays cached: the build would serve
    /// the measured order and the prediction would fall back to the model. Shared and
    /// immutable, so a catalog copy costs a reference count rather than a record copy and
    /// is safe to read from any thread.
    std::shared_ptr<const WinnerRecord> measuredRecord;
    BoundTokens bound; ///< What graph-scoped matchers resolved, merged across packs.
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
