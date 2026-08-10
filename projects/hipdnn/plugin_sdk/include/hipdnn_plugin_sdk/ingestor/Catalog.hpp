// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <vector>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

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
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
