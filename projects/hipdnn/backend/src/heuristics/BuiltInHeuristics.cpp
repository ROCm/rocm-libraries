// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file BuiltInHeuristics.cpp
 * @brief Registers backend-internal heuristic policies with HeuristicPluginManager.
 *
 * Each built-in lives in its own subdirectory under backend/src/heuristics/ and
 * exposes a populateFunctionTable() that yields a fully-populated
 * HeuristicPluginFunctionTable. This file wires those tables into the manager
 * via HeuristicPlugin::createBuiltIn so they flow through the same validation
 * and registration path as dlopen-loaded plugins.
 *
 * Add a new built-in by including its header and adding a single
 * registerPlugin(...) call below.
 */

#include "plugin/HeuristicPluginManager.hpp"

#include "config/ConfigBuiltIn.hpp"
#include "static_ordering/StaticOrderingBuiltIn.hpp"
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "uhd/UhdBuiltIn.hpp"
#endif

namespace hipdnn_backend::plugin
{

void HeuristicPluginManager::registerBuiltIns()
{
    registerPlugin(
        HeuristicPlugin::createBuiltIn(hipdnn_backend::heuristics::config::populateFunctionTable(),
                                       "built-in:SelectionHeuristic::Config"));

    registerPlugin(HeuristicPlugin::createBuiltIn(
        hipdnn_backend::heuristics::static_ordering::populateFunctionTable(),
        "built-in:SelectionHeuristic::StaticOrdering"));

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    // RFC 0019 UHD selection ships with the kernel ingestor (RFC 0017).
    registerPlugin(
        HeuristicPlugin::createBuiltIn(hipdnn_backend::heuristics::uhd::populateFunctionTable(),
                                       "built-in:SelectionHeuristic::UHD"));
#endif
}

} // namespace hipdnn_backend::plugin
