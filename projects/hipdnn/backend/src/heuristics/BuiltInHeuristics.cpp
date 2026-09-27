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
#include "prediction/PredictionBuiltIn.hpp"
#include "static_ordering/StaticOrderingBuiltIn.hpp"

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

    // ModeA/ModeB (RFC 0019 prediction ranking) are backend built-ins like the two
    // above: RFC 0007 §5.3.5/§10.1 make built-in registration the home for
    // first-party policies. One function table serves both policy IDs —
    // getAllPolicyIds reports them together and HeuristicPluginManager records
    // every ID a plugin exposes — so a single registration is enough.
    registerPlugin(HeuristicPlugin::createBuiltIn(
        hipdnn_backend::heuristics::prediction::populateFunctionTable(),
        "built-in:SelectionHeuristic::ModeA+ModeB"));
}

} // namespace hipdnn_backend::plugin
