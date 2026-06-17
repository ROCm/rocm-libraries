// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/HeuristicPlugin.hpp"

namespace hipdnn_backend::heuristics::static_ordering
{

// Build a fully-populated heuristic plugin function table that exposes the
// SelectionHeuristic::StaticOrdering policy as a backend built-in. The table
// references file-static functions in the matching .cpp; no symbols are
// exported and no shared library is involved. The built-in is registered with
// HeuristicPluginManager via HeuristicPlugin::createBuiltIn at construction
// time.
hipdnn_backend::plugin::HeuristicPluginFunctionTable populateFunctionTable();

} // namespace hipdnn_backend::heuristics::static_ordering
