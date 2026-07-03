// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/HeuristicPlugin.hpp"

namespace hipdnn_backend::heuristics::config
{

// Build a fully-populated heuristic plugin function table that exposes the
// SelectionHeuristic::Config policy as a backend built-in. The policy reads
// HIPDNN_HEUR_CONFIG_PATH, parses an EngineOverrideConfig JSON file,
// walks conv nodes in the serialized graph and reorders the candidate engine
// IDs so the rule-matched engine sits first. The policy declines (outApplied
// = 0) when the env var is unset, the file fails to load, no rule matches,
// or the matched engine is not in the candidate list — in those cases the
// outer policy loop continues to the next plugin.
hipdnn_backend::plugin::HeuristicPluginFunctionTable populateFunctionTable();

} // namespace hipdnn_backend::heuristics::config
