// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "plugin/HeuristicPlugin.hpp"

namespace hipdnn_backend::heuristics::uhd
{

/// Populate a HeuristicPluginFunctionTable with C ABI entry points for
/// the SelectionHeuristic::UHD built-in policy.
hipdnn_backend::plugin::HeuristicPluginFunctionTable populateFunctionTable();

} // namespace hipdnn_backend::heuristics::uhd
