// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {

/// Scan each basic block for kCdna5HazardRules producer->consumer pairs and
/// report the cycle gap between them (using real issueCycles/latencyCycles, not
/// instruction count). Prints a per-rule summary and, per consumer, the tightest
/// producer gap. Exits with a non-zero status if any gap is below the rule threshold.
///
/// Usage:  stinkytofu-opt --arch gfx1250 kernel.s --HazardGapAnalysisPass
///
/// Optional args (comma-separated after '='):
///   verbose   — print every producer->consumer pair, not just violations
STINKYTOFU_EXPORT std::unique_ptr<Pass> createHazardGapAnalysisPass(bool verbose = false);

}  // namespace stinkytofu
