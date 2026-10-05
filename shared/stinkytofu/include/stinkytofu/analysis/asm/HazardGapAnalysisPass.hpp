// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {

struct StinkyInstruction;

/// One producer->consumer pair under a write->read, cycle-counted rule of
/// kCdna5HazardRules. Positions index the sequence that was measured.
struct HazardGap {
    int ruleIdx;
    unsigned producer;
    unsigned consumer;
    /// Cycles strictly between the two: a matrix op counts its latency, anything
    /// else its issue cycles.
    int gap;
};

/// Every hazard pair in \p instructions, rule by rule and in consumer order. Each
/// consumer pairs with the latest writer of its source registers, and only when
/// that writer is a producer under the rule.
STINKYTOFU_EXPORT std::vector<HazardGap> measureHazardGaps(
    const std::vector<const StinkyInstruction*>& instructions);

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
