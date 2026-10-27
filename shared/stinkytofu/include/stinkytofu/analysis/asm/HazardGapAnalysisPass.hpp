// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
class BasicBlock;
class Function;
struct StinkyInstruction;

/// One producer -> consumer pair a kCdna5HazardRules rule measures. Only rules
/// counted as a write followed by a read, in cycles, are measured.
struct HazardGap {
    const char* rule = "";
    /// Cycles the rule needs between the two.
    int required = 0;
    /// Cycles of the instructions strictly between them.
    int gap = 0;
    const BasicBlock* block = nullptr;
    const StinkyInstruction* producer = nullptr;
    const StinkyInstruction* consumer = nullptr;
    /// Positions in the block, counted as isIndexedInstruction() does.
    int producerIndex = -1;
    int consumerIndex = -1;

    bool ok() const {
        return gap >= required;
    }
};

/// Every measured pair in \p function: block by block, and within a block rule
/// by rule. Pairs are intra-block, as the scheduler's hazard gate is.
STINKYTOFU_EXPORT std::vector<HazardGap> findHazardGaps(const Function& function);

/// Report findHazardGaps() for each function on stderr: every pair below its
/// rule's threshold, then a per-rule summary. Reports an analysis failure if any
/// gap is below the threshold.
///
/// Usage:  stinkytofu-opt --arch gfx1250 kernel.s --HazardGapAnalysisPass
///
/// Optional args (comma-separated after '='):
///   verbose   — print every producer->consumer pair, not just violations
STINKYTOFU_EXPORT std::unique_ptr<Pass> createHazardGapAnalysisPass(bool verbose = false);

}  // namespace stinkytofu
