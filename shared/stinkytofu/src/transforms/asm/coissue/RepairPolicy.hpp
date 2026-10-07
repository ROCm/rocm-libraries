// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The write-back policy of the co-issue repair (WAITCNT_COISSUE.md 7.11, rules 2, 4 and 5).
// Robust acceptance and the inserted-instruction limit (rules 1 and 3) decide each move in
// the engine; this decides whether a loop's moves are written back at all.

#include <optional>
#include <string>
#include <vector>

namespace stinkytofu::coissue {

enum class RepairMode { Off, Shadow, Apply };

std::optional<RepairMode> parseRepairMode(const std::string& text);

struct PolicyInput {
    RepairMode mode = RepairMode::Off;
    double marginPercent = 0.5;
    bool trustUncalibrated = false;
    /// Every matrix op of the loop has a calibrated form under the primary profile.
    bool calibrated = false;
    size_t moves = 0;
    /// Predicted gain per profile, in cycles per iteration.
    std::vector<int> gains;
    /// The loop's predicted cycles per iteration under the primary profile, before the moves.
    int loopCycles = 0;
};

struct PolicyDecision {
    bool writeBack = false;
    /// "off", "nothing to do", "below the 0.5% margin", "matrix ops not calibrated",
    /// "would apply (shadow mode)" or "apply".
    std::string reason;
    int worstGain = 0;
    double worstPercent = 0.0;
};

/// Write back only in apply mode, only for a calibrated loop (or with trustUncalibrated),
/// and only if the worst-case gain over the profiles reaches the margin.
PolicyDecision decide(const PolicyInput& input);

}  // namespace stinkytofu::coissue
