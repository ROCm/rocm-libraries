// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RepairPolicy.hpp"

#include <algorithm>
#include <sstream>

namespace stinkytofu::coissue {

std::optional<RepairMode> parseRepairMode(const std::string& text) {
    if (text == "off") return RepairMode::Off;
    if (text == "shadow") return RepairMode::Shadow;
    if (text == "apply") return RepairMode::Apply;
    return std::nullopt;
}

PolicyDecision decide(const PolicyInput& in) {
    PolicyDecision d;
    if (!in.gains.empty()) d.worstGain = *std::min_element(in.gains.begin(), in.gains.end());
    d.worstPercent = in.loopCycles > 0 ? 100.0 * d.worstGain / in.loopCycles : 0.0;
    if (in.mode == RepairMode::Off) {
        d.reason = "off";
        return d;
    }
    if (in.moves == 0) {
        d.reason = "nothing to do";
        return d;
    }
    // Rule 2: a gain under the margin can't be confirmed against hardware noise.
    if (d.worstGain * 100.0 < in.marginPercent * in.loopCycles) {
        std::ostringstream os;
        os << "below the " << in.marginPercent << "% margin";
        d.reason = os.str();
        return d;
    }
    // Rule 4: a form nobody measured can be wrong in ways the profiles don't cover.
    if (!in.calibrated && !in.trustUncalibrated) {
        d.reason = "matrix ops not calibrated";
        return d;
    }
    // Rule 5: shadow mode reports only.
    if (in.mode != RepairMode::Apply) {
        d.reason = "would apply (shadow mode)";
        return d;
    }
    d.writeBack = true;
    d.reason = "apply";
    return d;
}

}  // namespace stinkytofu::coissue
