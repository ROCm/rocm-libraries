// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Where the predicted final code of a loop loses matrix-pipe time, window by window, against
// the scheduler's plan (WAITCNT_COISSUE.md 7.7). The repair's rules propose moves only for
// the windows listed here, worst first.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "IssueTimeline.hpp"

namespace stinkytofu::coissue {

struct WindowDamage {
    /// Index of the window's matrix op in the trip.
    int window = 0;
    /// Matrix-pipe idle from this op's end to the next op's start.
    int idleAfter = 0;
    /// Cycles this op waited in the queue: how far the wave ran ahead of the pipe.
    int lead = 0;
    /// The window's issue length minus the plan's.
    int extraIssue = 0;
    /// Loop-order positions of the VALUs planned in this window that lost their slot.
    std::vector<size_t> slipped;

    /// A window that issues shorter than planned can still have lost a VALU slot.
    int severity(bool byIdle) const {
        return std::max(0, byIdle ? idleAfter : extraIssue) + static_cast<int>(slipped.size());
    }
};

/// Which windows count as damaged.
enum class DamageTrigger {
    /// The pipe idles after the window under the primary profile.
    PipeIdle,
    /// The window's issue length grew against the plan.
    IssueGrowth,
};

struct DamageReport {
    /// Every window of the trip, in order.
    std::vector<WindowDamage> all;
    /// The damaged windows, worst first (ties: earlier window first).
    std::vector<const WindowDamage*> damaged;
};

/// `kinds[k]` is the k-th instruction of the loop order. `plan[k]` / `final[k]` are its
/// placements in the plan's and in the predicted final steady trip (cycle -1 when it has
/// none); `planTrip` / `finalTrip` are those trips.
DamageReport buildDamageReport(const std::vector<const TimedInst*>& order,
                               const std::vector<Placement>& plan,
                               const std::vector<Placement>& final, const TripTiming& planTrip,
                               const TripTiming& finalTrip, DamageTrigger trigger);

}  // namespace stinkytofu::coissue
