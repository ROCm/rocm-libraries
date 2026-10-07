// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DamageReport.hpp"

#include <algorithm>
#include <tuple>

namespace stinkytofu::coissue {

DamageReport buildDamageReport(const std::vector<const TimedInst*>& order,
                               const std::vector<Placement>& plan,
                               const std::vector<Placement>& final, const TripTiming& planTrip,
                               const TripTiming& finalTrip, DamageTrigger trigger) {
    DamageReport report;
    const size_t windows = finalTrip.pipe.size();
    report.all.resize(windows);
    for (size_t k = 0; k < windows; ++k) {
        WindowDamage& w = report.all[k];
        w.window = static_cast<int>(k);
        const PipeOp& op = finalTrip.pipe[k];
        w.lead = op.start - op.issue;
        if (k + 1 < windows) {
            w.idleAfter = std::max(0, finalTrip.pipe[k + 1].start - op.end);
            if (k + 1 < planTrip.pipe.size())
                w.extraIssue = (finalTrip.pipe[k + 1].issue - op.issue) -
                               (planTrip.pipe[k + 1].issue - planTrip.pipe[k].issue);
        }
    }
    // A VALU lost its slot when it sits later in its window, or in a later window, than the
    // scheduler planned it.
    for (size_t k = 0; k < order.size(); ++k) {
        if (order[k]->isLabel || order[k]->kind != IssueClass::Valu) continue;
        if (plan[k].cycle < 0 || final[k].cycle < 0) continue;
        if (std::make_tuple(final[k].window, final[k].pos) <=
            std::make_tuple(plan[k].window, plan[k].pos))
            continue;
        if (plan[k].window >= 0 && static_cast<size_t>(plan[k].window) < windows)
            report.all[plan[k].window].slipped.push_back(k);
    }
    const bool byIdle = trigger == DamageTrigger::PipeIdle;
    for (const WindowDamage& w : report.all)
        if (w.severity(byIdle) > 0) report.damaged.push_back(&w);
    std::stable_sort(report.damaged.begin(), report.damaged.end(),
                     [byIdle](const WindowDamage* a, const WindowDamage* b) {
                         return a->severity(byIdle) > b->severity(byIdle);
                     });
    return report;
}

}  // namespace stinkytofu::coissue
