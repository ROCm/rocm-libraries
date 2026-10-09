/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * ************************************************************************ */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace stinkytofu {

/// A half-open range [begin, end) of region clock cycles.
struct DsPaceWindow {
    int begin = 0;
    int end = 0;
};

/// ds_load pacing plan for one scheduling region.
///
/// The scheduler normally issues ds_loads ASAP, held back only by a rate ceiling. This plans a
/// release time for each of the \p count ds_loads of a region instead: spread evenly over
/// [\p start, \p end) at a uniform rate, except inside \p windows (the TDM landing window, where
/// ds_loads stall against the TDM's LDS writes) where the rate is at most \p windowRate. The
/// count is kept: the rate outside the windows rises to make up what the windows give up.
///
///   rate(t) = r        outside the windows,   min(r, windowRate) inside
///   count   = integral of rate over [start, end)
///
/// Rates are in ds_loads per cycle. The k-th ds_load issued in the region may not issue before
/// result[k] (cycles from the region start; non-decreasing).
///
/// Returns an empty plan (no pacing, i.e. ASAP) when it is not feasible: nothing to plan, an empty
/// range, or the outside-window rate needed to keep the count would exceed \p maxRate (the
/// normal ds cap rate).
inline std::vector<int> planDsRelease(int count, int start, int end, double maxRate,
                                      double windowRate, std::vector<DsPaceWindow> windows) {
    std::vector<int> plan;
    if (count <= 0 || end <= start || maxRate <= 0) return plan;
    windowRate = std::max(0.0, windowRate);

    // Clip the windows to [start, end) and merge them.
    std::vector<DsPaceWindow> w;
    std::sort(windows.begin(), windows.end(),
              [](const DsPaceWindow& a, const DsPaceWindow& b) { return a.begin < b.begin; });
    for (DsPaceWindow x : windows) {
        x.begin = std::max(x.begin, start);
        x.end = std::min(x.end, end);
        if (x.end <= x.begin) continue;
        if (!w.empty() && x.begin <= w.back().end)
            w.back().end = std::max(w.back().end, x.end);
        else
            w.push_back(x);
    }
    double tWin = 0;
    for (const DsPaceWindow& x : w) tWin += x.end - x.begin;
    const double tAvail = end - start;
    const double tOut = tAvail - tWin;

    // Uniform rate r. If it does not exceed windowRate the windows do not bind.
    double r = count / tAvail;
    if (r > windowRate) {
        if (tOut <= 0) return plan;  // the whole range is a window and it cannot hold the count
        r = (count - windowRate * tWin) / tOut;
        if (r > maxRate) return plan;  // cannot keep the count under the ceiling
    }
    if (r <= 0) return plan;

    // Walk the segments and invert the cumulative count: result[k] = t where count(t) = k.
    struct Seg {
        int begin, end;
        double rate;
    };
    std::vector<Seg> segs;
    int cur = start;
    for (const DsPaceWindow& x : w) {
        if (x.begin > cur) segs.push_back({cur, x.begin, r});
        segs.push_back({x.begin, x.end, std::min(r, windowRate)});
        cur = x.end;
    }
    if (cur < end) segs.push_back({cur, end, r});

    plan.reserve(count);
    double acc = 0;  // ds_loads released before the current segment
    size_t s = 0;
    for (int k = 0; k < count; ++k) {
        while (s < segs.size()) {
            const double cap = acc + segs[s].rate * (segs[s].end - segs[s].begin);
            if (segs[s].rate > 0 && k < cap - 1e-9) break;
            acc = cap;
            ++s;
        }
        if (s >= segs.size()) {
            plan.push_back(end);
            continue;
        }
        const double t = segs[s].begin + (k - acc) / segs[s].rate;
        plan.push_back(
            std::max(plan.empty() ? start : plan.back(), static_cast<int>(std::ceil(t - 1e-9))));
    }
    return plan;
}

}  // namespace stinkytofu
