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

#include "InFlightQueue.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {

// The ds cap is `cap` per `span` cycles; the hide-budget model sizes ds work per budget
// window (one WMMA batch window). The cap that fits one budget window is cap * window /
// span, rounded up and at least 1. The default span is the window, which gives `cap`.
inline int dsCapPerBudgetWindow(int cap, int window, int span) {
    if (span <= 0 || window <= 0) return cap;
    return std::max(1, (cap * window + span - 1) / span);
}

// Rule (4) ds_load issue cap: at most `depth` (A) ds_loads per `span` (X) cycles.
// Both modes answer the same questions (full / minResidual) on the same clock;
// they differ only in when an issued ds_load stops counting.
//
//  Sliding  - each ds_load frees its slot `span` cycles after its OWN issue
//             (a sliding window on the real timeline).
//  Periodic - a period opens at its first ds_load and every slot frees `span`
//             cycles after that one, so the cap is exactly A per X-cycle period.
class DsIssueCap {
   public:
    using Mode = PassFeatureConfig::DsIssueCapMode;

    DsIssueCap() = default;
    DsIssueCap(Mode mode, int depth) : mode_(mode), depth_(depth), sliding_(depth) {}

    void advance(int cycles) {
        sliding_.advance(cycles);
        now_ += cycles;
    }

    void push(int span) {
        if (mode_ == Mode::Sliding) {
            sliding_.push(span);
            return;
        }
        if (count_ == 0 || now_ >= periodEnd_) {
            count_ = 0;
            periodEnd_ = now_ + span;
        }
        ++count_;
    }

    bool full() const {
        if (mode_ == Mode::Sliding) return sliding_.full();
        return depth_ > 0 && count_ >= depth_ && now_ < periodEnd_;
    }

    // Cycles until a slot frees (meaningful while full()).
    int minResidual() const {
        if (mode_ == Mode::Sliding) return sliding_.minResidual();
        return std::max(0, periodEnd_ - now_);
    }

    int depth() const {
        return depth_;
    }

    Mode mode() const {
        return mode_;
    }

   private:
    Mode mode_ = Mode::Sliding;
    int depth_ = 0;
    InFlightQueue sliding_;
    int now_ = 0;
    int count_ = 0;
    int periodEnd_ = 0;
};

}  // namespace stinkytofu
