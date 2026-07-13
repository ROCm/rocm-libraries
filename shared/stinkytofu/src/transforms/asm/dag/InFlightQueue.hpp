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
#include <deque>

namespace stinkytofu {

// Simulation of a finite-depth hardware pipeline queue.
//
// Entries are tracked by absolute expiry time (currentTime_ + drainLatency).
// advance() is O(1) — it only increments the clock. Expired entries are
// evicted lazily on push() and full(). Each entry carries its own drain
// latency to support per-entry math-model variation.
class InFlightQueue {
   public:
    InFlightQueue() = default;
    explicit InFlightQueue(int depth) : depth_(depth) {}

    void advance(int cycles) {
        currentTime_ += cycles;
    }

    void push(int drainLatency) {
        evict();
        expiries_.push_back(currentTime_ + drainLatency);
    }

    bool full() const {
        evict();
        return depth_ > 0 && (int)expiries_.size() >= depth_;
    }

    bool empty() const {
        evict();
        return expiries_.empty();
    }

    int size() const {
        evict();
        return (int)expiries_.size();
    }

    int depth() const {
        return depth_;
    }

    void clear() {
        expiries_.clear();
        currentTime_ = 0;
    }

    // Seed with `count` entries each expiring `residual` cycles from now.
    void seed(int count, int residual) {
        expiries_.clear();
        for (int i = 0; i < count; ++i) expiries_.push_back(currentTime_ + residual);
    }

    // Remaining cycles until the oldest in-flight entry expires.
    int minResidual() const {
        evict();
        if (expiries_.empty()) return 0;
        return std::max(0, expiries_.front() - currentTime_);
    }

    // Remaining cycles until the longest-lived in-flight entry expires.
    int maxResidual() const {
        evict();
        if (expiries_.empty()) return 0;
        return std::max(0, *std::max_element(expiries_.begin(), expiries_.end()) - currentTime_);
    }

   private:
    void evict() const {
        while (!expiries_.empty() && expiries_.front() <= currentTime_) expiries_.pop_front();
    }

    int depth_ = 0;
    int currentTime_ = 0;
    mutable std::deque<int> expiries_;
};

}  // namespace stinkytofu
