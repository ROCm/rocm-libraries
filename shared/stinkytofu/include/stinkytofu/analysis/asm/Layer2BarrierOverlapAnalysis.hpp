/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
 *
 * ************************************************************************ */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <utility>

#include "stinkytofu/core/AnalysisManager.hpp"

namespace stinkytofu {
struct StinkyInstruction;

/// Scheduler-produced and final-order-validated Layer 2 overlap relationships
/// between barrier groups.
///
/// The relation is directional: first is a barrier in the exclusive "after"
/// group and second is a barrier in the overlapping exclusive "before" group.
struct Layer2BarrierOverlapAnalysis {
    STINKYTOFU_ANALYSIS_KEY("Layer2BarrierOverlapAnalysis")

    using BarrierPair = std::pair<const StinkyInstruction*, const StinkyInstruction*>;

    struct BarrierPairHash {
        size_t operator()(const BarrierPair& pair) const {
            const size_t first = std::hash<const StinkyInstruction*>{}(pair.first);
            const size_t second = std::hash<const StinkyInstruction*>{}(pair.second);
            return first ^ (second + 0x9e3779b9U + (first << 6U) + (first >> 2U));
        }
    };

    class Result {
       public:
        void record(const StinkyInstruction* barrierAfter, const StinkyInstruction* barrierBefore) {
            pairs_.emplace(barrierAfter, barrierBefore);
        }

        bool contains(const StinkyInstruction* barrierAfter,
                      const StinkyInstruction* barrierBefore) const {
            return pairs_.contains({barrierAfter, barrierBefore});
        }

        const std::unordered_set<BarrierPair, BarrierPairHash>& pairs() const {
            return pairs_;
        }

        bool empty() const {
            return pairs_.empty();
        }

       private:
        std::unordered_set<BarrierPair, BarrierPairHash> pairs_;
    };

    // StinkyDAGSchedulerPass updates this cached result after final-order
    // validation. Querying it before scheduling yields no eligible pairs.
    static Result run(Function&, AnalysisManager&) {
        return {};
    }
};

}  // namespace stinkytofu
