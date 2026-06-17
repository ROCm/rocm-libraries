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
 *
 * ************************************************************************ */
#pragma once

// Optimizer that promotes a single overly-strict merge-anchor wait into
// per-pred tail drains plus a relaxed anchor wait.
//
// Motivation: when paths converging at a join have different per-counter
// FIFO depths, the conservative dataflow emits the strictest single
// anchor wait, which over-drains the deeper paths. Pre-draining the
// SHALLOWER paths at the pred's tail lets the anchor relax to the
// deepest path's wait without losing safety. See
// shared/stinkytofu/tests/filecheck/waitcnt_insertion_tensor_per_path_tail_compensation.stir
// for a worked example.

#include "stinkytofu/transforms/asm/waitcnt/WaitPlanOptimizer.hpp"

namespace stinkytofu {
namespace waitcnt {

class ShallowPredPromotion : public WaitPlanOptimizer {
   public:
    const char* getName() const override {
        return "ShallowPredPromotion";
    }

    void rewrite(WaitInsertionPlan& plan, const DataflowResult& dfr, Function& func) override;
};

}  // namespace waitcnt
}  // namespace stinkytofu
