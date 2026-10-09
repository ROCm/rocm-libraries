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

// Optimizer that keeps a run of back-to-back matrix instructions free of waits.
//
// The WMMA FIFO only treats WMMAs issued back-to-back as one batch; any
// instruction between them, an s_wait_* included, splits it. The dataflow plans
// each WMMA's own wait, so a scheduled batch comes out as
//   s_wait_dscnt 56 W  s_wait_dscnt 52 W  s_wait_dscnt 48 W
// No memory op issues inside the run, so every counter's queue is the same at each
// member and the waits compare directly: the run gets one wait, the per-counter
// minimum, before its first WMMA. finalizePlan then finds the rest already drained.

#include "stinkytofu/transforms/asm/waitcnt/WaitPlanOptimizer.hpp"

namespace stinkytofu {
namespace waitcnt {

class WmmaRunWaitMerge : public WaitPlanOptimizer {
   public:
    const char* getName() const override {
        return "WmmaRunWaitMerge";
    }

    void rewrite(WaitInsertionPlan& plan, const DataflowResult& dfr, Function& func) override;
};

}  // namespace waitcnt
}  // namespace stinkytofu
