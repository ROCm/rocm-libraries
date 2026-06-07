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

// Interface for post-dataflow rewriters that improve a WaitInsertionPlan.
//
// Optimizers run AFTER WaitDataflow has produced a sound (conservative) plan
// and BEFORE the emit phase materialises waits in IR. They may relax anchor
// waits and record predecessor-tail drains, but they must never make the
// plan less safe.

#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
class Function;

namespace waitcnt {

class WaitPlanOptimizer {
   public:
    virtual ~WaitPlanOptimizer() = default;

    /// Human-readable name for debug / logging.
    virtual const char* getName() const = 0;

    /// Rewrite `plan` in place. `dfr` is the converged dataflow result
    /// (per-block entry/exit states). `func` gives access to CFG structure.
    virtual void rewrite(WaitInsertionPlan& plan, const DataflowResult& dfr, Function& func) = 0;
};

}  // namespace waitcnt
}  // namespace stinkytofu
