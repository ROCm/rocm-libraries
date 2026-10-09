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

// Optimizer that moves the tensor wait of an LDS-ring publishing barrier into
// the predecessor arms that join at it, one drain per arm.
//
// When the arms are the two sides of a wave-role branch, each arm only carries
// the queues of its own waves, so each wave waits for its own ring depth
// instead of the shallowest depth of any wave. Must run after
// ShallowPredPromotion, which replaces the plan's tail drains.

#include "stinkytofu/transforms/asm/waitcnt/WaitPlanOptimizer.hpp"

namespace stinkytofu {
namespace waitcnt {

class RoleSplitDrain : public WaitPlanOptimizer {
   public:
    explicit RoleSplitDrain(const WaitDataflow& dataflow) : df(dataflow) {}

    const char* getName() const override {
        return "RoleSplitDrain";
    }

    void rewrite(WaitInsertionPlan& plan, const DataflowResult& dfr, Function& func) override;

   private:
    const WaitDataflow& df;
};

}  // namespace waitcnt
}  // namespace stinkytofu
