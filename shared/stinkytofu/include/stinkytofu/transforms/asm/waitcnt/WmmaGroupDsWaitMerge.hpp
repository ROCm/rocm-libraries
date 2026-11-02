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

// Optimizer that gives every group of N WMMAs one s_wait_dscnt.
//
// Unlike WmmaRunWaitMerge, the WMMAs need not be back-to-back: other
// instructions, ds_loads included, may sit between them. Within a block, N
// WMMAs in a row that each carry their own dscnt wait become one group:
//   s_wait_dscnt 57 W  ..  s_wait_dscnt 53 W  ds_load x4  s_wait_dscnt 53 W  ..  s_wait_dscnt 49 W
// is planned as a single wait before the first WMMA,
//   s_wait_dscnt 45 W  ..  W  ds_load x4  W  ..  W
// A later member's count is lowered by the DS ops issued between the first WMMA
// and it, so the merged wait drains everything each member waited for. A WMMA
// without a dscnt wait ends the group; a group shorter than N keeps its waits.
// Groups are cut every N WMMAs. Only dscnt moves; the other counters stay put.

#include "stinkytofu/transforms/asm/waitcnt/WaitPlanOptimizer.hpp"

namespace stinkytofu {
namespace waitcnt {

class WmmaGroupDsWaitMerge : public WaitPlanOptimizer {
   public:
    explicit WmmaGroupDsWaitMerge(int groupSize) : groupSize_(groupSize) {}

    const char* getName() const override {
        return "WmmaGroupDsWaitMerge";
    }

    void rewrite(WaitInsertionPlan& plan, const DataflowResult& dfr, Function& func) override;

   private:
    int groupSize_;
};

}  // namespace waitcnt
}  // namespace stinkytofu
