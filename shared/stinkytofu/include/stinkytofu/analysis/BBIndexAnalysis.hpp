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

#include <unordered_map>
#include <vector>

#include "stinkytofu/core/AnalysisManager.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"

namespace stinkytofu {
/// RPO-indexed map from BasicBlock* to unsigned index and vice versa.
struct BBIndexMap {
    std::vector<BasicBlock*> rpo;
    std::unordered_map<const BasicBlock*, unsigned> index;
};

/// Analysis pass that computes an RPO-indexed BasicBlock map.
struct BBIndexAnalysis {
    static inline AnalysisKey Key;

    static AnalysisKey* ID() {
        return &Key;
    }

    static const char* name() {
        return "BBIndexAnalysis";
    }

    using Result = BBIndexMap;

    static Result run(Function& F, AnalysisManager& /*AM*/) {
        BBIndexMap result;
        unsigned idx = 0;
        traverseCFGInRPO(F, [&](BasicBlock* bb) {
            result.rpo.push_back(bb);
            result.index[bb] = idx++;
        });
        return result;
    }
};

}  // namespace stinkytofu
