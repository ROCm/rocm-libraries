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

#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"

namespace stinkytofu {
/// Register all built-in analyses with an AnalysisManager.
/// Call this once per PM at pipeline setup time.
inline void registerAllAnalyses(AnalysisManager& AM) {
    AM.registerPass<BBIndexAnalysis>();
    AM.registerPass<DominanceAnalysis>();
    AM.registerPass<LoopAnalysis>();
}

/// Convenience: build a PreservedAnalyses that keeps CFG analyses.
/// Use when a pass reorders instructions but does not add/remove BBs or edges.
inline PreservedAnalyses preserveCFGAnalyses() {
    PreservedAnalyses PA;
    PA.preserve<BBIndexAnalysis>();
    PA.preserve<DominanceAnalysis>();
    PA.preserve<LoopAnalysis>();
    return PA;
}

}  // namespace stinkytofu
