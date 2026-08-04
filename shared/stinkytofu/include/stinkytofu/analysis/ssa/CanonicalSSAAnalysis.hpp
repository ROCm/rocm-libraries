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

#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {

/// Canonical SSA over a function's physical register operands.
///
/// The graph is a cache, not IR state: it is a pure function of the CFG,
/// instruction order, and register operands, so the pass manager evicting it
/// whenever one of those changes is exactly the right lifecycle. That is what
/// makes a stale graph, whose instruction pointers may dangle, unreachable.
///
/// Failure is ordinary rather than exceptional: a function using accumulator
/// registers, True16 halves, calls, or unreachable blocks cannot be lifted, so
/// the result carries either a graph or the reason there is none. Consumers must
/// check before dereferencing.
///
/// Requesting this recomputes the graph on demand, which is usually not what a
/// consumer of an already-lifted function wants. Prefer
/// getCachedResult<CanonicalSSAAnalysis>(), which answers "was this lifted"
/// instead of "lift it now", and which fails loudly when a pass in between
/// forgot to preserve the result.
///
/// This cannot apply LiftAsmRegistersToSSAOptions, because an analysis factory
/// takes no arguments. A pipeline needing non-default options runs
/// LiftAsmRegistersToSSAPass, which lifts with its own options and seeds the
/// result here.
///
/// The function must already be free of def-use analysis state; construction
/// reads the function without modifying it, so a leftover `GFX::PHI` is an
/// error. Run RemoveDefUseAnalysisPass first.
struct CanonicalSSAAnalysis {
    STINKYTOFU_ANALYSIS_KEY("CanonicalSSAAnalysis")

    using Result = Expected<CanonicalSSA>;

    static Result run(Function& F, AnalysisManager& AM) {
        return liftAsmRegistersToSSA(F, AM.getResult<DominanceAnalysis>(F));
    }
};

}  // namespace stinkytofu
