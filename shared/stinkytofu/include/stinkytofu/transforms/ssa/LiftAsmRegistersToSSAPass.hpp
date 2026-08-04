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

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"

namespace stinkytofu {
class Function;
class Pass;

/// True when any function in \p functions contains a call site.
///
/// A kernel must be recoloured as a whole or not at all. Caller and callee agree
/// on registers only through the convention the producer used, and nothing
/// records that agreement yet, so recolouring one side would silently break it.
/// A pipeline enabling allocation therefore preflights the whole kernel with
/// this and keeps the legacy path for all of it when the answer is true, rather
/// than deciding function by function.
STINKYTOFU_EXPORT bool kernelHasCallSites(const std::vector<const Function*>& functions);

/// Creates a pass that lifts a function's physical registers to canonical SSA
/// and seeds the result into CanonicalSSAAnalysis.
///
/// Running this rather than letting the analysis compute lazily is what applies
/// LiftAsmRegistersToSSAOptions and what produces the located
/// missed-optimization remark, neither of which an analysis factory can express.
///
/// The pass is function-wide: PHI placement and renaming need every block, so
/// it refuses to run at all when basic-block filtering excludes any block.
///
/// The function must already be free of def-use analysis state; run
/// RemoveDefUseAnalysisPass first.
///
/// Failure is seeded too, as an error rather than a graph, so a consumer finds
/// the reason instead of a result describing an earlier version of the IR.
/// Unsupported input is a missed-optimization remark, not a hard error, so
/// consumers read getCachedResult<CanonicalSSAAnalysis>() and fall back to the
/// physical path when it is absent or holds an error.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createLiftAsmRegistersToSSAPass(
    const LiftAsmRegistersToSSAOptions& options = {});

}  // namespace stinkytofu
