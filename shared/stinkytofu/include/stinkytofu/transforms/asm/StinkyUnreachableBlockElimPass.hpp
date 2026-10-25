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

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// Creates a pass that erases basic blocks not reachable from the function entry
/// along CFG successor edges.
///
/// Reachability is the CFG as it stands, not layout order and not unresolved
/// indirect-branch targets. Run it after CFGBuilderPass and
/// LongBranchLoweringPass; running it on an incomplete CFG deletes live code.
///
/// The pass is function-wide. PHI placement and dominance need every remaining
/// block, and a region adaptor's temporary Function has a different entry, so
/// it refuses to run when basic-block filtering excludes any block.
///
/// Deleting blocks mutates the CFG. Attached SSA is cleared when anything is
/// erased, because incoming edges and the shape fingerprint would be stale.
///
/// This overload is for a caller that keeps no instruction-group ranges.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyUnreachableBlockElimPass();

/// Same, and additionally guards \p module's instruction-group ranges.
///
/// A group range is a pair of raw boundary nodes, and erasing a block frees the
/// instructions it holds. This overload checks the groups while those nodes are
/// still alive and clears any that would lose a boundary, so findGroupRange()
/// reports nothing and a later ScopeAdaptor no-ops instead of walking freed
/// memory. It is the only correct moment for the check — afterwards, reading a
/// node to ask whether it is still listed is itself undefined.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyUnreachableBlockElimPass(
    StinkyAsmModule& module);

}  // namespace stinkytofu
