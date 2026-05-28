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

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Function;
class Pass;
struct DominanceInfo;

/// Insert PHI instructions at CFG join points where a physical register
/// has definitions reaching from multiple control flow paths.
///
/// Algorithm (Cytron et al.):
///  1. Compute dominator tree (Cooper-Harvey-Kennedy iterative algorithm)
///  2. Compute dominance frontiers
///  3. For each defined register, place PHIs at the iterated dominance frontier
///  4. Resolve PHI operands using dominator-tree-based reaching definitions
///
/// Each inserted PHI has:
///  - destRegs[0] : the merged register (RegType, idx, num=1)
///  - sources[i] : definition reaching from predecessor i
///                      (nullptr if undefined on that edge)
///  - srcRegs[i] : corresponding source register (literal 0 if undefined)
///
/// The pass only builds PHI -> operand-def chains (and adds each PHI to its
/// operand-defs' user lists).  Chains for non-PHI instructions are NOT touched;
/// call buildUseDefChain() to build the full def-use graph.
///
/// @param clearExisting  When true, existing PHI instructions are removed
///                        before new ones are inserted.  Pass false on a
///                        fresh function that has no PHIs yet.
///
/// Assumes: non-SSA form (physical registers), CFG already built.
STINKYTOFU_EXPORT void insertPhiInstructions(Function& func, bool clearExisting);

/// Overload that accepts pre-computed dominance info.
STINKYTOFU_EXPORT void insertPhiInstructions(Function& func, const DominanceInfo& domInfo,
                                             bool clearExisting);

/// Creates a Pass that inserts PHI instructions via insertPhiInstructions().
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertPhiPass();

}  // namespace stinkytofu
