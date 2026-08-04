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

// Lowering: writes an allocation back into the physical register operands.
//
// This is the only component that changes a register operand. Lifting leaves
// them alone and an allocator only produces an AllocationResult, so without
// this step an allocation would never reach the emitted program.

#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSAAllocation.hpp"

namespace stinkytofu {
class Function;

/// Everything that stopped SSA destruction, in deterministic order.
struct STINKYTOFU_EXPORT SSADestructionResult {
    std::vector<std::string> errors;

    bool ok() const {
        return errors.empty();
    }

    std::string toString() const;
};

/// Rewrite \p function's physical operands from \p allocation, as described by
/// \p ssa.
///
/// This is the single lowering path shared by every allocation result, so
/// legacy replay and a real allocator differ only in the colouring they are
/// given, never in how it is applied.
///
/// The graph is passed in rather than read off the function: SSA value IDs mean
/// something only relative to one graph, so the caller has to name the graph its
/// allocation was computed against. A graph that no longer describes \p function,
/// or an allocation computed against a different graph, is reported instead of
/// being applied. Discarding the graph afterwards is the caller's job.
///
/// The rewrite is atomic: every operand is validated before any is modified, so
/// a rejected function keeps its original registers.
///
/// A PHI whose inputs and result do not all land on the same register needs a
/// copy on the incoming edge. Copy insertion, parallel-copy sequencing, and
/// critical-edge splitting are not implemented, so that case is reported rather
/// than mis-lowered. Legacy replay never hits it: every version of a register
/// colours back to that same register.
STINKYTOFU_EXPORT SSADestructionResult destroyCanonicalSSA(Function& function,
                                                           const CanonicalSSA& ssa,
                                                           const AllocationResult& allocation);

}  // namespace stinkytofu
