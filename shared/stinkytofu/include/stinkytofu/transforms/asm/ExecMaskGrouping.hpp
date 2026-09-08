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

#include <cstdint>
#include <unordered_set>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class BasicBlock;
class AsmIRBuilder;
struct StinkyInstruction;

/// Collapse each narrow-exec-write..full-mask-reset span into a single opaque
/// ExecMaskGroup pseudo-instruction so the DAG scheduler cannot reorder into or
/// out of the span. Call expandExecMaskedGroups() after scheduling to restore.
STINKYTOFU_EXPORT void collapseExecMaskedRegions(BasicBlock& bb, AsmIRBuilder& builder,
                                                 uint32_t wavefrontSize);

/// Inverse of collapseExecMaskedRegions.
STINKYTOFU_EXPORT void expandExecMaskedGroups(BasicBlock& bb);

/// The instructions a narrow exec mask covers: those between a narrow exec
/// write and the full-mask reset closing it, excluding the two writes
/// themselves. Spans nest, and the predicates are the ones
/// collapseExecMaskedRegions uses, so the two cannot disagree about where a
/// span is. A *vector* write in that set updates the active lanes and leaves
/// the rest of its destination as it was, which makes the destination a
/// read-modify-write at lane granularity.
///
/// A narrow write with no reset before the end of its block leaves the mask
/// narrow on the way out. The rest of that block is reported as covered, its
/// successors are not, and \p unmatched says so.
STINKYTOFU_EXPORT std::unordered_set<const StinkyInstruction*> execMaskedInstructions(
    const BasicBlock& bb, uint32_t wavefrontSize, bool* unmatched = nullptr);

}  // namespace stinkytofu
