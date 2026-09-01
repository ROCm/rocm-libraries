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

#include <cstdint>
#include <string>
#include <vector>

#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"

namespace stinkytofu {
class Function;
struct StinkyInstruction;

/// One operand destruction rewrote, with both identities. `beforeType`/`beforeIdx`
/// are what the producer wrote; `afterType`/`afterIdx` are the allocation result.
struct RewrittenOperand {
    StinkyInstruction* instruction = nullptr;
    bool isDestination = false;
    size_t operand = 0;
    RegType beforeType = RegType::UNKNOWN;
    uint32_t beforeIdx = 0;
    RegType afterType = RegType::UNKNOWN;
    uint32_t afterIdx = 0;
};

/// Everything that stopped SSA destruction, in deterministic order.
struct SSADestructionResult {
    std::vector<std::string> errors;
    /// Populated only on success, in the order destruction applied them.
    std::vector<RewrittenOperand> rewritten;

    bool ok() const {
        return errors.empty();
    }

    std::string toString() const;
};

/// Rewrite \p function's physical operands from \p allocation.
///
/// This is the single lowering path shared by every allocation result, so the
/// producer's colouring and a real allocator differ only in the colouring they
/// are given, never in how it is applied.
///
/// A function whose attached SSA no longer matches its physical shape, or an
/// allocation computed against a different lift, is reported instead of being
/// applied. On success, attached SSA is cleared after the rewrite.
///
/// The rewrite is atomic: every operand is validated before any is modified, so
/// a rejected function keeps its original registers and its attached SSA.
///
/// A block argument whose inputs and result do not all land on the same
/// register needs a copy on the incoming edge. Copy insertion, parallel-copy
/// sequencing, and critical-edge splitting are not implemented, so that case is
/// reported rather than mis-lowered. The producer's colouring never hits it:
/// every version of a register colours back to that same register.
SSADestructionResult destroyAttachedSSA(Function& function, const AllocationResult& allocation);

}  // namespace stinkytofu
