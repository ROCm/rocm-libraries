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

// The colouring-policy seam.
//
// A policy decides which value to colour next and which candidate to take.
// It does not derive legality from operands, does not verify, and does not
// write srcRegs/destRegs. Those belong to AllocationConstraints,
// AllocationVerifier, and destroyAttachedSSA.

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationScope.hpp"

namespace stinkytofu {

/// Everything an allocator may read. Const on purpose: no allocator mutates IR.
struct AllocationContext {
    const Function& function;
    const SSALiveIntervals& intervals;
    const AsmTargetRegisters& target;
    const AllocationConstraints& constraints;
    const std::vector<Loop>& loops;

    /// What this architecture forbids and prefers. A policy queries these and
    /// never learns which rule or which chip it is honouring, which is what
    /// keeps every present and future policy subject to whatever it runs on.
    const AllocationRules& rules;

    /// What this run may relocate: register classes and, optionally, a slot
    /// prefix of the function. Values outside the scope keep their lifted
    /// register; the verifier and destruction still need a total colouring.
    AllocationScope scope;
};

/// What lowering must support for this allocator's output to be applicable.
struct AllocatorCapabilities {
    bool mayRecolourMerges = false;  // needs copy insertion on merge edges
    bool maySpill = false;           // needs scratch and waitcnt integration
};

class STINKYTOFU_EXPORT RegisterAllocator {
   public:
    virtual ~RegisterAllocator() = default;
    virtual const char* name() const = 0;
    virtual AllocatorCapabilities capabilities() const = 0;
    virtual Expected<AllocationResult> allocate(const AllocationContext& context) = 0;
};

}  // namespace stinkytofu
