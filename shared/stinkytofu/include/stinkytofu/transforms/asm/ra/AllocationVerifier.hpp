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

// Legality of any AllocationResult, including the identity colouring.
//
// Independent of the colourer. A colouring the verifier rejects is never
// applied, so a new allocator inherits the whole check.

#include <string>
#include <vector>

#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocator.hpp"

namespace stinkytofu {

struct AllocationVerificationResult {
    std::vector<std::string> errors;

    bool ok() const {
        return errors.empty();
    }

    std::string toString() const;
};

/// Check that \p result is a legal colouring of \p function under \p context.
///
/// \p context.function must be \p function. The checks are: shape match, every
/// value assigned, assignment in an allocatable unit of the value's class,
/// pinned values and scope-immobile values keep their lifted register, no
/// overlapping intervals sharing a unit, tuple runs consecutive in operand
/// order, affinity sets one colour, reserved units unused.
AllocationVerificationResult verifyAllocation(const Function& function,
                                              const AllocationResult& result,
                                              const AllocationContext& context);

}  // namespace stinkytofu
