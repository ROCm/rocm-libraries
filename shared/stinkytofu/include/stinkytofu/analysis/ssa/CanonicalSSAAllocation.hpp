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

#include <cstddef>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"

namespace stinkytofu {

/// Physical register chosen for each canonical SSA value.
///
/// This is the interface between allocation policy and SSA destruction: any
/// allocator produces one of these, and the same lowering path consumes it.
/// Keeping policy and lowering separate is what lets legacy replay and a real
/// allocator be compared without the comparison being muddied by differences in
/// how the result is applied.
class STINKYTOFU_EXPORT AllocationResult {
   public:
    AllocationResult() = default;

    /// Sizes the result for a graph. Values start unassigned.
    explicit AllocationResult(const CanonicalSSA& ssa);

    void assign(SSAValueID id, RegKey physical);

    bool isAssigned(SSAValueID id) const;

    /// Assigned register; only valid when isAssigned() is true.
    RegKey assignmentOf(SSAValueID id) const;

    /// Number of values the result was sized for.
    size_t valueCount() const;

    /// Values still without a physical register.
    size_t unassignedCount() const;

    /// Fingerprint of the graph this result was computed against. SSA value IDs
    /// only mean something relative to one graph, so SSA destruction compares
    /// this rather than trusting the caller to pair them correctly.
    uint64_t shape() const;

   private:
    // Indexed by value ID; RegType::UNKNOWN marks an unassigned slot.
    std::vector<RegKey> byValue_;
    uint64_t shape_ = kUnstampedShape;
};

/// Assign every value the physical register it was lifted from.
///
/// This reproduces the producer's original allocation exactly, which makes it
/// the reference point for differential testing: lifting, colouring, and SSA
/// destruction must together be an identity transform on the physical program.
/// Any difference is a defect in that machinery rather than in allocation
/// policy, which is why this gate runs before any real allocator is evaluated.
STINKYTOFU_EXPORT AllocationResult createLegacyColoring(const CanonicalSSA& ssa);

}  // namespace stinkytofu
