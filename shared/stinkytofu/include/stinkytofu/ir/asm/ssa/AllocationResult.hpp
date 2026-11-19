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
#include <cstdint>
#include <string>
#include <vector>

#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"

namespace stinkytofu {

class Function;

/// Physical register chosen for each attached SSA value.
///
/// This is the interface between allocation policy and SSA destruction: any
/// allocator produces one of these, and the same lowering path consumes it.
/// Keeping policy and lowering separate is what lets the identity colouring and
/// a real allocator be compared without the comparison being muddied by
/// differences in how the result is applied. It lives beside the SSA data model
/// rather than with any one allocator because it is the SSA subsystem's exit
/// interface: a mapping over SSAValueID, not a policy.
class AllocationResult {
   public:
    AllocationResult() = default;

    /// Sizes the result for the function's current SSA arena. Values start
    /// unassigned. Copies the arena shape so destruction can reject a result
    /// computed against a different lift.
    explicit AllocationResult(const Function& function);

    void assign(SSAValueID id, RegKey physical);

    bool isAssigned(SSAValueID id) const;

    /// Assigned register; only valid when isAssigned() is true.
    RegKey assignmentOf(SSAValueID id) const;

    /// Number of values the result was sized for.
    size_t valueCount() const;

    /// Values still without a physical register.
    size_t unassignedCount() const;

    /// Fingerprint of the attached SSA this result was computed against.
    uint64_t shape() const;

    /// Register classes the lift this result was computed against covered.
    ///
    /// Two lifts of one program under different scopes share a shape but number
    /// their values differently, so the shape alone cannot tell them apart.
    const RegClassSet& liftedClasses() const {
        return liftedClasses_;
    }

    /// Deterministic dump: a counts line, then one line per value in ID order.
    /// An unassigned value prints "-", so a partial colouring is readable rather
    /// than silently short.
    ///
    /// Reads alongside SSALiveIntervals::toString(), which keys its lines the
    /// same way, so a colouring and the ranges it was derived from can be diffed
    /// value by value.
    std::string toString() const;

   private:
    // Indexed by value ID; RegType::UNKNOWN marks an unassigned slot.
    std::vector<RegKey> byValue_;
    uint64_t shape_ = kUnstampedShape;
    RegClassSet liftedClasses_ = RegClassSet::all();
};

}  // namespace stinkytofu
