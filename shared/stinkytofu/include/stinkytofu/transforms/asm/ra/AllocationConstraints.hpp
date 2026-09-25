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

// Shared, policy-independent legality view of attached SSA.
//
// Built once from srcRegs/destRegs with liftedSSAUnits() and from
// SSABlockArgument.incoming. Every allocator reads this instead of walking
// operands itself, so tuple and merge rules cannot drift between policies.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"

namespace stinkytofu {

class AllocationRules;
class AsmTargetRegisters;
class Function;

/// Value IDs that must occupy consecutive physical units, in operand order.
struct TupleRun {
    std::vector<SSAValueID> units;

    bool operator==(const TupleRun& other) const {
        return units == other.units;
    }
};

/// A block argument and the incoming values that must share its colour until
/// copy insertion exists.
struct AffinitySet {
    std::vector<SSAValueID> members;

    bool operator==(const AffinitySet& other) const {
        return members == other.members;
    }
};

class AllocationConstraints {
   public:
    /// Recover constraints from \p function, letting \p rules append the offset
    /// relations the architecture requires. \p target must outlive this object:
    /// isAllocatable() asks it at query time, so a later reserve() is visible.
    ///
    /// The architecture contributes through build() rather than by mutating the
    /// result, which keeps this object immutable once built and means a
    /// rule-imposed run reaches OffsetUnion and the verifier through exactly the
    /// path an IR-derived one takes.
    static AllocationConstraints build(const Function& function, const AsmTargetRegisters& target,
                                       const AllocationRules& rules);

    /// Register class of \p id, or UNKNOWN when the id is missing.
    RegType classOf(SSAValueID id) const;

    /// True when \p id is a value in an allocatable class. Does not consult the
    /// hint: a value whose original register is reserved is still a candidate.
    bool isAllocatable(SSAValueID id) const;

    /// PhysicalBinding of \p id as a preferred register, if it has one.
    std::optional<RegKey> hintFor(SSAValueID id) const;

    /// True when \p id must keep the register it was lifted from, so hintFor() is
    /// a requirement rather than a preference.
    ///
    /// Function live-ins are pinned: their value arrives in a specific register
    /// placed by the dispatch before any instruction runs, so nothing in the
    /// function defines them and moving one changes what the kernel reads. Lifting
    /// models them as block arguments with no incoming edges.
    ///
    /// This is legality, not policy. A colourer that ignores it produces wrong
    /// code rather than a slower kernel.
    bool isPinned(SSAValueID id) const;

    std::span<const TupleRun> tupleRuns() const {
        return tupleRuns_;
    }

    std::span<const AffinitySet> affinitySets() const {
        return affinitySets_;
    }

    std::string toString() const;

   private:
    const AsmTargetRegisters* target_ = nullptr;
    std::vector<RegType> classByValue_;
    std::vector<std::optional<RegKey>> hintByValue_;
    std::vector<bool> pinnedByValue_;
    std::vector<TupleRun> tupleRuns_;
    std::vector<AffinitySet> affinitySets_;
};

}  // namespace stinkytofu
