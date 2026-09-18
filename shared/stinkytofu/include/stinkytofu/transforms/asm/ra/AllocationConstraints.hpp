// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
class RegClassSet;
struct StinkyInstruction;

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

/// Two values an architecture rule would rather see placed in some relation to
/// each other: sharing a register, avoiding one, avoiding a bank.
///
/// Soft, and structurally so. A preference is consulted only to order candidate
/// registers that placement has already accepted, so it can change which
/// colouring comes out and never whether one does.
///
/// The pair carries no meaning of its own. Which relation was wanted lives on
/// the rule that asked, which is what lets a new kind of preference be a new
/// table row rather than a new field here.
struct Preference {
    SSAValueID a = kInvalidSSAValueID;
    SSAValueID b = kInvalidSSAValueID;
    /// Index into AllocationRules::all(), so the relation and the report line
    /// come from the same place.
    size_t rule = 0;
    /// What satisfying it is worth, charged as a penalty when it goes unmet.
    /// Comparable only with other preferences.
    ///
    /// Must never be negative, for the same reason AllocationRule::baseCost
    /// must not be: placement stops searching once a base costs nothing.
    double benefit = 1.0;

    bool operator==(const Preference& other) const = default;
};

/// SSA values behind every register operand of one instruction, both sides.
///
/// Shared by AllocationConstraints::build() and auditRules() so a pin rule
/// sees the same operand-to-value map the colourer honoured.
struct OperandGroups {
    std::vector<std::vector<SSAValueID>> dest;
    std::vector<std::vector<SSAValueID>> src;

    std::span<const SSAValueID> at(size_t operand, bool isDest) const {
        const std::vector<std::vector<SSAValueID>>& groups = isDest ? dest : src;
        if (operand >= groups.size()) return {};
        return groups[operand];
    }
};

/// SSA values behind each register operand of \p instruction, using the same
/// lifted-DWORD walk as tuple collection.
OperandGroups operandGroupsOf(const StinkyInstruction& instruction, const RegClassSet& classes);

/// Stamp every entry live-in the dispatch does not fill as undefined.
///
/// Recorded on the value because three consumers need it: pinning and affinity
/// in build() below, and the phi check in SSA destruction, which is given only a
/// function and a colouring. Run before build(); forgetting leaves every live-in
/// pinned, an optimisation lost rather than correctness.
void markUndefinedLiveIns(Function& function, const AsmTargetRegisters& target);

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
    /// An Active pinToProducer rule pins the values it names the same way: the
    /// colourer must keep the producer's register rather than pick another legal
    /// one. Compacting still honours that; it is not a hint it may ignore.
    ///
    /// This is legality, not policy. A colourer that ignores it produces wrong
    /// code rather than a slower kernel.
    bool isPinned(SSAValueID id) const;

    /// Why isPinned() is true, or nullptr. `"a function live-in"`, or the name
    /// of the pinToProducer rule that asked.
    const char* pinReason(SSAValueID id) const;

    /// Highest index \p id may occupy, or no limit when nothing constrains it.
    ///
    /// Lower than the register file when the value is used through an operand
    /// field that cannot select a VGPR bank. Such a field has eight bits of
    /// index and no selector, so it reaches one bank only, and a value it names
    /// has to live inside that bank.
    ///
    /// Legality, not policy, for the same reason isPinned() is. A colourer that
    /// ignores it emits an operand naming a register the instruction cannot
    /// reach, which is wrong arithmetic rather than a slower kernel.
    uint32_t maxIndexFor(SSAValueID id) const;

    /// Live-ins left unpinned: read before anything defines them, but above
    /// where the dispatch stops writing, so they arrive holding nothing.
    ///
    /// Exposed rather than merely acted on, because "nothing defines it" and "it
    /// is undefined" differ by whether lifting saw every definition, so a run
    /// that moves one should be able to say which.
    ///
    /// Collected from isUndefined(), so empty unless markUndefinedLiveIns() ran.
    std::span<const SSAValueID> undefinedLiveIns() const {
        return undefinedLiveIns_;
    }

    std::span<const TupleRun> tupleRuns() const {
        return tupleRuns_;
    }

    std::span<const AffinitySet> affinitySets() const {
        return affinitySets_;
    }

    /// Soft pairings the architecture asked for. Empty on a chip with no
    /// pairing rule, which is what keeps placement on its first-fit path.
    std::span<const Preference> preferences() const {
        return preferences_;
    }

    std::string toString() const;

   private:
    const AsmTargetRegisters* target_ = nullptr;
    std::vector<RegType> classByValue_;
    std::vector<std::optional<RegKey>> hintByValue_;
    std::vector<bool> pinnedByValue_;
    std::vector<const char*> pinReasonByValue_;
    std::vector<uint32_t> maxIndexByValue_;
    std::vector<SSAValueID> undefinedLiveIns_;
    std::vector<TupleRun> tupleRuns_;
    std::vector<AffinitySet> affinitySets_;
    std::vector<Preference> preferences_;
};

}  // namespace stinkytofu
