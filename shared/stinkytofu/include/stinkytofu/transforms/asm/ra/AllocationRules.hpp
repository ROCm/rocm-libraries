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

// What one architecture requires of a register allocation.
//
// A rule is one row in a table: a name, a status, and exactly one function
// saying what it forbids or what it prefers. Which function you fill in decides
// what kind of rule it is, and that is the only decision to make:
//
//   forbidsBase   a value may not sit at some index          (hard)
//   clobbersEarly an instruction writes before it reads      (hard)
//   addRelations  two values sit at a fixed offset           (hard)
//   baseCost      an index is legal but worse                (soft)
//
// To add a rule, append a row. Nothing else changes: the queries below walk the
// table and skip rules that are not Active, so a rule never tests its own
// status, and a policy never sees a rule or an architecture at all.
// An unregistered triple yields an empty table.
// See docs/developer/register-allocation.md §14.

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"

namespace stinkytofu {

class Function;
struct StinkyInstruction;

/// Deployment state, not strength.
///
/// A rule cannot go straight to Active. verifyAllocation runs on every
/// colouring including the producer's own, so a hard rule the input already
/// violates turns those kernels silently uncoloured. Audit answers "does the
/// input already break this" first, without enforcing anything.
///
/// Soft rules have no Audit step: there is no violation to report, only a
/// slower kernel, which the shadow report already measures.
enum class RuleStatus : uint8_t {
    Off,
    Audit,
    Active,
};

/// Derived from which function a rule filled in, so it can never disagree with
/// the rule's own behaviour.
enum class RuleKind : uint8_t {
    Empty,         ///< Nothing filled in; the rule does nothing.
    Placement,     ///< forbidsBase
    Interference,  ///< clobbersEarly
    Offset,        ///< addRelations
    Preference,    ///< baseCost
};

/// One architecture rule.
///
/// Fill in exactly one of the four functions. A rule that is both a veto and a
/// price would be invisible to diagnostics and to the lifecycle, so the table
/// rejects it.
struct AllocationRule {
    std::string_view name;
    std::string_view description;
    RuleStatus status = RuleStatus::Off;

    /// May a \p width -wide block of \p regClass start at \p base? Return true
    /// to forbid it.
    ///
    /// No instruction is in scope, and that is structural rather than an
    /// omission: placement is decided per block of tied values, any one of
    /// which may have many uses, so there is no single instruction to pass. A
    /// rule here therefore constrains every block of that class and width --
    /// the same trade LLVM makes by putting SGPR pair alignment on the register
    /// class rather than on an operand.
    std::function<bool(RegType regClass, uint32_t base, uint32_t width)> forbidsBase;

    /// Does \p inst write its results before it has finished reading its
    /// sources? Its destinations then conflict with the sources dying there.
    std::function<bool(const StinkyInstruction& inst)> clobbersEarly;

    /// Append the offset relations this architecture requires. These reach
    /// OffsetUnion and the verifier by exactly the path an IR-derived relation
    /// takes, so neither needs to know a rule produced them.
    std::function<void(const Function& function, std::vector<TupleRun>& tupleRuns,
                       std::vector<AffinitySet>& affinitySets)>
        addRelations;

    /// Relative penalty for starting a \p width -wide \p regClass block at
    /// \p base. Lower is better; 0.0 means no opinion.
    ///
    /// Only a cost that depends on `base % k` can change a colouring: ascending
    /// first-fit already returns the cheapest base for any cost that merely
    /// grows with the index. Alignment-shaped preferences are the useful space.
    std::function<double(RegType regClass, uint32_t base, uint32_t width)> baseCost;

    RuleKind kind() const;
};

/// Forced statuses, ignoring whatever gate the architecture applied. A testing
/// hatch like noVerify: a standalone run has no rocisa capabilities, so a
/// filecheck test needs a way to switch a rule on by name.
struct RuleOverrides {
    bool disableAll = false;            ///< Behave as if the chip had no rules.
    bool auditAll = false;              ///< Every rule to Audit.
    bool activateAll = false;           ///< Every rule to Active.
    std::vector<std::string> activate;  ///< These rules, by name, to Active.
};

/// The rules one architecture imposes, as a value. No rules is an empty table,
/// which is the right answer for a chip nobody has written rules for -- so
/// there is no null to check anywhere.
class AllocationRules {
   public:
    AllocationRules() = default;

    /// Rules in declaration order. A row filling in more than one function, or
    /// none, is dropped and named in `problems()`; a table is data, so a
    /// mistake in it is reported rather than asserted.
    explicit AllocationRules(std::vector<AllocationRule> rules);

    bool empty() const {
        return rules_.empty();
    }

    /// Every rule, whatever its status. Reporting an Off rule rather than
    /// hiding it is what separates "rule present, capability unset" from "no
    /// rule": identical colourings, completely different fixes.
    std::span<const AllocationRule> all() const {
        return rules_;
    }

    /// Rows rejected by the constructor, if any. Empty in a correct table.
    std::span<const std::string> problems() const {
        return problems_;
    }

    // ---- Queries. Each walks the table in declaration order and considers
    // only Active rules, so the answer to "should anything be done about this"
    // is decided in exactly one place. A returned rule is always Active, which
    // is why callers need no status check of their own.

    /// First Active rule forbidding a \p width -wide \p regClass block at
    /// \p base; nullptr when none does. Declaration order decides which rule a
    /// diagnostic names, which is stable across builds.
    const AllocationRule* forbidsBase(RegType regClass, uint32_t base, uint32_t width) const;

    /// First Active rule under which \p inst clobbers early; nullptr when none.
    const AllocationRule* clobbersEarly(const StinkyInstruction& inst) const;

    /// Append the relations every Active rule requires.
    void addRelations(const Function& function, std::vector<TupleRun>& tupleRuns,
                      std::vector<AffinitySet>& affinitySets) const;

    /// Summed penalty from every Active preference. 0.0 when there are none.
    double baseCost(RegType regClass, uint32_t base, uint32_t width) const;

    /// True when any Active rule has a cost, so a policy can keep plain
    /// first-fit -- and today's colouring -- on a chip with no preference.
    bool prices() const {
        return prices_;
    }

    /// Names in \p overrides this table does not declare. Checked separately so
    /// a misspelling is an error rather than a silent no-op: a test that
    /// enables nothing still passes, which is the worst way to find out.
    std::vector<std::string> unknownNames(const RuleOverrides& overrides) const;

    /// Apply \p overrides. Precedence is disableAll, then activation, then
    /// auditAll. Just mutation -- being a value is what makes this a few lines
    /// instead of a forwarding wrapper.
    void force(const RuleOverrides& overrides);

    /// One row per rule: name, kind, status, description.
    std::string toString() const;

   private:
    void refresh();

    std::vector<AllocationRule> rules_;
    std::vector<std::string> problems_;
    bool prices_ = false;
};

/// "off", "audit" or "active".
STINKYTOFU_EXPORT const char* ruleStatusName(RuleStatus status);

/// "placement", "interference", "offset", "preference" or "empty".
STINKYTOFU_EXPORT const char* ruleKindName(RuleKind kind);

/// \p base with every early-clobber destination made live from its
/// instruction's 'u' point, so it overlaps the sources dying there and
/// PhysRegMatrix refuses to share a register between them.
///
/// Returns \p base unchanged when no clobbersEarly rule is Active. Build this
/// beside the pure intervals rather than in place of them: widening raises peak
/// pressure, and the shadow report presents that as a property of the program.
STINKYTOFU_EXPORT SSALiveIntervals applyEarlyClobber(const Function& function,
                                                     const SSALiveIntervals& base,
                                                     const AllocationRules& rules);

/// One message per place \p coloring already violates a rule of \p rules,
/// whatever that rule's status. Empty when nothing does.
///
/// The pre-flight check a hard rule needs before it can be promoted: run it
/// over the producer's own colouring, and if it fires, the finding is about the
/// input rather than the allocator.
STINKYTOFU_EXPORT std::vector<std::string> auditRules(const Function& function,
                                                      const AllocationResult& coloring,
                                                      const AllocationRules& rules);

}  // namespace stinkytofu
