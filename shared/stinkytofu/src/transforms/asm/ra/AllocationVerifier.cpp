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
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/analysis/asm/ssa/SSAFunctionShape.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/ra/PhysRegMatrix.hpp"

namespace stinkytofu {
namespace {

std::string valueName(SSAValueID id) {
    return "%" + std::to_string(id);
}

std::string joinIds(const std::vector<SSAValueID>& ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) out << ", ";
        out << valueName(ids[i]);
    }
    return out.str();
}

/// How wide a run each value starts, and which values are interior to one.
///
/// A placement rule constrains where a *run* begins, so checking each value's own
/// index would flag the odd members of a legitimately-placed tuple. Interior
/// members are therefore skipped and a run's leader is checked at the run's
/// full width -- the same block the allocator placed.
struct RunShape {
    std::vector<uint32_t> widthAt;  ///< run length led by this value, else 0
    std::vector<bool> interior;     ///< appears at position > 0 in some run
};

RunShape runShapeOf(const AllocationConstraints& constraints, size_t valueCount) {
    RunShape shape;
    shape.widthAt.assign(valueCount + 1, 0);
    shape.interior.assign(valueCount + 1, false);
    for (const TupleRun& run : constraints.tupleRuns()) {
        if (run.units.empty()) continue;
        const SSAValueID leader = run.units.front();
        if (leader != kInvalidSSAValueID && leader <= valueCount) {
            // Overlapping runs share a leader; the widest one is the block.
            shape.widthAt[leader] =
                std::max(shape.widthAt[leader], static_cast<uint32_t>(run.units.size()));
        }
        for (size_t unit = 1; unit < run.units.size(); ++unit) {
            const SSAValueID id = run.units[unit];
            if (id != kInvalidSSAValueID && id <= valueCount) shape.interior[id] = true;
        }
    }
    return shape;
}

}  // namespace

std::string AllocationVerificationResult::toString() const {
    std::ostringstream out;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) out << '\n';
        out << errors[i];
    }
    return out.str();
}

AllocationVerificationResult verifyAllocation(const Function& function,
                                              const AllocationResult& result,
                                              const AllocationContext& context) {
    AllocationVerificationResult checked;
    auto error = [&](std::string message) { checked.errors.push_back(std::move(message)); };

    const std::string prefix = "@" + function.getName() + ": ";

    if (&context.function != &function) {
        error(prefix + "verification context describes a different function");
        return checked;
    }

    const uint64_t attachedShape = function.ssaArena().shape();
    if (attachedShape != kUnstampedShape && computeFunctionShape(function) != attachedShape) {
        error(prefix +
              "the function changed after it was lifted, so the attached SSA describes a "
              "different program");
    }
    if (result.shape() != kUnstampedShape && attachedShape != kUnstampedShape &&
        result.shape() != attachedShape) {
        error(prefix +
              "the allocation was computed against a different graph, so its SSA value IDs do "
              "not mean the same thing here");
    }
    // Two scopes share a shape but number values differently; destruction rejects
    // that, so this must too.
    const RegClassSet& lifted = function.ssaArena().liftedClasses();
    if (result.shape() != kUnstampedShape && result.liftedClasses() != lifted) {
        error(prefix + "the allocation was computed against a lift of " +
              result.liftedClasses().toString() + " but this SSA covers " + lifted.toString() +
              ", so its SSA value IDs do not mean the same thing here");
    }

    PhysRegMatrix matrix(context.target);
    const AllocationConstraints& constraints = context.constraints;
    const RunShape runs = runShapeOf(constraints, function.ssaArena().valueCount());

    for (StinkySSAValue* value : function.ssaArena().values()) {
        if (value == nullptr) continue;
        const SSAValueID id = value->valueId();
        const std::string where = prefix + valueName(id);

        if (!result.isAssigned(id)) {
            error(where + " has no physical register");
            continue;
        }

        const RegKey physical = result.assignmentOf(id);
        if (physical.half != RegHalf::NONE) {
            error(where + " is assigned the sub-DWORD register " + regKeyToString(physical) +
                  ", which cannot be written back to a full-DWORD operand");
            continue;
        }

        const RegType expectedClass = constraints.classOf(id);
        if (expectedClass != RegType::UNKNOWN && physical.type != expectedClass) {
            error(where + " is " + regKeyToString(physical) + " but the value is class " +
                  regTypeToString(expectedClass));
            continue;
        }

        if (!context.target.isAllocatable(physical.type, physical.idx)) {
            if (context.target.isReserved(physical.type, physical.idx)) {
                error(where + " is assigned reserved " + regKeyToString(physical));
            } else {
                error(where + " is assigned " + regKeyToString(physical) +
                      ", which is not allocatable");
            }
            continue;
        }

        // The enforcement point for a placement rule, not the honouring one: a
        // policy that ignores one must produce a refused colouring rather than
        // wrong code, and that is what makes a rule bind every policy.
        if (!runs.interior[id]) {
            const uint32_t width = runs.widthAt[id] != 0
                                       ? runs.widthAt[id]
                                       : std::max<uint32_t>(1, value->type().dwordWidth);
            if (const AllocationRule* rule =
                    context.rules.forbidsBase(physical.type, physical.idx, width);
                rule != nullptr) {
                error(where + " is " + regKeyToString(physical) + " but rule " +
                      std::string(rule->name) + " forbids it: " + std::string(rule->description));
                continue;
            }
        }

        // Pin and scope are remit checks, not a reason to skip occupancy. A
        // live-in that kept its hint still occupies the register, and a mobile
        // value assigned the same unit over an overlapping range is illegal.
        if (constraints.isPinned(id)) {
            const std::optional<RegKey> hint = constraints.hintFor(id);
            if (!hint.has_value()) {
                error(where + " is a function live-in but has no physical register recorded");
            } else if (physical != *hint) {
                error(where + " is a function live-in and must keep " + regKeyToString(*hint) +
                      " but is assigned " + regKeyToString(physical));
            }
        } else if (const char* reason = context.scope.immobileReason(id)) {
            const std::optional<RegKey> hint = constraints.hintFor(id);
            if (!hint.has_value()) {
                error(where + " is " + reason + " but has no physical register recorded");
            } else if (physical != *hint) {
                error(where + " is " + reason + " and must keep " + regKeyToString(*hint) +
                      " but is assigned " + regKeyToString(physical));
            }
        }

        // The other half of holding a register: the check above keeps the
        // occupant, this one turns away a newcomer. Needed separately because
        // the register is free wherever the occupant is dead.
        if (context.scope.isPinnedRegister(physical.type, physical.idx)) {
            const std::optional<RegKey> hint = constraints.hintFor(id);
            if (!hint.has_value() || !(*hint == physical)) {
                error(where + " is assigned " + regKeyToString(physical) +
                      ", which this run is holding for the value lifted from it");
            }
        }

        const LiveRange& range = context.intervals.rangeOf(id);
        if (!matrix.available(physical.type, physical.idx, range)) {
            std::vector<SSAValueID> conflicts;
            matrix.collectConflicts(physical.type, physical.idx, range, conflicts);
            std::ostringstream out;
            out << where << " overlaps";
            if (conflicts.empty()) {
                out << " another value";
            } else {
                out << ' ' << joinIds(conflicts);
            }
            out << " on " << regKeyToString(physical);
            error(out.str());
            continue;
        }
        matrix.bind(physical.type, physical.idx, id, range);
    }

    for (const TupleRun& run : constraints.tupleRuns()) {
        if (run.units.empty()) continue;
        const SSAValueID firstId = run.units.front();
        if (!result.isAssigned(firstId)) continue;
        const RegKey first = result.assignmentOf(firstId);
        for (size_t unit = 1; unit < run.units.size(); ++unit) {
            const SSAValueID id = run.units[unit];
            if (!result.isAssigned(id)) continue;
            const RegKey physical = result.assignmentOf(id);
            if (physical.type == first.type && physical.idx == first.idx + unit) continue;
            error(prefix + "tuple [" + joinIds(run.units) + "]: unit 0 is " +
                  regKeyToString(first) + " but unit " + std::to_string(unit) + " is " +
                  regKeyToString(physical) +
                  "; a range operand must be consecutive in operand "
                  "order");
            break;
        }
    }

    for (const AffinitySet& set : constraints.affinitySets()) {
        if (set.members.empty()) continue;
        const SSAValueID firstId = set.members.front();
        if (!result.isAssigned(firstId)) continue;
        const RegKey first = result.assignmentOf(firstId);
        for (size_t i = 1; i < set.members.size(); ++i) {
            const SSAValueID id = set.members[i];
            if (!result.isAssigned(id)) continue;
            const RegKey physical = result.assignmentOf(id);
            if (physical == first) continue;
            error(prefix + "affinity {" + joinIds(set.members) + "}: " + valueName(id) + " is " +
                  regKeyToString(physical) + " but " + valueName(firstId) + " is " +
                  regKeyToString(first) +
                  "; a merge and its incoming values must share one colour");
            break;
        }
    }

    return checked;
}

}  // namespace stinkytofu
