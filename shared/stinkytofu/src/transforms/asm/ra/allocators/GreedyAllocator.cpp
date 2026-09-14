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
#include "GreedyAllocator.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "GreedyPlacement.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/PhysRegMatrix.hpp"

namespace stinkytofu {
namespace {

/// Weight multiplier per enclosing loop level, and the level it stops growing.
///
/// The plan writes the priority as useCount * loopDepth / intervalLength. Read
/// literally that zeroes every value outside a loop, so depth enters as a
/// multiplier instead: a range in a loop body is worth an order of magnitude more
/// per use than the same range in straight-line code, which is the usual
/// block-frequency intuition without a frequency analysis to draw on.
constexpr double kLoopWeight = 10.0;
constexpr uint32_t kMaxLoopDepth = 4;

/// How many times one block may be evicted before the colouring is abandoned.
///
/// Eviction only ever moves work to a strictly lighter block, but a requeued
/// block can displace a third one, so a cap is what guarantees termination
/// rather than an appeal to that ordering.
constexpr uint32_t kMaxEvictionsPerBlock = 2;

/// Union-find over value IDs that also carries a fixed offset to the root.
///
/// Both constraints an allocator must honour have the shape "value b sits delta
/// units from value a": a tuple run wants unit i at base + i, and an affinity set
/// wants every member on the same register, delta 0. Carrying them in one
/// structure is not a convenience -- tuple runs overlap in real code, because a
/// wide load followed by a narrow read of its first half shares values, so
/// honouring each run on its own would let a later placement break an earlier
/// one.
class OffsetUnion {
   public:
    explicit OffsetUnion(size_t count) : parent_(count), offset_(count, 0) {
        for (size_t i = 0; i < count; ++i) parent_[i] = static_cast<SSAValueID>(i);
    }

    /// Root of \p id, and \p id's offset relative to it.
    std::pair<SSAValueID, int64_t> find(SSAValueID id) {
        SSAValueID root = id;
        int64_t toRoot = 0;
        while (parent_[root] != root) {
            toRoot += offset_[root];
            root = parent_[root];
        }
        // Compress, rewriting each node on the path to point straight at the root
        // with its own accumulated offset.
        SSAValueID cursor = id;
        int64_t remaining = toRoot;
        while (parent_[cursor] != root) {
            const SSAValueID next = parent_[cursor];
            const int64_t step = offset_[cursor];
            parent_[cursor] = root;
            offset_[cursor] = remaining;
            remaining -= step;
            cursor = next;
        }
        return {root, toRoot};
    }

    /// Requires offset(b) - offset(a) == delta. False when that contradicts what
    /// is already known, which means no assignment can satisfy every operand.
    bool relate(SSAValueID a, SSAValueID b, int64_t delta) {
        const auto [rootA, offsetA] = find(a);
        const auto [rootB, offsetB] = find(b);
        if (rootA == rootB) return offsetB - offsetA == delta;
        parent_[rootB] = rootA;
        offset_[rootB] = offsetA + delta - offsetB;
        return true;
    }

   private:
    std::vector<SSAValueID> parent_;
    std::vector<int64_t> offset_;
};

std::string valueName(SSAValueID id) {
    return "%" + std::to_string(id);
}

class Greedy {
   public:
    /// \p followHints decides whether a block prefers the register it was lifted
    /// from. With hints the colouring reproduces the producer's numbering wherever
    /// there is room; without them placement packs from the bottom, which is the
    /// only way the high-water mark can come down.
    /// \p mayFoldPairs says whether this run may fold a pairing, which means
    /// putting both values in one block so they share a register by
    /// construction. Scoring is the other way to grant a pairing, and it is
    /// always available. So this adds a mechanism rather than choosing one:
    /// anything not folded is still scored.
    ///
    /// This names no rule and no architecture. Each rule answers for itself
    /// through its own satisfiedBy.
    Greedy(const AllocationContext& context, bool followHints, const PlacementPolicy& policy,
           bool mayFoldPairs)
        : context_(context),
          followHints_(followHints),
          policy_(policy),
          mayFoldPairs_(mayFoldPairs) {}

    /// Did this run fold anything? Folding can make a colouring refuse that
    /// would otherwise have worked, so the caller needs to know whether a
    /// refusal is worth retrying without it.
    bool foldedAny() const {
        return foldedAny_;
    }

    Expected<AllocationResult> run() {
        if (context_.function.ssaArena().valueCount() == 0)
            return AllocationResult(context_.function);

        if (!buildBlocks()) return fail(error_);
        if (mayFoldPairs_) foldPairs();
        measure();
        if (!place()) return fail(error_);

        AllocationResult result(context_.function);
        for (const Block& block : blocks_) {
            for (const Member& member : block.members)
                result.assign(member.value,
                              RegKey{block.regClass, block.base + member.offset, RegHalf::NONE});
        }
        return result;
    }

   private:
    Expected<AllocationResult> fail(const std::string& message) {
        return Expected<AllocationResult>::Error("@" + context_.function.getName() + ": " +
                                                 message);
    }

    const LiveRange& rangeOf(SSAValueID id) const {
        return context_.intervals.rangeOf(id);
    }

    /// Fold every tuple run and affinity set into offset-related blocks, then
    /// check each one can physically exist before any placement is attempted.
    bool buildBlocks() {
        const size_t valueCount = context_.function.ssaArena().valueCount();
        OffsetUnion offsets(valueCount + 1);

        for (const TupleRun& run : context_.constraints.tupleRuns()) {
            for (size_t unit = 1; unit < run.units.size(); ++unit) {
                if (offsets.relate(run.units.front(), run.units[unit], static_cast<int64_t>(unit)))
                    continue;
                error_ = "operands disagree about where " + valueName(run.units[unit]) +
                         " sits relative to " + valueName(run.units.front()) +
                         "; no assignment can satisfy both";
                return false;
            }
        }
        for (const AffinitySet& set : context_.constraints.affinitySets()) {
            for (size_t i = 1; i < set.members.size(); ++i) {
                if (offsets.relate(set.members.front(), set.members[i], 0)) continue;
                error_ = "a merge needs " + valueName(set.members[i]) + " and " +
                         valueName(set.members.front()) +
                         " on one register, but an operand needs them apart";
                return false;
            }
        }

        // Group by root, keeping raw offsets; they are normalised below.
        std::vector<int64_t> rawOffset(valueCount + 1, 0);
        std::vector<SSAValueID> rootOf(valueCount + 1, kInvalidSSAValueID);
        std::vector<size_t> blockOfRoot(valueCount + 1, kNoBlock);
        for (SSAValueID id = 1; id <= valueCount; ++id) {
            const auto [root, offset] = offsets.find(id);
            rootOf[id] = root;
            rawOffset[id] = offset;
        }

        for (SSAValueID id = 1; id <= valueCount; ++id) {
            const SSAValueID root = rootOf[id];
            if (blockOfRoot[root] == kNoBlock) {
                blockOfRoot[root] = blocks_.size();
                blocks_.push_back({});
                blocks_.back().regClass = context_.constraints.classOf(id);
                blocks_.back().leader = id;
            }
            Block& block = blocks_[blockOfRoot[root]];
            if (context_.constraints.classOf(id) != block.regClass) {
                error_ = valueName(id) + " is class " +
                         regTypeToString(context_.constraints.classOf(id)) + " but is tied to " +
                         valueName(block.leader) + " in class " + regTypeToString(block.regClass);
                return false;
            }
            block.members.push_back({id, 0});
            blockIndexOf_.resize(valueCount + 1, kNoBlock);
            blockIndexOf_[id] = blockOfRoot[root];
        }

        for (Block& block : blocks_) {
            int64_t lowest = rawOffset[block.members.front().value];
            for (const Member& member : block.members)
                lowest = std::min(lowest, rawOffset[member.value]);
            int64_t highest = lowest;
            for (Member& member : block.members) {
                member.offset = static_cast<uint32_t>(rawOffset[member.value] - lowest);
                highest = std::max(highest, rawOffset[member.value]);
            }
            block.width = static_cast<uint32_t>(highest - lowest) + 1;
            // Deterministic member order, and a leader that does not depend on
            // which value happened to become the union-find root.
            std::sort(block.members.begin(), block.members.end(),
                      [](const Member& a, const Member& b) {
                          return a.offset != b.offset ? a.offset < b.offset : a.value < b.value;
                      });
            for (const Member& member : block.members)
                block.leader = std::min(block.leader, member.value);

            // Before feasibility, because the ceiling bounds the base scan that
            // both the check and placementFreedom come out of.
            block.maxIndex = ceilingOf(block);
            if (!checkFeasible(block)) return false;
        }
        return true;
    }

    /// Fold every pairing that one register would satisfy, where that is
    /// possible. This is the stronger of the two mechanisms.
    ///
    /// Scoring cannot grant these reliably. A pair takes two placements. The
    /// one placed second may find the register already taken by a third value
    /// that outlives the first. Folding makes it one placement, so there is
    /// nothing left to lose. The hand-written producer does the same thing: it
    /// keeps an accumulator chain in one register range for the whole kernel.
    ///
    /// A whole chain folds, not just its first link. Each pair is checked
    /// against the block as it stands, so the third value joins the block the
    /// first two already formed.
    ///
    /// This cannot produce a wrong colouring. Two values are folded only when
    /// their live ranges are disjoint at every register they would share. It
    /// can still ask more of placement: one run of registers where two shorter
    /// runs would have fitted. If that makes placement refuse, the caller runs
    /// again without folding.
    void foldPairs() {
        for (const Preference& preference : context_.constraints.preferences())
            foldPair(preference);
        if (foldedAny_) compactBlocks();
    }

    /// Fold the two blocks \p preference names, so its two values share a
    /// register. Returns false when the fold is refused, which happens when the
    /// rule wanted something other than one register, when the two are already
    /// in one block, when the two values are live at the same time, or when the
    /// folded block would have no legal base.
    bool foldPair(const Preference& preference) {
        const SSAValueID a = preference.a;
        const SSAValueID b = preference.b;
        if (a == kInvalidSSAValueID || b == kInvalidSSAValueID) return false;
        if (a >= blockIndexOf_.size() || b >= blockIndexOf_.size()) return false;
        const size_t hostIndex = blockIndexOf_[a];
        const size_t guestIndex = blockIndexOf_[b];
        if (hostIndex == kNoBlock || guestIndex == kNoBlock) return false;
        if (hostIndex == guestIndex) return false;

        const Block& host = blocks_[hostIndex];
        const Block& guest = blocks_[guestIndex];
        if (host.regClass != guest.regClass) return false;

        // Folding puts both values on one register. So asking the rule about a
        // key against itself asks exactly the right question: would one
        // register satisfy you? Nothing here needs to know what a rule means.
        // A rule that wants its pair merely near, or apart, answers no, and
        // scoring handles it instead.
        const RegKey shared{host.regClass, 0, RegHalf::NONE};
        if (!context_.rules.satisfiedBy(preference, shared, shared)) return false;
        // A pinned block already has to be where it is. Folding would drag its
        // partner onto that register, which decides the colouring instead of
        // preferring one. Leave it to scoring.
        if (pinReasonOf(host) != nullptr || pinReasonOf(guest) != nullptr) return false;
        // Only fold two blocks under the same ceiling. A ceiling belongs to the
        // one value whose operand field cannot reach past it, and folding would
        // apply it to every member of the block: one capped scale operand could
        // confine a whole accumulator chain to the first bank. That is a large
        // price for one pairing, so it is left to scoring too.
        if (host.maxIndex != guest.maxIndex) return false;

        const std::optional<uint32_t> hostAt = offsetOf(host, a);
        const std::optional<uint32_t> guestAt = offsetOf(guest, b);
        if (!hostAt.has_value() || !guestAt.has_value()) return false;

        // How far the guest's members move once b sits on a. This can be
        // negative, when the guest reaches below the host's base. That is why
        // the offsets below stay signed until normalise() rebases them.
        const int64_t shift = static_cast<int64_t>(*hostAt) - static_cast<int64_t>(*guestAt);

        std::vector<std::pair<SSAValueID, int64_t>> placed;
        placed.reserve(host.members.size() + guest.members.size());
        for (const Member& member : host.members)
            placed.emplace_back(member.value, static_cast<int64_t>(member.offset));
        for (const Member& member : guest.members) {
            const int64_t offset = static_cast<int64_t>(member.offset) + shift;
            // Two members conflict only if they land on the same register, so
            // compare per offset rather than across the whole block.
            for (const Member& other : host.members) {
                if (static_cast<int64_t>(other.offset) != offset) continue;
                if (rangeOf(other.value).overlaps(rangeOf(member.value))) return false;
            }
            placed.emplace_back(member.value, offset);
        }

        const Block folded = normalise(host.regClass, placed);
        const BaseRejection rejected = rejectionFor(folded);
        if (rejected.legalBases == 0) return false;

        blocks_[guestIndex].members.clear();
        blocks_[hostIndex] = folded;
        blocks_[hostIndex].placementFreedom = rejected.legalBases;
        for (const Member& member : folded.members) blockIndexOf_[member.value] = hostIndex;
        foldedAny_ = true;
        return true;
    }

    /// Offset of \p value inside \p block, or nullopt when it is not a member.
    static std::optional<uint32_t> offsetOf(const Block& block, SSAValueID value) {
        for (const Member& member : block.members) {
            if (member.value == value) return member.offset;
        }
        return std::nullopt;
    }

    /// Build a block from \p placed: rebase the offsets on zero, and sort the
    /// members the way placement and the feasibility check both expect.
    Block normalise(RegType regClass,
                    const std::vector<std::pair<SSAValueID, int64_t>>& placed) const {
        int64_t lowest = placed.front().second;
        int64_t highest = lowest;
        for (const auto& [value, offset] : placed) {
            lowest = std::min(lowest, offset);
            highest = std::max(highest, offset);
        }

        Block block;
        block.regClass = regClass;
        block.width = static_cast<uint32_t>(highest - lowest) + 1;
        for (const auto& [value, offset] : placed)
            block.members.push_back({value, static_cast<uint32_t>(offset - lowest)});
        std::sort(block.members.begin(), block.members.end(), [](const Member& a, const Member& b) {
            return a.offset != b.offset ? a.offset < b.offset : a.value < b.value;
        });
        block.leader = block.members.front().value;
        for (const Member& member : block.members)
            block.leader = std::min(block.leader, member.value);
        block.maxIndex = ceilingOf(block);
        return block;
    }

    /// Remove the blocks that folding emptied, then point every value at the
    /// index its block now has.
    void compactBlocks() {
        std::vector<Block> kept;
        kept.reserve(blocks_.size());
        for (Block& block : blocks_) {
            if (block.members.empty()) continue;
            kept.push_back(std::move(block));
        }
        blocks_ = std::move(kept);
        std::fill(blockIndexOf_.begin(), blockIndexOf_.end(), kNoBlock);
        for (size_t index = 0; index < blocks_.size(); ++index) {
            for (const Member& member : blocks_[index].members) blockIndexOf_[member.value] = index;
        }
    }

    /// What placement rules have to say about \p block, for diagnostics.
    ///
    /// Both facts come from one scan because both callers want the same walk:
    /// checkFeasible needs to know a block can never be placed, and place()
    /// needs to name a rule that narrowed the search rather than blame pressure.
    struct BaseRejection {
        const AllocationRule* first = nullptr;  ///< first rule to forbid any base
        bool everyBase = false;                 ///< no rule-legal base survives
        uint32_t legalBases = 0;                ///< how many do, before occupancy
    };

    /// Highest base \p block could take, given the file and any index ceiling.
    /// Nullopt when even base 0 would put a member out of reach.
    std::optional<uint32_t> highestBaseFor(const Block& block) const {
        const uint32_t indexes = context_.target.indexCount(block.regClass);
        if (block.width == 0 || block.width > indexes) return std::nullopt;
        uint32_t highest = indexes - block.width;
        // The ceiling binds on the member sitting furthest from the base, so the
        // whole range has to end at or below it.
        if (block.maxIndex.has_value()) {
            if (*block.maxIndex + 1 < block.width) return std::nullopt;
            highest = std::min(highest, *block.maxIndex + 1 - block.width);
        }
        return highest;
    }

    /// What the rules and the ceiling leave of \p block's candidate bases.
    ///
    /// The count is the block's placementFreedom. It costs nothing extra: this
    /// scan already runs once per block for checkFeasible, so counting what
    /// survives is a branch inside a loop that was happening anyway.
    BaseRejection rejectionFor(const Block& block) const {
        BaseRejection found;
        const std::optional<uint32_t> highest = highestBaseFor(block);
        if (!highest.has_value()) return found;

        // No rule can forbid anything, so every base in range survives and there
        // is nothing to walk.
        if (context_.rules.empty()) {
            found.legalBases = *highest + 1;
            return found;
        }

        found.everyBase = true;
        for (uint32_t base = 0; base <= *highest; ++base) {
            const AllocationRule* rule =
                context_.rules.forbidsBase(block.regClass, base, block.width);
            if (rule == nullptr) {
                found.everyBase = false;
                ++found.legalBases;
            } else if (found.first == nullptr) {
                found.first = rule;
            }
        }
        return found;
    }

    /// What the block builder actually produced, by shape.
    ///
    /// Widths say how much of the function needs a run rather than a register,
    /// and freedoms say how finely the placement order can tell those blocks
    /// apart: an order can only separate blocks whose counts differ, so a
    /// histogram with few distinct values is an order with few distinct
    /// opinions. `shared` counts blocks holding more values than they are
    /// wide. Three things produce those: overlapping tuple runs, affinity sets
    /// and folded pairs. A block that is one plain tuple run is not counted.
    std::string shapeSummary() const {
        std::map<uint32_t, uint32_t> widths;
        std::map<uint32_t, uint32_t> freedoms;
        uint32_t shared = 0;
        for (const Block& block : blocks_) {
            if (block.regClass != RegType::V) continue;
            ++widths[block.width];
            ++freedoms[block.placementFreedom];
            if (block.members.size() > block.width) ++shared;
        }

        auto render = [](const std::map<uint32_t, uint32_t>& counts) {
            std::string text;
            for (const auto& [key, count] : counts) {
                if (!text.empty()) text += " ";
                text += std::to_string(key) + "x" + std::to_string(count);
            }
            return text;
        };
        return "; v blocks by width [" + render(widths) + "], by freedom [" + render(freedoms) +
               "], " + std::to_string(shared) + " holding more values than their width";
    }

    /// Why every base was refused, and whether the blockers could have moved.
    ///
    /// This is the question recoloring turns on. A block that cannot be placed
    /// is rescuable only if the blocks sitting in its way have somewhere else to
    /// go, and "somewhere else" is what placementFreedom measures: a blocker
    /// freer than the block it blocks has more candidate bases to retreat to.
    /// If the blockers are pinned, or no freer than this block, no amount of
    /// reshuffling helps and the function needs splitting instead.
    ///
    /// Only built on the refusal path, so the scan costs nothing in a run that
    /// colours.
    std::string blockerSummary(const Block& block) const {
        const std::optional<uint32_t> highest = highestBaseFor(block);
        if (!highest.has_value()) return {};

        uint32_t ruledOut = 0;
        uint32_t occupied = 0;
        std::vector<size_t> blockers;
        for (uint32_t base = 0; base <= *highest; ++base) {
            if (!reachableAt(block, base)) {
                ++ruledOut;
                continue;
            }
            if (availableAt(block, base)) continue;
            ++occupied;
            for (size_t index : occupantsAt(block, base)) {
                if (std::find(blockers.begin(), blockers.end(), index) == blockers.end())
                    blockers.push_back(index);
            }
        }

        uint32_t pinned = 0;
        uint32_t freer = 0;
        for (size_t index : blockers) {
            if (blocks_[index].pinReason != nullptr) ++pinned;
            if (blocks_[index].placementFreedom > block.placementFreedom) ++freer;
        }

        return "; " + std::to_string(ruledOut) + " base(s) ruled out before occupancy, " +
               std::to_string(occupied) + " occupied by " + std::to_string(blockers.size()) +
               " block(s), of which " + std::to_string(pinned) + " pinned and " +
               std::to_string(freer) + " freer than this one";
    }

    /// Two members can legitimately share an offset: overlapping tuple runs force
    /// a value written before a partial overwrite onto the same register as the
    /// value that replaces it. That is only sound while their ranges are disjoint.
    bool checkFeasible(Block& block) {
        const uint32_t indexes = context_.target.indexCount(block.regClass);
        if (indexes == 0) {
            error_ = valueName(block.leader) + " is class " + regTypeToString(block.regClass) +
                     ", which this target does not allocate";
            return false;
        }
        if (block.width > indexes) {
            error_ = "values tied to " + valueName(block.leader) + " span " +
                     std::to_string(block.width) + " registers, more than the " +
                     std::to_string(indexes) + " " + regTypeToString(block.regClass) +
                     " registers this target can encode";
            return false;
        }
        if (block.maxIndex.has_value() && *block.maxIndex + 1 < block.width) {
            error_ = "values tied to " + valueName(block.leader) + " span " +
                     std::to_string(block.width) +
                     " registers but their operands cannot address "
                     "past index " +
                     std::to_string(*block.maxIndex);
            return false;
        }

        const BaseRejection rejected = rejectionFor(block);
        block.placementFreedom = rejected.legalBases;
        // A block no base can ever satisfy should name the rule rather than
        // exhaust every base and report the generic "no register is free".
        if (rejected.everyBase && rejected.first != nullptr) {
            const AllocationRule* rule = rejected.first;
            error_ = "no " + regTypeToString(block.regClass) + " base is legal for " +
                     valueName(block.leader) + ": rule " + std::string(rule->name) + " (" +
                     std::string(rule->description) + ")";
            return false;
        }
        for (size_t i = 0; i < block.members.size(); ++i) {
            for (size_t j = i + 1; j < block.members.size(); ++j) {
                if (block.members[i].offset != block.members[j].offset) break;
                if (!rangeOf(block.members[i].value).overlaps(rangeOf(block.members[j].value)))
                    continue;
                error_ = valueName(block.members[i].value) + " and " +
                         valueName(block.members[j].value) +
                         " are forced onto one register by their operands but are live at the "
                         "same point";
                return false;
            }
        }
        return true;
    }

    /// Loop nesting depth per block, and the block each value is defined in, so a
    /// value's weight can reflect how often its uses execute.
    void measure() {
        std::unordered_map<const BasicBlock*, uint32_t> depth;
        for (const BasicBlock& bb : context_.function) {
            uint32_t nesting = 0;
            for (const Loop& loop : context_.loops) {
                if (loop.contains(&bb)) ++nesting;
            }
            depth.emplace(&bb, nesting);
        }

        const size_t valueCount = context_.function.ssaArena().valueCount();
        std::vector<uint32_t> depthOfValue(valueCount + 1, 0);
        for (const BasicBlock& bb : context_.function) {
            const uint32_t nesting = depth[&bb];
            for (const SSABlockArgument& argument : bb.ssaArguments()) {
                if (argument.value != nullptr) depthOfValue[argument.value->valueId()] = nesting;
            }
            for (const IRBase& ir : bb) {
                const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
                if (instruction == nullptr || !instruction->hasAttachedSSA()) continue;
                for (size_t i = 0; i < instruction->getNumSSAResults(); ++i) {
                    const StinkySSAValue* value = instruction->getSSAResult(i);
                    if (value != nullptr) depthOfValue[value->valueId()] = nesting;
                }
            }
        }

        const SSAArena& arena = context_.function.ssaArena();
        for (Block& block : blocks_) {
            for (const Member& member : block.members) {
                const StinkySSAValue* value = arena.get(member.value);
                const double uses = value == nullptr ? 0.0 : static_cast<double>(value->useCount());
                double factor = 1.0;
                const uint32_t nesting = std::min(depthOfValue[member.value], kMaxLoopDepth);
                for (uint32_t level = 0; level < nesting; ++level) factor *= kLoopWeight;
                const SlotIndex length = rangeOf(member.value).length();
                block.weight += uses * factor / static_cast<double>(std::max<SlotIndex>(1, length));
            }
            block.hintBase = hintBaseOf(block);
            block.pinReason = pinReasonOf(block);
        }
        indexPreferences();
    }

    /// Hand every block the pairings naming one of its members.
    ///
    /// Built from the value side, so a preference is found from either end: the
    /// block being placed asks about it, and the partner may be anywhere.
    void indexPreferences() {
        const std::span<const Preference> preferences = context_.constraints.preferences();
        if (preferences.empty()) return;

        for (size_t index = 0; index < preferences.size(); ++index) {
            // A pairing that folding already granted has nothing left to
            // score. Skipping it keeps the block on the first-fit path with
            // its early exit. Without this, every folded block would scan the
            // whole register file only to learn it had already won.
            if (alreadyShared(preferences[index])) continue;
            for (const SSAValueID value : {preferences[index].a, preferences[index].b}) {
                if (value == kInvalidSSAValueID || value >= blockIndexOf_.size()) continue;
                const size_t block = blockIndexOf_[value];
                if (block == kNoBlock) continue;
                if (std::find(blocks_[block].preferences.begin(), blocks_[block].preferences.end(),
                              index) == blocks_[block].preferences.end())
                    blocks_[block].preferences.push_back(index);
            }
        }
    }

    /// Did folding already put both ends of \p preference on one register, by
    /// placing them in one block at one offset?
    bool alreadyShared(const Preference& preference) const {
        if (preference.a == kInvalidSSAValueID || preference.b == kInvalidSSAValueID) return false;
        if (preference.a >= blockIndexOf_.size() || preference.b >= blockIndexOf_.size())
            return false;
        const size_t index = blockIndexOf_[preference.a];
        if (index == kNoBlock || index != blockIndexOf_[preference.b]) return false;
        const std::optional<uint32_t> here = offsetOf(blocks_[index], preference.a);
        const std::optional<uint32_t> there = offsetOf(blocks_[index], preference.b);
        return here.has_value() && here == there;
    }

    /// One pairing this block can act on now. It records where the block's own
    /// end sits, and which register the other end already took.
    ///
    /// Only a pairing whose partner is already placed appears here. If the
    /// partner is still waiting there is no register to aim at, so the pairing
    /// is skipped rather than guessed. This is why scoring settles a chain
    /// fully when its members are placed in order, and only partly otherwise.
    struct OpenPairing {
        size_t index = 0;      ///< Into AllocationConstraints::preferences().
        uint32_t offset = 0;   ///< This block's member, inside this block.
        RegKey partner;        ///< Where the other end is.
        bool mineIsA = false;  ///< Which side of the pair is this block's.
    };

    /// Resolve the pairings \p block can act on, once per placement.
    ///
    /// None of this depends on the candidate base. Doing it inside the base
    /// loop would walk the block's members twice per pairing per base, which
    /// is quadratic in the width of the block and buys nothing.
    std::vector<OpenPairing> openPairings(const Block& block) const {
        std::vector<OpenPairing> open;
        for (const size_t index : block.preferences) {
            const Preference& preference = context_.constraints.preferences()[index];
            OpenPairing pairing;
            pairing.index = index;
            const std::optional<uint32_t> mine = offsetOf(block, preference.a);
            pairing.mineIsA = mine.has_value();
            const SSAValueID theirs = pairing.mineIsA ? preference.b : preference.a;
            const std::optional<uint32_t> here =
                pairing.mineIsA ? mine : offsetOf(block, preference.b);
            if (!here.has_value()) continue;
            pairing.offset = *here;

            if (theirs == kInvalidSSAValueID || theirs >= blockIndexOf_.size()) continue;
            const size_t partnerIndex = blockIndexOf_[theirs];
            if (partnerIndex == kNoBlock) continue;
            const Block& partner = blocks_[partnerIndex];
            if (!partner.placed) continue;
            const std::optional<uint32_t> there = offsetOf(partner, theirs);
            if (!there.has_value()) continue;

            pairing.partner = RegKey{partner.regClass, partner.base + *there, RegHalf::NONE};
            open.push_back(pairing);
        }
        return open;
    }

    /// What \p block gives up by starting at \p base: the architecture's own
    /// price for that base, plus every pairing in \p open it leaves unmet.
    ///
    /// The comparison uses member registers, not block bases. Two paired
    /// values can sit at different offsets inside their own blocks, so
    /// comparing the bases would relate the wrong pair of registers.
    double costAt(const Block& block, uint32_t base, std::span<const OpenPairing> open) const {
        double cost = context_.rules.baseCost(block.regClass, base, block.width);
        for (const OpenPairing& pairing : open) {
            const RegKey mine{block.regClass, base + pairing.offset, RegHalf::NONE};
            const RegKey a = pairing.mineIsA ? mine : pairing.partner;
            const RegKey b = pairing.mineIsA ? pairing.partner : mine;
            const Preference& preference = context_.constraints.preferences()[pairing.index];
            if (!context_.rules.satisfiedBy(preference, a, b)) cost += preference.benefit;
        }
        return cost;
    }

    /// The expensive half of a refusal message, or nothing when this attempt
    /// is going to be thrown away.
    ///
    /// Both summaries walk every base and every block. A run that folded
    /// something runs again without folding when it refuses, so nobody reads
    /// its message. The second run is the one the caller sees, and it folds
    /// nothing, so it builds the message in full.
    std::string refusalDetail(const Block& block) const {
        if (foldedAny_) return {};
        std::string detail = blockerSummary(block) + shapeSummary();
        // A rule narrowing the search is worth naming: without it the message
        // blames pressure for a base a rule ruled out.
        if (const AllocationRule* rule = rejectionFor(block).first; rule != nullptr)
            detail += " (rule " + std::string(rule->name) + " also forbids some bases)";
        return detail;
    }

    /// Tightest index any member of \p block may occupy, or nullopt when no
    /// member is limited.
    ///
    /// A block is placed as a unit, so one limited member limits all of them.
    /// That is why a capped block goes early: the bank it is confined to is
    /// shared with every block that packs low, and there is no second choice.
    std::optional<uint32_t> ceilingOf(const Block& block) const {
        std::optional<uint32_t> tightest;
        for (const Member& member : block.members) {
            const uint32_t ceiling = context_.constraints.maxIndexFor(member.value);
            if (ceiling == std::numeric_limits<uint32_t>::max()) continue;
            if (!tightest.has_value() || ceiling < *tightest) tightest = ceiling;
        }
        return tightest;
    }

    /// Why \p block cannot move, or null when it may.
    ///
    /// A pinned member fixes the whole block, since its members sit at fixed
    /// offsets from one another. Live-ins bind regardless of policy: relocating a
    /// register the dispatch filled changes what the kernel reads, so this is not
    /// something a compacting run may trade away for a lower high-water mark.
    const char* pinReasonOf(const Block& block) const {
        for (const Member& member : block.members) {
            if (context_.constraints.isPinned(member.value)) return "a function live-in";
        }
        for (const Member& member : block.members) {
            if (const char* reason = context_.scope.immobileReason(member.value)) return reason;
        }
        return nullptr;
    }

    /// Base implied by the members' original registers, when they all agree.
    std::optional<uint32_t> hintBaseOf(const Block& block) const {
        std::optional<uint32_t> base;
        for (const Member& member : block.members) {
            const std::optional<RegKey> hint = context_.constraints.hintFor(member.value);
            if (!hint.has_value() || hint->type != block.regClass) return std::nullopt;
            if (hint->idx < member.offset) return std::nullopt;
            const uint32_t candidate = hint->idx - member.offset;
            if (base.has_value() && *base != candidate) return std::nullopt;
            base = candidate;
        }
        return base;
    }

    /// Every register this block would use at \p base exists and may be handed
    /// out, ignoring who currently holds it. Separate from availableAt() so a
    /// refusal can say whether the register is off limits or merely occupied.
    bool reachableAt(const Block& block, uint32_t base) const {
        if (base + block.width > context_.target.indexCount(block.regClass)) return false;
        // The single funnel every candidate base passes through, so one call
        // here covers placement, eviction and hint-following at once. Only
        // Active rules can answer, so there is no status check to get wrong.
        if (context_.rules.forbidsBase(block.regClass, base, block.width) != nullptr) return false;
        for (const Member& member : block.members) {
            const uint32_t idx = base + member.offset;
            // Per member, so a multi-DWORD operand cannot half-fit.
            if (idx > context_.constraints.maxIndexFor(member.value)) return false;
            if (!context_.target.isAllocatable(block.regClass, idx)) return false;
            if (!mayOccupy(block.regClass, idx, member.value)) return false;
        }
        return true;
    }

    /// A held register takes no newcomers: only the value lifted from it may sit
    /// there. Asked from reachableAt so placement, eviction and hint-following
    /// all inherit it, the same reason forbidsBase lives there.
    bool mayOccupy(RegType regClass, uint32_t idx, SSAValueID value) const {
        if (!context_.scope.isPinnedRegister(regClass, idx)) return true;
        const std::optional<RegKey> hint = context_.constraints.hintFor(value);
        return hint.has_value() && hint->type == regClass && hint->idx == idx;
    }

    bool availableAt(const Block& block, uint32_t base) const {
        if (!reachableAt(block, base)) return false;
        for (const Member& member : block.members) {
            if (!matrix_.available(block.regClass, base + member.offset, rangeOf(member.value)))
                return false;
        }
        return true;
    }

    void bindAt(Block& block, uint32_t base) {
        for (const Member& member : block.members)
            matrix_.bind(block.regClass, base + member.offset, member.value, rangeOf(member.value));
        block.base = base;
        block.placed = true;
    }

    void unbind(Block& block) {
        for (const Member& member : block.members)
            matrix_.unbind(block.regClass, block.base + member.offset, member.value);
        block.placed = false;
    }

    /// Blocks already holding any register this one needs at \p base.
    std::vector<size_t> occupantsAt(const Block& block, uint32_t base) const {
        std::vector<SSAValueID> conflicts;
        for (const Member& member : block.members) {
            matrix_.collectConflicts(block.regClass, base + member.offset, rangeOf(member.value),
                                     conflicts);
        }
        std::vector<size_t> occupants;
        for (SSAValueID value : conflicts) {
            const size_t index = blockIndexOf_[value];
            if (std::find(occupants.begin(), occupants.end(), index) == occupants.end())
                occupants.push_back(index);
        }
        return occupants;
    }

    bool place() {
        matrix_ = PhysRegMatrix(context_.target);

        // Pinned first: a freely placed block has to see the registers it cannot
        // move as already taken.
        for (Block& block : blocks_) {
            if (block.pinReason == nullptr) continue;
            const std::string unmoved = valueName(block.leader) + " is " + block.pinReason +
                                        ", so it must keep its original register, but ";
            if (!block.hintBase.has_value()) {
                error_ = unmoved + "it has none recorded";
                return false;
            }
            if (!availableAt(block, *block.hintBase)) {
                error_ = unmoved + regTypeToString(block.regClass) +
                         std::to_string(*block.hintBase) +
                         (reachableAt(block, *block.hintBase) ? " is already taken"
                                                              : " is not allocatable");
                return false;
            }
            bindAt(block, *block.hintBase);
        }

        // Capped blocks next, before anything free to go elsewhere.
        //
        // pickBase scans upward, so every block prefers a low base. Left in the
        // weight-ordered queue, a capped block competes for its one reachable
        // bank against blocks that could have gone anywhere, and under pressure
        // finds it full. Order is the fix here, not the constraint.
        std::vector<size_t> capped;
        for (size_t index = 0; index < blocks_.size(); ++index) {
            Block& block = blocks_[index];
            if (block.pinReason != nullptr || !policy_.placesEarly(block)) continue;
            capped.push_back(index);
        }
        std::sort(capped.begin(), capped.end(), [this](size_t a, size_t b) {
            return policy_.placesBefore(blocks_[a], blocks_[b]);
        });
        for (size_t index : capped) {
            Block& block = blocks_[index];
            if (tryPlace(block)) continue;
            // Distinct from the pressure refusal below: nothing has been placed
            // here except pinned blocks, so the reachable bank really is full
            // rather than merely full by the time this block was reached.
            error_ = "no " + regTypeToString(block.regClass) + " register at or below index " +
                     std::to_string(*block.maxIndex) + " is free for " + valueName(block.leader) +
                     ", which its operands cannot address past (" +
                     std::to_string(block.placementFreedom) + " legal base(s) before occupancy)" +
                     refusalDetail(block);
            return false;
        }

        std::vector<size_t> queue;
        for (size_t index = 0; index < blocks_.size(); ++index) {
            if (blocks_[index].pinReason == nullptr && !policy_.placesEarly(blocks_[index]))
                queue.push_back(index);
        }
        std::sort(queue.begin(), queue.end(), [this](size_t a, size_t b) {
            return policy_.placesBefore(blocks_[a], blocks_[b]);
        });

        uint32_t evictionBudget = static_cast<uint32_t>(blocks_.size()) * kMaxEvictionsPerBlock + 1;
        std::vector<size_t> worklist(queue.rbegin(), queue.rend());
        while (!worklist.empty()) {
            const size_t index = worklist.back();
            worklist.pop_back();
            Block& block = blocks_[index];
            if (block.placed) continue;

            if (tryPlace(block)) continue;

            const std::optional<uint32_t> evictBase = findEvictableBase(block);
            if (!evictBase.has_value() || evictionBudget == 0) {
                error_ = "no " + regTypeToString(block.regClass) + " register is free for " +
                         valueName(block.leader) +
                         (block.width > 1 ? " and the " + std::to_string(block.width - 1) +
                                                " register(s) tied to it"
                                          : "") +
                         "; splitting and spilling are not implemented" +
                         // How many bases it ever had, so a reader can tell a
                         // full file from a shape with almost nowhere to go.
                         " (" + std::to_string(block.placementFreedom) +
                         " legal base(s) before occupancy)" + refusalDetail(block);
                return false;
            }

            for (size_t occupant : occupantsAt(block, *evictBase)) {
                Block& evicted = blocks_[occupant];
                unbind(evicted);
                ++evicted.evictions;
                --evictionBudget;
                worklist.push_back(occupant);
            }
            bindAt(block, *evictBase);
        }
        return true;
    }

    /// Hint first, so a colouring that has room lands where the producer had it
    /// and a simple function reproduces the legacy assignment exactly.
    ///
    /// Skipping the hint is what a compacting run does. The producer's numbering
    /// is legal by construction, so with hints on, every block finds its hint free
    /// and first-fit below is never reached - which is also why a hint-following
    /// run can never lower the high-water mark.
    bool tryPlace(Block& block) {
        // Taking the hint skips scoring, so a preference cannot pull a block
        // off the register it was lifted from. That is deliberate. The hint is
        // the producer's own answer, and a run asked to reproduce it should
        // keep it. A compacting run passes no hints, so it sees every
        // preference.
        if (followHints_ && block.hintBase.has_value() && availableAt(block, *block.hintBase)) {
            bindAt(block, *block.hintBase);
            return true;
        }
        const std::optional<uint32_t> base =
            pickBase(block, [&](uint32_t candidate) { return availableAt(block, candidate); });
        if (!base.has_value()) return false;
        bindAt(block, *base);
        return true;
    }

    /// A base whose occupants are all lighter than \p block and have not already
    /// been evicted too often. Weight strictly decreasing is what stops two
    /// blocks trading the same register forever.
    std::optional<uint32_t> findEvictableBase(const Block& block) const {
        return pickBase(block, [&](uint32_t base) { return evictableAt(block, base); });
    }

    bool evictableAt(const Block& block, uint32_t base) const {
        if (!reachableAt(block, base)) return false;
        const std::vector<size_t> occupants = occupantsAt(block, base);
        if (occupants.empty()) return false;  // tryPlace already refused this base
        for (size_t occupant : occupants) {
            const Block& other = blocks_[occupant];
            // Pinning and the eviction cap are invariants rather than policy, so
            // they are decided here. Whether this block has the stronger claim
            // is the policy's question.
            if (other.pinReason != nullptr || other.evictions >= kMaxEvictionsPerBlock)
                return false;
            if (!policy_.mayEvict(block, other)) return false;
        }
        return true;
    }

    /// The base to take among those \p acceptable admits.
    ///
    /// Placement and eviction share this on purpose. A preference honoured in
    /// one and ignored in the other is worse than no preference at all: the
    /// allocator would spend an eviction to reach a base it was told to avoid.
    ///
    /// Without an Active preference this is plain ascending first-fit, early
    /// exit included, so a chip with no preference keeps exactly today's
    /// colouring. With one it is the cheapest acceptable base, ties going to the
    /// lower index so the result stays deterministic.
    template <typename Acceptable>
    std::optional<uint32_t> pickBase(const Block& block, Acceptable acceptable) const {
        const uint32_t indexes = context_.target.indexCount(block.regClass);
        const std::vector<OpenPairing> open =
            block.preferences.empty() ? std::vector<OpenPairing>{} : openPairings(block);
        // Asked per block, not per table. A chip can declare a pairing rule
        // while most of its blocks are named by none of it, and those keep
        // first-fit and the colouring they had before any of this existed. A
        // block whose partners are all still unplaced scores the same at every
        // base, so it takes this path too instead of scanning to find out.
        if (!context_.rules.prices() && open.empty()) {
            for (uint32_t base = 0; base + block.width <= indexes; ++base) {
                if (acceptable(base)) return base;
            }
            return std::nullopt;
        }
        std::optional<uint32_t> best;
        double bestCost = 0.0;
        for (uint32_t base = 0; base + block.width <= indexes; ++base) {
            // Scoring only sees bases that placement had already accepted. So
            // a preference can change which colouring comes out, and never
            // whether one comes out at all.
            if (!acceptable(base)) continue;
            const double cost = costAt(block, base, open);
            if (!best.has_value() || cost < bestCost) {
                best = base;
                bestCost = cost;
                // Costs are penalties and never negative, so zero is the best
                // any base can do. Stopping here saves a block with a
                // satisfiable pairing from scanning the rest of the file only
                // to confirm what it already found.
                if (bestCost <= 0.0) break;
            }
        }
        return best;
    }

    static constexpr size_t kNoBlock = static_cast<size_t>(-1);

    const AllocationContext& context_;
    PhysRegMatrix matrix_{context_.target};
    bool followHints_ = true;
    const PlacementPolicy& policy_;
    bool mayFoldPairs_ = false;
    bool foldedAny_ = false;
    std::vector<Block> blocks_;
    std::vector<size_t> blockIndexOf_;
    std::string error_;
};

/// Heaviest first, with capped blocks taken ahead of the queue.
///
/// The early phase exists because a capped block is confined to one bank that
/// every low-packing block competes for, and weight order alone reaches it too
/// late. freedomPolicy dissolves the phase by ordering on that fact directly.
bool weightPlacesEarly(const Block& block) {
    return block.maxIndex.has_value();
}

bool weightPlacesBefore(const Block& lhs, const Block& rhs) {
    if (lhs.weight != rhs.weight) return lhs.weight > rhs.weight;
    return lhs.leader < rhs.leader;
}

bool weightMayEvict(const Block& evictor, const Block& occupant) {
    // A capped block is as immovable as a pinned one here. Evicting it returns
    // it to the worklist, where it is re-placed after the queue has filled its
    // bank -- the ordering failure, reintroduced one block at a time.
    if (occupant.maxIndex.has_value()) return false;
    return occupant.weight < evictor.weight;
}

}  // namespace

Expected<AllocationResult> runGreedyPlacement(const AllocationContext& context, bool followHints,
                                              const PlacementPolicy& policy) {
    // No folding while following hints. That mode exists to reproduce the
    // producer's numbering. Folding two blocks the producer kept apart leaves
    // them with no hint they agree on, so the run would stop reproducing
    // anything, which is the one thing it is for.
    Greedy greedy(context, followHints, policy, /*mayFoldPairs=*/!followHints);
    Expected<AllocationResult> result = greedy.run();
    if (result.hasValue() || !greedy.foldedAny()) return result;

    // Folding asks placement for one run of registers where two shorter runs
    // would have done. If that is what refused the colouring, the preference
    // gives way. A soft rule may not decide whether a kernel colours.
    return Greedy(context, followHints, policy, /*mayFoldPairs=*/false).run();
}

const PlacementPolicy& weightPolicy() {
    static const PlacementPolicy policy{"weight", weightPlacesEarly, weightPlacesBefore,
                                        weightMayEvict};
    return policy;
}

const char* GreedyAllocator::name() const {
    return "greedy";
}

AllocatorCapabilities GreedyAllocator::capabilities() const {
    // Affinity sets keep a merge and its inputs on one register, and no range is
    // ever split, so lowering needs neither copy insertion nor scratch.
    return {};
}

Expected<AllocationResult> GreedyAllocator::allocate(const AllocationContext& context) {
    return runGreedyPlacement(context, /*followHints=*/true, weightPolicy());
}

const char* CompactingGreedyAllocator::name() const {
    return "greedy-compact";
}

AllocatorCapabilities CompactingGreedyAllocator::capabilities() const {
    return {};
}

Expected<AllocationResult> CompactingGreedyAllocator::allocate(const AllocationContext& context) {
    return runGreedyPlacement(context, /*followHints=*/false, weightPolicy());
}

namespace {
struct GreedyRegistrar {
    GreedyRegistrar() {
        AllocatorRegistry::registerAllocator("greedy",
                                             [] { return std::make_unique<GreedyAllocator>(); });
        AllocatorRegistry::registerAllocator(
            "greedy-compact", [] { return std::make_unique<CompactingGreedyAllocator>(); });
    }
};
static GreedyRegistrar s_greedyRegistrar;
}  // namespace

void anchorGreedyAllocator() {}  // NOLINT(misc-use-internal-linkage)

}  // namespace stinkytofu
