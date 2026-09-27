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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

struct Member {
    SSAValueID value = kInvalidSSAValueID;
    uint32_t offset = 0;
};

/// One block of values placed as a unit, each at its own fixed offset.
struct Block {
    RegType regClass = RegType::UNKNOWN;
    std::vector<Member> members;
    uint32_t width = 1;
    double weight = 0.0;
    /// Smallest member ID, so ordering is stable when weights tie.
    SSAValueID leader = kInvalidSSAValueID;
    /// Base the original registers imply, tried before any first-fit candidate.
    std::optional<uint32_t> hintBase;
    /// Must land on hintBase rather than prefer it. Null when free to move;
    /// otherwise the reason, for the diagnostic if that register is unusable.
    const char* pinReason = nullptr;
    uint32_t evictions = 0;
    bool placed = false;
    uint32_t base = 0;
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
    Greedy(const AllocationContext& context, bool followHints)
        : context_(context), followHints_(followHints) {}

    Expected<AllocationResult> run() {
        if (context_.function.ssaArena().valueCount() == 0)
            return AllocationResult(context_.function);

        if (!buildBlocks()) return fail(error_);
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

            if (!checkFeasible(block)) return false;
        }
        return true;
    }

    /// What placement rules have to say about \p block, for diagnostics.
    ///
    /// Both facts come from one scan because both callers want the same walk:
    /// checkFeasible needs to know a block can never be placed, and place()
    /// needs to name a rule that narrowed the search rather than blame pressure.
    struct BaseRejection {
        const AllocationRule* first = nullptr;  ///< first rule to forbid any base
        bool everyBase = false;                 ///< no base survives at all
    };

    /// Only interesting when a rule is in play, so the common no-rules case does
    /// not pay for the scan.
    BaseRejection rejectionFor(const Block& block) const {
        BaseRejection found;
        if (context_.rules.empty()) return found;

        const uint32_t indexes = context_.target.indexCount(block.regClass);
        found.everyBase = true;
        for (uint32_t base = 0; base + block.width <= indexes; ++base) {
            const AllocationRule* rule =
                context_.rules.forbidsBase(block.regClass, base, block.width);
            if (rule == nullptr) {
                found.everyBase = false;
            } else if (found.first == nullptr) {
                found.first = rule;
            }
        }
        return found;
    }

    /// Two members can legitimately share an offset: overlapping tuple runs force
    /// a value written before a partial overwrite onto the same register as the
    /// value that replaces it. That is only sound while their ranges are disjoint.
    bool checkFeasible(const Block& block) {
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
        // A block no base can ever satisfy should name the rule rather than
        // exhaust every base and report the generic "no register is free".
        if (const BaseRejection rejected = rejectionFor(block); rejected.everyBase) {
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

        std::vector<size_t> queue;
        for (size_t index = 0; index < blocks_.size(); ++index) {
            if (blocks_[index].pinReason == nullptr) queue.push_back(index);
        }
        std::sort(queue.begin(), queue.end(), [this](size_t a, size_t b) {
            const Block& lhs = blocks_[a];
            const Block& rhs = blocks_[b];
            if (lhs.weight != rhs.weight) return lhs.weight > rhs.weight;
            return lhs.leader < rhs.leader;
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
                         "; splitting and spilling are not implemented";
                // A rule narrowing the search is worth naming here: without it
                // the message blames pressure for a base a rule ruled out.
                if (const AllocationRule* rule = rejectionFor(block).first; rule != nullptr) {
                    error_ += " (rule " + std::string(rule->name) + " also forbids some bases)";
                }
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
            if (other.pinReason != nullptr || other.weight >= block.weight ||
                other.evictions >= kMaxEvictionsPerBlock)
                return false;
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
        if (!context_.rules.prices()) {
            for (uint32_t base = 0; base + block.width <= indexes; ++base) {
                if (acceptable(base)) return base;
            }
            return std::nullopt;
        }
        std::optional<uint32_t> best;
        double bestCost = 0.0;
        for (uint32_t base = 0; base + block.width <= indexes; ++base) {
            if (!acceptable(base)) continue;
            const double cost = context_.rules.baseCost(block.regClass, base, block.width);
            if (!best.has_value() || cost < bestCost) {
                best = base;
                bestCost = cost;
            }
        }
        return best;
    }

    static constexpr size_t kNoBlock = static_cast<size_t>(-1);

    const AllocationContext& context_;
    PhysRegMatrix matrix_{context_.target};
    bool followHints_ = true;
    std::vector<Block> blocks_;
    std::vector<size_t> blockIndexOf_;
    std::string error_;
};

}  // namespace

const char* GreedyAllocator::name() const {
    return "greedy";
}

AllocatorCapabilities GreedyAllocator::capabilities() const {
    // Affinity sets keep a merge and its inputs on one register, and no range is
    // ever split, so lowering needs neither copy insertion nor scratch.
    return {};
}

Expected<AllocationResult> GreedyAllocator::allocate(const AllocationContext& context) {
    return Greedy(context, /*followHints=*/true).run();
}

const char* CompactingGreedyAllocator::name() const {
    return "greedy-compact";
}

AllocatorCapabilities CompactingGreedyAllocator::capabilities() const {
    return {};
}

Expected<AllocationResult> CompactingGreedyAllocator::allocate(const AllocationContext& context) {
    return Greedy(context, /*followHints=*/false).run();
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
