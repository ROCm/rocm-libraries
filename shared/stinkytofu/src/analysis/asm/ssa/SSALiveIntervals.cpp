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
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

constexpr SlotIndex kInvalid = SSASlotIndexes::kInvalidSlot;

/// Index plus the half of the step it names: 'u' where operands are read, 'd'
/// where results are written. Slots alternate, so the low bit decides.
std::string slotToString(SlotIndex slot) {
    if (slot == kInvalid) return "invalid";
    return std::to_string(slot) + (slot % 2 == 0 ? 'u' : 'd');
}

/// Dense set of SSA value IDs. Liveness runs over these rather than pointer
/// sets: value IDs are dense and 1-based, and a GEMM kernel has enough values
/// that repeated hashing would dominate the analysis.
class ValueSet {
   public:
    explicit ValueSet(size_t bits) : words_((bits + 63) / 64, 0ull) {}

    bool test(uint32_t bit) const {
        return ((words_[bit >> 6] >> (bit & 63)) & 1ull) != 0ull;
    }

    void set(uint32_t bit) {
        words_[bit >> 6] |= 1ull << (bit & 63);
    }

    /// this |= other. Returns true when a bit was added.
    bool orInto(const ValueSet& other) {
        bool changed = false;
        for (size_t word = 0; word < words_.size(); ++word) {
            const uint64_t merged = words_[word] | other.words_[word];
            changed = changed || merged != words_[word];
            words_[word] = merged;
        }
        return changed;
    }

    /// this |= (other & ~mask). Returns true when a bit was added.
    bool orExceptInto(const ValueSet& other, const ValueSet& mask) {
        bool changed = false;
        for (size_t word = 0; word < words_.size(); ++word) {
            const uint64_t merged = words_[word] | (other.words_[word] & ~mask.words_[word]);
            changed = changed || merged != words_[word];
            words_[word] = merged;
        }
        return changed;
    }

    template <typename Fn>
    void forEachSet(Fn&& fn) const {
        for (size_t word = 0; word < words_.size(); ++word) {
            uint64_t bits = words_[word];
            while (bits != 0ull) {
                const uint32_t bit = static_cast<uint32_t>(word * 64 + __builtin_ctzll(bits));
                fn(bit);
                bits &= bits - 1;
            }
        }
    }

   private:
    std::vector<uint64_t> words_;
};

/// Post-order block sequence: a block is visited after its successors, which is
/// the order a backward dataflow wants. Unreachable blocks are appended so a
/// hand-built function still gets ranges.
std::vector<const BasicBlock*> postOrderBlocks(const Function& function) {
    std::vector<const BasicBlock*> order;
    std::unordered_set<const BasicBlock*> visited;

    const BasicBlock* entry = function.getEntryBlock();
    if (entry != nullptr) {
        // Explicit stack: a long straight-line kernel is as deep as its block count.
        std::vector<std::pair<const BasicBlock*, size_t>> stack;
        stack.emplace_back(entry, 0);
        visited.insert(entry);
        while (!stack.empty()) {
            const BasicBlock* block = stack.back().first;
            const std::vector<BasicBlock*>& successors = block->getSuccessors();
            if (stack.back().second < successors.size()) {
                const BasicBlock* successor = successors[stack.back().second++];
                if (visited.insert(successor).second) stack.emplace_back(successor, 0);
                continue;
            }
            order.push_back(block);
            stack.pop_back();
        }
    }

    for (const BasicBlock& block : function) {
        if (!visited.contains(&block)) order.push_back(&block);
    }
    return order;
}

/// Post-order from a reverse-post-order the caller already computed.
std::vector<const BasicBlock*> postOrderBlocks(const Function& function,
                                               const DominanceInfo& dominance) {
    std::vector<const BasicBlock*> order;
    order.reserve(dominance.rpo.size());
    for (auto it = dominance.rpo.rbegin(); it != dominance.rpo.rend(); ++it) order.push_back(*it);

    for (const BasicBlock& block : function) {
        if (!dominance.rpoIndex.contains(&block)) order.push_back(&block);
    }
    return order;
}

/// Where each value is defined, and what class it belongs to.
struct DefinitionMap {
    std::vector<const BasicBlock*> block;
    std::vector<SlotIndex> point;
    std::vector<RegType> regClass;
    std::vector<uint16_t> width;
};

DefinitionMap buildDefinitionMap(const Function& function, const SSASlotIndexes& slots,
                                 size_t valueCount) {
    DefinitionMap defs;
    defs.block.assign(valueCount + 1, nullptr);
    defs.point.assign(valueCount + 1, kInvalid);
    defs.regClass.assign(valueCount + 1, RegType::UNKNOWN);
    defs.width.assign(valueCount + 1, 1);

    const SSAArena& arena = function.ssaArena();
    for (StinkySSAValue* value : arena.values()) {
        if (value == nullptr) continue;
        const uint32_t id = value->valueId();
        if (id == kInvalidSSAValueID || id > valueCount) continue;
        defs.regClass[id] = value->type().regType;
        defs.width[id] = value->type().dwordWidth == 0 ? 1 : value->type().dwordWidth;
    }

    for (const BasicBlock& block : function) {
        for (const SSABlockArgument& argument : block.ssaArguments()) {
            if (argument.value == nullptr) continue;
            const uint32_t id = argument.value->valueId();
            if (id == kInvalidSSAValueID || id > valueCount) continue;
            defs.block[id] = &block;
            defs.point[id] = slots.blockArgDef(&block);
        }
        for (const IRBase& ir : block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr || !instruction->hasAttachedSSA()) continue;
            for (size_t result = 0; result < instruction->getNumSSAResults(); ++result) {
                const StinkySSAValue* value = instruction->getSSAResult(result);
                if (value == nullptr) continue;
                const uint32_t id = value->valueId();
                if (id == kInvalidSSAValueID || id > valueCount) continue;
                defs.block[id] = &block;
                defs.point[id] = slots.defSlot(instruction);
            }
        }
    }
    return defs;
}

}  // namespace

// ---------------------------------------------------------------------------
// LiveRange
// ---------------------------------------------------------------------------

SlotIndex LiveRange::start() const {
    return segments_.empty() ? kInvalid : segments_.front().start;
}

SlotIndex LiveRange::end() const {
    return segments_.empty() ? kInvalid : segments_.back().end;
}

SlotIndex LiveRange::length() const {
    SlotIndex total = 0;
    for (const LiveSegment& segment : segments_) total += segment.end - segment.start;
    return total;
}

bool LiveRange::covers(SlotIndex point) const {
    for (const LiveSegment& segment : segments_) {
        if (segment.start > point) return false;
        if (segment.covers(point)) return true;
    }
    return false;
}

bool LiveRange::overlaps(const LiveRange& other) const {
    size_t lhs = 0;
    size_t rhs = 0;
    while (lhs < segments_.size() && rhs < other.segments_.size()) {
        const LiveSegment& a = segments_[lhs];
        const LiveSegment& b = other.segments_[rhs];
        if (a.start < b.end && b.start < a.end) return true;
        if (a.end <= b.end) {
            ++lhs;
        } else {
            ++rhs;
        }
    }
    return false;
}

void LiveRange::addSegment(SlotIndex start, SlotIndex end) {
    if (end <= start) return;
    segments_.push_back({start, end});
}

void LiveRange::finalize() {
    if (segments_.size() < 2) return;
    std::sort(segments_.begin(), segments_.end(), [](const LiveSegment& a, const LiveSegment& b) {
        return a.start < b.start || (a.start == b.start && a.end < b.end);
    });

    std::vector<LiveSegment> merged;
    merged.reserve(segments_.size());
    for (const LiveSegment& segment : segments_) {
        // Adjacent segments are merged too: a value live out of one block and
        // live into the next occupies numerically contiguous points.
        if (!merged.empty() && segment.start <= merged.back().end) {
            merged.back().end = std::max(merged.back().end, segment.end);
            continue;
        }
        merged.push_back(segment);
    }
    segments_ = std::move(merged);
}

std::string LiveRange::toString() const {
    if (segments_.empty()) return "-";
    std::string text;
    for (const LiveSegment& segment : segments_) {
        if (!text.empty()) text += " ";
        text += "[" + slotToString(segment.start) + "," + slotToString(segment.end) + ")";
    }
    return text;
}

// ---------------------------------------------------------------------------
// SSALiveIntervals
// ---------------------------------------------------------------------------

const LiveRange& SSALiveIntervals::rangeOf(SSAValueID id) const {
    static const LiveRange kEmpty;
    if (id == kInvalidSSAValueID || id >= byValue_.size()) return kEmpty;
    return byValue_[id];
}

bool SSALiveIntervals::overlap(SSAValueID a, SSAValueID b) const {
    if (a == b) return !rangeOf(a).empty();
    return rangeOf(a).overlaps(rangeOf(b));
}

uint32_t SSALiveIntervals::peakPressure(RegType regClass) const {
    auto it = peakByClass_.find(regClass);
    return it == peakByClass_.end() ? 0u : it->second;
}

std::string SSALiveIntervals::toString() const {
    std::string text = "slots=" + std::to_string(slots_.slotCount()) +
                       " values=" + std::to_string(valueCount()) + "\n";
    for (size_t id = 1; id < byValue_.size(); ++id) {
        text += "%" + std::to_string(id) + ":" + regTypeToString(classByValue_[id]) + " " +
                byValue_[id].toString() + "\n";
    }
    for (const auto& [regClass, peak] : peakByClass_) {
        text += "peak " + regTypeToString(regClass) + "=" + std::to_string(peak) + "\n";
    }
    return text;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Assembles the result. A struct rather than a free function so one friend
/// declaration covers construction.
struct SSALiveIntervalsBuilder {
    static SSALiveIntervals build(const Function& function,
                                  const std::vector<const BasicBlock*>& blockOrder);
};

SSALiveIntervals SSALiveIntervalsBuilder::build(const Function& function,
                                                const std::vector<const BasicBlock*>& blockOrder) {
    SSALiveIntervals intervals;
    if (!function.hasAttachedSSA()) return intervals;

    const size_t valueCount = function.ssaArena().valueCount();
    if (valueCount == 0) return intervals;

    const SSASlotIndexes slots = computeSSASlotIndexes(function);
    const DefinitionMap defs = buildDefinitionMap(function, slots, valueCount);

    const size_t bits = valueCount + 1;
    const size_t blockCount = blockOrder.size();
    std::unordered_map<const BasicBlock*, size_t> indexOf;
    indexOf.reserve(blockCount);
    for (size_t i = 0; i < blockCount; ++i) indexOf.emplace(blockOrder[i], i);

    std::vector<ValueSet> defined;
    std::vector<ValueSet> liveIn;
    std::vector<ValueSet> liveOut;
    defined.reserve(blockCount);
    liveIn.reserve(blockCount);
    liveOut.reserve(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        defined.emplace_back(bits);
        liveIn.emplace_back(bits);   // seeded with upward-exposed uses
        liveOut.emplace_back(bits);  // seeded with uses on outgoing edges
    }

    // Local defs, and uses whose definition is in another block.
    for (size_t i = 0; i < blockCount; ++i) {
        const BasicBlock* block = blockOrder[i];
        for (const SSABlockArgument& argument : block->ssaArguments()) {
            if (argument.value != nullptr) defined[i].set(argument.value->valueId());
        }
        for (const IRBase& ir : *block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr || !instruction->hasAttachedSSA()) continue;
            for (size_t operand = 0; operand < instruction->getNumSSAOperands(); ++operand) {
                const StinkySSAValue* value = instruction->getSSAOperandValue(operand);
                if (value == nullptr) continue;
                const uint32_t id = value->valueId();
                if (id == kInvalidSSAValueID || id > valueCount) continue;
                if (defs.block[id] != block) liveIn[i].set(id);
            }
            for (size_t result = 0; result < instruction->getNumSSAResults(); ++result) {
                const StinkySSAValue* value = instruction->getSSAResult(result);
                if (value != nullptr) defined[i].set(value->valueId());
            }
        }
    }

    // A block argument's incoming value is consumed on the predecessor edge, so
    // it is live out of that predecessor and not live inside the join. This is
    // what lets a merge share a register with its inputs.
    for (const BasicBlock& block : function) {
        for (const SSABlockArgument& argument : block.ssaArguments()) {
            for (const SSABlockIncoming& incoming : argument.incoming) {
                if (incoming.use == nullptr) continue;
                const StinkySSAValue* value = incoming.use->value();
                if (value == nullptr) continue;
                const uint32_t id = value->valueId();
                if (id == kInvalidSSAValueID || id > valueCount) continue;
                auto it = indexOf.find(incoming.predecessor);
                if (it == indexOf.end()) continue;
                liveOut[it->second].set(id);
                if (defs.block[id] != incoming.predecessor) liveIn[it->second].set(id);
            }
        }
    }

    // liveOut[B] = union of liveIn[successors] (plus the edge uses seeded above)
    // liveIn[B]  = local uses + (liveOut[B] - defs[B])
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < blockCount; ++i) {
            const BasicBlock* block = blockOrder[i];
            for (const BasicBlock* successor : block->getSuccessors()) {
                auto it = indexOf.find(successor);
                if (it == indexOf.end()) continue;
                changed = liveOut[i].orInto(liveIn[it->second]) || changed;
            }
            changed = liveIn[i].orExceptInto(liveOut[i], defined[i]) || changed;
        }
    }

    intervals.byValue_.assign(valueCount + 1, LiveRange{});
    intervals.classByValue_ = defs.regClass;

    // One segment per block a value is live in. Points inside a block are
    // contiguous, so a block contributes at most one segment.
    std::vector<SlotIndex> firstTouch(valueCount + 1, kInvalid);
    std::vector<SlotIndex> lastTouch(valueCount + 1, kInvalid);
    std::vector<uint32_t> touched;

    for (size_t i = 0; i < blockCount; ++i) {
        const BasicBlock* block = blockOrder[i];
        const SlotIndex blockStart = slots.blockStart(block);
        const SlotIndex blockEnd = slots.blockEnd(block);
        if (blockStart == kInvalid || blockEnd == kInvalid) continue;

        touched.clear();
        auto touch = [&](uint32_t id, SlotIndex point) {
            if (firstTouch[id] == kInvalid) {
                touched.push_back(id);
                firstTouch[id] = point;
                lastTouch[id] = point;
                return;
            }
            firstTouch[id] = std::min(firstTouch[id], point);
            lastTouch[id] = std::max(lastTouch[id], point);
        };

        for (const SSABlockArgument& argument : block->ssaArguments()) {
            if (argument.value != nullptr)
                touch(argument.value->valueId(), slots.blockArgDef(block));
        }
        for (const IRBase& ir : *block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr || !instruction->hasAttachedSSA()) continue;
            const SlotIndex useSlot = slots.useSlot(instruction);
            const SlotIndex defSlot = slots.defSlot(instruction);
            for (size_t operand = 0; operand < instruction->getNumSSAOperands(); ++operand) {
                const StinkySSAValue* value = instruction->getSSAOperandValue(operand);
                if (value != nullptr) touch(value->valueId(), useSlot);
            }
            for (size_t result = 0; result < instruction->getNumSSAResults(); ++result) {
                const StinkySSAValue* value = instruction->getSSAResult(result);
                if (value != nullptr) touch(value->valueId(), defSlot);
            }
        }

        auto emit = [&](uint32_t id) {
            const bool isLiveIn = liveIn[i].test(id);
            SlotIndex start = kInvalid;
            if (isLiveIn) {
                start = blockStart;
            } else {
                start = defs.point[id];
                if (start == kInvalid || start < blockStart || start >= blockEnd)
                    start = firstTouch[id];
            }
            if (start == kInvalid) return;

            SlotIndex end = blockEnd;
            if (!liveOut[i].test(id)) {
                // Dies here: a use at its instruction's use point keeps the
                // value up to, but not including, that instruction's def point.
                end = lastTouch[id] == kInvalid ? start + 1 : lastTouch[id] + 1;
            }
            intervals.byValue_[id].addSegment(start, end);
        };

        for (uint32_t id : touched) emit(id);
        liveIn[i].forEachSet([&](uint32_t id) {
            if (id == kInvalidSSAValueID || id > valueCount) return;
            if (firstTouch[id] != kInvalid) return;  // already emitted
            emit(id);
        });

        for (uint32_t id : touched) {
            firstTouch[id] = kInvalid;
            lastTouch[id] = kInvalid;
        }
    }

    for (LiveRange& range : intervals.byValue_) range.finalize();

    intervals.widthByValue_ = defs.width;
    intervals.slots_ = slots;
    intervals.recomputePeakPressure();
    return intervals;
}

void SSALiveIntervals::recomputePeakPressure() {
    peakByClass_.clear();
    if (slots_.slotCount() == 0) return;

    // Peak pressure per class, in DWORDs, from the segments themselves, so no
    // separate pressure analysis exists to disagree with them.
    std::map<RegType, std::vector<int32_t>> deltas;
    for (size_t id = 1; id < byValue_.size(); ++id) {
        const LiveRange& range = byValue_[id];
        if (range.empty()) continue;
        const RegType regClass = id < classByValue_.size() ? classByValue_[id] : RegType::UNKNOWN;
        const int32_t width = id < widthByValue_.size() ? widthByValue_[id] : 1;
        std::vector<int32_t>& delta = deltas[regClass];
        if (delta.empty()) delta.assign(slots_.slotCount() + 1, 0);
        for (const LiveSegment& segment : range.segments()) {
            delta[segment.start] += width;
            delta[segment.end] -= width;
        }
    }
    for (const auto& [regClass, delta] : deltas) {
        int32_t live = 0;
        int32_t peak = 0;
        for (int32_t step : delta) {
            live += step;
            peak = std::max(peak, live);
        }
        peakByClass_[regClass] = static_cast<uint32_t>(peak);
    }
}

SSALiveIntervals SSALiveIntervals::withEarlierStarts(
    const SSALiveIntervals& base, std::span<const std::pair<SSAValueID, SlotIndex>> earliestStart) {
    SSALiveIntervals adjusted = base;
    if (earliestStart.empty()) return adjusted;

    bool changed = false;
    for (const auto& [id, start] : earliestStart) {
        if (id == kInvalidSSAValueID || id >= adjusted.byValue_.size()) continue;
        const LiveRange& range = adjusted.byValue_[id];
        if (range.empty()) continue;

        const std::span<const LiveSegment> segments = range.segments();
        if (start >= segments.front().start) continue;  // already live by then

        // Segments are immutable once built, so rebuild the range with the
        // first one reaching further back and the rest carried over.
        LiveRange widened;
        widened.addSegment(start, segments.front().end);
        for (size_t i = 1; i < segments.size(); ++i) {
            widened.addSegment(segments[i].start, segments[i].end);
        }
        widened.finalize();
        adjusted.byValue_[id] = std::move(widened);
        changed = true;
    }

    if (changed) adjusted.recomputePeakPressure();
    return adjusted;
}

SSALiveIntervals computeSSALiveIntervals(const Function& function) {
    return SSALiveIntervalsBuilder::build(function, postOrderBlocks(function));
}

SSALiveIntervals computeSSALiveIntervals(const Function& function, const DominanceInfo& dominance) {
    return SSALiveIntervalsBuilder::build(function, postOrderBlocks(function, dominance));
}

}  // namespace stinkytofu
