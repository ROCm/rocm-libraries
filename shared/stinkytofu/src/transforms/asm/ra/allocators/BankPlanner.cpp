// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BankPlanner.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

namespace stinkytofu {
namespace {

/// The least weight a cost model gives a use inside a loop (BankCostModel.hpp).
constexpr double kInLoop = kBankCostLoopWeight;

/// Costs are whole numbers (BankCostModel.hpp), so a real improvement is at
/// least 1.
constexpr double kImprovement = -0.5;

/// The search stops after this many passes over the planned blocks or this
/// many cost evaluations, whichever comes first.
constexpr uint32_t kMaxPasses = 32;
constexpr uint64_t kMaxEvaluations = 100000;

/// Live registers per slot in one bank: add over a range of slots, and the
/// most over a range. Each node keeps its range's maximum including its own
/// pending add, so nothing is pushed down.
class BankLoad {
   public:
    explicit BankLoad(SlotIndex slots)
        : size_(std::max<SlotIndex>(slots, 1)), nodes_(4 * static_cast<size_t>(size_)) {}

    void add(SlotIndex start, SlotIndex end, int32_t amount) {
        end = std::min(end, size_);
        if (start < end) add(1, 0, size_, start, end, amount);
    }

    int32_t most(SlotIndex start, SlotIndex end) const {
        end = std::min(end, size_);
        return start < end ? most(1, 0, size_, start, end) : 0;
    }

   private:
    static constexpr int32_t kOutside = std::numeric_limits<int32_t>::min();

    struct Node {
        int32_t most = 0;
        int32_t pending = 0;
    };

    void add(size_t node, SlotIndex lo, SlotIndex hi, SlotIndex start, SlotIndex end,
             int32_t amount) {
        if (end <= lo || hi <= start) return;
        if (start <= lo && hi <= end) {
            nodes_[node].most += amount;
            nodes_[node].pending += amount;
            return;
        }
        const SlotIndex mid = lo + (hi - lo) / 2;
        add(2 * node, lo, mid, start, end, amount);
        add(2 * node + 1, mid, hi, start, end, amount);
        nodes_[node].most =
            std::max(nodes_[2 * node].most, nodes_[2 * node + 1].most) + nodes_[node].pending;
    }

    int32_t most(size_t node, SlotIndex lo, SlotIndex hi, SlotIndex start, SlotIndex end) const {
        if (end <= lo || hi <= start) return kOutside;
        if (start <= lo && hi <= end) return nodes_[node].most;
        const SlotIndex mid = lo + (hi - lo) / 2;
        const int32_t below =
            std::max(most(2 * node, lo, mid, start, end), most(2 * node + 1, mid, hi, start, end));
        return below == kOutside ? kOutside : below + nodes_[node].pending;
    }

    SlotIndex size_;
    std::vector<Node> nodes_;
};

/// Slots [start, end) in which a block holds \p count registers.
struct Piece {
    SlotIndex start = 0;
    SlotIndex end = 0;
    int32_t count = 0;
};

/// How many registers \p block holds over time: its members' live segments
/// summed, in as few pieces as the counts allow.
std::vector<Piece> footprintOf(const Block& block, const SSALiveIntervals& intervals) {
    std::vector<std::pair<SlotIndex, int32_t>> edges;
    for (const Member& member : block.members) {
        for (const LiveSegment& segment : intervals.rangeOf(member.value).segments()) {
            edges.emplace_back(segment.start, +1);
            edges.emplace_back(segment.end, -1);
        }
    }
    std::sort(edges.begin(), edges.end());
    std::vector<Piece> pieces;
    int32_t count = 0;
    SlotIndex from = 0;
    for (size_t index = 0; index < edges.size();) {
        const SlotIndex at = edges[index].first;
        if (count > 0 && at > from) {
            if (!pieces.empty() && pieces.back().end == from && pieces.back().count == count)
                pieces.back().end = at;
            else
                pieces.push_back({from, at, count});
        }
        while (index < edges.size() && edges[index].first == at) count += edges[index++].second;
        from = at;
    }
    return pieces;
}

struct Unit {
    size_t block = 0;
    /// The members the model prices, ascending.
    std::vector<SSAValueID> priced;
    /// The heaviest use among them.
    double weight = 0.0;
    std::vector<Piece> footprint;
};

class Planner {
   public:
    Planner(std::vector<Block>& blocks, const AllocationContext& context)
        : blocks_(blocks),
          context_(context),
          model_(*context.bankCost),
          bankOf_(context.function.ssaArena().valueCount() + 1, kUnknownBank) {
        const uint32_t indexes = context.target.indexCount(RegType::V);
        bankCount_ = static_cast<uint8_t>(
            std::min<uint32_t>((indexes + kVgprBankSize - 1) / kVgprBankSize, kUnknownBank));
        // A held register is lost to every other value for the whole function,
        // so it is left out of its bank's budget rather than counted as load.
        for (uint32_t bank = 0; bank < bankCount_; ++bank) {
            int32_t free = 0;
            const uint32_t end = std::min(indexes, (bank + 1) * kVgprBankSize);
            for (uint32_t idx = bank * kVgprBankSize; idx < end; ++idx) {
                if (context.target.isAllocatable(RegType::V, idx) &&
                    !context.scope.isPinnedRegister(RegType::V, idx))
                    ++free;
            }
            capacity_.push_back(free);
            loads_.emplace_back(context.intervals.slots().slotCount());
        }
    }

    uint32_t run() {
        if (bankCount_ == 0) return 0;
        fixBlocks();
        collectUnits();
        if (units_.empty()) return 0;
        unitBank_.assign(units_.size(), kUnknownBank);

        // Two starting points, kept by whichever leaves fewer blocks without a
        // bank and then costs less: the producer's banks, and a walk in program
        // order. A block left without one is placed wherever it fits, which the
        // model cannot foresee, so planning every block comes first.
        seedFromHints();
        const std::vector<uint8_t> fromHints = unitBank_;
        const std::pair<size_t, double> hintsScore = score();
        clearUnits();
        seedByCost();
        if (hintsScore < score()) {
            clearUnits();
            restore(fromHints);
        }
        improve();

        uint32_t planned = 0;
        for (size_t unit = 0; unit < units_.size(); ++unit) {
            if (unitBank_[unit] == kUnknownBank) continue;
            blocks_[units_[unit].block].plannedBank = unitBank_[unit];
            ++planned;
        }
        return planned;
    }

   private:
    const Block& blockOf(size_t unit) const {
        return blocks_[units_[unit].block];
    }

    std::optional<uint8_t> hintBankOf(size_t unit) const {
        const Block& block = blockOf(unit);
        if (!block.hintBase.has_value() || *block.hintBase / kVgprBankSize >= bankCount_)
            return std::nullopt;
        return static_cast<uint8_t>(*block.hintBase / kVgprBankSize);
    }

    /// Pinned blocks sit where their hints are, and capped blocks in bank 0:
    /// banks they take room in and that the model prices as known. A ceiling
    /// comes from an operand field with no bank selector, so it lies in bank 0;
    /// a block capped higher could sit in any bank below it, and stays undecided.
    void fixBlocks() {
        for (const Block& block : blocks_) {
            if (block.regClass != RegType::V) continue;
            if (block.pinReason != nullptr) {
                if (!block.hintBase.has_value()) continue;
                for (const Member& member : block.members) {
                    const uint32_t idx = *block.hintBase + member.offset;
                    const uint32_t bank = idx / kVgprBankSize;
                    if (bank >= bankCount_) continue;
                    bankOf_[member.value] = static_cast<uint8_t>(bank);
                    if (!context_.scope.isPinnedRegister(RegType::V, idx))
                        occupy(member.value, bank, +1);
                }
            } else if (block.maxIndex.has_value() && *block.maxIndex < kVgprBankSize) {
                for (const Member& member : block.members) {
                    bankOf_[member.value] = 0;
                    occupy(member.value, 0, +1);
                }
            }
        }
    }

    void collectUnits() {
        for (size_t index = 0; index < blocks_.size(); ++index) {
            const Block& block = blocks_[index];
            if (block.regClass != RegType::V || block.pinReason != nullptr ||
                block.maxIndex.has_value() || block.width > kVgprBankSize)
                continue;
            Unit unit;
            unit.block = index;
            for (const Member& member : block.members) {
                const double weight = model_.useWeight(member.value);
                if (weight <= 0.0) continue;
                unit.priced.push_back(member.value);
                unit.weight = std::max(unit.weight, weight);
            }
            if (unit.weight < kInLoop) continue;
            std::sort(unit.priced.begin(), unit.priced.end());
            unit.footprint = footprintOf(block, context_.intervals);
            units_.push_back(std::move(unit));
        }
        std::sort(units_.begin(), units_.end(), [this](const Unit& a, const Unit& b) {
            if (a.weight != b.weight) return a.weight > b.weight;
            return blocks_[a.block].leader < blocks_[b.block].leader;
        });
    }

    /// Unit indexes, most register-time first.
    std::vector<size_t> bySize() const {
        std::vector<size_t> order(units_.size());
        for (size_t unit = 0; unit < order.size(); ++unit) order[unit] = unit;
        std::sort(order.begin(), order.end(), [this](size_t a, size_t b) {
            const Block& lhs = blockOf(a);
            const Block& rhs = blockOf(b);
            const uint64_t lhsSize = static_cast<uint64_t>(lhs.width) * lhs.span;
            const uint64_t rhsSize = static_cast<uint64_t>(rhs.width) * rhs.span;
            if (lhsSize != rhsSize) return lhsSize > rhsSize;
            return lhs.leader < rhs.leader;
        });
        return order;
    }

    /// Unit indexes, in the order they become live.
    std::vector<size_t> byStart() const {
        std::vector<size_t> order(units_.size());
        for (size_t unit = 0; unit < order.size(); ++unit) order[unit] = unit;
        std::sort(order.begin(), order.end(), [this](size_t a, size_t b) {
            const Block& lhs = blockOf(a);
            const Block& rhs = blockOf(b);
            if (lhs.start != rhs.start) return lhs.start < rhs.start;
            return lhs.leader < rhs.leader;
        });
        return order;
    }

    void occupy(SSAValueID value, uint32_t bank, int32_t amount) {
        for (const LiveSegment& segment : context_.intervals.rangeOf(value).segments())
            loads_[bank].add(segment.start, segment.end, amount);
    }

    void occupyUnit(size_t unit, uint32_t bank, int32_t amount) {
        for (const Piece& piece : units_[unit].footprint)
            loads_[bank].add(piece.start, piece.end, piece.count * amount);
    }

    bool fitsIn(size_t unit, uint32_t bank) const {
        for (const Piece& piece : units_[unit].footprint) {
            if (loads_[bank].most(piece.start, piece.end) > capacity_[bank]) return false;
        }
        return true;
    }

    void assign(size_t unit, uint8_t bank) {
        unitBank_[unit] = bank;
        for (const SSAValueID value : units_[unit].priced) bankOf_[value] = bank;
    }

    /// Put \p unit, in no bank yet, into \p bank if it has room.
    bool tryPlaceUnit(size_t unit, uint8_t bank) {
        occupyUnit(unit, bank, +1);
        if (fitsIn(unit, bank)) {
            assign(unit, bank);
            return true;
        }
        occupyUnit(unit, bank, -1);
        return false;
    }

    void removeUnit(size_t unit) {
        if (unitBank_[unit] == kUnknownBank) return;
        occupyUnit(unit, unitBank_[unit], -1);
        assign(unit, kUnknownBank);
    }

    bool tryMove(size_t unit, uint8_t to) {
        const uint8_t from = unitBank_[unit];
        removeUnit(unit);
        if (tryPlaceUnit(unit, to)) return true;
        if (from != kUnknownBank) {
            [[maybe_unused]] const bool back = tryPlaceUnit(unit, from);
            assert(back && "a unit no longer fits the bank it just left");
        }
        return false;
    }

    void clearUnits() {
        for (size_t unit = 0; unit < units_.size(); ++unit) removeUnit(unit);
    }

    void restore(const std::vector<uint8_t>& banks) {
        for (size_t unit = 0; unit < units_.size(); ++unit) {
            if (banks[unit] == kUnknownBank) continue;
            [[maybe_unused]] const bool placed = tryPlaceUnit(unit, banks[unit]);
            assert(placed && "a plan that fitted once no longer fits");
        }
    }

    /// Blocks left without a bank, then the cost: lower is better.
    std::pair<size_t, double> score() const {
        const auto unplanned =
            static_cast<size_t>(std::count(unitBank_.begin(), unitBank_.end(), kUnknownBank));
        return {unplanned, model_.cost(bankOf_)};
    }

    /// The bank of the producer's register, largest blocks first, or the next
    /// bank with room.
    void seedFromHints() {
        for (const size_t unit : bySize()) {
            std::vector<uint8_t> order;
            if (const std::optional<uint8_t> hinted = hintBankOf(unit)) order.push_back(*hinted);
            for (uint8_t bank = 0; bank < bankCount_; ++bank) {
                if (order.empty() || bank != order.front()) order.push_back(bank);
            }
            for (const uint8_t bank : order) {
                if (tryPlaceUnit(unit, bank)) break;
            }
        }
    }

    /// In the order blocks become live, the bank that costs least next to the
    /// blocks already given one. A block not yet given one is undecided, which
    /// prices every bank alike, so this follows the neighbours already chosen.
    void seedByCost() {
        for (const size_t unit : byStart()) {
            const std::vector<std::pair<double, uint8_t>> choices = cheapestBanks(unit);
            for (const auto& [change, bank] : choices) {
                if (tryPlaceUnit(unit, bank)) break;
            }
        }
    }

    /// Every bank for \p unit, cheapest first; ties go to the producer's bank,
    /// then the lower one.
    std::vector<std::pair<double, uint8_t>> cheapestBanks(size_t unit) {
        std::vector<std::pair<double, uint8_t>> choices;
        for (uint8_t bank = 0; bank < bankCount_; ++bank) {
            choices.emplace_back(model_.delta(bankOf_, units_[unit].priced, bank), bank);
            ++evaluations_;
        }
        const std::optional<uint8_t> hinted = hintBankOf(unit);
        std::sort(choices.begin(), choices.end(), [&](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            if ((hinted == a.second) != (hinted == b.second)) return hinted == a.second;
            return a.second < b.second;
        });
        return choices;
    }

    /// Move each block to a cheaper bank with room. A cheaper bank that is full
    /// gets a second chance as a swap with a block of the same width, which
    /// makes room both ways at once. A block no bank had room for when it was
    /// seeded takes the cheapest one that has room now.
    void improve() {
        for (uint32_t pass = 0; pass < kMaxPasses && evaluations_ < kMaxEvaluations; ++pass) {
            bool improved = false;
            std::vector<std::pair<size_t, uint8_t>> blocked;
            for (size_t unit = 0; unit < units_.size(); ++unit) {
                const uint8_t from = unitBank_[unit];
                if (from == kUnknownBank) {
                    for (const auto& [change, bank] : cheapestBanks(unit)) {
                        if (tryPlaceUnit(unit, bank)) {
                            improved = true;
                            break;
                        }
                    }
                    continue;
                }
                std::vector<std::pair<double, uint8_t>> cheaper;
                for (uint8_t to = 0; to < bankCount_; ++to) {
                    if (to == from) continue;
                    const double change = model_.delta(bankOf_, units_[unit].priced, to);
                    ++evaluations_;
                    if (change < kImprovement) cheaper.emplace_back(change, to);
                }
                std::sort(cheaper.begin(), cheaper.end());
                for (const auto& [change, to] : cheaper) {
                    if (tryMove(unit, to)) {
                        improved = true;
                        break;
                    }
                    blocked.emplace_back(unit, to);
                }
            }
            for (const auto& [unit, to] : blocked) {
                if (evaluations_ >= kMaxEvaluations) break;
                if (unitBank_[unit] == to || unitBank_[unit] == kUnknownBank) continue;
                for (size_t other = 0; other < units_.size(); ++other) {
                    if (other == unit || unitBank_[other] != to) continue;
                    if (blockOf(other).width != blockOf(unit).width) continue;
                    if (trySwap(unit, other)) {
                        improved = true;
                        break;
                    }
                }
            }
            if (!improved) break;
        }
    }

    /// Exchange the banks of two blocks of one width when that costs less and
    /// both fit.
    bool trySwap(size_t unit, size_t other) {
        const uint8_t here = unitBank_[unit];
        const uint8_t there = unitBank_[other];
        const double first = model_.delta(bankOf_, units_[unit].priced, there);
        for (const SSAValueID value : units_[unit].priced) bankOf_[value] = there;
        const double second = model_.delta(bankOf_, units_[other].priced, here);
        for (const SSAValueID value : units_[unit].priced) bankOf_[value] = here;
        evaluations_ += 2;
        if (first + second >= kImprovement) return false;

        removeUnit(unit);
        removeUnit(other);
        if (tryPlaceUnit(unit, there)) {
            if (tryPlaceUnit(other, here)) return true;
            removeUnit(unit);
        }
        [[maybe_unused]] const bool unitBack = tryPlaceUnit(unit, here);
        [[maybe_unused]] const bool otherBack = tryPlaceUnit(other, there);
        assert(unitBack && otherBack && "a swap that failed could not be undone");
        return false;
    }

    std::vector<Block>& blocks_;
    const AllocationContext& context_;
    const BankCostModel& model_;
    std::vector<uint8_t> bankOf_;
    uint8_t bankCount_ = 0;
    std::vector<int32_t> capacity_;
    std::vector<BankLoad> loads_;
    std::vector<Unit> units_;
    std::vector<uint8_t> unitBank_;
    uint64_t evaluations_ = 0;
};

}  // namespace

uint32_t planBanks(std::vector<Block>& blocks, const AllocationContext& context) {
    if (context.bankCost == nullptr) return 0;
    return Planner(blocks, context).run();
}

}  // namespace stinkytofu
